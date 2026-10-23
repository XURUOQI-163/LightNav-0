"""Official Grounding DINO Swin-T in an isolated persistent GPU worker.

Boxes are region hypotheses, not room boundaries. Scores are mean sigmoid token
similarities over the matching room phrase, never softmax class probabilities.
"""
import argparse
import asyncio
import base64
import contextlib
import json
import math
import os
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .semantics import (
    RoomSemanticClassifier,
    RoomSemanticResult,
    SemanticClassificationError,
    SemanticFrame,
    _validate_frames,
)

DEFAULT_REPO = str(Path(__file__).resolve().parents[3] / 'GroundingDINO')
DEFAULT_MODEL = str(Path(DEFAULT_REPO) / 'weights/groundingdino_swint_ogc.pth')
DEFAULT_CONFIG = str(Path(DEFAULT_REPO) / 'groundingdino/config/GroundingDINO_SwinT_OGC.py')
DEFAULT_BERT = str(Path(DEFAULT_REPO) / 'weights/bert-base-uncased')
DEFAULT_PYTHON = str(Path.home() / '.conda/envs/groundingdino/bin/python')
DEFAULT_LABELS = ('bedroom', 'living room', 'bathroom', 'kitchen', 'dining room', 'corridor')


def decode_regions(boxes, scores, labels, width, height, threshold):
    boxes, scores = np.asarray(boxes, float), np.asarray(scores, float)
    if (boxes.ndim != 2 or boxes.shape[1] != 4 or scores.shape != (len(boxes), len(labels))
            or min(width, height) <= 0 or not 0 <= threshold <= 1
            or not np.isfinite(boxes).all() or not np.isfinite(scores).all()
            or (scores < 0).any() or (scores > 1).any()):
        raise ValueError('invalid detector boxes/scores/image size')
    detections = []
    for box, vector in zip(boxes, scores):
        winner = int(vector.argmax())
        score = float(vector[winner])
        if score < threshold:
            continue
        cx, cy, w, h = box
        x1, x2 = np.clip([(cx - w / 2) * width, (cx + w / 2) * width], 0, width)
        y1, y2 = np.clip([(cy - h / 2) * height, (cy + h / 2) * height], 0, height)
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append(dict(bbox_xyxy=[float(x1), float(y1), float(x2), float(y2)],
            center_pixel=[float((x1 + x2) / 2), float((y1 + y2) / 2)],
            room_type=labels[winner], score=score,
            class_scores=dict(zip(labels, vector.tolist()))))
    return sorted(detections, key=lambda d: d['score'], reverse=True)


def summarize_frames(results, labels=DEFAULT_LABELS):
    per_view = []
    for frame in results:
        per_view.append([max((d['score'] for d in frame['detections'] if d['room_type'] == label), default=0.)
                         for label in labels])
    pooled = np.mean(per_view, axis=0)
    if pooled.sum() <= 0:
        return RoomSemanticResult('unknown', {'unknown': 1.}, 0., results[0]['frame_id'])
    winner = int(pooled.argmax())
    representative = int(np.argmax(np.asarray(per_view)[:, winner]))
    return RoomSemanticResult(labels[winner], dict(zip(labels, (pooled / pooled.sum()).tolist())),
                              float(pooled[winner]), results[representative]['frame_id'])


def annotate(jpeg, detections):
    from .semantic_projection import ROOM_COLORS
    image = Image.open(BytesIO(jpeg)).convert('RGB')
    draw = ImageDraw.Draw(image)
    for item in reversed(detections):
        box = item['bbox_xyxy']
        color = ROOM_COLORS.get(item['room_type'], '#ffffff')
        draw.rectangle(box, outline=color, width=max(2, image.width // 320))
        label = f"{item['room_type']} {item['score']:.3f}"
        x, y = box[0], max(0, box[1] - 14)
        bounds = draw.textbbox((x, y), label)
        draw.rectangle(bounds, fill=color)
        draw.text((x, y), label, fill='black')
    out = BytesIO()
    image.save(out, 'JPEG', quality=90)
    return out.getvalue()


class GroundingModel:
    def __init__(self, model_path=DEFAULT_MODEL, config_path=DEFAULT_CONFIG,
                 bert_path=DEFAULT_BERT, labels=DEFAULT_LABELS, device='cuda:0',
                 num_threads=2, box_threshold=.25):
        import torch
        from groundingdino import _C
        from groundingdino.datasets import transforms as T
        from groundingdino.models import build_model
        from groundingdino.util.misc import clean_state_dict
        from groundingdino.util.slconfig import SLConfig

        if not device.startswith('cuda') or not torch.cuda.is_available():
            raise RuntimeError('Grounding DINO GPU inference required; CUDA unavailable')
        for path in (model_path, config_path, bert_path):
            if not Path(path).exists():
                raise FileNotFoundError(path)
        torch.set_num_threads(num_threads)
        self.torch, self.device, self.labels = torch, device, list(labels)
        self.box_threshold = box_threshold
        args = SLConfig.fromfile(config_path)
        args.device = device
        args.text_encoder_type = bert_path  # Offline copy of the official BERT encoder.
        self.model = build_model(args)
        state = torch.load(model_path, map_location='cpu')
        incompatible = self.model.load_state_dict(clean_state_dict(state['model']), strict=False)
        if incompatible.missing_keys:
            raise RuntimeError('checkpoint missing model parameters: ' + str(incompatible.missing_keys[:8]))
        self.model.eval().to(device)
        self.transform = T.Compose([T.RandomResize([800], max_size=1333), T.ToTensor(),
                                    T.Normalize([.485, .456, .406], [.229, .224, .225])])
        self.caption = ' . '.join(self.labels) + ' .'
        encoded = self.model.tokenizer(self.caption)
        self.token_indices = []
        offset = 0
        for label in self.labels:
            indices = sorted({encoded.char_to_token(i) for i in range(offset, offset + len(label))
                              if encoded.char_to_token(i) is not None})
            if not indices:
                raise ValueError('empty token span: ' + label)
            self.token_indices.append(indices)
            offset += len(label) + 3
        self.extension_path = _C.__file__

    def detect(self, frame):
        image = Image.open(BytesIO(frame.jpeg)).convert('RGB')
        tensor, _ = self.transform(image, None)
        with self.torch.inference_mode():
            output = self.model(tensor[None].to(self.device), captions=[self.caption])
            logits = output['pred_logits'][0].sigmoid()
            scores = self.torch.stack([logits[:, indices].mean(-1) for indices in self.token_indices], -1)
            detections = decode_regions(output['pred_boxes'][0].cpu().numpy(), scores.cpu().numpy(),
                                        self.labels, *image.size, self.box_threshold)
        return dict(frame_id=frame.frame_id, width=image.width, height=image.height,
                    detections=detections, prompt=self.caption, device=self.device,
                    score_kind='mean_sigmoid_phrase_tokens', cuda_extension=self.extension_path)


class GroundingDINORecognizer(RoomSemanticClassifier):
    def __init__(self, python_executable=DEFAULT_PYTHON, model_path=DEFAULT_MODEL,
                 config_path=DEFAULT_CONFIG, bert_path=DEFAULT_BERT, labels=DEFAULT_LABELS,
                 device='cuda:0', num_threads=2, box_threshold=.25, timeout=120.):
        if (not 2 <= len(labels) <= 16 or len(set(labels)) != len(labels)
                or any(not isinstance(s, str) or not s.strip() or '.' in s or len(s) > 60 for s in labels)):
            raise ValueError('invalid room labels')
        if not isinstance(device, str) or not (device == 'cuda' or (device.startswith('cuda:') and device[5:].isdigit())):
            raise ValueError('Grounding DINO device must be cuda or cuda:N')
        if type(num_threads) is not int or not 1 <= num_threads <= 16:
            raise ValueError('num_threads must be 1..16')
        if not math.isfinite(box_threshold) or not 0 < box_threshold <= 1 or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('invalid threshold/timeout')
        self.options = dict(model_path=str(Path(model_path).expanduser().resolve()),
            config_path=str(Path(config_path).expanduser().resolve()), bert_path=str(Path(bert_path).expanduser().resolve()),
            labels=list(labels), device=device, num_threads=num_threads, box_threshold=box_threshold)
        self.python_executable, self.timeout = python_executable, timeout
        self.process = self._lock = self._request_task = None
        self._closed = False

    def metadata(self):
        return dict(backend='groundingdino', **self.options, score_kind='mean_sigmoid_phrase_tokens',
                    aggregation='mean_per_view_max_detected_score')

    async def classify(self, frames):
        results = await self._request(frames)
        return summarize_frames(results, self.options['labels'])

    async def detect(self, frame):
        return (await self._request([frame]))[0]

    async def _request(self, frames):
        frames = tuple(frames)
        _validate_frames(frames)
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            if self._closed:
                raise SemanticClassificationError('Grounding DINO recognizer closed')
            try:
                self._request_task = asyncio.create_task(self._remote(frames))
                return await asyncio.wait_for(self._request_task, self.timeout)
            except asyncio.TimeoutError:
                await self._stop_process()
                raise SemanticClassificationError(f"Grounding DINO inference timed out after {self.timeout:g}s") from None
            except BaseException:
                await self._stop_process()
                raise
            finally:
                self._request_task = None

    async def _read_response(self):
        line = await self.process.stdout.readline()
        if not line:
            raise SemanticClassificationError('Grounding DINO worker exited')
        value = json.loads(line)
        if 'error' in value:
            raise SemanticClassificationError('Grounding DINO: ' + value['error'])
        return value

    async def _remote(self, frames):
        if self.process is None or self.process.returncode is not None:
            env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]),
                       PYTHONNOUSERSITE='1', HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
            self.process = await asyncio.create_subprocess_exec(self.python_executable, '-u', '-m',
                'room_memory.groundingdino_recognizer', '--worker', '--options', json.dumps(self.options),
                env=env, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, limit=16_000_000)
            if not (await self._read_response()).get('ready'):
                raise SemanticClassificationError('Grounding DINO worker not ready')
        request = [dict(frame_id=f.frame_id, jpeg=base64.b64encode(f.jpeg).decode('ascii')) for f in frames]
        self.process.stdin.write((json.dumps(request) + '\n').encode())
        await self.process.stdin.drain()
        result = (await self._read_response())['frames']
        if [r['frame_id'] for r in result] != [f.frame_id for f in frames]:
            raise SemanticClassificationError('Grounding DINO frame identity mismatch')
        return result

    async def _stop_process(self):
        process, self.process = self.process, None
        if process is not None:
            if process.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    process.terminate()
            try:
                await asyncio.wait_for(process.wait(), 2)
            except asyncio.TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
                await process.wait()

    async def aclose(self):
        self._closed = True
        if self._request_task is not None:
            self._request_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._request_task
        if self._lock is None:
            await self._stop_process()
        else:
            async with self._lock:
                await self._stop_process()


def main():
    parser = argparse.ArgumentParser(description='Official Swin-T room-region detection on one RGB image')
    parser.add_argument('image', nargs='?')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--box-threshold', type=float, default=.25)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--annotated', type=Path)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--options', default='{}', help=argparse.SUPPRESS)
    args = parser.parse_args()
    options = dict(device=args.device, box_threshold=args.box_threshold)
    options.update(json.loads(args.options))
    try:
        # Official model prints diagnostics; stdout is reserved for JSON IPC.
        with contextlib.redirect_stdout(sys.stderr):
            model = GroundingModel(**options)
        if args.worker:
            print(json.dumps({'ready': True}), flush=True)
            while True:
                line = sys.stdin.buffer.readline(70_000_001)
                if not line:
                    break
                if len(line) > 70_000_000:
                    raise ValueError('request exceeds 70 MB')
                request = json.loads(line)
                frames = [SemanticFrame(f['frame_id'], base64.b64decode(f['jpeg'], validate=True)) for f in request]
                _validate_frames(frames)
                with contextlib.redirect_stdout(sys.stderr):
                    result = [model.detect(f) for f in frames]
                print(json.dumps({'frames': result}, allow_nan=False), flush=True)
        else:
            if not args.image:
                parser.error('provide an RGB image')
            frame = SemanticFrame('F0', Path(args.image).read_bytes())
            with contextlib.redirect_stdout(sys.stderr):
                result = model.detect(frame)
            value = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
            print(value)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(value)
            if args.annotated:
                args.annotated.parent.mkdir(parents=True, exist_ok=True)
                args.annotated.write_bytes(annotate(frame.jpeg, result['detections']))
    except Exception as exc:
        print(json.dumps({'error': f'{type(exc).__name__}: {str(exc)[:300]}'}), flush=True)
        raise


if __name__ == '__main__':
    main()
