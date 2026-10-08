"""Local OpenAI CLIP scene recognition; Habitat never imports torch or CLIP.

The persistent worker reuses the existing omnivla environment and local weights.
Class scores are relative to the supplied text labels, not calibrated accuracy.
"""

import argparse
import asyncio
import base64
import contextlib
import json
import math
import os
import sys
from dataclasses import asdict
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from .semantics import (
    RoomSemanticClassifier,
    RoomSemanticResult,
    SemanticClassificationError,
    SemanticFrame,
    _validate_frames,
    _validate_result,
)

DEFAULT_PYTHON = str(Path.home() / ".conda/envs/omnivla/bin/python")
DEFAULT_MODEL = str(Path.home() / ".cache/clip/ViT-B-32.pt")
DEFAULT_LABELS = (
    "bedroom",
    "living room",
    "bathroom",
    "kitchen",
    "dining room",
    "office room",
    "gym",
    "lounge",
    "laundry room",
    "corridor",
)
DEFAULT_TEMPLATES = (
    "a photo of a {}.",
    "a photo of the interior of a {}.",
    "an indoor scene of a {}.",
)


def validate_options(labels, templates):
    if (
        not isinstance(labels, (list, tuple))
        or not 2 <= len(labels) <= 64
        or any(
            not isinstance(s, str) or not s.strip() or len(s) > 80 or s == "unknown" for s in labels
        )
        or len(set(labels)) != len(labels)
    ):
        raise ValueError("CLIP labels must contain 2..64 unique, nonempty scene names")
    if (
        not isinstance(templates, (list, tuple))
        or not 1 <= len(templates) <= 8
        or any(not isinstance(s, str) or s.count("{}") != 1 or len(s) > 240 for s in templates)
    ):
        raise ValueError("CLIP templates require 1..8 prompts with one {} each")


def fuse_predictions(frames, labels, probabilities):
    scores = np.asarray(probabilities, dtype=float)
    if (
        not 1 <= len(frames) <= 5
        or scores.shape != (len(frames), len(labels))
        or not np.isfinite(scores).all()
        or (scores < 0).any()
        or not np.allclose(scores.sum(axis=1), 1.0, atol=1e-5)
    ):
        raise ValueError("invalid CLIP per-view probabilities")
    pooled = scores.mean(axis=0)
    pooled /= pooled.sum()
    winner = int(pooled.argmax())
    return RoomSemanticResult(
        labels[winner],
        dict(zip(labels, pooled.tolist())),
        float(pooled[winner]),
        frames[int(scores[:, winner].argmax())].frame_id,
    )


class _CLIPModel:
    def __init__(self, model_path, labels, templates, device, num_threads):
        # Only a local file is accepted; clip.load must not download during navigation.
        if not Path(model_path).is_file():
            raise FileNotFoundError("CLIP local checkpoint not found: " + model_path)
        import clip
        import torch

        validate_options(labels, templates)
        torch.set_num_threads(num_threads)
        self.torch, self.device, self.labels = torch, device, labels
        self.model, self.preprocess = clip.load(model_path, device=device, jit=False)
        self.model.eval()
        prompts = [template.replace("{}", label) for label in labels for template in templates]
        tokens = clip.tokenize(prompts).to(device)
        with torch.inference_mode():
            features = self.model.encode_text(tokens).float()
            features /= features.norm(dim=-1, keepdim=True)
            features = features.reshape(len(labels), len(templates), -1).mean(dim=1)
            self.text_features = features / features.norm(dim=-1, keepdim=True)

    def classify(self, frames):
        _validate_frames(frames)
        images = []
        for frame in frames:
            with Image.open(BytesIO(frame.jpeg)) as image:
                images.append(self.preprocess(image.convert("RGB")))
        with self.torch.inference_mode():
            features = self.model.encode_image(self.torch.stack(images).to(self.device)).float()
            features /= features.norm(dim=-1, keepdim=True)
            logits = self.model.logit_scale.exp() * (features @ self.text_features.T)
            probabilities = logits.softmax(dim=-1).cpu().numpy()
        return fuse_predictions(frames, self.labels, probabilities)


class CLIPRecognizer(RoomSemanticClassifier):
    def __init__(
        self,
        model_path=DEFAULT_MODEL,
        python_executable=DEFAULT_PYTHON,
        labels=DEFAULT_LABELS,
        templates=DEFAULT_TEMPLATES,
        device="cpu",
        num_threads=2,
        timeout=60.0,
    ):
        validate_options(labels, templates)
        if not isinstance(device, str) or not (
            device in ("cpu", "cuda") or (device.startswith("cuda:") and device[5:].isdigit())
        ):
            raise ValueError("CLIP device must be cpu, cuda or cuda:N")
        if type(num_threads) is not int or not 1 <= num_threads <= 32:
            raise ValueError("CLIP num_threads must be 1..32")
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("CLIP timeout must be finite and positive")
        self.options = dict(
            model_path=str(Path(model_path).expanduser().resolve()),
            labels=list(labels),
            templates=list(templates),
            device=device,
            num_threads=num_threads,
        )
        self.python_executable = str(Path(python_executable).expanduser())
        self.timeout = timeout
        self.process = None
        self._lock = None
        self._closed = False
        self._request_task = None

    def metadata(self):
        return dict(
            backend="clip",
            **self.options,
            aggregation="mean_per_view_probabilities",
            score_kind="softmax_over_candidate_labels",
        )

    async def classify(self, frames):
        frames = tuple(frames)
        _validate_frames(frames)
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            if self._closed:
                raise SemanticClassificationError("CLIP recognizer is closed")
            try:
                self._request_task = asyncio.create_task(self._remote(frames))
                return await asyncio.wait_for(self._request_task, self.timeout)
            except asyncio.CancelledError:
                await self._stop_process()
                raise
            except asyncio.TimeoutError:
                await self._stop_process()
                raise SemanticClassificationError("CLIP inference timed out") from None
            except Exception as exc:
                await self._stop_process()
                if isinstance(exc, SemanticClassificationError):
                    raise
                raise SemanticClassificationError(
                    f"CLIP inference failed: {type(exc).__name__}: {str(exc)[:140]}"
                ) from None
            finally:
                self._request_task = None

    async def _read_response(self):
        line = await self.process.stdout.readline()
        if not line:
            raise SemanticClassificationError("CLIP worker exited before returning a result")
        result = json.loads(line)
        if not isinstance(result, dict):
            raise SemanticClassificationError("CLIP worker returned an invalid response")
        if "error" in result:
            raise SemanticClassificationError("CLIP: " + str(result["error"])[:160])
        return result

    async def _remote(self, frames):
        if self.process is None or self.process.returncode is not None:
            env = dict(
                os.environ,
                PYTHONPATH=str(Path(__file__).resolve().parents[1]),
                PYTHONNOUSERSITE="1",
            )
            if self.options["device"] == "cpu":
                env["CUDA_VISIBLE_DEVICES"] = ""
            self.process = await asyncio.create_subprocess_exec(
                self.python_executable,
                "-u",
                "-m",
                "room_memory.clip_recognizer",
                "--worker",
                "--options",
                json.dumps(self.options),
                env=env,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                limit=65536,
            )
            if (await self._read_response()).get("ready") is not True:
                raise SemanticClassificationError("CLIP worker did not become ready")
        request = [
            dict(frame_id=f.frame_id, jpeg=base64.b64encode(f.jpeg).decode("ascii")) for f in frames
        ]
        self.process.stdin.write((json.dumps(request) + "\n").encode())
        await self.process.stdin.drain()
        return _validate_result(
            await self._read_response(), {f.frame_id for f in frames}, self.options["labels"]
        )

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
        request = self._request_task
        if request is not None:
            request.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await request
        if self._lock is None:
            await self._stop_process()
        else:
            # classify owns cancellation cleanup; wait until its child has exited.
            async with self._lock:
                await self._stop_process()


def main():
    parser = argparse.ArgumentParser(
        description="Local CLIP room Top-1 over a fixed scene vocabulary"
    )
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--options", default="{}", help=argparse.SUPPRESS)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("images", nargs="*")
    args = parser.parse_args()
    try:
        options = dict(
            model_path=args.model_path,
            device=args.device,
            num_threads=args.num_threads,
            labels=list(DEFAULT_LABELS),
            templates=list(DEFAULT_TEMPLATES),
        )
        options.update(json.loads(args.options))
        model = _CLIPModel(**options)
        if not args.worker:
            frames = [
                SemanticFrame(f"F{i}", Path(path).read_bytes())
                for i, path in enumerate(args.images)
            ]
            print(json.dumps(asdict(model.classify(frames))))
            return
        print(json.dumps({"ready": True}), flush=True)
        while True:
            line = sys.stdin.buffer.readline(70_000_001)
            if not line:
                break
            if len(line) > 70_000_000:
                raise ValueError("request exceeds 70 MB")
            request = json.loads(line)
            if not isinstance(request, list) or not 1 <= len(request) <= 5:
                raise ValueError("expected 1 to 5 frames")
            frames = [
                SemanticFrame(item["frame_id"], base64.b64decode(item["jpeg"], validate=True))
                for item in request
            ]
            print(json.dumps(asdict(model.classify(frames))), flush=True)
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {str(exc)[:160]}"}), flush=True)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
