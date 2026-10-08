"""Local Places365 ResNet18 inference behind the room semantic interface.

The default persistent stdio worker uses the existing places365 conda Python;
Habitat needs neither torch nor torchvision. No HTTP port or downloads are used.
Preprocessing/checkpoint loading follows CSAILVision/places365's basic demo;
see config/PLACES365_LICENSE.txt for its MIT notice.
"""

import argparse
import asyncio
import base64
import contextlib
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from .semantics import (
    DEFAULT_ROOM_TYPES,
    RoomSemanticClassifier,
    RoomSemanticResult,
    SemanticClassificationError,
    SemanticFrame,
    _validate_frames,
    _validate_result,
)

DEFAULT_REPO = str(Path.home() / "projects/places365")
DEFAULT_PYTHON = str(Path.home() / ".conda/envs/places365/bin/python")
DEFAULT_MAPPING = Path(__file__).parent / "config/places365_room_mapping.json"


def load_mapping(path=None):
    mapping = json.loads(Path(path or DEFAULT_MAPPING).expanduser().read_text())
    if (
        not isinstance(mapping, dict)
        or not mapping
        or any(
            not isinstance(label, str) or not label or room not in DEFAULT_ROOM_TYPES
            for label, room in mapping.items()
        )
    ):
        raise ValueError("Places365 mapping must map scene names to room taxonomy labels")
    return mapping


def aggregate_predictions(frames, labels, probabilities, mapping=None):
    """Return the raw scene Top-1 after averaging views, without label/score filtering.

    ``mapping`` remains accepted for callers of the older adapter; it no longer
    changes predictions. A low Top-1 score is retained as a score, never unknown.
    """
    scores = np.asarray(probabilities, dtype=float)
    if (
        not 1 <= len(frames) <= 5
        or not labels
        or len(set(labels)) != len(labels)
        or scores.shape != (len(frames), len(labels))
        or not np.isfinite(scores).all()
        or (scores < 0).any()
        or not np.allclose(scores.sum(axis=1), 1.0, atol=1e-5)
    ):
        raise ValueError("invalid Places365 per-view probabilities")
    pooled = scores.mean(axis=0)
    pooled /= pooled.sum()  # Remove softmax floating-point normalization drift.
    winner = int(pooled.argmax())
    representative = frames[int(scores[:, winner].argmax())].frame_id
    return RoomSemanticResult(
        labels[winner],
        dict(zip(labels, pooled.tolist())),
        float(pooled[winner]),
        representative,
    )


def load_labels(repo_path):
    entries = [
        line.split()
        for line in (Path(repo_path) / "categories_places365.txt").read_text().splitlines()
    ]
    if len(entries) != 365 or any(
        len(row) != 2 or int(row[1]) != i for i, row in enumerate(entries)
    ):
        raise ValueError("Places365 categories must have 365 ordered indices")
    labels = [row[0][3:] for row in entries]
    if len(set(labels)) != 365 or any(not label or label == "unknown" for label in labels):
        raise ValueError("Places365 categories must have 365 unique scene names")
    return labels


class _PlacesModel:
    """Only constructed in the inference worker (or the opt-in local thread)."""

    def __init__(self, repo_path, device, num_threads, mapping_path):
        import torch
        from torchvision import models, transforms

        torch.set_num_threads(num_threads)
        self.torch = torch
        self.device = torch.device(device)
        repo = Path(repo_path)
        self.mapping = load_mapping(mapping_path)
        self.labels = load_labels(repo)
        if set(self.mapping) - set(self.labels):
            raise ValueError("Places365 mapping contains unknown scene labels")
        self.model = models.resnet18(weights=None, num_classes=365)
        checkpoint = torch.load(
            repo / "resnet18_places365.pth.tar", map_location="cpu", weights_only=True
        )
        state = {
            key.removeprefix("module."): value for key, value in checkpoint["state_dict"].items()
        }
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def classify(self, frames):
        probabilities = self.predict_probabilities(frames)
        return aggregate_predictions(frames, self.labels, probabilities, self.mapping)

    def predict_probabilities(self, frames):
        _validate_frames(frames)
        tensors = []
        for frame in frames:
            with Image.open(BytesIO(frame.jpeg)) as image:
                tensors.append(self.transform(image.convert("RGB")))
        with self.torch.inference_mode():
            logits = self.model(self.torch.stack(tensors).to(self.device))
            probabilities = self.torch.softmax(logits, dim=1).cpu().numpy()
        return probabilities


class Places365Recognizer(RoomSemanticClassifier):
    """Async adapter with bounded requests, persistent weights and explicit cleanup.

    mode='subprocess' reuses another Python environment without importing torch in
    Habitat. mode='in_process' is for a Python that already has compatible torch
    and torchvision; inference still runs off the asyncio/navigation thread.
    """

    def __init__(
        self,
        repo_path=DEFAULT_REPO,
        python_executable=DEFAULT_PYTHON,
        mode="subprocess",
        device="cpu",
        num_threads=2,
        timeout=60.0,
        mapping_path=None,
    ):
        if mode not in ("subprocess", "in_process"):
            raise ValueError("Places365 mode must be subprocess or in_process")
        if not isinstance(device, str) or not (
            device == "cpu"
            or device == "cuda"
            or (device.startswith("cuda:") and device[5:].isdigit())
        ):
            raise ValueError("Places365 device must be cpu, cuda, or cuda:N")
        if (
            isinstance(num_threads, bool)
            or not isinstance(num_threads, int)
            or not 1 <= num_threads <= 32
        ):
            raise ValueError("Places365 num_threads must be between 1 and 32")
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("Places365 timeout must be positive and finite")
        self.repo_path = str(Path(repo_path).expanduser().resolve())
        self.python_executable = str(Path(python_executable).expanduser())
        self.mapping_path = str(Path(mapping_path or DEFAULT_MAPPING).expanduser().resolve())
        load_mapping(self.mapping_path)  # Validate the small mapping before starting Habitat.
        self.mode, self.device, self.num_threads, self.timeout = mode, device, num_threads, timeout
        self.process = None
        self._lock = None  # Bind to the running loop, including on Python 3.9.
        self._executor = None
        self._model = None
        self._labels = None
        self._closed = False

    def metadata(self):
        return dict(
            backend="places365",
            model_path=str(Path(self.repo_path) / "resnet18_places365.pth.tar"),
            label_space="places365_365_raw",
            device=self.device,
            aggregation="mean_per_view_probabilities",
            score_kind="softmax_over_365_scenes",
        )

    async def classify(self, frames):
        frames = tuple(frames)
        _validate_frames(frames)
        if self._closed:
            raise SemanticClassificationError("Places365 recognizer is closed")
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            try:
                if self.mode == "in_process":
                    if self._executor is None:
                        self._executor = ThreadPoolExecutor(
                            max_workers=1, thread_name_prefix="places365"
                        )
                    return await asyncio.wait_for(
                        asyncio.get_running_loop().run_in_executor(
                            self._executor, self._local, frames
                        ),
                        self.timeout,
                    )
                return await asyncio.wait_for(self._remote(frames), self.timeout)
            except asyncio.CancelledError:
                await self._stop_process()
                raise
            except asyncio.TimeoutError:
                await self._stop_process()
                raise SemanticClassificationError("Places365 inference timed out") from None
            except Exception as exc:
                await self._stop_process()
                if isinstance(exc, SemanticClassificationError):
                    raise
                raise SemanticClassificationError(
                    f"Places365 inference failed: {type(exc).__name__}: {str(exc)[:120]}"
                ) from None

    def _local(self, frames):
        if self._model is None:
            self._model = _PlacesModel(
                self.repo_path, self.device, self.num_threads, self.mapping_path
            )
        return self._model.classify(frames)

    async def _read_response(self):
        line = await self.process.stdout.readline()
        if not line:
            raise SemanticClassificationError("Places365 worker exited before returning a result")
        response = json.loads(line)
        if not isinstance(response, dict):
            raise SemanticClassificationError("Places365 worker returned an invalid response")
        if "error" in response:
            raise SemanticClassificationError("Places365: " + str(response["error"])[:140])
        return response

    async def _remote(self, frames):
        if self._labels is None:
            self._labels = load_labels(self.repo_path)
        if self.process is None or self.process.returncode is not None:
            env = dict(os.environ)
            env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
            env["PYTHONNOUSERSITE"] = "1"
            if self.device == "cpu":
                env["CUDA_VISIBLE_DEVICES"] = ""  # Do not reserve navigation model VRAM.
            self.process = await asyncio.create_subprocess_exec(
                self.python_executable,
                "-u",
                "-m",
                "room_memory.places365_recognizer",
                "--worker",
                "--repo-path",
                self.repo_path,
                "--device",
                self.device,
                "--num-threads",
                str(self.num_threads),
                "--mapping-path",
                self.mapping_path,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                env=env,
                limit=65536,
            )
            if (await self._read_response()).get("ready") is not True:
                raise SemanticClassificationError("Places365 worker did not become ready")
        request = [
            {"frame_id": frame.frame_id, "jpeg": base64.b64encode(frame.jpeg).decode("ascii")}
            for frame in frames
        ]
        self.process.stdin.write((json.dumps(request) + "\n").encode())
        await self.process.stdin.drain()
        return _validate_result(
            await self._read_response(), {frame.frame_id for frame in frames}, self._labels
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
        await self._stop_process()
        if self._executor is not None:
            # Local C++ inference cannot be killed; finish the bounded batch off-loop.
            await asyncio.to_thread(self._executor.shutdown, wait=True, cancel_futures=True)
            self._executor = None
        self._model = None


def main():
    parser = argparse.ArgumentParser(
        description="Places365 offline inference or persistent stdio worker"
    )
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--repo-path", default=DEFAULT_REPO)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--mapping-path", default=str(DEFAULT_MAPPING))
    parser.add_argument("images", nargs="*")
    args = parser.parse_args()
    try:
        model = _PlacesModel(args.repo_path, args.device, args.num_threads, args.mapping_path)
        if not args.worker:
            frames = [
                SemanticFrame(f"F{i}", Path(path).read_bytes())
                for i, path in enumerate(args.images)
            ]
            probabilities = model.predict_probabilities(frames)
            result = aggregate_predictions(frames, model.labels, probabilities, model.mapping)
            pooled = probabilities.mean(axis=0)
            output = asdict(result)
            output["places365_top5"] = [
                {"label": model.labels[i], "probability": float(pooled[i])}
                for i in np.argsort(-pooled)[:5]
            ]
            # Legacy diagnostic only; never used for the label or confidence.
            output["unmapped_probability"] = float(
                sum(pooled[i] for i, label in enumerate(model.labels) if label not in model.mapping)
            )
            print(json.dumps(output))
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
