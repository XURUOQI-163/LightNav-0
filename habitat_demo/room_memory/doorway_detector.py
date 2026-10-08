"""YOLOE-26s RGB doorway candidates in a cancellable, persistent stdio worker.

Habitat imports only Pillow here; torch and Ultralytics stay in the configured
Python child. Candidates are visual evidence, not proof of a traversable opening.
Run this module with --prepare once to cache model-specific text embeddings.
Normal workers require local assets and never install packages or change models.
"""

import argparse
import asyncio
import base64
import contextlib
import hashlib
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image

MODEL_DIR = Path(__file__).resolve().parents[1] / ".models" / "yoloe"
MAX_JPEG_BYTES = 16_000_000
MAX_REQUEST_BYTES = 22_000_000
MAX_RESPONSE_BYTES = 2_000_000


@dataclass(frozen=True)
class DoorwayDetection:
    """Coordinates refer to the original, unresized RGB image's pixels."""

    bbox_xyxy: Tuple[float, float, float, float]
    confidence: float
    label: str
    polygon: Optional[Tuple[Tuple[float, float], ...]] = None


class DoorwayDetectionError(RuntimeError):
    """Unavailable model, invalid frame/protocol, or bounded inference failure."""


def _number(value, name, low, high):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not low <= value <= high
    ):
        raise ValueError(f"YOLOE {name} must be finite and between {low} and {high}")
    return value


def _options(options):
    supplied = dict(options or {})
    model_path = (
        Path(supplied.get("model_path", MODEL_DIR / "yoloe-26s-seg.pt")).expanduser().resolve()
    )
    if model_path.name != "yoloe-26s-seg.pt":
        raise ValueError("YOLOE doorway detector requires yoloe-26s-seg.pt")
    defaults = {
        "python_executable": os.environ.get("ROOM_MEMORY_YOLOE_PYTHON", sys.executable),
        "model_path": str(model_path),
        "prompt_cache_path": str(model_path.parent / "doorway-prompts.pt"),
        "labels": ["doorway", "open door", "door"],
        "device": "cpu",
        "cpu_threads": 2,
        "confidence": 0.20,
        "imgsz": 640,
        "timeout_seconds": 45.0,
        "startup_timeout_seconds": 240.0,
        "max_detections": 12,
    }
    defaults.update(supplied)
    defaults["model_path"] = str(model_path)
    defaults["prompt_cache_path"] = str(Path(defaults["prompt_cache_path"]).expanduser().resolve())
    defaults["python_executable"] = str(Path(defaults["python_executable"]).expanduser())
    labels = defaults["labels"]
    if (
        not isinstance(labels, (list, tuple))
        or not 1 <= len(labels) <= 16
        or any(
            not isinstance(label, str) or not label.strip() or len(label) > 80 for label in labels
        )
        or len(set(labels)) != len(labels)
    ):
        raise ValueError("YOLOE labels must be 1 to 16 unique nonempty prompts")
    defaults["labels"] = list(labels)
    device = defaults["device"]
    if not isinstance(device, str) or not (
        device == "cpu" or device == "cuda" or (device.startswith("cuda:") and device[5:].isdigit())
    ):
        raise ValueError("YOLOE device must be cpu, cuda, or cuda:N")
    for key, low, high in (("cpu_threads", 1, 32), ("imgsz", 32, 2048), ("max_detections", 1, 100)):
        value = _number(defaults[key], key, low, high)
        if not isinstance(value, int):
            raise ValueError(f"YOLOE {key} must be an integer")
    _number(defaults["confidence"], "confidence", 0.001, 1)
    for key in ("timeout_seconds", "startup_timeout_seconds"):
        _number(defaults[key], key, 0.001, 1800)
    return defaults


def _image_size(jpeg):
    if not isinstance(jpeg, bytes) or not 1 <= len(jpeg) <= MAX_JPEG_BYTES:
        raise DoorwayDetectionError("YOLOE image must be nonempty JPEG bytes under 16 MB")
    try:
        with Image.open(BytesIO(jpeg)) as image:
            if image.format != "JPEG" or not (
                1 <= image.width <= 4096 and 1 <= image.height <= 4096
            ):
                raise ValueError("expected JPEG with dimensions at most 4096 pixels")
            image.verify()
            return image.size
    except Exception as exc:
        raise DoorwayDetectionError(f"YOLOE invalid image: {exc}") from None


def _detections(response, request_id, image_size, options):
    if response.get("id") != request_id or response.get("image_size") != list(image_size):
        raise DoorwayDetectionError("YOLOE response does not match request/image dimensions")
    rows = response.get("detections")
    if not isinstance(rows, list) or len(rows) > options["max_detections"]:
        raise DoorwayDetectionError("YOLOE returned invalid detections")
    width, height = image_size
    output = []
    for row in rows:
        if not isinstance(row, dict) or row.get("label") not in options["labels"]:
            raise DoorwayDetectionError("YOLOE returned an unknown label")
        bbox = row.get("bbox_xyxy")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise DoorwayDetectionError("YOLOE returned an invalid box")
        x1, y1, x2, y2 = [float(_number(v, "box coordinate", 0, max(width, height))) for v in bbox]
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise DoorwayDetectionError("YOLOE box lies outside original image")
        confidence = float(_number(row.get("confidence"), "confidence", 0, 1))
        polygon = row.get("polygon")
        if polygon is not None:
            if not isinstance(polygon, list) or not 3 <= len(polygon) <= 1024:
                raise DoorwayDetectionError("YOLOE returned an invalid polygon")
            points = []
            for point in polygon:
                if not isinstance(point, list) or len(point) != 2:
                    raise DoorwayDetectionError("YOLOE returned an invalid polygon point")
                points.append(
                    (
                        float(_number(point[0], "polygon x", 0, width)),
                        float(_number(point[1], "polygon y", 0, height)),
                    )
                )
            polygon = tuple(points)
        if confidence >= options["confidence"]:
            output.append(DoorwayDetection((x1, y1, x2, y2), confidence, row["label"], polygon))
    return output


class YOLOEDoorwayDetector:
    """Lazy subprocess adapter; timeouts/cancellation terminate the owned child."""

    def __init__(self, options=None):
        self.options = _options(options)
        self.process = None
        self._lock = None
        self._spawn_task = None
        self._closed = False
        self._request_id = 0

    def _worker_command(self):
        return [
            self.options["python_executable"],
            "-u",
            str(Path(__file__).resolve()),
            "--worker",
            "--options",
            json.dumps(self.options),
        ]

    async def _read_response(self):
        line = await self.process.stdout.readline()
        if not line:
            raise DoorwayDetectionError("YOLOE worker exited before returning a result")
        response = json.loads(line)
        if not isinstance(response, dict):
            raise DoorwayDetectionError("YOLOE worker returned an invalid response")
        if "error" in response:
            raise DoorwayDetectionError("YOLOE worker: " + str(response["error"])[:300])
        return response

    async def _start(self):
        if self.process is not None and self.process.returncode is None:
            return
        env = dict(os.environ, PYTHONNOUSERSITE="1", YOLO_AUTOINSTALL="false")
        env["OMP_NUM_THREADS"] = str(self.options["cpu_threads"])
        env["MKL_NUM_THREADS"] = str(self.options["cpu_threads"])
        if self.options["device"] == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
        self._spawn_task = asyncio.create_task(
            asyncio.create_subprocess_exec(
                *self._worker_command(),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                env=env,
                limit=MAX_RESPONSE_BYTES,
            )
        )
        # Shield creation so cancellation can reap even a child created mid-await.
        self.process = await asyncio.shield(self._spawn_task)
        self._spawn_task = None
        if (await self._read_response()).get("ready") is not True:
            raise DoorwayDetectionError("YOLOE worker did not become ready")

    async def _request(self, jpeg, image_size):
        self._request_id += 1
        request = {
            "id": self._request_id,
            "image_size": list(image_size),
            "jpeg": base64.b64encode(jpeg).decode("ascii"),
        }
        self.process.stdin.write((json.dumps(request) + "\n").encode())
        await self.process.stdin.drain()
        return _detections(await self._read_response(), self._request_id, image_size, self.options)

    async def detect(self, jpeg):
        if self._closed:
            raise DoorwayDetectionError("YOLOE detector is closed")
        image_size = _image_size(jpeg)
        if self._lock is None:
            self._lock = asyncio.Lock()
        # Waiting callers are bounded, but cancelling one must not kill another's inference.
        try:
            await asyncio.wait_for(self._lock.acquire(), self.options["timeout_seconds"])
        except asyncio.TimeoutError:
            raise DoorwayDetectionError("YOLOE detector queue timed out") from None
        try:
            if self._closed:
                raise DoorwayDetectionError("YOLOE detector is closed")
            await asyncio.wait_for(self._start(), self.options["startup_timeout_seconds"])
            return await asyncio.wait_for(
                self._request(jpeg, image_size), self.options["timeout_seconds"]
            )
        except asyncio.CancelledError:
            await self._stop_process()
            raise
        except asyncio.TimeoutError:
            await self._stop_process()
            raise DoorwayDetectionError("YOLOE inference or startup timed out") from None
        except Exception as exc:
            await self._stop_process()
            if isinstance(exc, DoorwayDetectionError):
                raise
            raise DoorwayDetectionError(
                f"YOLOE inference failed: {type(exc).__name__}: {str(exc)[:300]}"
            ) from None
        finally:
            self._lock.release()

    async def _stop_process(self):
        if self._spawn_task is not None:
            task, self._spawn_task = self._spawn_task, None
            with contextlib.suppress(Exception):
                self.process = await asyncio.shield(task)
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


class _YOLOEModel:
    """Constructed only inside the selected inference environment."""

    def __init__(self, options, prepare=False):
        os.environ["YOLO_AUTOINSTALL"] = "false"
        import torch
        import ultralytics
        from ultralytics import YOLOE

        self.options = options
        torch.set_num_threads(options["cpu_threads"])
        model_path, cache_path = Path(options["model_path"]), Path(options["prompt_cache_path"])
        if not model_path.is_file():
            raise FileNotFoundError(f"Download official yoloe-26s-seg.pt to {model_path} first")
        model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
        self.model = YOLOE(str(model_path))
        if (
            self.model.task != "segment"
            or self.model.model.yaml.get("scale") != "s"
            or type(self.model.model.model[-1]).__name__ != "YOLOESegment26"
        ):
            raise ValueError("Expected the YOLOE-26s segmentation checkpoint")
        expected = {
            "model_sha256": model_hash,
            "labels": options["labels"],
            "ultralytics_version": ultralytics.__version__,
        }
        if prepare:
            # No encoder is retained by YOLOE's cache_clip_model=False path.
            original_cwd = Path.cwd()
            try:
                os.chdir(model_path.parent)
                with torch.inference_mode():
                    embeddings = self.model.model.get_text_pe(
                        options["labels"], cache_clip_model=False
                    ).cpu()
            finally:
                os.chdir(original_cwd)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_name(cache_path.name + f".{os.getpid()}.tmp")
            try:
                torch.save(dict(expected, embeddings=embeddings), temporary)
                temporary.replace(cache_path)
            finally:
                temporary.unlink(missing_ok=True)
        else:
            if not cache_path.is_file():
                raise FileNotFoundError(
                    f"YOLOE prompt cache missing: run --prepare for {cache_path}"
                )
            cache = torch.load(cache_path, map_location="cpu", weights_only=True)
            if not isinstance(cache, dict) or any(cache.get(k) != v for k, v in expected.items()):
                raise ValueError(
                    "YOLOE prompt cache does not match checkpoint/prompts/version; run --prepare"
                )
            embeddings = cache.get("embeddings")
        if (
            not isinstance(embeddings, torch.Tensor)
            or embeddings.ndim != 3
            or embeddings.shape[:2] != (1, len(options["labels"]))
            or not torch.isfinite(embeddings).all()
        ):
            raise ValueError("Invalid YOLOE text embeddings")
        self.model.set_classes(options["labels"], embeddings=embeddings)
        self.predict(Image.new("RGB", (options["imgsz"], options["imgsz"])))

    def predict(self, image):
        result = self.model.predict(
            source=image,
            device=self.options["device"],
            imgsz=self.options["imgsz"],
            conf=self.options["confidence"],
            max_det=self.options["max_detections"],
            verbose=False,
            save=False,
            retina_masks=True,
        )[0]
        if tuple(result.orig_shape) != (image.height, image.width):
            raise ValueError("YOLOE prediction dimensions do not match original image")
        polygons = result.masks.xy if result.masks is not None else None
        output = []
        for index, box in enumerate(result.boxes):
            polygon = None
            if polygons is not None and len(polygons[index]) >= 3:
                points = polygons[index]
                points = points[:: max(1, math.ceil(len(points) / 1024))]
                polygon = tuple(tuple(float(v) for v in point) for point in points)
            output.append(
                DoorwayDetection(
                    tuple(float(v) for v in box.xyxy[0].cpu().tolist()),
                    float(box.conf[0].cpu()),
                    result.names[int(box.cls[0].cpu())],
                    polygon,
                )
            )
        return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--options", default="{}", help="JSON detector options")
    parser.add_argument("--config", help="Room-memory JSON; use its doorway.detector section")
    parser.add_argument("--image", help="Inspect one RGB image and exit (no Habitat or navigation)")
    args = parser.parse_args()
    protocol = sys.stdout
    # Third-party progress/logging must never corrupt the JSON stdout protocol.
    with contextlib.redirect_stdout(sys.stderr):
        try:
            supplied = json.loads(args.options)
            if args.config:
                supplied = {
                    **json.loads(Path(args.config).read_text())["doorway"]["detector"],
                    **supplied,
                }
            options = _options(supplied)
            model = _YOLOEModel(options, prepare=args.prepare)
            if args.prepare:
                print(
                    json.dumps({"prepared": options["prompt_cache_path"]}),
                    file=protocol,
                    flush=True,
                )
                return
            if args.image:
                with Image.open(args.image) as image:
                    rgb = image.convert("RGB")
                    output = model.predict(rgb)
                    response = {
                        "image_size": list(rgb.size),
                        "detections": [asdict(item) for item in output],
                    }
                print(json.dumps(response, allow_nan=False), file=protocol, flush=True)
                return
            if not args.worker:
                raise ValueError("Specify --worker or --prepare")
            print(json.dumps({"ready": True}), file=protocol, flush=True)
            while True:
                line = sys.stdin.buffer.readline(MAX_REQUEST_BYTES + 1)
                if not line:
                    break
                if len(line) > MAX_REQUEST_BYTES:
                    raise ValueError("YOLOE request exceeds 22 MB")
                request = json.loads(line)
                jpeg = base64.b64decode(request["jpeg"], validate=True)
                size = _image_size(jpeg)
                if request.get("image_size") != list(size):
                    raise ValueError("YOLOE request image dimensions mismatch")
                with Image.open(BytesIO(jpeg)) as image:
                    output = model.predict(image.convert("RGB"))
                response = {
                    "id": request["id"],
                    "image_size": list(size),
                    "detections": [asdict(item) for item in output],
                }
                print(json.dumps(response, allow_nan=False), file=protocol, flush=True)
        except Exception as exc:
            print(
                json.dumps({"error": f"{type(exc).__name__}: {str(exc)[:400]}"}),
                file=protocol,
                flush=True,
            )
            raise SystemExit(1) from None


if __name__ == "__main__":
    main()
