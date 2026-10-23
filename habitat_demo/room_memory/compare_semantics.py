"""Compare independent room classifiers on identical saved RGB views, without navigation."""

import argparse
import asyncio
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

from .clip_recognizer import CLIPRecognizer
from .config import load_config
from .groundingdino_recognizer import GroundingDINORecognizer
from .places365_recognizer import Places365Recognizer
from .semantics import SemanticFrame, _validate_frames


async def compare_images(paths, config=None, backends=("clip", "groundingdino")):
    constructors = {"clip": CLIPRecognizer, "groundingdino": GroundingDINORecognizer, "places365": Places365Recognizer}
    if not backends or any(b not in constructors for b in backends) or len(set(backends)) != len(backends):
        raise ValueError("choose unique clip/groundingdino/places365 backends")
    frames, images = [], []
    for i, path in enumerate(paths):
        path = Path(path)
        jpeg = path.read_bytes()
        frames.append(SemanticFrame(f"F{i}", jpeg))
        images.append(
            dict(
                frame_id=f"F{i}", path=str(path.resolve()), sha256=hashlib.sha256(jpeg).hexdigest()
            )
        )
    _validate_frames(frames)
    report = dict(
        images=images,
        results={},
        note="1..5 views of ONE room; backend scores are not comparable accuracy estimates.",
    )
    for backend in backends:
        constructor = constructors[backend]
        _, _, options = load_config(config, semantic_backend=backend)
        for key in ("backend", "live_enabled", "live_interval"):
            options.pop(key, None)
        recognizer = constructor(**options)
        try:
            start = time.perf_counter()
            await recognizer.classify(frames)
            first_ms = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            result = await recognizer.classify(frames)
            warm_ms = (time.perf_counter() - start) * 1000
            report["results"][backend] = dict(
                asdict(result),
                metadata=recognizer.metadata(),
                first_request_ms=first_ms,
                warm_request_ms=warm_ms,
            )
        finally:
            await recognizer.aclose()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path, help="1..5 JPEG views of the same room")
    parser.add_argument("--config", type=Path, help="Optional room-memory JSON")
    parser.add_argument("--output", type=Path, required=True, help="JSON comparison report")
    parser.add_argument("--backends", nargs="+", choices=("clip", "groundingdino", "places365"), default=["clip", "groundingdino"])
    args = parser.parse_args()
    if not 1 <= len(args.images) <= 5:
        parser.error("provide 1..5 images of one room; compare different rooms separately")
    report = asyncio.run(compare_images(args.images, args.config, args.backends))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    for backend, result in report["results"].items():
        print(
            f"{backend}: {result['room_type']} | score={result['confidence']:.4f} | warm={result['warm_request_ms']:.1f} ms"
        )
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
