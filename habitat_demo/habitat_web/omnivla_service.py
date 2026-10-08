"""Minimal, language-only HTTP adapter around original full OmniVLA inference.

Run in the OmniVLA environment, with habitat_demo on PYTHONPATH. The original
model uses one observation, not session history; reset therefore acknowledges
the episode boundary without retaining images, prompts, or model caches.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import importlib.util
import io
import logging
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from aiohttp import web
from PIL import Image, UnidentifiedImageError

LOG = logging.getLogger(__name__)
MAX_IMAGE_BYTES = 4 * 1024 * 1024
MAX_IMAGE_PIXELS = 2048 * 2048


class FullPolicy:
    """Reuse the upstream loader, preprocessing and L1 head without overrides."""

    def __init__(self, repo: Path, checkpoint: Path, resume_step: int = 120000):
        repo, checkpoint = repo.resolve(), checkpoint.resolve()
        source = repo / "inference" / "run_omnivla.py"
        if not source.is_file() or not checkpoint.is_dir():
            raise FileNotFoundError("OmniVLA repository or checkpoint directory is missing")
        # All weights must be local; fail instead of downloading missing artifacts.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        sys.path.insert(0, str(repo))
        import numpy as np
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Full OmniVLA requires a CUDA GPU")
        spec = importlib.util.spec_from_file_location("habitat_original_omnivla", source)
        full = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = full
        spec.loader.exec_module(full)
        self.full = full
        full.pose_goal = full.satellite = full.image_goal = False
        full.lan_prompt = True
        cfg = full.InferenceConfig()
        cfg.vla_path = str(checkpoint)
        cfg.resume_step = resume_step
        (
            self.vla,
            self.action_head,
            self.pose_projector,
            self.device_id,
            self.num_patches,
            self.action_tokenizer,
            self.processor,
        ) = full.define_model(cfg)
        self.vla.eval()
        self.action_head.eval()
        self.pose_projector.eval()
        self.goal_image = Image.new("RGB", (224, 224))
        self.goal_pose = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        # The upstream signature needs goal placeholders. Modality 7 masks both.
        self.inference = full.Inference(
            save_dir=str(repo / "inference"),
            lan_inst_prompt="",
            goal_utm=(0.0, 0.0),
            goal_compass=0.0,
            goal_image_PIL=self.goal_image,
            action_tokenizer=self.action_tokenizer,
            processor=self.processor,
        )

    def predict(self, image: Image.Image, prompt: str, session_id: str):
        batch = self.inference.data_transformer_omnivla(
            image,
            prompt,
            self.goal_image,
            self.goal_pose,
            prompt_builder=self.full.PurePromptBuilder,
            action_tokenizer=self.action_tokenizer,
            processor=self.processor,
        )
        actions, _ = self.inference.run_forward_pass(
            vla=self.vla,
            action_head=self.action_head,
            noisy_action_projector=None,
            pose_projector=self.pose_projector,
            batch=batch,
            action_tokenizer=self.action_tokenizer,
            device_id=self.device_id,
            use_l1_regression=True,
            use_diffusion=False,
            use_film=False,
            num_patches=self.num_patches,
            compute_diffusion_l1=False,
            num_diffusion_steps_train=None,
            mode="train",
            idrun=0,
        )
        # Preserve original normalized x/y and heading cos/sin (0.1 m per unit).
        return actions.float().detach().cpu().numpy()[0].tolist()


def _text(body, key, limit):
    value = body.get(key)
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"{key} must be a nonempty string of at most {limit} characters")
    return value.strip()


def _decode_image(encoded):
    if not isinstance(encoded, str) or len(encoded) > (MAX_IMAGE_BYTES * 4 // 3 + 4):
        raise ValueError("image_jpeg_base64 is missing or too large")
    try:
        raw = base64.b64decode(encoded, validate=True)
        if not raw or len(raw) > MAX_IMAGE_BYTES:
            raise ValueError("JPEG is empty or too large")
        with Image.open(io.BytesIO(raw)) as image:
            if image.format != "JPEG" or image.width * image.height > MAX_IMAGE_PIXELS:
                raise ValueError("Expected a JPEG of at most 2048 x 2048 pixels in area")
            return image.convert("RGB")
    except (binascii.Error, OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise ValueError("Invalid JPEG image") from exc


def create_app(policy):
    """Accept an injected policy for CPU tests; no model is loaded on import."""
    app = web.Application(client_max_size=6 * 1024 * 1024)
    worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="omnivla")
    pending = None

    async def body_for(request):
        try:
            body = await request.json()
        except (ValueError, UnicodeError) as exc:
            raise web.HTTPBadRequest(text="Invalid JSON") from exc
        if not isinstance(body, dict):
            raise web.HTTPBadRequest(text="Expected a JSON object")
        return body

    def infer(encoded, prompt, session_id):
        started = time.perf_counter()
        image = _decode_image(encoded)
        rows = policy.predict(image, prompt, session_id)
        if not isinstance(rows, list) or not 1 <= len(rows) <= 64:
            raise RuntimeError("Model returned invalid waypoints")
        for row in rows:
            if len(row) != 4 or not all(math.isfinite(float(value)) for value in row):
                raise RuntimeError("Model returned invalid waypoints")
        return {
            "waypoints": rows,
            "latency_ms": (time.perf_counter() - started) * 1000,
            "stop": False,
        }

    async def predict(request):
        nonlocal pending
        body = await body_for(request)
        try:
            prompt = _text(body, "prompt", 1024)
            session_id = _text(body, "session_id", 128)
            encoded = body.get("image_jpeg_base64")
            if not isinstance(encoded, str):
                raise ValueError("image_jpeg_base64 must be a string")
            if pending is not None and not pending.done():
                raise web.HTTPServiceUnavailable(text="OmniVLA is busy; retry after completion")
            pending = asyncio.get_running_loop().run_in_executor(
                worker,
                infer,
                encoded,
                prompt,
                session_id,
            )
            # A disconnected client must not cancel or overlap the GPU operation.
            pending.add_done_callback(lambda future: future.exception())
            result = await asyncio.shield(pending)
        except ValueError as exc:
            raise web.HTTPBadRequest(text=str(exc)) from exc
        except web.HTTPException:
            raise
        except Exception as exc:
            LOG.exception("OmniVLA inference failed")
            raise web.HTTPInternalServerError(
                text="OmniVLA inference failed; see service log"
            ) from exc
        return web.json_response(result)

    async def reset(request):
        body = await body_for(request)
        try:
            session_id = _text(body, "session_id", 128)
        except ValueError as exc:
            raise web.HTTPBadRequest(text=str(exc)) from exc
        return web.json_response({"ok": True, "session_id": session_id})

    async def health(request):
        return web.json_response({"ok": True, "model": "omnivla"})

    async def cleanup(app):
        if pending is not None:
            await asyncio.gather(asyncio.shield(pending), return_exceptions=True)
        worker.shutdown(wait=True)

    app.router.add_get("/health", health)
    app.router.add_post("/predict", predict)
    app.router.add_post("/reset", reset)
    app.on_cleanup.append(cleanup)
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--resume-step", type=int, default=120000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8891)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    policy = FullPolicy(args.repo, args.checkpoint, args.resume_step)
    web.run_app(create_app(policy), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
