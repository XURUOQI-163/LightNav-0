"""Async wire adapters; weights remain in their original model environments."""

import asyncio
import base64
import contextlib
import math
import time
import uuid
from dataclasses import dataclass

import aiohttp

from .control import parse_waypoints


@dataclass
class Prediction:
    waypoints: tuple
    stop: bool
    latency_ms: float
    model_latency_ms: float = None
    pointing: dict = None


class ModelClient:
    def __init__(self, backend, url, timeout=30.0):
        self.backend, self.url, self.timeout = backend, url, timeout
        self.session_id = "habitat-" + uuid.uuid4().hex
        self.http = None
        self.ws = None
        self.seq = 0
        self._reader_task = None
        self._reader_error = None
        self._responses = None

    async def __aenter__(self):
        self.http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        try:
            if self.backend == "lightnav":
                self.ws = await self.http.ws_connect(self.url, heartbeat=10, max_msg_size=2**20)
                self._responses = asyncio.Queue(maxsize=2)
                self._reader_error = None
                self._reader_task = asyncio.create_task(self._read_responses(), name='lightnav-ws-reader')
                await self.ws.send_json({"action": "login", "data": {"clientId": self.session_id}})
                await self._receive("login")
            return self
        except BaseException:
            await self.__aexit__()
            raise

    async def __aexit__(self, *_):
        if self._reader_task is not None:
            self._reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reader_task
        try:
            if self.ws is not None:
                await self.ws.close()
        finally:
            if self.http is not None:
                await self.http.close()

    async def _read_responses(self):
        """Single socket reader, also draining ping/pong during recovery/rendering.

        aiohttp autoping runs inside receive(), not in predict(). Application
        responses are queued separately so idle navigation keeps the same model
        session alive without submitting fake frames or resetting model history.
        """
        try:
            while True:
                message = await self.ws.receive()
                if message.type == aiohttp.WSMsgType.TEXT:
                    if self._responses.full():
                        raise ValueError('unexpected queued model response')
                    self._responses.put_nowait(message.json())
                elif message.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED,
                                      aiohttp.WSMsgType.CLOSING, aiohttp.WSMsgType.ERROR):
                    raise ConnectionError(
                        f'LightNav 模型连接已断开（关闭码 {self.ws.close_code}）；'
                        '请检查模型终端 A，服务正常后重新开始导航'
                    ) from self.ws.exception()
                else:
                    raise ValueError('unexpected model WebSocket frame')
        except Exception as exc:
            self._reader_error = exc
            # Wake an in-flight request immediately; preserve a final response
            # already received before a server close. The error remains sticky.
            if self._responses.empty():
                self._responses.put_nowait(exc)

    async def _receive(self, action):
        if self._responses.empty() and self._reader_error is not None:
            raise self._reader_error
        message = await asyncio.wait_for(self._responses.get(), self.timeout)
        if isinstance(message, Exception):
            raise message
        if not isinstance(message, dict) or message.get("action") != action:
            raise ValueError("unexpected model response")
        data = message.get("data", {})
        if data.get("rc") != 0:
            raise ValueError(str(data.get("msg", "model failed")))
        return data

    async def predict(self, jpeg, instruction):
        start = time.monotonic()
        encoded = base64.b64encode(jpeg).decode("ascii")
        if self.backend == "lightnav":
            if self._reader_error is not None:
                raise self._reader_error
            if self.ws.closed:
                raise ConnectionError('LightNav 模型连接已关闭；请检查模型终端 A 后重新开始导航')
            self.seq += 1
            try:
                await self.ws.send_json(
                    {
                        "action": "next",
                        "data": {"seq": self.seq, "image": encoded, "instruction": instruction},
                    }
                )
            except (aiohttp.ClientConnectionError, ConnectionResetError) as exc:
                raise ConnectionError(
                    '向 LightNav 模型发送图像时连接已断开；请检查模型终端 A 后重新开始导航'
                ) from exc
            data = await self._receive("next")
            if data.get("seq") != self.seq:
                raise ValueError("model sequence mismatch")
            raw = data.get("actions")
        else:
            async with self.http.post(
                self.url.rstrip("/") + "/predict",
                json={
                    "image_jpeg_base64": encoded,
                    "prompt": instruction,
                    "session_id": self.session_id,
                },
            ) as response:
                response.raise_for_status()
                data = await response.json()
            if not isinstance(data, dict):
                raise ValueError("model response must be an object")
            raw = data.get("waypoints")
        stop = data.get("stop", False)
        if not isinstance(stop, bool):
            raise ValueError("model stop must be boolean")
        points = parse_waypoints(raw, self.backend)
        latency = data.get("latency_ms")
        if latency is not None:
            if (
                isinstance(latency, bool)
                or not isinstance(latency, (int, float))
                or not math.isfinite(latency)
                or latency < 0
            ):
                raise ValueError("invalid model latency")
        pointing = parse_pointing(data.get("pointing")) if self.backend == "lightnav" else None
        return Prediction(points, stop, (time.monotonic() - start) * 1000, latency, pointing)


def parse_pointing(value):
    """Optional display metadata: invalid pixels are omitted, never used for motion."""
    if not isinstance(value, dict):
        return None
    size = value.get("frame_size")
    if (
        not isinstance(size, list)
        or len(size) != 2
        or any(isinstance(n, bool) or not isinstance(n, int) or not 0 < n <= 16384 for n in size)
    ):
        return None
    result = {"frame_size": size[:]}
    for channel in ("apos", "opos"):
        pixel = value.get(channel + "_px")
        valid = (
            isinstance(pixel, list)
            and len(pixel) == 2
            and all(
                not isinstance(n, bool)
                and isinstance(n, (int, float))
                and math.isfinite(n)
                and 0 <= n < limit
                for n, limit in zip(pixel, size)
            )
        )
        result[channel + "_px"] = pixel[:] if valid else None
        state = value.get(channel + "_state")
        result[channel + "_state"] = (
            state
            if state in ("point", "rot_left", "rot_right", "stop", "not_visible", "none")
            else "none"
        )
        result[channel + "_clamped"] = value.get(channel + "_clamped") is True
    return result
