"""Synchronous Scene contract backed by one private, spawned Isaac process."""

import multiprocessing as mp
import time

from .worker import worker_main


class IsaacScene:
    def __init__(self, config, *, startup_timeout=1200, command_timeout=120,
                 worker_target=worker_main, progress_callback=None, cancel_event=None):
        self.last_progress = None
        self.progress_callback = progress_callback
        self.cancel_event = cancel_event
        self.command_timeout = command_timeout
        self.closed = False
        self.preview_callback = None
        ctx = mp.get_context("spawn")
        self.pipe, child = ctx.Pipe()
        self.process = ctx.Process(target=worker_target, args=(child, config),
                                   name="isaac-carter-worker", daemon=True)
        self.process.start()
        child.close()
        try:
            self._receive(startup_timeout)
        except BaseException:
            self.close(force=True)
            raise

    def _receive(self, timeout):
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if self.cancel_event is not None and self.cancel_event.is_set():
                raise RuntimeError("Isaac startup/service cancelled")
            if remaining <= 0:
                raise TimeoutError(f"Isaac worker exceeded {timeout:g}s; inspect terminal/log")
            if not self.pipe.poll(min(remaining, .2)):
                continue
            try:
                response = self.pipe.recv()
            except EOFError as exc:
                self.process.join(.1)
                phase = (self.last_progress or {}).get("phase", "unknown")
                raise RuntimeError(
                    f"Isaac worker exited before replying (pid={self.process.pid}, "
                    f"exitcode={self.process.exitcode}, phase={phase}); "
                    "inspect the traceback printed before Simulation App Shutting Down "
                    "and this run's Kit log"
                ) from exc
            if response.get("event") == "progress":
                self.last_progress = response["result"]
                if self.progress_callback is not None:
                    self.progress_callback(response["result"])
                continue
            if response.get("event") == "preview":
                if self.preview_callback is not None:
                    self.preview_callback(response["result"])
                continue
            if not response["ok"]:
                raise RuntimeError(response["error"])
            return response.get("result")

    def _request(self, method, *args):
        if self.closed:
            raise RuntimeError("Isaac worker is closed; restart the web service")
        try:
            self.pipe.send((method, args))
            return self._receive(self.command_timeout)
        except BaseException:
            self.close(force=True)
            raise

    def reset(self):
        return self._request("reset")

    def observe(self):
        return self._request("observe")

    def step(self, linear, angular, dt):
        return self._request("step", linear, angular, dt)

    def close(self, force=False):
        if self.closed:
            return
        self.closed = True
        if not force and self.process.is_alive():
            try:
                self.pipe.send(("close", ()))
            except (BrokenPipeError, EOFError, OSError):
                pass
            self.process.join(15)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(5)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(5)
        self.pipe.close()
