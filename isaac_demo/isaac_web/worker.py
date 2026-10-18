"""Kit imports and SimulationApp construction happen only in the child main thread."""

import contextlib
import traceback


def worker_main(pipe, config):
    scene = None
    try:
        from .scene import CarterScene

        scene = CarterScene(
            config, progress_callback=lambda progress: pipe.send(
                {"event": "progress", "result": progress}
            ),
            error_callback=lambda message: pipe.send({"ok": False, "error": message}),
        )
        scene.preview_callback = lambda frame: pipe.send({"event": "preview", "result": frame})
        pipe.send({"ok": True})
        while True:
            method, args = pipe.recv()
            if method == "close":
                break
            if method not in ("reset", "observe", "step"):
                raise ValueError("unsupported scene command")
            pipe.send({"ok": True, "result": getattr(scene, method)(*args)})
    except (EOFError, BrokenPipeError):
        pass
    except BaseException:
        message = traceback.format_exc()
        print(message, flush=True)
        with contextlib.suppress(BrokenPipeError, EOFError, OSError):
            pipe.send({"ok": False, "error": message})
    finally:
        if scene is not None:
            scene.close()
        pipe.close()
