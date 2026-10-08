"""Small HTTP UI: immutable latest-frame reads and serialized run commands."""

import json
from pathlib import Path
from urllib.parse import urlsplit

from aiohttp import web

WEB_DIR = Path(__file__).with_name("web")


def create_app(runtime):
    app = web.Application(client_max_size=8192)

    async def startup(_app):
        await runtime.initialize()

    async def cleanup(_app):
        await runtime.close()

    async def index(_request):
        return web.Response(
            text=(WEB_DIR / "index.html").read_text(),
            content_type="text/html",
            headers={"Cache-Control": "no-store"},
        )

    async def state(_request):
        return web.json_response(runtime.state, headers={"Cache-Control": "no-store"})

    async def camera(_request):
        if runtime.jpeg is None:
            raise web.HTTPServiceUnavailable(text="camera loading")
        return web.Response(
            body=runtime.view_jpeg,
            content_type="image/jpeg",
            headers={"Cache-Control": "no-store", "X-Pointing": json.dumps(runtime.pointing)},
        )

    async def command(request):
        origin = request.headers.get("Origin")
        if origin and urlsplit(origin).netloc != request.host:
            raise web.HTTPForbidden(text="invalid origin")
        try:
            data = await request.json()
            if not isinstance(data, dict):
                raise ValueError("expected an object")
            if request.match_info["action"] == "start":
                await runtime.start(data.get("backend"), data.get("instruction"))
            elif request.match_info["action"] == "stop":
                await runtime.stop()
            elif request.match_info["action"] == "reset":
                await runtime.reset()
            else:
                raise web.HTTPNotFound()
            return web.json_response({"ok": True})
        except (ValueError, TypeError) as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)

    async def memory_state(request):
        memory = runtime.room_memory
        payload = memory.cache if memory else {"enabled": False}
        headers = {"Cache-Control": "no-store"}
        if request.path.endswith("/export"):
            headers["Content-Disposition"] = 'attachment; filename="room-graph.json"'
        return web.json_response(payload, headers=headers)

    async def memory_frame(request):
        if runtime.room_memory is None:
            raise web.HTTPNotFound()
        image = await runtime.room_memory.image(request.match_info["frame_id"])
        if image is None:
            raise web.HTTPNotFound()
        return web.Response(
            body=image, content_type="image/jpeg", headers={"Cache-Control": "no-store"}
        )

    app.router.add_get("/", index)
    app.router.add_get("/api/state", state)
    app.router.add_get("/api/camera.jpg", camera)
    app.router.add_get("/api/room-memory", memory_state)
    app.router.add_get("/api/room-memory/export", memory_state)
    app.router.add_get(r"/api/room-memory/frames/{frame_id:F\d+}.jpg", memory_frame)
    app.router.add_post("/api/{action:start|stop|reset}", command)
    app.router.add_static("/static", WEB_DIR)
    app.on_startup.append(startup)
    app.on_cleanup.append(cleanup)
    return app
