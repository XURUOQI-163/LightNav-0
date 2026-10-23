"""Small HTTP UI: immutable latest-frame reads and serialized run commands."""

import io
import json
from pathlib import Path
from urllib.parse import urlsplit

from aiohttp import web

WEB_DIR = Path(__file__).with_name("web")


def create_app(runtime):
    app = web.Application(client_max_size=8192)

    async def startup(_app):
        initialize = getattr(runtime, "begin_initialize", runtime.initialize)
        await initialize()

    async def cleanup(_app):
        await runtime.close()

    async def index(_request):
        return web.Response(
            text=(WEB_DIR / "index.html").read_text(),
            content_type="text/html",
            headers={"Cache-Control": "no-store"},
        )

    async def state(_request):
        snapshot = getattr(runtime, "status_snapshot", lambda: runtime.state)()
        memory = getattr(runtime, "room_memory", None)
        snapshot = dict(snapshot, room_detection_enabled=bool(
            memory and getattr(memory, "semantic_backend", None) == "groundingdino"),
            room_detection_session_id=getattr(memory, "session_id", None))
        return web.json_response(snapshot, headers={"Cache-Control": "no-store"})

    async def camera(_request):
        if runtime.jpeg is None:
            raise web.HTTPServiceUnavailable(text="camera loading")
        return web.Response(
            body=runtime.view_jpeg,
            content_type="image/jpeg",
            headers={
                "Cache-Control": "no-store",
                "X-Pointing": json.dumps(runtime.pointing),
                "X-Waypoints": json.dumps(runtime.waypoint_overlay),
                "X-Frame-Id": str(runtime.state["frame_id"]),
            },
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
            elif request.match_info["action"] == "manual" and callable(
                getattr(runtime, "manual", None)
            ):
                await runtime.manual(
                    data.get("linear"), data.get("angular"), data.get("duration")
                )
            elif request.match_info['action'] == 'local-navigation-mode' and callable(
                getattr(runtime, 'navigation_mode', None)
            ):
                await runtime.navigation_mode(data.get('mode'))
            else:
                raise web.HTTPNotFound()
            return web.json_response({"ok": True})
        except (ValueError, TypeError) as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)

    async def sensors(_request):
        observation = getattr(runtime, "observation", None)
        if not observation or observation.get("depth_camera") is None:
            raise web.HTTPServiceUnavailable(text="sensors loading")
        return web.json_response(
            {
                "frame_id": runtime.observation_frame_id,
                "sim_time": observation["sim_time"],
                "pose": observation["pose"],
                "pose_frame": observation.get("pose_frame", "unspecified"),
                "chassis_pose": observation.get("chassis_pose"),
                "floor_elevation": observation.get("floor_elevation"),
                "depth_camera": observation["depth_camera"],
                "source": "latest_physical_observation",
                "rgb_url": "/api/rgb.jpg",
                "depth_url": "/api/depth.npy",
                "bundle_url": "/api/observation.npz",
                "depth_definition": observation.get(
                    "depth_definition", "camera_forward_z_meters"
                ),
            },
            headers={"Cache-Control": "no-store"},
        )

    def sensor_headers(observation):
        # The viewer may retain the model's input frame to align APOS/OPOS.
        # Sensor endpoints always use the most recent physical observation.
        return {
            "Cache-Control": "no-store",
            "X-Frame-Id": str(runtime.observation_frame_id),
            "X-Sim-Time": str(observation["sim_time"]),
        }

    async def rgb(_request):
        observation = getattr(runtime, "observation", None)
        if not observation or observation.get("jpeg") is None:
            raise web.HTTPServiceUnavailable(text="rgb loading")
        return web.Response(
            body=observation["jpeg"], content_type="image/jpeg",
            headers=sensor_headers(observation),
        )

    async def display(request):
        view = request.query.get("view", "first")
        if view not in ("first", "third"):
            raise web.HTTPBadRequest(text="view must be first or third")
        frame = runtime.display_frame
        key = "jpeg" if view == "first" else "third_jpeg"
        if frame is None or not frame.get(key):
            raise web.HTTPServiceUnavailable(text="display camera loading")
        headers = {"Cache-Control": "no-store", "X-Frame-Id": str(runtime.display_frame_id),
                   "X-Sim-Time": str(frame["sim_time"])}
        if request.query.get("after") == headers["X-Frame-Id"]:
            return web.Response(status=204, headers=headers)
        return web.Response(body=frame[key], content_type="image/jpeg", headers=headers)

    async def depth(_request):
        observation = getattr(runtime, "observation", None)
        if not observation or observation.get("depth") is None:
            raise web.HTTPServiceUnavailable(text="depth loading")
        import numpy as np

        buffer = io.BytesIO()
        np.save(buffer, observation["depth"], allow_pickle=False)
        return web.Response(
            body=buffer.getvalue(),
            content_type="application/octet-stream",
            headers=sensor_headers(observation),
        )

    async def observation_bundle(_request):
        observation = getattr(runtime, "observation", None)
        if not observation or observation.get("depth") is None:
            raise web.HTTPServiceUnavailable(text="sensors loading")
        import numpy as np
        from PIL import Image

        # Build one coherent packet without advancing the simulator. The JPEG
        # and metric depth originate in the same worker capture; pose is XYZ+yaw.
        with Image.open(io.BytesIO(observation["jpeg"])) as image:
            image_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        camera = observation["depth_camera"]
        pose = observation["pose"]
        extra = {}
        if observation.get("chassis_pose"):
            extra["chassis_position"] = np.array(observation["chassis_pose"]["position"])
            extra["chassis_orientation_wxyz"] = np.array(
                observation["chassis_pose"]["orientation_wxyz"]
            )
        buffer = io.BytesIO()
        np.savez(
            buffer,
            rgb=image_rgb,
            depth=np.asarray(observation["depth"], dtype=np.float32),
            K=np.array([[camera["fx"], 0, camera["cx"]],
                        [0, camera["fy"], camera["cy"]], [0, 0, 1]], dtype=np.float64),
            camera_to_robot=np.asarray(camera["camera_to_robot"], dtype=np.float64),
            pose=np.array([pose["x"], pose["y"], pose["z"], pose["yaw"]], dtype=np.float64),
            pose_frame=np.array(observation.get("pose_frame", "unspecified")),
            sim_time=np.float64(observation["sim_time"]),
            frame_id=np.int64(runtime.observation_frame_id),
            depth_definition=np.array(observation.get(
                "depth_definition", "camera_forward_z_meters"
            )),
            **extra,
        )
        return web.Response(
            body=buffer.getvalue(), content_type="application/octet-stream",
            headers=sensor_headers(observation),
        )

    async def memory_state(request):
        memory = runtime.room_memory
        payload = memory.cache if memory else {"enabled": False}
        headers = {"Cache-Control": "no-store"}
        if request.path.endswith("/export"):
            headers["Content-Disposition"] = 'attachment; filename="room-graph.json"'
        return web.json_response(payload, headers=headers)

    async def local_navigation(_request):
        return web.json_response(getattr(runtime, 'local_snapshot', {'enabled': False}),
                                 headers={'Cache-Control': 'no-store'})

    async def detection_image(request):
        memory = getattr(runtime, "room_memory", None)
        frame = memory.projection.detection_frame if memory else None
        if frame is None:
            raise web.HTTPServiceUnavailable(text="room detector waiting for source frame")
        headers = {"Cache-Control": "no-store", "X-Frame-Id": frame["frame_id"],
                   "X-Sim-Time": str(frame["timestamp"])}
        if request.query.get("after") == frame["frame_id"]:
            return web.Response(status=204, headers=headers)
        return web.Response(body=frame["jpeg"], content_type="image/jpeg", headers=headers)

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
    app.router.add_get("/api/room-memory/detection.jpg", detection_image)
    app.router.add_get('/api/local-navigation', local_navigation)
    app.router.add_get("/api/room-memory/export", memory_state)
    app.router.add_get(r"/api/room-memory/frames/{frame_id:F\d+}.jpg", memory_frame)
    actions = "start|stop|reset"
    if callable(getattr(runtime, "manual", None)):
        actions += "|manual"
    if callable(getattr(runtime, 'navigation_mode', None)):
        actions += '|local-navigation-mode'
    app.router.add_post("/api/{action:" + actions + "}", command)
    if getattr(runtime, "sensor_api", False):
        app.router.add_get("/api/sensors", sensors)
        app.router.add_get("/api/rgb.jpg", rgb)
        app.router.add_get("/api/depth.npy", depth)
        app.router.add_get("/api/observation.npz", observation_bundle)
    if hasattr(runtime, "display_frame"):
        app.router.add_get("/api/display.jpg", display)
    app.router.add_static("/static", WEB_DIR)
    app.on_startup.append(startup)
    app.on_cleanup.append(cleanup)
    return app
