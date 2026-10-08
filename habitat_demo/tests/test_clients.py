import base64

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from habitat_web.clients import ModelClient


@pytest.mark.asyncio
async def test_lightnav_login_sequence_units_and_stop():
    async def socket(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        login = await ws.receive_json()
        assert login["action"] == "login"
        assert login["data"]["clientId"].startswith("habitat-")
        await ws.send_json({"action": "login", "data": {"rc": 0}})
        message = await ws.receive_json()
        assert message["data"]["instruction"] == "forward"
        assert base64.b64decode(message["data"]["image"]) == b"jpeg"
        await ws.send_json(
            {
                "action": "next",
                "data": {
                    "rc": 0,
                    "seq": message["data"]["seq"],
                    "actions": [{"actions": [[0.3, 0.2, 0.1]]}],
                    "stop": True,
                    "latency_ms": 123.0,
                    "pointing": {
                        "frame_size": [640, 480],
                        "apos_px": [120, 80],
                        "opos_px": [400, 300],
                        "apos_state": "point",
                        "opos_state": "point",
                    },
                },
            }
        )
        async for _ in ws:
            pass
        return ws

    app = web.Application()
    app.router.add_get("/", socket)
    async with TestServer(app) as server:
        async with ModelClient(
            "lightnav", str(server.make_url("/")).replace("http:", "ws:")
        ) as client:
            result = await client.predict(b"jpeg", "forward")
            assert result.waypoints == ((0.3, 0.2, 0.1),)
            assert result.stop
            assert result.model_latency_ms == 123.0
            assert result.pointing["apos_px"] == [120, 80]
            assert result.pointing["opos_px"] == [400, 300]


@pytest.mark.asyncio
async def test_omnivla_uses_stateless_simple_http_contract():
    async def predict(request):
        body = await request.json()
        assert body["prompt"] == "door"
        assert "backtrack" not in body and "robot_pose" not in body
        return web.json_response(
            {"waypoints": [[2.0, 1.0, 1.0, 0.0]], "stop": False, "latency_ms": 5.0}
        )

    app = web.Application()
    app.router.add_post("/predict", predict)
    async with TestServer(app) as server:
        async with ModelClient("omnivla", str(server.make_url("/"))) as client:
            result = await client.predict(b"jpeg", "door")
            assert result.waypoints == ((0.2, 0.1, 0.0),)
            assert not result.stop


@pytest.mark.asyncio
async def test_bad_model_sequence_fails_before_motion():
    async def socket(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.receive_json()
        await ws.send_json({"action": "login", "data": {"rc": 0}})
        await ws.receive_json()
        await ws.send_json(
            {"action": "next", "data": {"rc": 0, "seq": 999, "actions": [[1, 0, 0]]}}
        )
        async for _ in ws:
            pass
        return ws

    app = web.Application()
    app.router.add_get("/", socket)
    async with TestServer(app) as server:
        async with ModelClient(
            "lightnav", str(server.make_url("/")).replace("http:", "ws:")
        ) as client:
            with pytest.raises(ValueError, match="sequence"):
                await client.predict(b"jpeg", "go")
