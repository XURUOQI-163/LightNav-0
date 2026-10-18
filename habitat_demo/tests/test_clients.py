import asyncio
import base64

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from habitat_web.clients import ModelClient


@pytest.mark.asyncio
async def test_idle_model_connection_answers_server_ping_without_prediction():
    pong_seen = asyncio.Event()

    async def socket(request):
        ws = web.WebSocketResponse(autoping=False)
        await ws.prepare(request)
        await ws.receive_json()
        await ws.send_json({'action': 'login', 'data': {'rc': 0}})
        await ws.ping(b'recovery-idle')
        async for message in ws:
            if message.type == aiohttp.WSMsgType.PONG and message.data == b'recovery-idle':
                pong_seen.set()
        return ws

    app = web.Application()
    app.router.add_get('/', socket)
    async with TestServer(app) as server:
        async with ModelClient('lightnav', str(server.make_url('/')).replace('http:', 'ws:')):
            # Recovery/slow rendering does not call predict, but must answer pings.
            await asyncio.wait_for(pong_seen.wait(), .5)


@pytest.mark.asyncio
async def test_predict_after_idle_longer_than_client_heartbeat(monkeypatch):
    connect = aiohttp.ClientSession.ws_connect

    def fast_heartbeat(self, *args, **kwargs):
        kwargs['heartbeat'] = .1  # Accelerate the production 10 s heartbeat.
        return connect(self, *args, **kwargs)

    monkeypatch.setattr(aiohttp.ClientSession, 'ws_connect', fast_heartbeat)

    async def socket(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for message in ws:
            data = message.json()
            payload = {'rc': 0}
            if data['action'] == 'next':
                payload.update(seq=data['data']['seq'], actions=[[.2, 0, 0]], stop=False)
            await ws.send_json({'action': data['action'], 'data': payload})
        return ws

    app = web.Application()
    app.router.add_get('/', socket)
    async with TestServer(app) as server:
        async with ModelClient('lightnav', str(server.make_url('/')).replace('http:', 'ws:')) as client:
            await asyncio.sleep(.4)
            result = await client.predict(b'jpeg-after-turn', 'go')
            assert result.waypoints == ((.2, 0., 0.),)
        assert client._reader_task.done() and client.http.closed and client.ws.closed


@pytest.mark.asyncio
async def test_server_close_wakes_pending_prediction_with_connection_error():
    async def socket(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.receive_json()
        await ws.send_json({'action': 'login', 'data': {'rc': 0}})
        await ws.receive_json()
        await ws.close(code=1011, message=b'engine unavailable')
        return ws

    app = web.Application()
    app.router.add_get('/', socket)
    async with TestServer(app) as server:
        async with ModelClient('lightnav', str(server.make_url('/')).replace('http:', 'ws:')) as client:
            with pytest.raises(ConnectionError, match='LightNav 模型连接已断开'):
                await asyncio.wait_for(client.predict(b'jpeg', 'go'), .5)


@pytest.mark.asyncio
async def test_cancelled_prediction_context_closes_reader_and_transport():
    request_seen = asyncio.Event()

    async def socket(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.receive_json()
        await ws.send_json({'action': 'login', 'data': {'rc': 0}})
        await ws.receive_json()
        request_seen.set()  # Never return a prediction; simulate an in-flight model call.
        async for _ in ws:
            pass
        return ws

    app = web.Application()
    app.router.add_get('/', socket)
    async with TestServer(app) as server:
        client = ModelClient('lightnav', str(server.make_url('/')).replace('http:', 'ws:'))

        async def run():
            async with client:
                await client.predict(b'jpeg', 'go')

        task = asyncio.create_task(run())
        await asyncio.wait_for(request_seen.wait(), .5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, .5)
        assert client._reader_task.done() and client.http.closed and client.ws.closed


@pytest.mark.asyncio
async def test_final_response_before_server_close_is_not_lost():
    async def socket(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.receive_json()
        await ws.send_json({'action': 'login', 'data': {'rc': 0}})
        message = await ws.receive_json()
        await ws.send_json({'action': 'next', 'data': {
            'rc': 0, 'seq': message['data']['seq'], 'actions': [[.2, 0, 0]], 'stop': True}})
        await ws.close()
        return ws

    app = web.Application()
    app.router.add_get('/', socket)
    async with TestServer(app) as server:
        async with ModelClient('lightnav', str(server.make_url('/')).replace('http:', 'ws:')) as client:
            assert (await client.predict(b'jpeg', 'go')).stop
            await asyncio.wait_for(client._reader_task, .5)
            with pytest.raises(ConnectionError):
                await client.predict(b'next-jpeg', 'go')


@pytest.mark.asyncio
async def test_login_timeout_closes_background_reader():
    async def socket(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for _ in ws:
            pass  # No login reply.
        return ws

    app = web.Application()
    app.router.add_get('/', socket)
    async with TestServer(app) as server:
        client = ModelClient('lightnav', str(server.make_url('/')).replace('http:', 'ws:'), timeout=.1)
        with pytest.raises(asyncio.TimeoutError):
            await client.__aenter__()
        assert client._reader_task.done() and client.http.closed and client.ws.closed


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
