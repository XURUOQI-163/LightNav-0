import asyncio
import base64
import json
from contextlib import asynccontextmanager
from io import BytesIO

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from PIL import Image
from room_memory.semantics import (
    QwenRoomSemanticClassifier,
    SemanticClassificationError,
    SemanticFrame,
)


def jpeg_bytes(color="red"):
    output = BytesIO()
    Image.new("RGB", (32, 32), color).save(output, format="JPEG")
    return output.getvalue()


def semantic_answer(**changes):
    answer = {
        "room_type": "bedroom",
        "probabilities": {"bedroom": 3, "unknown": 1},
        "confidence": 0.75,
        "representative_frame_id": "view-b",
    }
    answer.update(changes)
    return answer


def chat_response(answer):
    return {
        "id": "test-completion",
        "object": "chat.completion",
        "created": 1,
        "model": "qwen-vl-plus",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": json.dumps(answer)},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 30, "completion_tokens": 40, "total_tokens": 70},
    }


@asynccontextmanager
async def api_server(handler):
    app = web.Application()
    app.router.add_post("/compatible-mode/v1/chat/completions", handler)
    async with TestServer(app) as server:
        yield str(server.make_url("/compatible-mode/v1"))


@pytest.mark.asyncio
async def test_multiple_views_are_sent_together_and_probabilities_are_normalized():
    observed = []

    async def classify(request):
        observed.append((await request.json(), request.headers.get("Authorization")))
        return web.json_response(chat_response(semantic_answer()))

    frames = [SemanticFrame("view-a", jpeg_bytes()), SemanticFrame("view-b", jpeg_bytes("blue"))]
    async with api_server(classify) as base_url:
        result = await QwenRoomSemanticClassifier(api_key="test-key", base_url=base_url).classify(frames)

    assert result.room_type == "bedroom"
    assert result.confidence == 0.75
    assert result.representative_frame_id == "view-b"
    assert result.probabilities["bedroom"] == 0.75
    assert result.probabilities["unknown"] == 0.25
    assert result.probabilities["kitchen"] == 0.0
    assert sum(result.probabilities.values()) == pytest.approx(1.0)
    assert len(observed) == 1
    body, authorization = observed[0]
    assert authorization == "Bearer test-key"
    assert body["model"] == "qwen-vl-plus"
    assert body["response_format"] == {"type": "json_object"}
    content = body["messages"][0]["content"]
    images = [item["image_url"]["url"] for item in content if item["type"] == "image_url"]
    assert len(images) == 2
    assert all(image.startswith("data:image/jpeg;base64,") for image in images)
    assert [base64.b64decode(image.split(",", 1)[1]) for image in images] == [frame.jpeg for frame in frames]
    prompt = " ".join(item["text"] for item in content if item["type"] == "text")
    assert "view-a" in prompt and "view-b" in prompt
    assert "unknown" in prompt and "representative_frame_id" in prompt


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changes",
    [
        {"room_type": "made-up-room"},
        {"room_type": ["bedroom"]},
        {"probabilities": {"bedroom": -0.2, "unknown": 1.2}},
        {"probabilities": {"bedroom": float("nan")}},
        {"probabilities": {"bedroom": float("inf")}},
        {"probabilities": {"bedroom": True}},
        {"probabilities": {"bedroom": "0.9"}},
        {"probabilities": {"bedroom": 0.0, "unknown": 0.0}},
        {"probabilities": {"made-up-room": 1.0}},
        {"probabilities": {}},
        {"probabilities": None},
        {"confidence": -0.1},
        {"confidence": 1.1},
        {"confidence": float("nan")},
        {"confidence": True},
        {"representative_frame_id": "unseen-view"},
        {"representative_frame_id": ["view-b"]},
    ],
)
async def test_malformed_semantic_fields_are_rejected(changes):
    async def classify(request):
        return web.json_response(chat_response(semantic_answer(**changes)))

    async with api_server(classify) as base_url:
        classifier = QwenRoomSemanticClassifier(api_key="test-key", base_url=base_url)
        with pytest.raises(SemanticClassificationError, match="invalid"):
            await classifier.classify([SemanticFrame("view-b", jpeg_bytes())])


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", ["room_type", "probabilities", "confidence", "representative_frame_id"])
async def test_missing_semantic_fields_are_rejected(missing):
    async def classify(request):
        answer = semantic_answer()
        del answer[missing]
        return web.json_response(chat_response(answer))

    async with api_server(classify) as base_url:
        with pytest.raises(SemanticClassificationError, match="invalid"):
            await QwenRoomSemanticClassifier(api_key="test-key", base_url=base_url).classify(
                [SemanticFrame("view-b", jpeg_bytes())]
            )


@pytest.mark.asyncio
async def test_missing_api_key_fails_without_network(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    classifier = QwenRoomSemanticClassifier(base_url="http://127.0.0.1:1")
    with pytest.raises(SemanticClassificationError, match="DASHSCOPE_API_KEY"):
        await classifier.classify([SemanticFrame("view-b", jpeg_bytes())])


@pytest.mark.asyncio
async def test_environment_api_key_is_supported(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-environment-key")
    authorizations = []

    async def classify(request):
        authorizations.append(request.headers.get("Authorization"))
        return web.json_response(chat_response(semantic_answer()))

    async with api_server(classify) as base_url:
        await QwenRoomSemanticClassifier(base_url=base_url).classify([SemanticFrame("view-b", jpeg_bytes())])
    assert authorizations == ["Bearer test-environment-key"]


@pytest.mark.asyncio
async def test_five_views_can_return_cautious_unknown_with_an_input_representative():
    async def classify(request):
        return web.json_response(chat_response(semantic_answer(
            room_type="unknown", probabilities={"unknown": 1},
            confidence=0.2, representative_frame_id="view-4",
        )))

    async with api_server(classify) as base_url:
        result = await QwenRoomSemanticClassifier(api_key="test-key", base_url=base_url).classify(
            [SemanticFrame("view-{}".format(index), jpeg_bytes()) for index in range(5)]
        )
    assert result.room_type == "unknown"
    assert result.probabilities["unknown"] == 1.0
    assert result.confidence == 0.2
    assert result.representative_frame_id == "view-4"


@pytest.mark.asyncio
async def test_connection_failure_has_a_sanitized_error():
    with pytest.raises(SemanticClassificationError, match="request failed") as caught:
        await QwenRoomSemanticClassifier(api_key="test-key", base_url="http://127.0.0.1:1").classify(
            [SemanticFrame("view-b", jpeg_bytes())]
        )
    assert "127.0.0.1" not in str(caught.value)
    assert "test-key" not in str(caught.value)
    assert caught.value.__suppress_context__


@pytest.mark.asyncio
async def test_http_errors_do_not_expose_response_body_or_api_key():
    calls = []

    async def classify(request):
        calls.append(1)
        return web.Response(status=429, text="private provider text test-key")

    async with api_server(classify) as base_url:
        with pytest.raises(SemanticClassificationError, match="429") as caught:
            await QwenRoomSemanticClassifier(api_key="test-key", base_url=base_url).classify(
                [SemanticFrame("view-b", jpeg_bytes())]
            )
    assert "private provider text" not in str(caught.value)
    assert "test-key" not in str(caught.value)
    assert calls == [1]


@pytest.mark.asyncio
async def test_timeout_is_bounded_and_does_not_retry():
    calls = []

    async def classify(request):
        calls.append(1)
        await asyncio.sleep(0.1)
        return web.json_response(chat_response(semantic_answer()))

    async with api_server(classify) as base_url:
        with pytest.raises(SemanticClassificationError, match="timed out"):
            await QwenRoomSemanticClassifier(api_key="test-key", base_url=base_url, timeout=0.02).classify(
                [SemanticFrame("view-b", jpeg_bytes())]
            )
    assert calls == [1]


@pytest.mark.asyncio
@pytest.mark.parametrize("frame_ids", [[], [str(i) for i in range(6)], ["same", "same"], [""]])
async def test_invalid_frame_count_or_ids_fail_before_network(frame_ids):
    classifier = QwenRoomSemanticClassifier(api_key="test-key", base_url="http://127.0.0.1:1")
    with pytest.raises(ValueError):
        await classifier.classify([SemanticFrame(frame_id, jpeg_bytes()) for frame_id in frame_ids])


@pytest.mark.asyncio
@pytest.mark.parametrize("jpeg", [b"", b"not a JPEG", None])
async def test_invalid_image_bytes_fail_before_network(jpeg):
    with pytest.raises(ValueError, match="JPEG"):
        await QwenRoomSemanticClassifier(api_key="test-key", base_url="http://127.0.0.1:1").classify(
            [SemanticFrame("view-b", jpeg)]
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body", [b"not JSON", b"{}", b"[]", b"x" * 70_000],
    ids=["invalid-json", "missing-choices", "wrong-envelope", "excessive-body"],
)
async def test_invalid_or_excessive_provider_responses_are_bounded(body):
    async def classify(request):
        return web.Response(body=body, content_type="application/json")

    async with api_server(classify) as base_url:
        with pytest.raises(SemanticClassificationError) as caught:
            await QwenRoomSemanticClassifier(api_key="test-key", base_url=base_url).classify(
                [SemanticFrame("view-b", jpeg_bytes())]
            )
    assert len(str(caught.value)) < 150


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan"), True])
def test_invalid_timeouts_cannot_disable_the_request_deadline(timeout):
    with pytest.raises(ValueError, match="timeout"):
        QwenRoomSemanticClassifier(api_key="test-key", timeout=timeout)
