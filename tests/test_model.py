import json
from types import SimpleNamespace

import pytest
from PIL import Image

from scopepilot.model import AnthropicBackend, ModelError, _parse_claude_code_output

SCHEMA = {"type": "object", "properties": {}, "additionalProperties": False}


class FakeMessages:
    def __init__(self, response):
        self.response = response
        self.request = None

    def create(self, **request):
        self.request = request
        return self.response


def fake_client(response):
    stable, beta = FakeMessages(response), FakeMessages(response)
    return SimpleNamespace(messages=stable, beta=SimpleNamespace(messages=beta)), stable, beta


def reply(text='{"ok": true}', stop_reason="end_turn"):
    return SimpleNamespace(
        stop_reason=stop_reason, content=[SimpleNamespace(type="text", text=text)]
    )


def test_request_carries_image_question_and_schema():
    client, _, beta = fake_client(reply())
    backend = AnthropicBackend(client=client)

    assert backend.ask("system", "question", [Image.new("RGB", (4, 4))], SCHEMA) == {"ok": True}

    request = beta.request
    image, text = request["messages"][0]["content"]
    assert image["type"] == "image" and image["source"]["media_type"] == "image/png"
    assert text == {"type": "text", "text": "question"}
    assert request["output_config"]["format"] == {"type": "json_schema", "schema": SCHEMA}
    assert request["fallbacks"] == "default"


def test_fallbacks_off_uses_the_stable_endpoint():
    client, stable, beta = fake_client(reply())

    AnthropicBackend(client=client, fallbacks=False).ask("s", "q", [], SCHEMA)

    assert beta.request is None
    assert "fallbacks" not in stable.request and "betas" not in stable.request


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
def test_unusable_responses_raise(stop_reason):
    client, _, _ = fake_client(reply(stop_reason=stop_reason))

    with pytest.raises(ModelError):
        AnthropicBackend(client=client).ask("s", "q", [], SCHEMA)


def test_claude_code_structured_output_is_preferred():
    stdout = json.dumps({"is_error": False, "result": "done", "structured_output": {"a": 1}})

    assert _parse_claude_code_output(stdout) == {"a": 1}


def test_claude_code_result_text_is_parsed_when_there_is_no_structured_output():
    assert _parse_claude_code_output(json.dumps({"result": '{"a": 2}'})) == {"a": 2}


@pytest.mark.parametrize(
    "stdout",
    ["not json", json.dumps({"is_error": True, "result": "boom"}), json.dumps({"result": "hi"})],
)
def test_claude_code_bad_output_raises(stdout):
    with pytest.raises(ModelError):
        _parse_claude_code_output(stdout)
