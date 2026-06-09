from __future__ import annotations

import importlib
import sys
import types

import pytest

from webwright.models.openai_responses_utils import (
    ensure_response_completed,
    extract_response_text,
)


def test_extract_response_text_ignores_reasoning_and_keeps_final_message() -> None:
    payload = {
        "status": "completed",
        "output": [
            {
                "type": "reasoning",
                "content": [{"type": "reasoning_text", "text": "internal reasoning"}],
            },
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "final answer"}],
            },
        ],
    }

    assert extract_response_text(payload) == "final answer"


def test_extract_response_text_strips_leading_whitespace() -> None:
    payload = {
        "status": "completed",
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "\n\nfinal answer"}],
            },
        ],
    }

    assert extract_response_text(payload) == "final answer"


def test_ensure_response_completed_accepts_completed_payload() -> None:
    payload = {"status": "completed", "incomplete_details": None}

    ensure_response_completed(payload)


def test_ensure_response_completed_rejects_incomplete_payload() -> None:
    payload = {
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "output": [],
    }

    with pytest.raises(RuntimeError, match="max_output_tokens"):
        ensure_response_completed(payload)


def _load_openai_model_module(monkeypatch: pytest.MonkeyPatch):
    fake_httpx = types.SimpleNamespace(
        TimeoutException=type("TimeoutException", (Exception,), {}),
        NetworkError=type("NetworkError", (Exception,), {}),
        RemoteProtocolError=type("RemoteProtocolError", (Exception,), {}),
    )
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)
    sys.modules.pop("webwright.models.base", None)
    sys.modules.pop("webwright.models.openai_model", None)
    return importlib.import_module("webwright.models.openai_model")


@pytest.mark.skipif(sys.version_info < (3, 10), reason="project requires Python 3.10+")
def test_openai_model_allows_missing_key_for_non_openai_endpoint(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    module = _load_openai_model_module(monkeypatch)

    model = module.OpenAIModel(
        model_name="Qwen/Qwen3.6-35B-A3B-FP8",
        openai_endpoint="http://192.168.1.171:8000/v1/responses",
    )

    assert model.config.openai_api_key == ""
    assert "Authorization" not in model._request_headers()


@pytest.mark.skipif(sys.version_info < (3, 10), reason="project requires Python 3.10+")
def test_openai_model_requires_key_for_api_openai(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    module = _load_openai_model_module(monkeypatch)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        module.OpenAIModel(
            model_name="gpt-5.4",
            openai_endpoint="https://api.openai.com/v1/responses",
        )
