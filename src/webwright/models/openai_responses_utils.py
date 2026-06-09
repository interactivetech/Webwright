from __future__ import annotations

from typing import Any


def extract_response_text(payload: dict[str, Any]) -> str:
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text:
        return output_text

    texts: list[str] = []
    for item in payload.get("output") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if not isinstance(content, dict):
                continue
            if isinstance(content.get("text"), str):
                texts.append(content["text"])
            elif isinstance(content.get("output_text"), str):
                texts.append(content["output_text"])
    return "\n".join(texts).lstrip()


def ensure_response_completed(payload: dict[str, Any]) -> None:
    status = str(payload.get("status", "") or "")
    incomplete_details = payload.get("incomplete_details")
    if status == "completed" and incomplete_details in (None, {}):
        return

    reason = ""
    if isinstance(incomplete_details, dict):
        reason = str(incomplete_details.get("reason", "") or "")
    if not reason and status:
        reason = status
    if not reason:
        reason = "unknown"
    if reason == "max_output_tokens":
        raise RuntimeError(
            "Responses API payload incomplete: max_output_tokens. "
            "Increase model.max_output_tokens or the tool call's max_output_tokens."
        )
    raise RuntimeError(f"Responses API payload incomplete: {reason}")


__all__ = ["ensure_response_completed", "extract_response_text"]
