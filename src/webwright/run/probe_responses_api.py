from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import urllib.request
from pathlib import Path
from typing import Any

from webwright.models.openai_responses_utils import (
    ensure_response_completed,
    extract_response_text,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe an OpenAI-compatible Responses API endpoint and inspect extraction."
    )
    parser.add_argument("--endpoint", required=True, help="Responses API endpoint URL.")
    parser.add_argument("--model", required=True, help="Model ID to call.")
    parser.add_argument("--prompt", required=True, help="User prompt text.")
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        help="Optional local image path. Repeat to attach multiple images.",
    )
    parser.add_argument(
        "--schema",
        default="",
        help="Optional path to a JSON schema object for strict JSON output.",
    )
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--api-key", default="", help="Optional bearer token.")
    parser.add_argument("--timeout-seconds", type=int, default=120)
    return parser


def _load_schema(path: str) -> dict[str, Any]:
    if not path:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Schema file must contain a JSON object: {path}")
    return data


def _build_payload(args: argparse.Namespace) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "input_text", "text": args.prompt}]
    for image_path in args.image:
        path = Path(image_path)
        mime_type, _ = mimetypes.guess_type(str(path))
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        content.append(
            {
                "type": "input_image",
                "image_url": f"data:{mime_type or 'image/png'};base64,{encoded}",
                "detail": "high",
            }
        )

    payload: dict[str, Any] = {
        "model": args.model,
        "input": [{"type": "message", "role": "user", "content": content}],
        "max_output_tokens": args.max_output_tokens,
    }

    schema = _load_schema(args.schema)
    if schema:
        payload["text"] = {
            "format": {
                "type": "json_schema",
                "name": "probe_schema",
                "schema": schema,
                "strict": True,
            }
        }
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    payload = _build_payload(args)
    request = urllib.request.Request(
        args.endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
    )
    with urllib.request.urlopen(request, timeout=args.timeout_seconds) as response:
        response_payload = json.loads(response.read().decode("utf-8"))

    extracted_text = ""
    completion_error = ""
    try:
        ensure_response_completed(response_payload)
        extracted_text = extract_response_text(response_payload)
    except Exception as exc:  # pragma: no cover - experiment helper
        completion_error = str(exc)

    print(
        json.dumps(
            {
                "status": response_payload.get("status"),
                "incomplete_details": response_payload.get("incomplete_details"),
                "extracted_text": extracted_text,
                "completion_error": completion_error,
                "response_payload": response_payload,
            },
            ensure_ascii=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
