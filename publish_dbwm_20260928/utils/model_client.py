"""Minimal OpenAI-compatible client for models served by micuapi.

Example:
    from utils.model_client import chat_text

    answer = chat_text("Return only the word: ok")
    print(answer)

For function-calling or structured outputs use ``chat``. It returns the full
OpenAI Chat Completions response dictionary, including ``choices[0]``.
"""

from __future__ import annotations

import json
import os
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


API_KEY = os.environ.get("MICUAPI_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
BASE_URL = "https://www.micuapi.ai/v1"
DEFAULT_MODEL = "deepseek-v4-pro"
DEFAULT_TIMEOUT_SECONDS = 120

# micuapi documents a browser User-Agent for external calls to Chinese models.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:149.0) "
    "Gecko/20100101 Firefox/149.0"
)


class ModelAPIError(RuntimeError):
    """An HTTP or protocol error returned by the model provider."""


def chat(
    messages: Sequence[Mapping[str, Any]],
    *,
    model: str = DEFAULT_MODEL,
    tools: Sequence[Mapping[str, Any]] | None = None,
    tool_choice: str | Mapping[str, Any] | None = None,
    temperature: float | None = 0.0,
    thinking: bool | None = None,
    max_tokens: int | None = None,
    response_format: Mapping[str, Any] | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    **extra: Any,
) -> dict[str, Any]:
    """Call ``POST /chat/completions`` and return its decoded JSON response.

    ``messages`` follows the OpenAI chat-completions schema. Optional keyword
    arguments are passed through so the same helper supports tool calls and
    JSON-mode requests used by the consistency-data generator. Set ``thinking``
    to ``True`` or ``False`` to explicitly enable or disable DeepSeek thinking;
    leave it as ``None`` to use the provider default.
    """
    if not API_KEY:
        raise ModelAPIError("Set MICUAPI_API_KEY or OPENAI_API_KEY before calling the model.")
    payload: dict[str, Any] = {"model": model, "messages": list(messages)}
    if temperature is not None:
        payload["temperature"] = temperature
    if thinking is not None:
        payload["thinking"] = {"type": "enabled" if thinking else "disabled"}
    if tools is not None:
        payload["tools"] = list(tools)
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if response_format is not None:
        payload["response_format"] = dict(response_format)
    payload.update(extra)

    request = Request(
        f"{BASE_URL}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ModelAPIError(f"micuapi returned HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise ModelAPIError(f"could not reach micuapi: {exc.reason}") from exc

    try:
        result = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ModelAPIError(f"micuapi returned non-JSON content: {body[:500]!r}") from exc
    if "error" in result:
        raise ModelAPIError(f"micuapi returned an error: {result['error']}")
    return result


def chat_text(
    prompt: str,
    *,
    system: str | None = None,
    model: str = DEFAULT_MODEL,
    **kwargs: Any,
) -> str:
    """Send one prompt and return the assistant text content only."""
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    response = chat(messages, model=model, **kwargs)
    try:
        content = response["choices"][0]["message"]["content"]
    except (IndexError, KeyError, TypeError) as exc:
        raise ModelAPIError(f"unexpected chat-completions response: {response}") from exc
    if not isinstance(content, str):
        raise ModelAPIError(f"expected text content, received: {content!r}")
    return content


if __name__ == "__main__":
    print(chat_text("Reply with exactly: ok", max_tokens=64))
