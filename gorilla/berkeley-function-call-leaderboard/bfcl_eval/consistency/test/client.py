"""OpenAI-compatible transport, including optional local vLLM endpoints."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen


def observed_reasoning_tokens(value):
    """Inspect nested gateway billing metadata as well as normalized usage."""
    if isinstance(value, dict):
        own = value.get("reasoning_tokens", 0)
        own = own if isinstance(own, (int, float)) else 0
        return max([own, *[observed_reasoning_tokens(v) for v in value.values()]])
    if isinstance(value, list):
        return max([0, *[observed_reasoning_tokens(v) for v in value]])
    return 0


def parse_observation(content):
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Empty or non-text model content")
    text = content.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if lines[0] in {"```", "```json"}:
            text = "\n".join(lines[1:-1])
    def invalid_constant(value):
        raise ValueError(f"Non-finite JSON value: {value}")
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    value = json.loads(text, parse_constant=invalid_constant, object_pairs_hook=unique_keys)
    json.dumps(value, allow_nan=False)  # Also reject exponent overflow (e.g. 1e400).
    return value


class ChatClient:
    def __init__(self, *, model, thinking="enabled", max_tokens=32768,
                 timeout=300, base_url=None, api_key_env="WM_API_KEY", retries=1,
                 api_format="openai", request_options=None):
        self.model, self.thinking = model, thinking
        self.max_tokens, self.timeout = max_tokens, timeout
        self.base_url, self.api_key_env, self.retries = base_url, api_key_env, retries
        self.api_format = api_format
        self.request_options = dict(request_options or {})
        protected = {"model", "messages", "system", "thinking", "max_tokens", "temperature", "timeout"}
        if protected & self.request_options.keys():
            raise ValueError("Request options cannot override experiment settings")

    @staticmethod
    def _gateway():
        workspace = Path(__file__).resolve().parents[5]
        if str(workspace) not in sys.path:
            sys.path.insert(0, str(workspace))
        from utils import model_client
        return model_client

    def _request_anthropic(self, messages):
        """Use native Messages when gateway Chat translation loses thinking controls."""
        headers = {"Content-Type": "application/json", "Accept": "application/json",
                   "anthropic-version": "2023-06-01"}
        base_url, key = self.base_url, os.environ.get(self.api_key_env)
        if base_url is None:
            gateway = self._gateway()
            base_url, key = gateway.BASE_URL, gateway.API_KEY
            headers["User-Agent"] = gateway.USER_AGENT
        if key:
            headers.update({"Authorization": f"Bearer {key}", "x-api-key": key})
        payload = {"model": self.model, "max_tokens": self.max_tokens,
                   "messages": [m for m in messages if m["role"] != "system"]}
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        if system:
            payload["system"] = system
        if self.thinking == "disabled":
            payload.update(thinking={"type": "disabled"}, temperature=0.0)
        elif self.thinking == "enabled":
            if self.max_tokens < 1025:
                raise ValueError("Anthropic thinking requires max_tokens >= 1025")
            payload["thinking"] = {"type": "enabled", "budget_tokens": min(8192, self.max_tokens - 1)}
        else:
            payload["temperature"] = 0.0
        payload.update(self.request_options)
        request = Request(base_url.rstrip("/") + "/messages",
                          data=json.dumps(payload).encode("utf-8"), headers=headers)
        with urlopen(request, timeout=self.timeout) as response:
            raw = json.load(response)
        if raw.get("error"):
            return raw
        content = "".join(b.get("text", "") for b in raw.get("content", []) if b.get("type") == "text")
        reasoning = "".join(b.get("thinking", "") for b in raw.get("content", []) if b.get("type") == "thinking")
        original_usage = raw.get("usage", {})
        prompt_tokens = sum(original_usage.get(k, 0) for k in
                            ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
        output_tokens = original_usage.get("output_tokens", 0)
        return {"id": raw.get("id"), "model": raw.get("model"),
                "choices": [{"message": {"role": "assistant", "content": content,
                                          "reasoning_content": reasoning},
                             "finish_reason": "length" if raw.get("stop_reason") == "max_tokens" else raw.get("stop_reason")}],
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": output_tokens,
                          "total_tokens": prompt_tokens + output_tokens},
                "provider_response": raw, "api_format": "anthropic"}

    def _request(self, messages):
        if self.api_format == "anthropic":
            return self._request_anthropic(messages)
        thinking = {"enabled": True, "disabled": False, "omit": None}[self.thinking]
        if self.base_url is None:
            # Reuse this workspace's existing gateway configuration; never copy
            # its credentials into prompts, manifests or generated source.
            return self._gateway().chat(messages, model=self.model, thinking=thinking,
                        max_tokens=self.max_tokens, timeout=self.timeout, temperature=0.0,
                        **self.request_options)
        payload = {"model": self.model, "messages": messages,
                   "max_tokens": self.max_tokens, "temperature": 0.0}
        if thinking is not None:
            payload["thinking"] = {"type": "enabled" if thinking else "disabled"}
        payload.update(self.request_options)
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if key := os.environ.get(self.api_key_env):
            headers["Authorization"] = f"Bearer {key}"
        request = Request(self.base_url.rstrip("/") + "/chat/completions",
                          data=json.dumps(payload).encode("utf-8"), headers=headers)
        with urlopen(request, timeout=self.timeout) as response:
            return json.load(response)

    def predict(self, messages):
        started = time.monotonic()
        errors = []
        for attempt in range(self.retries + 1):
            try:
                response = self._request(messages)
                if not isinstance(response, dict) or response.get("error"):
                    raise ValueError("Provider returned an invalid response or error")
                break
            except Exception as exc:
                # No response was usable. Only transport/protocol failures are
                # retried; never resample a syntactically valid bad prediction.
                errors.append(f"{type(exc).__name__}: {str(exc)[:1000]}")
                if attempt == self.retries:
                    return {"status": "api_error", "errors": errors,
                            "seconds": time.monotonic() - started}
                time.sleep(min(2 ** attempt, 8))
        result = {"response": response, "seconds": time.monotonic() - started,
                  "transport_errors": errors}
        try:
            choice = response["choices"][0]
            message = choice["message"]
            reasoning = message.get("reasoning_content") or message.get("reasoning")
            reasoning_tokens = observed_reasoning_tokens(response)
            native_blocks = response.get("provider_response", {}).get("content", [])
            native_thinking = any(b.get("type") in {"thinking", "redacted_thinking"} for b in native_blocks)
            result["thinking_observed"] = bool(reasoning or reasoning_tokens or native_thinking)
            result["reasoning_tokens_observed"] = reasoning_tokens
            if self.thinking == "disabled" and result["thinking_observed"]:
                result.update(status="thinking_not_disabled", error="Provider returned reasoning despite disabled thinking")
            elif choice.get("finish_reason") == "length":
                result.update(status="truncated", error="Model reached max_tokens")
            else:
                result.update(status="ok", observation=parse_observation(choice["message"]["content"]))
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            result.update(status="invalid_json", error=str(exc))
        return result
