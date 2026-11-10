"""Researcher model backends behind one small interface.

`lmstudio`  - any OpenAI-compatible local server (LM Studio, llama.cpp, vLLM).
`anthropic` - the Claude API via the official SDK.

Both map provider errors onto two classes so the agent loop can decide whether
to back off and retry (transient) or stop cleanly and keep its state (fatal).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol


class LLMTransientError(Exception):
    """Worth retrying after a pause: rate limit, overload, dropped connection."""


class LLMFatalError(Exception):
    """Retrying will not help: bad credentials, unknown model, refused request."""


@dataclass
class Reply:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0


class LLM(Protocol):
    name: str

    def complete(self, system: str, user: str) -> Reply: ...


_THINK = re.compile(r"<think>.*?</think>", re.S)


class LocalLLM:
    """OpenAI-compatible chat endpoint, by default LM Studio on localhost."""

    def __init__(self, model: str, base_url: str = "http://localhost:1234/v1",
                 max_tokens: int = 10000, temperature: float = 0.6,
                 reasoning_effort: str | None = "low"):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key="not-needed", timeout=1200, max_retries=0)
        self.model, self.max_tokens, self.temperature = model, max_tokens, temperature
        # Reasoning tokens count against max_tokens on local servers. Left at the
        # model default, a thinking model can spend the whole budget before it
        # writes any code, so the reply arrives empty.
        self.extra = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}
        self.name = f"local:{model}"

    def complete(self, system: str, user: str) -> Reply:
        import openai

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                **self.extra,
            )
        except (openai.AuthenticationError, openai.NotFoundError, openai.BadRequestError) as e:
            raise LLMFatalError(str(e)) from e
        except (openai.APIConnectionError, openai.RateLimitError, openai.APIStatusError) as e:
            raise LLMTransientError(str(e)) from e
        text = _THINK.sub("", resp.choices[0].message.content or "").strip()
        if resp.choices[0].finish_reason == "length":
            text += "\n[reply truncated at max_tokens]"
        usage = resp.usage
        return Reply(
            text,
            usage.prompt_tokens if usage else 0,
            usage.completion_tokens if usage else 0,
        )


class ClaudeLLM:
    """Claude via the Anthropic SDK. Credentials come from the environment."""

    def __init__(self, model: str = "claude-opus-5-5", effort: str = "high",
                 max_tokens: int = 32000):
        import anthropic

        self.client = anthropic.Anthropic(max_retries=3)
        self.model, self.effort, self.max_tokens = model, effort, max_tokens
        self.name = f"anthropic:{model}"

    def complete(self, system: str, user: str) -> Reply:
        import anthropic

        try:
            # The system prompt (task, contract, data profile) is identical on every
            # call in a run, so it is the cached prefix; the user turn is volatile.
            # Streaming keeps long thinking turns clear of HTTP timeouts.
            with self.client.beta.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                thinking={"type": "adaptive"},
                output_config={"effort": self.effort},
                # If a safety classifier declines a turn, let the API re-run it on
                # Anthropic's recommended fallback model instead of stalling the loop.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                system=[{"type": "text", "text": system,
                         "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
                messages=[{"role": "user", "content": user}],
            ) as stream:
                msg = stream.get_final_message()
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError,
                anthropic.NotFoundError, anthropic.BadRequestError) as e:
            raise LLMFatalError(f"{type(e).__name__}: {e}") from e
        except anthropic.RateLimitError as e:
            raise LLMTransientError(f"rate limited: {e}") from e
        except anthropic.APIStatusError as e:
            if e.status_code >= 500:
                raise LLMTransientError(f"server error {e.status_code}") from e
            raise LLMFatalError(f"API error {e.status_code}: {e}") from e
        except anthropic.APIConnectionError as e:
            raise LLMTransientError(f"connection error: {e}") from e

        if msg.stop_reason == "refusal":
            raise LLMFatalError(f"request refused: {msg.stop_details}")
        text = "".join(b.text for b in msg.content if b.type == "text")
        if msg.stop_reason == "max_tokens":
            text += "\n[reply truncated at max_tokens]"
        u = msg.usage
        read = getattr(u, "cache_read_input_tokens", 0) or 0
        written = getattr(u, "cache_creation_input_tokens", 0) or 0
        return Reply(text.strip(), u.input_tokens + read + written, u.output_tokens)


def make_llm(backend: str, model: str | None = None, base_url: str | None = None) -> LLM:
    if backend == "anthropic":
        return ClaudeLLM(model or "claude-opus-5-5")
    if backend == "lmstudio":
        if not model:
            raise ValueError("--model is required for the lmstudio backend")
        return LocalLLM(model, base_url or "http://localhost:1234/v1")
    raise ValueError(f"unknown backend {backend!r}")
