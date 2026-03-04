"""Model-agnostic LLM client using OpenAI-compatible API.

All LLM calls in the pipeline go through this client. Supports Cerebras (Qwen3),
OpenAI (GPT-4o-mini), Together AI (Kimi K2), or any OpenAI-compatible endpoint.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from openai import OpenAI

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "default.yaml"


@dataclass
class LLMCallRecord:
    """Record of a single LLM call for cost tracking."""

    model: str
    prompt_hash: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    success: bool
    error: str | None = None


@dataclass
class LLMClient:
    """Model-agnostic LLM client wrapping the OpenAI SDK.

    Args:
        api_key: API key for the provider.
        base_url: Provider endpoint URL.
        model: Model identifier string.
        temperature: Sampling temperature (0 for deterministic).
        max_retries: Maximum retry attempts on failure.
        timeout_seconds: Request timeout.
    """

    api_key: str
    base_url: str
    model: str
    temperature: float = 0.0
    max_retries: int = 3
    timeout_seconds: int = 30
    min_call_interval: float = 0.0  # seconds between calls (rate limit protection)
    call_log: list[LLMCallRecord] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout_seconds,
        )
        self._last_call_time: float = 0.0

    @classmethod
    def from_config(cls, api_key: str, config_path: str | Path | None = None) -> LLMClient:
        """Create an LLMClient from a YAML config file.

        Args:
            api_key: API key for the provider.
            config_path: Path to config YAML. Defaults to config/default.yaml.
        """
        path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
        with open(path) as f:
            cfg = yaml.safe_load(f)["llm"]

        return cls(
            api_key=api_key,
            base_url=cfg["base_url"],
            model=cfg["model"],
            temperature=cfg.get("temperature", 0),
            max_retries=cfg.get("max_retries", 3),
            timeout_seconds=cfg.get("timeout_seconds", 30),
            min_call_interval=cfg.get("min_call_interval", 0.0),
        )

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        json_mode: bool = False,
        max_tokens: int = 4096,
    ) -> str:
        """Send a completion request and return the response text.

        Args:
            prompt: The user message / prompt content.
            system: Optional system message.
            json_mode: If True, request JSON-formatted output.
            max_tokens: Maximum tokens in the response.

        Returns:
            The assistant's response text.

        Raises:
            RuntimeError: If all retry attempts are exhausted.
        """
        # Enforce minimum interval between calls (rate limit protection)
        if self.min_call_interval > 0 and self._last_call_time > 0:
            elapsed = time.monotonic() - self._last_call_time
            if elapsed < self.min_call_interval:
                time.sleep(self.min_call_interval - elapsed)

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:12]
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            start = time.monotonic()
            try:
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": max_tokens,
                }
                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}

                response = self._client.chat.completions.create(**kwargs)
                latency_ms = int((time.monotonic() - start) * 1000)

                usage = response.usage
                record = LLMCallRecord(
                    model=self.model,
                    prompt_hash=prompt_hash,
                    input_tokens=usage.prompt_tokens if usage else 0,
                    output_tokens=usage.completion_tokens if usage else 0,
                    latency_ms=latency_ms,
                    success=True,
                )
                self.call_log.append(record)

                logger.info(
                    "LLM call success",
                    extra={
                        "model": self.model,
                        "prompt_hash": prompt_hash,
                        "input_tokens": record.input_tokens,
                        "output_tokens": record.output_tokens,
                        "latency_ms": latency_ms,
                        "attempt": attempt,
                    },
                )

                content = response.choices[0].message.content
                self._last_call_time = time.monotonic()
                return content or ""

            except Exception as e:
                latency_ms = int((time.monotonic() - start) * 1000)
                last_error = e

                record = LLMCallRecord(
                    model=self.model,
                    prompt_hash=prompt_hash,
                    input_tokens=0,
                    output_tokens=0,
                    latency_ms=latency_ms,
                    success=False,
                    error=str(e),
                )
                self.call_log.append(record)

                logger.warning(
                    "LLM call failed (attempt %d/%d): %s",
                    attempt,
                    self.max_retries,
                    e,
                    extra={"model": self.model, "prompt_hash": prompt_hash},
                )

                self._last_call_time = time.monotonic()
                if attempt < self.max_retries:
                    backoff = 2 ** (attempt - 1)
                    time.sleep(backoff)

        raise RuntimeError(
            f"LLM call failed after {self.max_retries} attempts: {last_error}"
        )

    def complete_json(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        """Send a completion request and parse the response as JSON.

        Args:
            prompt: The user message / prompt content.
            system: Optional system message.
            max_tokens: Maximum tokens in the response.

        Returns:
            Parsed JSON dict from the response.

        Raises:
            RuntimeError: If all retries exhausted or JSON parsing fails after retry.
        """
        raw = self.complete(prompt, system=system, json_mode=True, max_tokens=max_tokens)

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Invalid JSON from LLM, retrying with fix prompt")
            fix_prompt = (
                "The following text was supposed to be valid JSON but has syntax errors. "
                "Fix it and return ONLY the corrected JSON, nothing else.\n\n"
                f"{raw}"
            )
            fixed = self.complete(fix_prompt, json_mode=True, max_tokens=max_tokens)
            return json.loads(fixed)

    def total_tokens(self) -> dict[str, int]:
        """Return cumulative token usage across all calls."""
        input_total = sum(r.input_tokens for r in self.call_log)
        output_total = sum(r.output_tokens for r in self.call_log)
        return {"input": input_total, "output": output_total, "total": input_total + output_total}
