from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol


DEFAULT_CLEANUP_MODEL = "google/gemini-2.5-flash"
DEFAULT_OPENROUTER_URL = "https://openrouter.ai/api/v1"
DEFAULT_CLEANUP_PROMPT = (
    "Rewrite this segment description to keep only the main visible human actions and "
    "interactions. Remove repeated background details and generic scene context. "
    "Be short, factual, and do not invent anything not clearly supported by the text."
)


class DescriptionCleanupModel(Protocol):
    def cleanup_description(self, description: str) -> str:
        """Return a cleaned, factual segment description."""


@dataclass
class RetryConfig:
    max_retries: int = 3
    backoff_base_seconds: float = 20.0
    max_backoff_seconds: float = 60.0


def _sleep_with_message(delay_seconds: float) -> None:
    if delay_seconds > 0:
        time.sleep(delay_seconds)


def _extract_retry_delay_seconds(details: str) -> float | None:
    text = str(details or "")
    patterns = [
        r'"retryDelay"\s*:\s*"(\d+)s"',
        r"retry in ([0-9]+(?:\.[0-9]+)?)s",
        r"retry shortly",
    ]
    for pattern in patterns[:2]:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    if re.search(patterns[2], text, flags=re.IGNORECASE):
        return None
    return None


def _compute_retry_delay_seconds(
    *,
    attempt_index: int,
    details: str,
    retry_config: RetryConfig,
) -> float:
    parsed_delay = _extract_retry_delay_seconds(details)
    if parsed_delay is not None:
        return min(parsed_delay, retry_config.max_backoff_seconds)

    exponential_delay = retry_config.backoff_base_seconds * (2 ** attempt_index)
    return min(exponential_delay, retry_config.max_backoff_seconds)


def _request_json_with_retry(
    request: urllib.request.Request,
    *,
    timeout_seconds: float,
    retry_config: RetryConfig,
    error_label: str,
) -> dict:
    total_attempts = max(1, int(retry_config.max_retries) + 1)

    for attempt_index in range(total_attempts):
        if retry_config.request_delay_seconds > 0:
            _sleep_with_message(retry_config.request_delay_seconds)

        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            is_retryable = int(exc.code) == 429
            is_last_attempt = attempt_index >= total_attempts - 1
            if not is_retryable or is_last_attempt:
                raise RuntimeError(f"{error_label} failed with HTTP {exc.code}: {details}") from exc

            retry_delay = _compute_retry_delay_seconds(
                attempt_index=attempt_index,
                details=details,
                retry_config=retry_config,
            )
            print(
                f"[retry] {error_label} hit HTTP 429, sleeping {retry_delay:.1f}s "
                f"before retry {attempt_index + 2}/{total_attempts}"
            )
            _sleep_with_message(retry_delay)
        except urllib.error.URLError as exc:
            raise RuntimeError(f"{error_label} failed: {exc}") from exc

    raise RuntimeError(f"{error_label} failed after exhausting retries.")


def _call_with_retry(
    *,
    retry_config: RetryConfig,
    error_label: str,
    request_fn,
):
    total_attempts = max(1, int(retry_config.max_retries) + 1)

    for attempt_index in range(total_attempts):
        try:
            return request_fn()
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            response = getattr(exc, "response", None)
            response_text = ""
            if response is not None:
                response_text = str(getattr(response, "text", "") or "")
            details = response_text or str(exc)
            is_retryable = int(status_code) == 429 if status_code is not None else "429" in details
            is_last_attempt = attempt_index >= total_attempts - 1
            if not is_retryable or is_last_attempt:
                if status_code is not None:
                    raise RuntimeError(f"{error_label} failed with HTTP {status_code}: {details}") from exc
                raise RuntimeError(f"{error_label} failed: {details}") from exc

            retry_delay = _compute_retry_delay_seconds(
                attempt_index=attempt_index,
                details=details,
                retry_config=retry_config,
            )
            print(
                f"[retry] {error_label} hit HTTP 429, sleeping {retry_delay:.1f}s "
                f"before retry {attempt_index + 2}/{total_attempts}"
            )
            _sleep_with_message(retry_delay)

    raise RuntimeError(f"{error_label} failed after exhausting retries.")


@dataclass
class GeminiCleanupModel:
    api_key: str
    model_name: str = DEFAULT_CLEANUP_MODEL
    prompt: str = DEFAULT_CLEANUP_PROMPT
    timeout_seconds: float = 60.0
    temperature: float = 0.1
    retry_config: RetryConfig = field(default_factory=RetryConfig)

    def cleanup_description(self, description: str) -> str:
        description = str(description or "").strip()
        if not description:
            return ""

        endpoint = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model_name}:generateContent?key={self.api_key}"
        )
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": (
                                f"{self.prompt}\n\n"
                                f"Original description:\n{description}\n\n"
                                "Return only the rewritten description."
                            )
                        }
                    ],
                }
            ],
            "generationConfig": {
                "temperature": self.temperature,
                "topP": 0.8,
                "topK": 20,
                "maxOutputTokens": 96,
            },
        }
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        response_payload = _request_json_with_retry(
            request,
            timeout_seconds=self.timeout_seconds,
            retry_config=self.retry_config,
            error_label="Gemini API request",
        )

        candidates = response_payload.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise RuntimeError(f"Gemini API returned no candidates: {response_payload}")

        first_candidate = candidates[0]
        content = first_candidate.get("content", {})
        parts = content.get("parts", [])
        texts = [part.get("text", "") for part in parts if isinstance(part, dict) and part.get("text")]
        cleaned = " ".join(text.strip() for text in texts if text.strip()).strip()
        if not cleaned:
            raise RuntimeError(f"Gemini API returned an empty cleanup response: {response_payload}")
        return cleaned


@dataclass
class OpenRouterCleanupModel:
    api_key: str
    base_url: str = DEFAULT_OPENROUTER_URL
    model_name: str = DEFAULT_CLEANUP_MODEL
    prompt: str = DEFAULT_CLEANUP_PROMPT
    timeout_seconds: float = 60.0
    temperature: float = 0.1
    referer: str | None = None
    title: str | None = None
    retry_config: RetryConfig = field(default_factory=RetryConfig)

    def cleanup_description(self, description: str) -> str:
        description = str(description or "").strip()
        if not description:
            return ""

        client = self._build_client()
        extra_headers: dict[str, str] = {}
        if self.referer:
            extra_headers["HTTP-Referer"] = self.referer
        if self.title:
            extra_headers["X-OpenRouter-Title"] = self.title

        completion = _call_with_retry(
            retry_config=self.retry_config,
            error_label="OpenRouter API request",
            request_fn=lambda: client.chat.completions.create(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=96,
                extra_headers=extra_headers or None,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You clean noisy surveillance segment descriptions. "
                            "Keep only the main visible human actions and interactions. "
                            "Remove repeated background details and generic filler. "
                            "Be short, factual, and do not invent anything."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"{self.prompt}\n\n"
                            f"Original description:\n{description}\n\n"
                            "Return only the rewritten description."
                        ),
                    },
                ],
            ),
        )

        choices = getattr(completion, "choices", None)
        if not choices:
            raise RuntimeError(f"OpenRouter API returned no choices: {completion}")

        first_choice = choices[0]
        message = getattr(first_choice, "message", None)
        cleaned = str(getattr(message, "content", "") or "").strip()
        if not cleaned:
            raise RuntimeError(f"OpenRouter API returned an empty cleanup response: {completion}")
        return cleaned

    def _build_client(self):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The `openai` package is required for the OpenRouter backend. "
                "Install it with `pip install openai` in this environment."
            ) from exc

        return OpenAI(
            base_url=self.base_url.rstrip("/") + "/",
            api_key=self.api_key,
            timeout=self.timeout_seconds,
        )


@dataclass
class PassthroughCleanupModel:
    prompt: str = DEFAULT_CLEANUP_PROMPT

    def cleanup_description(self, description: str) -> str:
        return str(description or "").strip()


def build_cleanup_model(
    backend: str = "openrouter",
    model_name: str = DEFAULT_CLEANUP_MODEL,
    prompt: str = DEFAULT_CLEANUP_PROMPT,
    timeout_seconds: float = 60.0,
    max_retries: int = 3,
    backoff_base_seconds: float = 20.0,
    max_backoff_seconds: float = 60.0,
) -> DescriptionCleanupModel:
    normalized_backend = backend.strip().lower()
    retry_config = RetryConfig(
        max_retries=max(0, int(max_retries)),
        backoff_base_seconds=max(0.0, float(backoff_base_seconds)),
        max_backoff_seconds=max(0.0, float(max_backoff_seconds)),
    )

    if normalized_backend == "openrouter":
        api_key = os.environ.get("LLM_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "LLM_API_KEY is not set. Export your OpenRouter-compatible API key before running cleanup, "
                "or use --cleanup-backend passthrough to test the pipeline without model rewriting."
            )
        base_url = os.environ.get("LLM_URL", DEFAULT_OPENROUTER_URL).strip() or DEFAULT_OPENROUTER_URL
        referer = os.environ.get("OPENROUTER_HTTP_REFERER", "").strip() or None
        title = os.environ.get("OPENROUTER_APP_TITLE", "").strip() or None
        return OpenRouterCleanupModel(
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
            prompt=prompt,
            timeout_seconds=timeout_seconds,
            referer=referer,
            title=title,
            retry_config=retry_config,
        )

    if normalized_backend == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Export your Gemini API key before running cleanup, "
                "or use --cleanup-backend passthrough to test the pipeline without model rewriting."
            )
        return GeminiCleanupModel(
            api_key=api_key,
            model_name=model_name,
            prompt=prompt,
            timeout_seconds=timeout_seconds,
            retry_config=retry_config,
        )

    if normalized_backend == "passthrough":
        return PassthroughCleanupModel(prompt=prompt)

    raise ValueError(f"Unsupported cleanup backend: {backend}")
