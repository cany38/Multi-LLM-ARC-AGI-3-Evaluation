from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import time
from typing import Any

from dotenv import load_dotenv
import requests

from bsp3.config import load_models_config


class OracleError(RuntimeError):
    """Raised when an oracle provider cannot return a valid response."""


@dataclass(frozen=True, slots=True)
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    thoughts_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True, slots=True)
class OracleResponse:
    text: str
    model: str
    provider: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    latency_ms: float = 0.0
    reasoning: str = ""
    settings: dict[str, Any] = field(default_factory=dict)


def _split_keys(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def _split_thought_tag(text: str) -> tuple[str, str]:
    stripped = text.lstrip()
    for tag in ("thought", "think"):
        open_tag, close_tag = f"<{tag}>", f"</{tag}>"
        if stripped.startswith(open_tag) and close_tag in stripped:
            head, tail = stripped[len(open_tag) :].split(close_tag, 1)
            return tail.strip(), head.strip()
    return text.strip(), ""


def _match_option(reply: str, options: Sequence[str]) -> str:
    raw = reply.strip()
    if raw.startswith("```"):
        lines = [line for line in raw.splitlines() if not line.strip().startswith("```")]
        raw = "\n".join(lines).strip()

    for opt in options:
        if raw == opt or raw.lower() == opt.lower():
            return opt

    cleaned = raw.strip("`\"' \n\t.")
    for opt in options:
        if cleaned == opt or cleaned.lower() == opt.lower():
            return opt

    if cleaned.isdigit():
        idx = int(cleaned) - 1
        if 0 <= idx < len(options):
            return options[idx]

    hits = [
        opt
        for opt in options
        if re.search(rf"(?<!\w){re.escape(opt)}(?!\w)", cleaned, re.IGNORECASE)
    ]
    if len(hits) == 1:
        return hits[0]

    raise OracleError(f"Could not match choice {reply!r} to {list(options)}")


def _parse_retry_after(headers: Any, default_s: float) -> float:
    raw = str(headers.get("Retry-After", "")).strip() if headers else ""
    if raw and raw.replace(".", "", 1).isdigit():
        return max(0.0, float(raw))
    return default_s


def _read_sse_chat_completion(resp: Any) -> dict[str, Any]:
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    model_name = ""
    finish_reason = "unknown"
    usage: dict[str, Any] = {}
    try:
        for raw_line in resp.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else str(raw_line)
            if not line.startswith("data: "):
                continue
            payload = line[6:].strip()
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            if chunk.get("model"):
                model_name = str(chunk["model"])
            if chunk.get("usage"):
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            if choice.get("finish_reason"):
                finish_reason = str(choice["finish_reason"])
            delta = choice.get("delta") or {}
            if delta.get("content"):
                content_parts.append(str(delta["content"]))
            extra_reason = delta.get("reasoning_content") or delta.get("reasoning")
            if extra_reason:
                reasoning_parts.append(str(extra_reason))
    finally:
        close_fn = getattr(resp, "close", None)
        if callable(close_fn):
            close_fn()
    return {
        "model": model_name,
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": {
                    "content": "".join(content_parts),
                    "reasoning_content": "".join(reasoning_parts),
                },
            }
        ],
        "usage": usage,
    }


class Oracle(ABC):
    """Common interface for LLM providers and the test mock."""

    provider: str
    model: str
    last_response: OracleResponse | None = None
    on_event: Callable[[dict[str, Any]], None] | None = None

    def _emit(self, event: str, **fields: Any) -> None:
        if self.on_event is not None:
            self.on_event({"event": event, "provider": self.provider, "model": self.model, **fields})

    @abstractmethod
    def query(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> OracleResponse:
        """Send a prompt and return the model response with metadata."""

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        response = self.query(
            prompt, system=system, temperature=temperature, max_tokens=max_tokens
        )
        self.last_response = response
        return response.text

    def choose(
        self,
        prompt: str,
        options: Sequence[str],
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        if not options:
            raise ValueError("options must not be empty")
        numbered = "\n".join(f"{i}. {opt}" for i, opt in enumerate(options, 1))
        full_prompt = (
            f"{prompt}\n\n"
            "Choose one option and reply with its exact text only:\n"
            f"{numbered}"
        )
        response = self.query(
            full_prompt, system=system, temperature=temperature, max_tokens=max_tokens
        )
        self.last_response = response
        return _match_option(response.text, options)


class MockOracle(Oracle):
    """Deterministic oracle for tests without API calls."""

    def __init__(
        self,
        answers: Sequence[str] | Callable[[str, str | None], str] = (),
        *,
        default: str = "OK",
        model: str = "mock",
    ) -> None:
        self.provider = "mock"
        self.model = model
        self._answers = list(answers) if not callable(answers) else answers
        self._default = default
        self._index = 0
        self.calls: list[dict[str, Any]] = []

    def query(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> OracleResponse:
        self.calls.append(
            {
                "prompt": prompt,
                "system": system,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        if callable(self._answers):
            text = self._answers(prompt, system)
        elif self._index < len(self._answers):
            text = self._answers[self._index]
            self._index += 1
        else:
            text = self._default

        response = OracleResponse(
            text=text,
            model=self.model,
            provider=self.provider,
            settings={"temperature": temperature, "max_tokens": max_tokens},
        )
        self.last_response = response
        return response


class _KeyedHttpOracle(Oracle, ABC):
    def __init__(
        self,
        *,
        provider: str,
        model: str,
        api_keys: Sequence[str],
        base_url: str,
        timeout_s: float = 60.0,
        min_interval_s: float = 0.0,
        cooldown_s: float = 15.0,
        max_wait_s: float = 5.0,
        max_tokens: int | None = None,
        attempts: int | None = None,
        session: requests.Session | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        clock_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        if not api_keys:
            raise ValueError(f"{provider} requires at least one API key")
        self._attempts = attempts
        self.provider = provider
        self.model = model
        self._keys = list(api_keys)
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._min_interval_s = min_interval_s
        self._cooldown_s = cooldown_s
        self._max_wait_s = max_wait_s
        self._max_tokens = max_tokens
        self._session = session or requests.Session()
        self._sleep = sleep_fn
        self._clock = clock_fn
        self._cursor = 0
        self._last_used: dict[int, float] = {}
        self._cooldown_until: dict[int, float] = {}
        self._dead: set[int] = set()

    def _ready_at(self, idx: int) -> float:
        cooldown = self._cooldown_until.get(idx, float("-inf"))
        if idx in self._last_used:
            return max(cooldown, self._last_used[idx] + self._min_interval_s)
        return cooldown

    def _acquire_key(self) -> tuple[int, str]:
        n = len(self._keys)
        now = self._clock()
        for offset in range(n):
            idx = (self._cursor + offset) % n
            if idx not in self._dead and self._ready_at(idx) <= now:
                self._cursor = (idx + 1) % n
                self._last_used[idx] = now
                return idx, self._keys[idx]

        live = [i for i in range(n) if i not in self._dead]
        if not live:
            raise OracleError(f"{self.provider}: all API keys failed authentication or quota")

        idx = min(live, key=self._ready_at)
        wait = self._ready_at(idx) - now
        if wait > self._max_wait_s:
            raise OracleError(
                f"{self.provider}: all keys in cooldown ({wait:.1f}s > {self._max_wait_s:.1f}s)"
            )
        if wait > 0:
            self._emit("api_wait", key_index=idx + 1, wait_s=round(wait, 2))
            self._sleep(wait)
        self._cursor = (idx + 1) % n
        self._last_used[idx] = self._clock()
        return idx, self._keys[idx]

    @abstractmethod
    def _build_request(
        self,
        api_key: str,
        prompt: str,
        system: str | None,
        temperature: float,
        max_tokens: int,
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        """Build URL, headers, and JSON body."""

    @abstractmethod
    def _parse_response(
        self,
        data: dict[str, Any],
        settings: dict[str, Any],
        latency_ms: float,
    ) -> OracleResponse:
        """Extract text, model, reasoning, and token counts."""

    def query(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> OracleResponse:
        token_limit = self._max_tokens if self._max_tokens is not None else max_tokens
        settings = {"temperature": temperature, "max_tokens": token_limit}
        attempts = self._attempts or (len(self._keys) if len(self._keys) > 1 else 4)
        errors: list[str] = []

        for attempt in range(1, attempts + 1):
            idx, api_key = self._acquire_key()
            url, headers, payload = self._build_request(
                api_key, prompt, system, temperature, token_limit
            )
            stream = bool(payload.get("stream"))
            self._emit(
                "api_request",
                key_index=idx + 1,
                attempt=attempt,
                attempts=attempts,
                stream=stream,
            )
            t0 = self._clock()
            try:
                resp = (
                    self._session.post(
                        url, headers=headers, json=payload, timeout=self._timeout_s, stream=True
                    )
                    if stream
                    else self._session.post(
                        url, headers=headers, json=payload, timeout=self._timeout_s
                    )
                )
                if resp.status_code == 200:
                    data = _read_sse_chat_completion(resp) if stream else resp.json()
                    latency_ms = (self._clock() - t0) * 1000.0
                    parsed = self._parse_response(data, settings, latency_ms)
                    self.last_response = parsed
                    self._emit(
                        "api_response",
                        key_index=idx + 1,
                        status=200,
                        latency_ms=round(latency_ms, 1),
                        total_tokens=parsed.usage.total_tokens,
                        thoughts_tokens=parsed.usage.thoughts_tokens,
                    )
                    return parsed
            except requests.RequestException as exc:
                self._cooldown_until[idx] = self._clock() + min(2.0, self._cooldown_s)
                self._emit("api_error", key_index=idx + 1, attempt=attempt, error=str(exc))
                errors.append(f"key#{idx + 1}: {exc}")
                continue
            except ValueError as exc:
                self._emit(
                    "api_error",
                    key_index=idx + 1,
                    attempt=attempt,
                    error=f"invalid JSON: {exc}",
                )
                raise OracleError(f"{self.provider}: invalid JSON response: {exc}") from exc

            snippet = resp.text[:240].strip()
            self._emit(
                "api_http_error",
                key_index=idx + 1,
                attempt=attempt,
                status=resp.status_code,
                detail=snippet[:120],
            )
            errors.append(f"key#{idx + 1}: HTTP {resp.status_code} ({snippet})")

            if resp.status_code in (401, 402, 403):
                self._dead.add(idx)
            elif resp.status_code == 429:
                wait = _parse_retry_after(resp.headers, self._cooldown_s)
                self._cooldown_until[idx] = self._clock() + wait
            elif resp.status_code >= 500:
                self._cooldown_until[idx] = self._clock() + min(
                    5.0, self._cooldown_s, self._max_wait_s
                )
            else:
                raise OracleError(f"{self.provider}: HTTP {resp.status_code} ({snippet})")

        raise OracleError(f"{self.provider} failed: {'; '.join(errors)}")


class GeminiOracle(_KeyedHttpOracle):
    """Google AI Studio client using native :generateContent."""

    # Native REST avoids /v1beta/openai/ 500 errors on temperature=0 and keeps thought parts separate.
    def __init__(
        self,
        api_keys: Sequence[str],
        *,
        model: str = "gemma-4-26b-a4b-it",
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        thinking_level: str | None = None,
        **kwargs: Any,
    ) -> None:
        clean_model = model.removeprefix("models/")
        super().__init__(
            provider="gemini",
            model=clean_model,
            api_keys=api_keys,
            base_url=base_url,
            **kwargs,
        )
        self._thinking_level = thinking_level

    def _build_request(
        self,
        api_key: str,
        prompt: str,
        system: str | None,
        temperature: float,
        max_tokens: int,
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        url = f"{self._base_url}/models/{self.model}:generateContent"
        headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
        body: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if self._thinking_level:
            body["generationConfig"]["thinkingConfig"] = {"thinkingLevel": self._thinking_level}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        return url, headers, body

    def _parse_response(
        self,
        data: dict[str, Any],
        settings: dict[str, Any],
        latency_ms: float,
    ) -> OracleResponse:
        candidates = data.get("candidates") or []
        if not candidates:
            raise OracleError(f"gemini: empty candidates in response: {data}")
        candidate = candidates[0]
        parts = candidate.get("content", {}).get("parts") or []
        text = "".join(p["text"] for p in parts if not p.get("thought") and p.get("text")).strip()
        reasoning = "\n".join(
            p["text"] for p in parts if p.get("thought") and p.get("text")
        ).strip()
        if not text:
            finish_reason = candidate.get("finishReason", "UNKNOWN")
            raise OracleError(f"gemini: returned empty text (finishReason={finish_reason})")

        meta = data.get("usageMetadata") or {}
        usage = TokenUsage(
            prompt_tokens=int(meta.get("promptTokenCount") or 0),
            completion_tokens=int(meta.get("candidatesTokenCount") or 0),
            thoughts_tokens=int(meta.get("thoughtsTokenCount") or 0),
            total_tokens=int(meta.get("totalTokenCount") or 0),
        )
        return OracleResponse(
            text=text,
            model=str(data.get("modelVersion") or self.model),
            provider=self.provider,
            usage=usage,
            latency_ms=latency_ms,
            reasoning=reasoning,
            settings={**settings, "thinking_level": self._thinking_level} if self._thinking_level else settings,
        )


class OpenAICompatOracle(_KeyedHttpOracle):
    """Client for OpenAI-compatible endpoints such as NVIDIA build and HPC vLLM."""

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        base_url: str,
        api_keys: Sequence[str] = ("",),
        extra_body: dict[str, Any] | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> None:
        self._extra_body = dict(extra_body) if extra_body else {}
        self._stream = bool(stream or self._extra_body.get("stream"))
        super().__init__(
            provider=provider,
            model=model,
            api_keys=api_keys,
            base_url=base_url,
            **kwargs,
        )

    def _build_request(
        self,
        api_key: str,
        prompt: str,
        system: str | None,
        temperature: float,
        max_tokens: int,
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        url = f"{self._base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if self._stream else "application/json",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        body: dict[str, Any] = {
            **self._extra_body,
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if self._stream:
            body["stream"] = True
            body.setdefault("stream_options", {"include_usage": True})
        return url, headers, body

    def _parse_response(
        self,
        data: dict[str, Any],
        settings: dict[str, Any],
        latency_ms: float,
    ) -> OracleResponse:
        choices = data.get("choices") or []
        if not choices:
            raise OracleError(f"{self.provider}: empty choices in response: {data}")
        msg = choices[0].get("message") or {}
        raw_text = msg.get("content") or ""
        text, tag_thought = _split_thought_tag(raw_text)
        extra_reasoning = str(msg.get("reasoning") or msg.get("reasoning_content") or "").strip()
        reasoning = "\n".join(x for x in (tag_thought, extra_reasoning) if x)
        if not text:
            finish_reason = choices[0].get("finish_reason", "unknown")
            raise OracleError(f"{self.provider}: returned empty text (finish_reason={finish_reason})")

        raw_usage = data.get("usage") or {}
        details = raw_usage.get("completion_tokens_details") or {}
        prompt_tokens = int(raw_usage.get("prompt_tokens") or 0)
        completion_tokens = int(raw_usage.get("completion_tokens") or 0)
        thoughts_tokens = int(details.get("reasoning_tokens") or 0)
        total_tokens = int(raw_usage.get("total_tokens") or (prompt_tokens + completion_tokens))
        full_settings = {**settings, **self._extra_body} if self._extra_body else settings
        return OracleResponse(
            text=text,
            model=str(data.get("model") or self.model),
            provider=self.provider,
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                thoughts_tokens=thoughts_tokens,
                total_tokens=total_tokens,
            ),
            latency_ms=latency_ms,
            reasoning=reasoning,
            settings=full_settings,
        )


class FallbackOracle(Oracle):
    """Try oracle providers in order until one succeeds."""

    def __init__(self, providers: Sequence[Oracle]) -> None:
        if not providers:
            raise ValueError("providers must not be empty")
        self._providers = list(providers)
        self.provider = providers[0].provider
        self.model = providers[0].model

    def query(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> OracleResponse:
        errors: list[str] = []
        for idx, oracle in enumerate(self._providers):
            prev_hook = oracle.on_event
            if self.on_event is not None:
                oracle.on_event = self.on_event
            try:
                response = oracle.query(
                    prompt, system=system, temperature=temperature, max_tokens=max_tokens
                )
            except OracleError as exc:
                errors.append(f"{oracle.provider}: {exc}")
                if idx + 1 < len(self._providers):
                    nxt = self._providers[idx + 1]
                    self._emit(
                        "provider_fallback",
                        from_provider=oracle.provider,
                        from_model=oracle.model,
                        to_provider=nxt.provider,
                        to_model=nxt.model,
                        reason=str(exc)[:160],
                    )
                continue
            finally:
                oracle.on_event = prev_hook
            self.last_response = response
            return response

        raise OracleError(f"All providers failed: {' | '.join(errors)}")


def _env_keys(names: Sequence[str]) -> list[str]:
    for name in names:
        keys = _split_keys(os.getenv(name))
        if keys:
            return keys
    return []


def oracle_from_config(
    profile: str,
    *,
    config_path: Path | str = "configs/models.toml",
    env_path: str = ".env",
    extra_body: dict[str, Any] | None = None,
) -> Oracle:
    """Build an Oracle from configs/models.toml and environment credentials."""
    load_dotenv(env_path)
    cfg = load_models_config(config_path)
    providers_cfg = cfg.get("providers", {})
    entries = cfg.get("models", {}).get(profile)
    if not entries:
        raise OracleError(f"Unknown model profile: {profile!r}")

    oracles: list[Oracle] = []
    for entry in entries:
        pname = entry["provider"]
        pcfg = providers_cfg.get(pname, {})
        kind = pcfg.get("kind", pname)
        model_name = (
            (os.getenv(entry["model_env"]) or "").strip() if entry.get("model_env") else ""
        ) or entry.get("default_model") or entry.get("model", "")

        if kind == "mock":
            oracles.append(MockOracle(model=model_name or "mock"))
            continue

        keys = _env_keys(pcfg.get("keys_env", ()))
        if not keys:
            continue

        base_url = (
            (os.getenv(pcfg["base_url_env"]) or "").strip() if pcfg.get("base_url_env") else ""
        ) or pcfg.get("default_base_url") or pcfg.get("base_url", "")
        raw_max_tokens = entry.get("max_tokens", pcfg.get("max_tokens"))
        common = {
            "model": model_name,
            "base_url": base_url,
            "timeout_s": float(pcfg.get("timeout_s", 60.0)),
            "cooldown_s": float(pcfg.get("cooldown_s", 15.0)),
            "max_wait_s": float(pcfg.get("max_wait_s", 5.0)),
            "min_interval_s": float(pcfg.get("min_interval_s", 0.0)),
            "max_tokens": int(raw_max_tokens) if raw_max_tokens is not None else None,
            "attempts": int(pcfg["attempts"]) if "attempts" in pcfg else None,
        }
        if kind == "gemini":
            oracles.append(GeminiOracle(keys, thinking_level=entry.get("thinking_level"), **common))
        elif kind == "openai_compat":
            merged_extra = {**entry.get("extra_body", {}), **(extra_body or {})}
            oracles.append(
                OpenAICompatOracle(
                    provider=pname,
                    api_keys=keys,
                    extra_body=merged_extra or None,
                    stream=bool(pcfg.get("stream", False)),
                    **common,
                )
            )
        else:
            raise OracleError(f"Unsupported provider kind {kind!r} for {pname!r}")

    if not oracles:
        raise OracleError(f"No API keys configured for model profile {profile!r}")
    if len(oracles) == 1:
        return oracles[0]
    return FallbackOracle(oracles)


def gemma_oracle_from_env(
    *,
    env_path: str = ".env",
    config_path: Path | str = "configs/models.toml",
    gemini_model: str | None = None,
) -> Oracle:
    """Build the Gemma 4 26B oracle from configs/models.toml and .env."""
    oracle = oracle_from_config("gemma", config_path=config_path, env_path=env_path)
    if gemini_model and isinstance(oracle, GeminiOracle):
        oracle.model = gemini_model.removeprefix("models/")
    return oracle


def qwen_oracle_from_env(
    *,
    env_path: str = ".env",
    config_path: Path | str = "configs/models.toml",
    enable_thinking: bool | None = None,
    timeout_s: float | None = None,
) -> OpenAICompatOracle:
    """Build the Qwen 3.5 9B client for the HPC vLLM tunnel described in docs/HPC.md."""
    extra = (
        {"chat_template_kwargs": {"enable_thinking": enable_thinking}}
        if enable_thinking is not None
        else None
    )
    oracle = oracle_from_config(
        "qwen", config_path=config_path, env_path=env_path, extra_body=extra
    )
    if not isinstance(oracle, OpenAICompatOracle):
        raise OracleError(f"Expected OpenAICompatOracle for qwen, got {type(oracle).__name__}")
    if timeout_s is not None:
        oracle._timeout_s = timeout_s
    return oracle
