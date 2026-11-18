import os
from typing import Any
import unittest
from unittest.mock import patch

from bsp3.oracle import (
    FallbackOracle,
    GeminiOracle,
    MockOracle,
    OpenAICompatOracle,
    OracleError,
    gemma_oracle_from_env,
    qwen_oracle_from_env,
)


class _FakeResponse:
    def __init__(
        self,
        status_code: int,
        data: Any = None,
        text: str = "",
        headers: dict[str, str] | None = None,
        lines: list[str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._data = data
        self.text = text
        self.headers = headers or {}
        self._lines = lines or []
        self.closed = False

    def json(self) -> Any:
        if isinstance(self._data, Exception):
            raise self._data
        return self._data

    def iter_lines(self, decode_unicode: bool = False):
        return iter(self._lines)

    def close(self) -> None:
        self.closed = True


class _FakeSession:
    def __init__(self, handler) -> None:
        self._handler = handler
        self.calls: list[dict[str, Any]] = []

    def post(
        self, url: str, headers=None, json=None, timeout=None, stream: bool = False
    ) -> _FakeResponse:
        call = {
            "url": url,
            "headers": headers or {},
            "json": json or {},
            "timeout": timeout,
            "stream": stream,
        }
        self.calls.append(call)
        return self._handler(call)


class OracleTests(unittest.TestCase):
    def test_mock_oracle_sequence_and_choose(self) -> None:
        oracle = MockOracle(["first", "ACTION2"])
        self.assertEqual(oracle.generate("hello"), "first")
        self.assertEqual(oracle.choose("pick", ["ACTION1", "ACTION2"]), "ACTION2")
        self.assertEqual(oracle.generate("after"), "OK")
        self.assertEqual(len(oracle.calls), 3)
        self.assertIsNotNone(oracle.last_response)
        self.assertEqual(oracle.last_response.provider, "mock")

    def test_choose_matching_variants_and_errors(self) -> None:
        self.assertEqual(
            MockOracle(["```text\naction1\n```"]).choose("p", ["ACTION1", "ACTION2"]),
            "ACTION1",
        )
        self.assertEqual(
            MockOracle(["2."]).choose("p", ["ACTION1", "ACTION2"]),
            "ACTION2",
        )
        self.assertEqual(
            MockOracle(["Pick ACTION1 now."]).choose("p", ["ACTION1", "ACTION10"]),
            "ACTION1",
        )
        with self.assertRaises(OracleError):
            MockOracle(["ACTION1 or ACTION2"]).choose("p", ["ACTION1", "ACTION2"])
        with self.assertRaises(ValueError):
            MockOracle(["ACTION1"]).choose("p", [])

    def test_gemini_parses_thoughts_and_usage(self) -> None:
        session = _FakeSession(
            lambda _: _FakeResponse(
                200,
                {
                    "modelVersion": "gemma-4-26b-a4b-it",
                    "candidates": [
                        {
                            "finishReason": "STOP",
                            "content": {
                                "parts": [
                                    {"text": "thinking step", "thought": True},
                                    {"text": '{"action": "UP"}'},
                                ]
                            },
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 20,
                        "candidatesTokenCount": 6,
                        "thoughtsTokenCount": 40,
                        "totalTokenCount": 66,
                    },
                },
            )
        )
        oracle = GeminiOracle(["g-key-1"], session=session)
        resp = oracle.query("move", system="be brief", temperature=0.0, max_tokens=512)

        self.assertEqual(resp.text, '{"action": "UP"}')
        self.assertEqual(resp.reasoning, "thinking step")
        self.assertEqual(resp.usage.prompt_tokens, 20)
        self.assertEqual(resp.usage.completion_tokens, 6)
        self.assertEqual(resp.usage.thoughts_tokens, 40)
        self.assertEqual(resp.usage.total_tokens, 66)
        sent = session.calls[0]
        self.assertTrue(sent["url"].endswith("/models/gemma-4-26b-a4b-it:generateContent"))
        self.assertEqual(sent["headers"]["x-goog-api-key"], "g-key-1")
        self.assertEqual(sent["json"]["systemInstruction"], {"parts": [{"text": "be brief"}]})

    def test_key_rotation_on_429_and_dead_on_401(self) -> None:
        now = [100.0]

        def handler(call: dict[str, Any]) -> _FakeResponse:
            key = call["headers"]["x-goog-api-key"]
            if key == "k1":
                return _FakeResponse(401, text="bad key")
            if key == "k2":
                return _FakeResponse(429, text="rate limit")
            return _FakeResponse(
                200,
                {"candidates": [{"content": {"parts": [{"text": "done"}]}}]},
            )

        session = _FakeSession(handler)
        oracle = GeminiOracle(
            ["k1", "k2", "k3"],
            session=session,
            clock_fn=lambda: now[0],
            sleep_fn=lambda dt: now.__setitem__(0, now[0] + dt),
        )
        self.assertEqual(oracle.generate("q1"), "done")
        self.assertEqual(
            [c["headers"]["x-goog-api-key"] for c in session.calls],
            ["k1", "k2", "k3"],
        )

        session.calls.clear()
        self.assertEqual(oracle.generate("q2"), "done")
        self.assertEqual([c["headers"]["x-goog-api-key"] for c in session.calls], ["k3"])

    def test_multi_key_503_burst_retries_after_pause(self) -> None:
        now = [100.0]
        statuses = [503, 503, 503, 503, 200]

        def handler(call: dict[str, Any]) -> _FakeResponse:
            status = statuses.pop(0)
            if status == 503:
                return _FakeResponse(503, text="high demand")
            return _FakeResponse(200, {"candidates": [{"content": {"parts": [{"text": "done"}]}}]})

        oracle = GeminiOracle(
            ["k1", "k2", "k3"],
            session=_FakeSession(handler),
            attempts=12,
            clock_fn=lambda: now[0],
            sleep_fn=lambda dt: now.__setitem__(0, now[0] + dt),
        )
        self.assertEqual(oracle.generate("q"), "done")
        self.assertGreaterEqual(now[0] - 100.0, 5.0)

        statuses[:] = [503] * 3
        short = GeminiOracle(
            ["k1", "k2", "k3"],
            session=_FakeSession(handler),
            clock_fn=lambda: now[0],
            sleep_fn=lambda dt: now.__setitem__(0, now[0] + dt),
        )
        with self.assertRaises(OracleError):
            short.generate("q")

    def test_single_key_pacing_and_retry_after(self) -> None:
        now = [0.0]
        sleeps: list[float] = []
        attempts = [0]

        def fake_sleep(dt: float) -> None:
            sleeps.append(dt)
            now[0] += dt

        def handler(_: dict[str, Any]) -> _FakeResponse:
            attempts[0] += 1
            if attempts[0] == 1:
                return _FakeResponse(429, text="slow down", headers={"Retry-After": "1"})
            if attempts[0] == 3:
                return _FakeResponse(429, text="too long", headers={"Retry-After": "30"})
            return _FakeResponse(
                200,
                {"choices": [{"message": {"content": "ok"}}]},
            )

        session = _FakeSession(handler)
        oracle = OpenAICompatOracle(
            provider="nvidia",
            model="minimax",
            base_url="https://example.invalid/v1",
            api_keys=["nv-1"],
            min_interval_s=1.5,
            max_wait_s=5.0,
            session=session,
            clock_fn=lambda: now[0],
            sleep_fn=fake_sleep,
        )
        self.assertEqual(oracle.generate("first"), "ok")
        self.assertEqual(sleeps, [1.5])

        with self.assertRaises(OracleError):
            oracle.generate("second")
        self.assertEqual(len(sleeps), 2)

    def test_openai_compat_thought_tags_and_400_fail_fast(self) -> None:
        session = _FakeSession(
            lambda _: _FakeResponse(
                200,
                {
                    "model": "google/gemma-4-26b-a4b-it",
                    "choices": [
                        {
                            "message": {
                                "content": "<thought>check</thought>ACTION3",
                                "reasoning": "",
                            }
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 12,
                        "completion_tokens": 5,
                        "total_tokens": 17,
                        "completion_tokens_details": {"reasoning_tokens": 3},
                    },
                },
            )
        )
        oracle = OpenAICompatOracle(
            provider="openrouter",
            model="google/gemma-4-26b-a4b-it:free",
            base_url="https://openrouter.ai/api/v1",
            api_keys=["or-1", "or-2"],
            session=session,
        )
        resp = oracle.query("step")
        self.assertEqual(resp.text, "ACTION3")
        self.assertEqual(resp.reasoning, "check")
        self.assertEqual(resp.model, "google/gemma-4-26b-a4b-it")
        self.assertEqual(resp.usage.thoughts_tokens, 3)

        bad_session = _FakeSession(lambda _: _FakeResponse(400, text="bad model"))
        bad_oracle = OpenAICompatOracle(
            provider="openrouter",
            model="bad",
            base_url="https://openrouter.ai/api/v1",
            api_keys=["or-1", "or-2"],
            session=bad_session,
        )
        with self.assertRaises(OracleError):
            bad_oracle.query("step")
        self.assertEqual(len(bad_session.calls), 1)

    def test_fallback_oracle_and_invalid_json(self) -> None:
        primary_session = _FakeSession(lambda _: _FakeResponse(429, text="limit"))
        secondary_session = _FakeSession(lambda _: _FakeResponse(200, data=ValueError("bad json")))
        backup = MockOracle(["YES"])

        gemini = GeminiOracle(["g-1"], session=primary_session, max_wait_s=0.0)
        openrouter = OpenAICompatOracle(
            provider="openrouter",
            model="google/gemma-4-26b-a4b-it:free",
            base_url="https://openrouter.ai/api/v1",
            api_keys=["or-1"],
            session=secondary_session,
            max_wait_s=0.0,
        )
        oracle = FallbackOracle([gemini, openrouter, backup])

        self.assertEqual(oracle.choose("confirm", ["YES", "NO"]), "YES")
        self.assertIsNotNone(oracle.last_response)
        self.assertEqual(oracle.last_response.provider, "mock")

        all_fail = FallbackOracle([gemini, openrouter])
        with self.assertRaises(OracleError):
            all_fail.generate("hi")

    def test_gemma_oracle_from_env(self) -> None:
        with patch("bsp3.oracle.load_dotenv"):
            with patch.dict(os.environ, {"GEMINI_API_KEYS": "g1,g2,g3"}, clear=True):
                oracle = gemma_oracle_from_env(env_path="missing.env")
                self.assertIsInstance(oracle, GeminiOracle)
                self.assertEqual(oracle.model, "gemma-4-26b-a4b-it")
                _, _, body = oracle._build_request("g1", "hi", None, 0.0, 4096)
                self.assertEqual(body["generationConfig"]["thinkingConfig"], {"thinkingLevel": "minimal"})

            with patch.dict(os.environ, {"GEMINI_API_KEY": "g1"}, clear=True):
                oracle = gemma_oracle_from_env(env_path="missing.env", gemini_model="models/other")
                self.assertIsInstance(oracle, GeminiOracle)
                self.assertEqual(oracle.model, "other")

            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(OracleError):
                    gemma_oracle_from_env(env_path="missing.env")

    def test_qwen_hpc_oracle_think_tags_and_enable_thinking(self) -> None:
        session = _FakeSession(
            lambda _: _FakeResponse(
                200,
                {
                    "model": "qwen3.5-9b",
                    "choices": [{"message": {"content": "<think>reason</think>hello"}}],
                    "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
                },
            )
        )
        with patch("bsp3.oracle.load_dotenv"):
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(OracleError):
                    qwen_oracle_from_env(env_path="missing.env")

            with patch.dict(
                os.environ,
                {"QWEN_API_KEY": "q-secret", "QWEN_BASE_URL": "http://localhost:8000/v1"},
                clear=True,
            ):
                oracle = qwen_oracle_from_env(env_path="missing.env", enable_thinking=False)
                oracle._session = session
                resp = oracle.query("Say hello")

        self.assertEqual(resp.provider, "hpc")
        self.assertEqual(resp.model, "qwen3.5-9b")
        self.assertEqual(resp.text, "hello")
        self.assertEqual(resp.reasoning, "reason")
        self.assertEqual(
            resp.settings["chat_template_kwargs"],
            {"enable_thinking": False},
        )
        sent = session.calls[0]
        self.assertEqual(sent["url"], "http://localhost:8000/v1/chat/completions")
        self.assertEqual(sent["headers"]["Authorization"], "Bearer q-secret")
        self.assertEqual(
            sent["json"]["chat_template_kwargs"],
            {"enable_thinking": False},
        )

    def test_openai_compat_streaming_and_reasoning_effort(self) -> None:
        sse_lines = [
            'data: {"model": "minimaxai/minimax-m3.1-flash", "choices": [{"delta": {"reasoning_content": "think "}, "finish_reason": null}]}',
            'data: {"choices": [{"delta": {"reasoning_content": "step", "content": "ACTION"}, "finish_reason": null}]}',
            'data: {"choices": [{"delta": {"content": "4"}, "finish_reason": "stop"}]}',
            'data: {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 6, "total_tokens": 16, "completion_tokens_details": {"reasoning_tokens": 4}}}',
            "data: [DONE]",
        ]
        fake_resp = _FakeResponse(200, lines=sse_lines)
        session = _FakeSession(lambda _: fake_resp)
        oracle = OpenAICompatOracle(
            provider="nvidia",
            model="minimaxai/minimax-m3.1-flash",
            base_url="https://integrate.api.nvidia.com/v1",
            api_keys=["nv-1"],
            extra_body={"reasoning_effort": "max"},
            max_tokens=16384,
            stream=True,
            session=session,
        )
        resp = oracle.query("move", max_tokens=1024)
        self.assertTrue(fake_resp.closed)
        self.assertEqual(resp.text, "ACTION4")
        self.assertEqual(resp.reasoning, "think step")
        self.assertEqual(resp.usage.thoughts_tokens, 4)
        self.assertEqual(resp.usage.total_tokens, 16)
        self.assertEqual(session.calls[0]["json"]["max_tokens"], 16384)
        self.assertTrue(session.calls[0]["stream"])
        self.assertTrue(session.calls[0]["json"]["stream"])
        self.assertEqual(session.calls[0]["json"]["reasoning_effort"], "max")
        self.assertEqual(session.calls[0]["headers"]["Accept"], "text/event-stream")

    def test_single_key_transient_503_recovers(self) -> None:
        now = [0.0]
        sleeps: list[float] = []
        calls = [0]

        def fake_sleep(dt: float) -> None:
            sleeps.append(dt)
            now[0] += dt

        def handler(_: dict[str, Any]) -> _FakeResponse:
            calls[0] += 1
            if calls[0] <= 3:
                return _FakeResponse(503, text="temporarily unavailable")
            return _FakeResponse(200, {"choices": [{"message": {"content": "ok"}}]})

        oracle = OpenAICompatOracle(
            provider="nvidia",
            model="minimax",
            base_url="https://example.invalid/v1",
            api_keys=["nv-1"],
            cooldown_s=20.0,
            max_wait_s=45.0,
            session=_FakeSession(handler),
            clock_fn=lambda: now[0],
            sleep_fn=fake_sleep,
        )
        self.assertEqual(oracle.generate("ping"), "ok")
        self.assertEqual(calls[0], 4)
        self.assertEqual(sleeps, [5.0, 5.0, 5.0])


if __name__ == "__main__":
    unittest.main()
