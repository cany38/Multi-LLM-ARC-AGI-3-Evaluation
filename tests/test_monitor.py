from dataclasses import dataclass, field
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock
from urllib.request import Request, urlopen

import numpy as np

from bsp3.games import Games
from bsp3.logger import Logger
from bsp3.monitor import INDEX_HTML, RunController, make_server, read_monitor_state
from bsp3.oracle import FallbackOracle, MockOracle, OpenAICompatOracle, OracleError
from bsp3.runner import Runner
from bsp3.store import Store


def _post_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request) as resp:
        return json.loads(resp.read().decode("utf-8"))


class _FakeResponse:
    def __init__(self, status_code: int, data=None, text: str = "", headers=None) -> None:
        self.status_code = status_code
        self._data = data
        self.text = text
        self.headers = headers or {}

    def json(self):
        return self._data


class _FakeSession:
    def __init__(self, responses) -> None:
        self._responses = list(responses)

    def post(self, url, headers=None, json=None, timeout=None, stream=False):
        del url, headers, json, timeout, stream
        return self._responses.pop(0)


@dataclass
class _FakeObs:
    frame: list[np.ndarray]
    levels_completed: int = 0
    win_levels: int = 3
    state: str = "NOT_FINISHED"
    available_actions: list[int] = field(default_factory=lambda: [1, 2])


class _FakeEnv:
    def __init__(self, game_id: str) -> None:
        self.info = SimpleNamespace(game_id=game_id, baseline_actions=[3, 3, 3])
        self.action_space: list[object] = []
        self._game = SimpleNamespace(level_index=0)
        self._step = 0

    def reset(self) -> _FakeObs:
        self._step = 0
        grid = np.zeros((8, 8), dtype=np.int8)
        grid[2, 2] = 9
        return _FakeObs(frame=[grid], levels_completed=0)

    def step(self, action: object, data=None) -> _FakeObs:
        del action, data
        self._step += 1
        grid = np.zeros((8, 8), dtype=np.int8)
        grid[2 + self._step, 2] = 9
        self._game.level_index += 1
        return _FakeObs(frame=[grid], levels_completed=1, state="NOT_FINISHED")


def _openai_ok(text: str = "ok") -> _FakeResponse:
    return _FakeResponse(
        200,
        {
            "model": "test-model",
            "choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        },
    )


class MonitorTests(unittest.TestCase):
    def test_oracle_emits_request_wait_http_error_and_provider_fallback(self) -> None:
        now = [0.0]

        def clock() -> float:
            return now[0]

        def sleep(dt: float) -> None:
            now[0] += dt

        primary = OpenAICompatOracle(
            api_keys=["key-a"],
            model="primary-model",
            base_url="https://primary.test/v1",
            provider="primary",
            cooldown_s=1.0,
            max_wait_s=5.0,
            session=_FakeSession(
                [
                    _FakeResponse(429, text="rate limit", headers={"Retry-After": "1"}),
                    _FakeResponse(503, text="busy"),
                    _FakeResponse(503, text="busy"),
                    _FakeResponse(503, text="busy"),
                ]
            ),
            clock_fn=clock,
            sleep_fn=sleep,
        )
        backup = OpenAICompatOracle(
            api_keys=["key-b"],
            model="backup-model",
            base_url="https://backup.test/v1",
            provider="backup",
            session=_FakeSession([_openai_ok("hello")]),
            clock_fn=clock,
            sleep_fn=sleep,
        )
        fallback = FallbackOracle([primary, backup])
        events: list[dict] = []
        fallback.on_event = events.append

        resp = fallback.query("hi")

        self.assertEqual(resp.text, "hello")
        self.assertIsNone(primary.on_event)
        self.assertIsNone(backup.on_event)
        names = [ev["event"] for ev in events]
        self.assertIn("api_request", names)
        self.assertIn("api_http_error", names)
        self.assertIn("api_wait", names)
        self.assertIn("provider_fallback", names)
        self.assertEqual(names[-1], "api_response")

    def test_runner_records_live_frame_events_and_serves_http(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runs_dir = Path(tmp) / "runs"
            store = Store(Path(tmp) / "knowledge")
            games = Games(arcade=SimpleNamespace(make=lambda gid, seed=0: _FakeEnv(gid)), offline=True)
            oracle = MockOracle(
                [
                    json.dumps({"plan": [{"action": "ACTION1"}], "reason": "move"}),
                    json.dumps([{"rule": "ACTION1 moves the player", "confidence": 0.9}]),
                    "YES",
                    "- ACTION1 moves the player",
                    "NO",
                ]
            )
            runner = Runner(
                games=games,
                oracle=oracle,
                store=store,
                runs_dir=runs_dir,
                model_profile="mock",
            )
            runner.run_task("c1", "ls20-9607627b", level=0, condition="notes", budget=2, seed=0)

            state = read_monitor_state(runs_dir)
            active = state["active"]
            self.assertIsNotNone(active)
            assert active is not None
            self.assertEqual(active["status"], "finished")
            self.assertEqual(active["game_id"], "ls20-9607627b")
            self.assertEqual(active["model_profile"], "mock")
            self.assertTrue(active["won"])
            self.assertEqual(len(active["grid"]), 8)
            self.assertEqual((runs_dir / "latest.txt").read_text(encoding="utf-8"), active["run_id"])

            event_names = [ev["event"] for ev in state["events"]]
            self.assertEqual(event_names[0], "run_start")
            self.assertIn("agent_plan", event_names)
            self.assertIn("step", event_names)
            self.assertIn("run_end", event_names)
            self.assertIn("knowledge_start", event_names)
            self.assertEqual(event_names[-1], "knowledge_done")

            server = make_server(runs_dir, port=0)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                port = server.server_address[1]
                with urlopen(f"http://127.0.0.1:{port}/") as resp:
                    html = resp.read().decode("utf-8")
                self.assertIn("BSP3 Live Monitor", html)
                with urlopen(f"http://127.0.0.1:{port}/api/state") as resp:
                    api_data = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(api_data["active"]["run_id"], active["run_id"])
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

    def test_monitor_reads_legacy_run_without_live_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runs_dir = Path(tmp) / "runs"
            logger = Logger(runs_dir, run_id="20261113_120000_000000_ls20_L0_nothing_s0")
            logger.record_step(
                {
                    "step": 1,
                    "action": "ACTION1",
                    "action_data": None,
                    "state": {"diff": {"changed_pixels": 4}},
                    "game_state": "NOT_FINISHED",
                    "model_call": {"provider": "nvidia", "model": "step5", "plan": [{"action": "ACTION1"}], "reason": "test"},
                }
            )
            logger.record_run({"game_id": "ls20-9607627b", "level": 0, "exact_model": "step5", "won": False, "outcome": "budget_exhausted"})

            state = read_monitor_state(runs_dir)
            self.assertEqual(state["active"]["status"], "finished")
            self.assertEqual(state["active"]["model"], "step5")
            self.assertEqual([ev["event"] for ev in state["events"]], ["agent_plan", "step"])

    def test_monitor_control_endpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            server = make_server(Path(tmp) / "runs", port=0)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base = f"http://127.0.0.1:{server.server_address[1]}"

                with urlopen(f"{base}/api/options") as resp:
                    options = json.loads(resp.read().decode("utf-8"))
                self.assertIn("ls20-9607627b", options["games"])
                self.assertNotIn("mock", options["models"])
                self.assertEqual(options["conditions"], ["nothing", "trajectory", "notes", "ontology"])

                with urlopen(f"{base}/api/state") as resp:
                    state = json.loads(resp.read().decode("utf-8"))
                self.assertFalse(state["control"]["running"])

                self.assertFalse(_post_json(f"{base}/api/stop", {})["ok"])
                self.assertFalse(_post_json(f"{base}/api/start", {"steps": []})["ok"])
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

    def test_run_controller_builds_command(self) -> None:
        command = RunController._build_command(
            {
                "steps": [
                    {"game": "ls20-9607627b", "level": 0, "model": "minimax"},
                    {"game": "ls20-9607627b", "level": 1, "model": "gemma"},
                ],
                "conditions": ["nothing", "notes"],
                "judge": "minimax",
                "budget": "5",
            }
        )
        self.assertIsNotNone(command)
        self.assertEqual(command[:4], ["uv", "run", "python", "scripts/run_experiment.py"])
        self.assertEqual(command[command.index("--conditions") + 1], "nothing,notes")
        self.assertEqual(command[command.index("--judge") + 1], "minimax")
        self.assertEqual(command[command.index("--budget") + 1], "5")
        spec = json.loads(command[command.index("--chain-json") + 1])
        self.assertEqual([task["model"] for task in spec["tasks"]], ["minimax", "gemma"])
        self.assertEqual([task["level"] for task in spec["tasks"]], [0, 1])

        self.assertIsNone(RunController._build_command({"steps": [], "conditions": ["nothing"]}))
        self.assertIsNone(
            RunController._build_command({"steps": [{"game": "a", "level": 0, "model": "m"}], "conditions": []})
        )
        self.assertIsNone(
            RunController._build_command(
                {"steps": [{"game": "a", "level": "x", "model": "m"}], "conditions": ["nothing"]}
            )
        )
        self.assertIsNone(
            RunController._build_command(
                {"steps": [{"game": "", "level": 0, "model": "m"}], "conditions": ["nothing"]}
            )
        )

    def test_build_command_drops_trajectory_on_same_level_chain(self) -> None:
        same_level = [
            {"game": "ls20-9607627b", "level": 0, "model": "a"},
            {"game": "ls20-9607627b", "level": 0, "model": "b"},
            {"game": "ls20-9607627b", "level": 0, "model": "c"},
        ]
        command = RunController._build_command(
            {"steps": same_level, "conditions": ["nothing", "trajectory", "notes"]}
        )
        self.assertIsNotNone(command)
        self.assertEqual(command[command.index("--conditions") + 1], "nothing,notes")

        self.assertIsNone(RunController._build_command({"steps": same_level, "conditions": ["trajectory"]}))

        mixed = [
            {"game": "ls20-9607627b", "level": 0, "model": "a"},
            {"game": "ls20-9607627b", "level": 1, "model": "b"},
        ]
        kept = RunController._build_command({"steps": mixed, "conditions": ["trajectory"]})
        self.assertIsNotNone(kept)
        self.assertEqual(kept[kept.index("--conditions") + 1], "trajectory")

        partial_repeat = mixed + [{"game": "ls20-9607627b", "level": 0, "model": "c"}]
        self.assertIsNone(RunController._build_command({"steps": partial_repeat, "conditions": ["trajectory"]}))

    def test_index_html_plays_one_level_up_per_step(self) -> None:
        self.assertIn("level: start + i", INDEX_HTML)
        self.assertNotIn("same level would leak the solution", INDEX_HTML)

    def test_run_controller_start_and_stop(self) -> None:
        class _FakeProc:
            pid = 4321

            def poll(self):
                return None

        with tempfile.TemporaryDirectory() as tmp:
            runs_dir = Path(tmp) / "runs"
            controller = RunController(Path(tmp), runs_dir)
            payload = {
                "steps": [
                    {"game": "ls20-9607627b", "level": 0, "model": "mock"},
                    {"game": "ls20-9607627b", "level": 1, "model": "mock"},
                ],
                "conditions": ["nothing"],
            }
            with mock.patch("bsp3.monitor._spawn_run", return_value=_FakeProc()), mock.patch("bsp3.monitor._kill_run"):
                started = controller.start(payload)
                self.assertTrue(started["ok"])
                self.assertTrue(controller.status()["running"])

                again = controller.start(payload)
                self.assertFalse(again["ok"])

                open_run = runs_dir / "run_a" / "live.json"
                open_run.parent.mkdir(parents=True)
                open_run.write_text(json.dumps({"status": "running", "step": 3}), encoding="utf-8")
                self.assertTrue(controller.stop()["ok"])
                self.assertFalse(controller.status()["running"])
                self.assertEqual(json.loads(open_run.read_text(encoding="utf-8")), {"status": "stopped", "step": 3})

    def test_runner_marks_error_when_oracle_raises(self) -> None:
        class _BrokenOracle(MockOracle):
            def query(self, prompt: str, *, system=None, temperature=0.0, max_tokens=1024):
                raise OracleError("boom")

        with tempfile.TemporaryDirectory() as tmp:
            runs_dir = Path(tmp) / "runs"
            games = Games(arcade=SimpleNamespace(make=lambda gid, seed=0: _FakeEnv(gid)), offline=True)
            runner = Runner(games=games, oracle=_BrokenOracle(), store=Store(Path(tmp) / "k"), runs_dir=runs_dir)
            with self.assertRaises(OracleError):
                runner.run_task("c1", "ls20-9607627b", level=0, condition="nothing", budget=2, seed=0)

            state = read_monitor_state(runs_dir)
            self.assertEqual(state["active"]["status"], "error")
            self.assertIn("boom", state["active"]["error"])
            self.assertEqual(state["events"][-1]["event"], "error")
