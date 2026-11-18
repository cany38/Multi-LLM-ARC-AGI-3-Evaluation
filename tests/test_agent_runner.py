import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from arcengine import GameAction, GameState
import numpy as np

from bsp3.agent import Agent
from bsp3.games import Games
from bsp3.logger import Logger
from bsp3.oracle import MockOracle
from bsp3.runner import Runner
from bsp3.store import Store

VALID_TTL = """@prefix arc: <http://bsp3.local/arc#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

arc:Player a owl:Class ."""


class _GridEnv:
    """Two-level toy game: ACTION4 moves right, reaching x=3 wins a level."""

    def __init__(self) -> None:
        self.action_space = [GameAction.RESET, GameAction.ACTION1, GameAction.ACTION4]
        self.info = SimpleNamespace(baseline_actions=[2, 2])
        self._x = 1
        self._level = 0
        self._completed = 0
        self._game = self
        self.level_index = 0
        self.current_level = self
        self.camera = self

    def get_sprites(self):
        return None

    def render(self, _):
        grid = np.zeros((5, 5), dtype=np.int8)
        grid[2, self._x] = 9
        grid[2, 3] = 5
        return grid

    def set_level(self, level: int) -> None:
        self._level = level
        self.level_index = level
        self._x = 1

    def reset(self):
        self._x = 1
        self._level = 0
        self.level_index = 0
        self._completed = 0
        return SimpleNamespace(
            frame=[self.render(None)],
            levels_completed=0,
            win_levels=2,
            state=GameState.NOT_FINISHED,
            available_actions=[1, 4],
        )

    def step(self, action: GameAction, data=None):
        if action == GameAction.ACTION4:
            self._x = min(3, self._x + 1)
        # ACTION1 intentionally hits top boundary (no visual change) to test replanning.
        state = GameState.NOT_FINISHED
        if self._x == 3:
            self._completed += 1
            if self._level + 1 >= 2:
                state = GameState.WIN
            else:
                self._level += 1
                self.level_index = self._level
                self._x = 1
        return SimpleNamespace(
            frame=[self.render(None)],
            levels_completed=self._completed,
            win_levels=2,
            state=state,
            available_actions=[1, 4],
        )


class _ToyArcade:
    def __init__(self) -> None:
        self._env = _GridEnv()

    def get_environments(self):
        return [SimpleNamespace(game_id="toy-1"), SimpleNamespace(game_id="toy-2")]

    def make(self, game_id: str, seed: int = 0):
        return self._env


class AgentAndRunnerTests(unittest.TestCase):
    def test_agent_wins_replans_on_noop_and_logs(self) -> None:
        # Step 1 asks for [ACTION1, ACTION4]; ACTION1 produces 0 changed pixels, forcing replan.
        oracle = MockOracle(
            [
                json.dumps({"plan": ["ACTION1", "ACTION4"], "reason": "probe"}),
                json.dumps({"plan": [{"action": "ACTION4"}, {"action": "ACTION4"}], "reason": "goal"}),
            ]
        )
        games = Games(arcade=_ToyArcade())
        agent = Agent.from_config(oracle)

        with tempfile.TemporaryDirectory() as tmp:
            logger = Logger.for_run(
                game_id="toy-1", level=0, condition="nothing", seed=42, runs_dir=tmp
            )
            res = agent.play(games, "toy-1", level=0, seed=42, logger=logger)
            self.assertTrue(res.won)
            self.assertEqual(res.outcome, "won")
            self.assertEqual(res.model_calls, 2)
            self.assertEqual(res.actions_taken, 3)

            summary = logger.read_summary()
            self.assertIsNotNone(summary)
            self.assertEqual(summary["seed"], 42)
            self.assertEqual(summary["outcome"], "won")
            self.assertEqual(len(logger.read_steps()), 3)

    def test_runner_four_conditions_end_to_end(self) -> None:
        tasks = [("toy-1", 0), ("toy-2", 0)]
        facts = [{"subject": "Player", "relation": "moves_right", "object": "ACTION4"}]

        for condition, extra_answers in [
            ("nothing", []),
            ("trajectory", []),
            (
                "notes",
                [
                    json.dumps(facts),
                    "YES",
                    "ACTION4 moves right.",
                    "NO",
                ],
            ),
            (
                "ontology",
                [
                    json.dumps(facts),
                    "YES",
                    VALID_TTL,
                    "NO",
                ],
            ),
        ]:
            # Player wins in 1 call per task: 2x ACTION4.
            answers = [
                json.dumps({"plan": ["ACTION4", "ACTION4"]}),
                *extra_answers,
                json.dumps({"plan": ["ACTION4", "ACTION4"]}),
            ]
            oracle = MockOracle(answers)
            with tempfile.TemporaryDirectory() as tmp:
                store = Store(Path(tmp) / "store")
                runner = Runner(
                    games=Games(arcade=_ToyArcade()),
                    oracle=oracle,
                    store=store,
                    runs_dir=Path(tmp) / "runs",
                )
                results = runner.run_chain("toy-chain", tasks, condition=condition, seed=1)
                self.assertEqual(len(results), 2)
                self.assertTrue(all(r.won for r in results))

                if condition == "nothing":
                    self.assertIn("Transferred knowledge: none.", oracle.calls[1]["prompt"])
                else:
                    second_prompt = oracle.calls[-1]["prompt"]
                    self.assertIn(f"Transferred knowledge ({condition}):", second_prompt)

    def test_runner_with_offline_ls20_game_and_mock_oracle(self) -> None:
        if not Path("environment_files/ls20").exists():
            self.skipTest("offline environment_files not downloaded")
        oracle = MockOracle(default=json.dumps({"plan": ["ACTION1", "ACTION4"]}))
        with tempfile.TemporaryDirectory() as tmp:
            runner = Runner(
                games=Games(offline=True),
                oracle=oracle,
                store=Store(Path(tmp) / "store"),
                runs_dir=Path(tmp) / "runs",
            )
            res = runner.run_task(
                "ls20-smoke",
                "ls20-9607627b",
                level=0,
                condition="nothing",
                budget=3,
                seed=0,
                extract_knowledge=False,
            )
            self.assertEqual(res.actions_taken, 3)
            self.assertEqual(res.outcome, "budget_exhausted")
            self.assertIsNotNone(res.run_dir)
            self.assertTrue((res.run_dir / "summary.json").exists())

    def test_runner_skips_unverified_and_leaked_knowledge(self) -> None:
        facts = [{"subject": "Player", "relation": "moves", "object": "ACTION4"}]
        # Rejected facts: filter_facts returns [] so as_notes is never saved.
        oracle_empty = MockOracle(
            [
                json.dumps({"plan": ["ACTION4", "ACTION4"]}),
                json.dumps(facts),
                "NO",
            ]
        )
        # Leak detected: judge_leak returns YES so notes are withheld from store.
        oracle_leak = MockOracle(
            [
                json.dumps({"plan": ["ACTION4", "ACTION4"]}),
                json.dumps(facts),
                "YES",
                "Press ACTION4 twice at x=1.",
                "YES",
            ]
        )
        for oracle in (oracle_empty, oracle_leak):
            with tempfile.TemporaryDirectory() as tmp:
                store = Store(Path(tmp) / "store")
                runner = Runner(
                    games=Games(arcade=_ToyArcade()),
                    oracle=oracle,
                    store=store,
                    runs_dir=Path(tmp) / "runs",
                )
                runner.run_task("leak-chain", "toy-1", level=0, condition="notes", seed=0)
                self.assertEqual(store.load("leak-chain", condition="notes"), [])

    def test_run_all_conditions_shares_t1_and_step_models(self) -> None:
        tasks = [("toy-1", 0), ("toy-2", 0)]
        facts = [{"subject": "Player", "relation": "moves_right", "object": "ACTION4"}]
        player_qwen = MockOracle([json.dumps({"plan": ["ACTION4", "ACTION4"]})], model="qwen-9b")
        player_gemma = MockOracle(
            [json.dumps({"plan": ["ACTION4", "ACTION4"]}) for _ in range(4)],
            model="gemma-26b",
        )
        writer_oracle = MockOracle(
            [json.dumps(facts), "ACTION4 moves right.", VALID_TTL],
            model="writer-model",
        )
        judge_oracle = MockOracle(["YES", "NO", "NO"], model="judge-model")
        oracles = {"qwen": player_qwen, "gemma": player_gemma}

        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "store")
            runner = Runner(
                games=Games(arcade=_ToyArcade()),
                oracle=player_qwen,
                knowledge_oracle=writer_oracle,
                judge_oracle=judge_oracle,
                model_profile="qwen",
                store=store,
                runs_dir=Path(tmp) / "runs",
                oracle_factory=lambda name: oracles[name],
            )
            results = runner.run_all_conditions(
                "shared-chain",
                tasks,
                seed=0,
                models=("qwen", "gemma"),
            )
            self.assertEqual(len(player_qwen.calls), 1)
            self.assertEqual(len(player_gemma.calls), 4)
            self.assertEqual(len(writer_oracle.calls), 3)
            self.assertEqual(len(judge_oracle.calls), 3)
            self.assertEqual(results["notes"][0].exact_model, "qwen-9b")
            self.assertEqual(results["notes"][1].exact_model, "gemma-26b")
            self.assertEqual(len(store.load("shared-chain", condition="trajectory")), 1)
            self.assertEqual(len(store.load("shared-chain", condition="notes")), 1)
            self.assertEqual(len(store.load("shared-chain", condition="ontology")), 1)

    def test_knowledge_writer_follows_step_player_model(self) -> None:
        tasks = [("toy-1", 0), ("toy-2", 0), ("toy-1", 0)]
        facts = [{"subject": "Player", "relation": "moves_right", "object": "ACTION4"}]
        plan = json.dumps({"plan": ["ACTION4", "ACTION4"]})
        facts_json = json.dumps(facts)

        player_a = MockOracle([plan, facts_json, "ACTION4 moves right."], model="model-a")
        player_b = MockOracle(
            [plan, facts_json, "ACTION4 moves right.", plan], model="model-b"
        )
        judge = MockOracle(["YES", "NO", "YES", "NO"], model="judge-model")
        oracles = {"model-a": player_a, "model-b": player_b}

        with tempfile.TemporaryDirectory() as tmp:
            runner = Runner(
                games=Games(arcade=_ToyArcade()),
                oracle=player_a,
                judge_oracle=judge,
                model_profile="model-a",
                store=Store(Path(tmp) / "store"),
                runs_dir=Path(tmp) / "runs",
                oracle_factory=lambda name: oracles[name],
            )
            results = runner.run_chain(
                "writer-chain",
                tasks,
                condition="notes",
                seed=0,
                models=("model-a", "model-b"),
            )

        self.assertEqual([r.exact_model for r in results], ["model-a", "model-b", "model-b"])
        self.assertEqual(len(player_a.calls), 3)
        self.assertEqual(len(player_b.calls), 4)
        self.assertEqual(len(judge.calls), 4)
        for oracle in (player_a, player_b):
            self.assertTrue(
                any("Reply with a JSON array only" in call["prompt"] for call in oracle.calls)
            )

    def test_explicit_knowledge_oracle_stays_fixed(self) -> None:
        tasks = [("toy-1", 0), ("toy-2", 0), ("toy-1", 0)]
        facts = [{"subject": "Player", "relation": "moves_right", "object": "ACTION4"}]
        plan = json.dumps({"plan": ["ACTION4", "ACTION4"]})
        facts_json = json.dumps(facts)

        player_a = MockOracle([plan], model="model-a")
        player_b = MockOracle([plan, plan], model="model-b")
        writer = MockOracle(
            [facts_json, "ACTION4 moves right.", facts_json, "ACTION4 moves right."],
            model="fixed-writer",
        )
        judge = MockOracle(["YES", "NO", "YES", "NO"], model="judge-model")
        oracles = {"model-a": player_a, "model-b": player_b}

        with tempfile.TemporaryDirectory() as tmp:
            runner = Runner(
                games=Games(arcade=_ToyArcade()),
                oracle=player_a,
                knowledge_oracle=writer,
                judge_oracle=judge,
                model_profile="model-a",
                store=Store(Path(tmp) / "store"),
                runs_dir=Path(tmp) / "runs",
                oracle_factory=lambda name: oracles[name],
            )
            runner.run_chain(
                "fixed-chain",
                tasks,
                condition="notes",
                seed=0,
                models=("model-a", "model-b"),
            )

        self.assertEqual(len(writer.calls), 4)
        self.assertEqual(len(player_a.calls), 1)
        self.assertEqual(len(player_b.calls), 2)


if __name__ == "__main__":
    unittest.main()
