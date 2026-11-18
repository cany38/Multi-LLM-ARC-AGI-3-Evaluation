from contextlib import redirect_stdout
from dataclasses import dataclass, field
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from bsp3.agent import Agent
from bsp3.experiment import (
    ChainSpec,
    check_model_profiles,
    make_store_key,
    resolve_chains,
    run_experiment,
)
from bsp3.games import Games
from bsp3.knowledge import validate_turtle
from bsp3.oracle import MockOracle, OracleError
from bsp3.runner import Runner
from bsp3.store import Store
from bsp3.summary import (
    aggregate_by_condition,
    aggregate_by_task,
    format_summary_markdown,
    load_run_summaries,
    write_summary_files,
)
from scripts.run_experiment import main as run_experiment_cli
from scripts.summarize_runs import main as summarize_cli


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
        grid[2, 2] = 3
        return _FakeObs(frame=[grid], levels_completed=0)

    def step(self, action: object, data: dict[str, int] | None = None) -> _FakeObs:
        del action, data
        self._step += 1
        grid = np.zeros((8, 8), dtype=np.int8)
        grid[2 + self._step, 2] = 3
        self._game.level_index += 1
        return _FakeObs(frame=[grid], levels_completed=1, state="NOT_FINISHED")


class _FakeArcade:
    def get_environments(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(game_id="gameA-1"), SimpleNamespace(game_id="gameB-2")]

    def make(self, game_id: str, seed: int = 0) -> _FakeEnv:
        del seed
        return _FakeEnv(game_id)


def _scripted_answer(seen_prompts: list[str], *, fail_tag: str | None = None):
    def _reply(prompt: str, system: str | None) -> str:
        del system
        seen_prompts.append(prompt)
        if fail_tag and fail_tag in prompt:
            raise OracleError("simulated failure")
        if "Reply with a JSON array only" in prompt:
            return '[{"subject": "Marker", "relation": "movesWith", "object": "ACTION1", "evidence": "clears level"}]'
        if "Check whether the candidate fact" in prompt:
            return "YES"
        if "Check whether the transferred knowledge below" in prompt:
            return "NO"
        if "Write short, clear plain-language notes" in prompt:
            return "- Rule from prior task: ACTION1 moves toward completion."
        if "Convert these verified game rules into a concise OWL ontology" in prompt:
            return (
                "@prefix arc: <http://bsp3.local/arc#> .\n"
                "@prefix owl: <http://www.w3.org/2002/07/owl#> .\n"
                "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n"
                "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n\n"
                "arc:MovementRule a owl:Class .\n"
                "arc:triggersTransition a owl:ObjectProperty .\n"
            )
        return '{"plan": ["ACTION1"], "reason": "move down once"}'

    return _reply


class ExperimentAndSummaryTests(unittest.TestCase):
    def test_resolve_chains_and_suites(self) -> None:
        pilot = resolve_chains(suite="pilot")
        self.assertEqual([c.name for c in pilot], ["within_tu93", "chain2_click_align"])
        self.assertEqual(pilot[0].tasks[0], ("tu93-0768757b", 0))

        cross = resolve_chains(suite="cross")
        self.assertEqual(len(cross), 5)

        picked = resolve_chains(["within_tu93", "within_tu93", "chain1_grid_nav"])
        self.assertEqual([c.name for c in picked], ["within_tu93", "chain1_grid_nav"])

        with self.assertRaises(ValueError):
            resolve_chains(suite="missing_suite")
        with self.assertRaises(ValueError):
            resolve_chains(["missing_chain"])

    def test_run_experiment_dry_run_and_seed_isolation(self) -> None:
        chains = [
            ChainSpec(
                name="mini_chain",
                group="test",
                description="test chain",
                tasks=(("gameA-1", 0), ("gameB-2", 0)),
            )
        ]
        planned = run_experiment(
            chains,
            model_profile="mock",
            conditions=("nothing", "trajectory", "notes", "ontology"),
            seeds=(0, 1),
            dry_run=True,
        )
        self.assertEqual(len(planned), 10)
        self.assertTrue(all(row["status"] == "planned" for row in planned))

        prompts: list[str] = []
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            runs_dir = tmp_path / "runs"
            store = Store(tmp_path / "knowledge")
            oracle = MockOracle(_scripted_answer(prompts))

            def make_runner() -> Runner:
                return Runner(
                    games=Games(arcade=_FakeArcade()),
                    oracle=oracle,
                    store=store,
                    model_profile="mock",
                    runs_dir=runs_dir,
                )

            records = run_experiment(
                chains,
                model_profile="mock",
                conditions=("nothing", "trajectory", "notes", "ontology"),
                seeds=(0, 1),
                runner_factory=make_runner,
            )
            self.assertEqual(len(records), 10)
            self.assertTrue(all(row["status"] == "ok" and row["won"] for row in records))

            t1_step_prompts = [
                p for p in prompts if "Game: gameA-1 | Level: 0 | Remaining action budget:" in p
            ]
            self.assertEqual(len(t1_step_prompts), 2)

            for seed in (0, 1):
                key = make_store_key("mini_chain", "mock", seed)
                self.assertEqual(len(store.load(key, condition="notes")), 1)
                self.assertEqual(len(store.load(key, condition="trajectory")), 1)
                ttl_path = store.store_dir / key / "ontology.ttl"
                self.assertTrue(validate_turtle(ttl_path.read_text(encoding="utf-8")))

            task2_notes_prompts = [
                p for p in prompts if "Game: gameB-2" in p and "Rule from prior task" in p
            ]
            self.assertEqual(len(task2_notes_prompts), 2)
            task2_nothing_prompts = [
                p for p in prompts if "Game: gameB-2" in p and "Transferred knowledge: none." in p
            ]
            self.assertEqual(len(task2_nothing_prompts), 2)

            summaries = load_run_summaries(runs_dir, complete_only=True)
            self.assertEqual(len(summaries), 10)
            by_cond = aggregate_by_condition(summaries)
            self.assertEqual([row["condition"] for row in by_cond], ["nothing", "trajectory", "notes", "ontology"])
            self.assertTrue(all(row["win_rate"] == 1.0 for row in by_cond))

    def test_resume_skips_completed_prefix_and_preserves_store(self) -> None:
        chains = [
            ChainSpec(
                name="resume_chain",
                group="test",
                description="resume test",
                tasks=(("gameA-1", 0), ("gameB-2", 0)),
            )
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            runs_dir = tmp_path / "runs"
            store = Store(tmp_path / "knowledge")

            fail_oracle = MockOracle(_scripted_answer([], fail_tag="Game: gameB-2"))
            first = run_experiment(
                chains,
                model_profile="mock",
                conditions=("notes",),
                seeds=(0,),
                continue_on_error=True,
                runner_factory=lambda: Runner(
                    games=Games(arcade=_FakeArcade()),
                    oracle=fail_oracle,
                    store=store,
                    model_profile="mock",
                    runs_dir=runs_dir,
                ),
            )
            self.assertEqual([r["status"] for r in first], ["ok", "error"])

            seen_resumed: list[str] = []
            ok_oracle = MockOracle(_scripted_answer(seen_resumed))
            resumed = run_experiment(
                chains,
                model_profile="mock",
                conditions=("notes",),
                seeds=(0,),
                resume=True,
                runner_factory=lambda: Runner(
                    games=Games(arcade=_FakeArcade()),
                    oracle=ok_oracle,
                    store=store,
                    model_profile="mock",
                    runs_dir=runs_dir,
                ),
            )
            self.assertEqual([r["status"] for r in resumed], ["skipped", "ok"])
            self.assertTrue(any("Rule from prior task" in p for p in seen_resumed))

    def test_check_model_profiles_and_summary_cli(self) -> None:
        def fake_factory(name: str) -> MockOracle:
            if name == "broken":
                raise OracleError("missing key")
            return MockOracle(["READY"], model=f"model-{name}")

        checks = check_model_profiles(["good", "broken"], oracle_factory=fake_factory)
        self.assertEqual(checks[0]["status"], "ok")
        self.assertEqual(checks[0]["reply"], "READY")
        self.assertEqual(checks[1]["status"], "error")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            runs_dir = tmp_path / "runs"
            r1 = runs_dir / "run_1"
            r1.mkdir(parents=True)
            (r1 / "summary.json").write_text(
                json.dumps(
                    {
                        "run_id": "run_1",
                        "chain": "c1",
                        "task_index": 0,
                        "game_id": "gameA-1",
                        "level": 0,
                        "condition": "nothing",
                        "seed": 0,
                        "model_profile": "qwen",
                        "actions_taken": 5,
                        "model_calls": 2,
                        "tokens": {"prompt_tokens": 100, "completion_tokens": 20, "thoughts_tokens": 10, "total_tokens": 130},
                        "elapsed_s": 1.5,
                        "won": False,
                        "complete": True,
                    }
                ),
                encoding="utf-8",
            )
            r2 = runs_dir / "run_2"
            r2.mkdir(parents=True)
            (r2 / "summary.json").write_text(
                json.dumps(
                    {
                        "run_id": "run_2",
                        "chain": "c1",
                        "task_index": 1,
                        "game_id": "gameB-2",
                        "level": 0,
                        "condition": "ontology",
                        "seed": 0,
                        "model_profile": "qwen",
                        "actions_taken": 3,
                        "model_calls": 1,
                        "tokens": {"prompt_tokens": 80, "completion_tokens": 15, "thoughts_tokens": 5, "total_tokens": 100},
                        "elapsed_s": 0.9,
                        "won": True,
                        "complete": True,
                    }
                ),
                encoding="utf-8",
            )

            csv_path = tmp_path / "summary.csv"
            md_path = tmp_path / "summary.md"
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = summarize_cli(["--runs-dir", str(runs_dir), "--csv", str(csv_path), "--md", str(md_path)])
            self.assertEqual(code, 0)
            self.assertTrue(csv_path.exists())
            self.assertIn("ontology", md_path.read_text(encoding="utf-8"))
            tasks = aggregate_by_task(load_run_summaries(runs_dir))
            self.assertEqual(len(tasks), 2)
            md = format_summary_markdown(load_run_summaries(runs_dir))
            self.assertIn("| qwen | ontology | 1 | 1 | 1.0 |", md)

            out_csv, out_md = write_summary_files([], csv_path=tmp_path / "empty.csv", md_path=tmp_path / "empty.md")
            self.assertTrue(out_csv.exists())
            self.assertIn("No runs matched.", out_md.read_text(encoding="utf-8"))

        with redirect_stdout(io.StringIO()):
            self.assertEqual(run_experiment_cli(["--suite", "pilot", "--dry-run", "--seeds", "0"]), 0)

    def test_multi_model_chain_and_knowledge_profile_from_config(self) -> None:
        chains = [
            ChainSpec(
                name="three_step",
                group="cross",
                description="qwen to gemma to minimax",
                tasks=(("gameA-1", 0), ("gameB-2", 0), ("gameA-1", 1)),
                models=("qwen", "gemma", "minimax"),
            )
        ]
        planned = run_experiment(
            chains,
            conditions=("nothing", "notes"),
            seeds=(0,),
            dry_run=True,
        )
        step_models = [(r["task_index"], r["condition"], r["model_profile"]) for r in planned]
        self.assertEqual(
            step_models,
            [
                (0, "nothing", "qwen"),
                (1, "nothing", "gemma"),
                (2, "nothing", "minimax"),
                (1, "notes", "gemma"),
                (2, "notes", "minimax"),
            ],
        )

        with tempfile.TemporaryDirectory() as tmp:
            exp_cfg = Path(tmp) / "experiment.toml"
            exp_cfg.write_text(
                '[judge]\nmodel_profile = "mock"\n',
                encoding="utf-8",
            )
            runner = Runner.from_config(
                model_profile="mock",
                knowledge_profile="mock",
                config_path=exp_cfg,
                models_config_path="configs/models.toml",
            )
            self.assertEqual(runner._writer._oracle.model, "mock")

    def test_empty_length_oracle_error_falls_back_to_choose(self) -> None:
        class _LengthThenChooseOracle(MockOracle):
            def query(self, prompt: str, *, system: str | None = None, temperature: float = 0.0, max_tokens: int = 1024):
                if "Choose one option" in prompt:
                    return super().query(prompt, system=system, temperature=temperature, max_tokens=max_tokens)
                raise OracleError("nvidia: returned empty text (finish_reason=length)")

        oracle = _LengthThenChooseOracle(["ACTION1"])
        result = Agent(oracle=oracle).play(Games(arcade=_FakeArcade()), "gameA-1", budget=1)
        self.assertTrue(result.won)
        self.assertEqual(result.actions_taken, 1)

        class _AuthFailOracle(MockOracle):
            def query(self, prompt: str, *, system: str | None = None, temperature: float = 0.0, max_tokens: int = 1024):
                del prompt, system, temperature, max_tokens
                raise OracleError("nvidia: all API keys failed authentication or quota")

        with self.assertRaises(OracleError):
            Agent(oracle=_AuthFailOracle()).play(Games(arcade=_FakeArcade()), "gameA-1", budget=1)


if __name__ == "__main__":
    unittest.main()
