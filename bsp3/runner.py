from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from bsp3.agent import Agent, RunResult
from bsp3.config import load_experiment_config
from bsp3.games import Games
from bsp3.judge import Judge
from bsp3.knowledge import KnowledgeWriter
from bsp3.logger import Logger
from bsp3.oracle import Oracle, oracle_from_config
from bsp3.store import Store


def _compact_trajectory(trajectory: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "step": item["step"],
            "action": item["action"],
            "action_data": item.get("action_data"),
            "diff": item["state"].get("diff"),
            "game_state": item.get("game_state"),
        }
        for item in trajectory
    ]


class Runner:
    """Runs chains of tasks across the four knowledge-transfer conditions."""

    def __init__(
        self,
        *,
        games: Games,
        oracle: Oracle,
        store: Store | None = None,
        knowledge_oracle: Oracle | None = None,
        judge_oracle: Oracle | None = None,
        model_profile: str | None = None,
        config_path: Path | str = "configs/experiment.toml",
        models_config_path: Path | str = "configs/models.toml",
        runs_dir: Path | str | None = None,
        oracle_factory: Callable[[str], Oracle] | None = None,
    ) -> None:
        exp = load_experiment_config(config_path)
        self._config_path = config_path
        self._models_config_path = models_config_path
        self._games = games
        self._oracle = oracle
        self._model_profile = model_profile
        self._oracle_factory = oracle_factory
        self._agent = Agent.from_config(oracle, config_path=config_path)
        self._agents: dict[str, Agent] = {}
        self._oracles: dict[str, Oracle] = {}
        if model_profile:
            self._agents[model_profile] = self._agent
        self._store = store or Store.from_config(config_path)
        self._knowledge_oracle = knowledge_oracle
        self._writer = KnowledgeWriter.from_config(
            knowledge_oracle or oracle, config_path=config_path
        )
        self._judge = Judge.from_config(judge_oracle or oracle, config_path=config_path)
        self._valid_conditions = tuple(
            exp.get("conditions", {}).get(
                "valid", ("nothing", "trajectory", "notes", "ontology")
            )
        )
        self._runs_dir = runs_dir or exp.get("paths", {}).get("runs_dir", "runs")

    @classmethod
    def from_config(
        cls,
        *,
        model_profile: str = "minimax",
        knowledge_profile: str | None = None,
        judge_profile: str | None = None,
        offline: bool = True,
        config_path: Path | str = "configs/experiment.toml",
        models_config_path: Path | str = "configs/models.toml",
    ) -> "Runner":
        exp = load_experiment_config(config_path)
        k_profile = knowledge_profile
        j_profile = judge_profile or exp.get("judge", {}).get("model_profile", model_profile)

        oracles: dict[str, Oracle] = {}

        def get_oracle(profile: str) -> Oracle:
            if profile not in oracles:
                oracles[profile] = oracle_from_config(profile, config_path=models_config_path)
            return oracles[profile]

        player = get_oracle(model_profile)
        knowledge_oracle = get_oracle(k_profile) if k_profile else None
        judge_oracle = get_oracle(j_profile)
        return cls(
            games=Games(offline=offline),
            oracle=player,
            knowledge_oracle=knowledge_oracle,
            judge_oracle=judge_oracle,
            model_profile=model_profile,
            config_path=config_path,
            models_config_path=models_config_path,
            oracle_factory=get_oracle,
        )

    def _agent_for_profile(self, profile: str | None) -> tuple[Agent, Oracle, str | None]:
        active = profile or self._model_profile
        if not active or active == self._model_profile:
            return self._agent, self._oracle, active
        if active not in self._agents:
            oracle = (
                self._oracle_factory(active)
                if self._oracle_factory is not None
                else oracle_from_config(active, config_path=self._models_config_path)
            )
            self._agents[active] = Agent.from_config(oracle, config_path=self._config_path)
            self._oracles[active] = oracle
        return self._agents[active], self._oracles[active], active

    def run_task(
        self,
        chain: str,
        game_id: str,
        *,
        level: int = 0,
        condition: str = "nothing",
        seed: int = 0,
        budget: int | None = None,
        extract_knowledge: bool = True,
        extract_conditions: Sequence[str] | None = None,
        store_chain: str | None = None,
        task_index: int | None = None,
        model_profile: str | None = None,
    ) -> RunResult:
        if condition not in self._valid_conditions:
            raise ValueError(f"Invalid condition {condition!r}; expected {self._valid_conditions}")

        agent, player_oracle, active_profile = self._agent_for_profile(model_profile)
        if self._knowledge_oracle is None:
            self._writer = KnowledgeWriter.from_config(
                player_oracle, config_path=self._config_path
            )
        store_key = store_chain or chain
        knowledge = self._store.load_text(store_key, condition=condition)
        metadata: dict[str, Any] = {"chain": chain, "store_chain": store_key}
        if task_index is not None:
            metadata["task_index"] = task_index
        if active_profile:
            metadata["model_profile"] = active_profile

        logger = Logger.for_run(
            game_id=game_id,
            level=level,
            condition=condition,
            seed=seed,
            config_path=self._config_path,
            runs_dir=self._runs_dir,
            metadata=metadata,
        )
        result = agent.play(
            self._games,
            game_id,
            level=level,
            knowledge=knowledge,
            condition=condition,
            budget=budget,
            seed=seed,
            logger=logger,
        )

        targets = (
            [c for c in extract_conditions if c != "nothing"]
            if extract_conditions is not None
            else ([condition] if extract_knowledge and condition != "nothing" else [])
        )
        if targets:
            logger.record_frame({"status": "extracting_knowledge"})
            logger.record_event(
                {"event": "knowledge_start", "conditions": targets, "store_chain": store_key}
            )
            hooks = []
            for oracle in (self._writer._oracle, self._judge._oracle):
                if all(existing is not oracle for existing, _ in hooks):
                    hooks.append((oracle, oracle.on_event))
                    oracle.on_event = logger.record_event
            try:
                self.save_knowledge(store_key, result, targets)
            except Exception as exc:
                logger.record_event({"event": "error", "error": str(exc)})
                logger.record_frame({"status": "error", "error": str(exc)})
                raise
            finally:
                for oracle, prev in hooks:
                    oracle.on_event = prev
            logger.record_event(
                {"event": "knowledge_done", "conditions": targets, "store_chain": store_key}
            )
            logger.record_frame({"status": "finished"})
        logger.update_summary({"complete": True})
        return result

    def save_knowledge(
        self,
        chain: str,
        result: RunResult,
        conditions: Sequence[str],
    ) -> None:
        targets = [c for c in conditions if c != "nothing"]
        if not targets:
            return
        compact = _compact_trajectory(result.trajectory)
        if "trajectory" in targets:
            self._store.add(
                chain,
                compact,
                condition="trajectory",
                game_id=result.game_id,
                level=result.level,
            )

        symbolic = [c for c in targets if c in ("notes", "ontology")]
        if not symbolic:
            return

        facts = self._writer.extract_facts(
            compact,
            game_id=result.game_id,
            level=result.level,
            outcome=result.outcome,
        )
        verified = self._judge.filter_facts(facts, compact)
        if not verified:
            return

        for cond in symbolic:
            rendered = (
                self._writer.as_notes(verified)
                if cond == "notes"
                else self._writer.as_owl(verified)
            )
            if rendered and not self._judge.check_leak(rendered):
                self._store.add(
                    chain,
                    rendered,
                    condition=cond,
                    game_id=result.game_id,
                    level=result.level,
                )

    @property
    def store(self) -> Store:
        return self._store

    @property
    def runs_dir(self) -> Path | str:
        return self._runs_dir

    def run_chain(
        self,
        chain: str,
        tasks: Sequence[tuple[str, int]],
        *,
        condition: str = "nothing",
        seed: int = 0,
        budget: int | None = None,
        store_chain: str | None = None,
        models: Sequence[str] | None = None,
    ) -> list[RunResult]:
        results: list[RunResult] = []
        for idx, (game_id, level) in enumerate(tasks):
            step_model = models[min(idx, len(models) - 1)] if models else None
            results.append(
                self.run_task(
                    chain,
                    game_id,
                    level=level,
                    condition=condition,
                    seed=seed,
                    budget=budget,
                    extract_knowledge=idx < len(tasks) - 1,
                    store_chain=store_chain,
                    task_index=idx,
                    model_profile=step_model,
                )
            )
        return results

    def run_all_conditions(
        self,
        chain: str,
        tasks: Sequence[tuple[str, int]],
        *,
        conditions: Sequence[str] | None = None,
        seed: int = 0,
        budget: int | None = None,
        store_chain: str | None = None,
        models: Sequence[str] | None = None,
    ) -> dict[str, list[RunResult]]:
        conds = tuple(conditions or self._valid_conditions)
        if not tasks:
            return {c: [] for c in conds}

        store_key = store_chain or chain
        for c in conds:
            if c != "nothing":
                self._store.clear(store_key, condition=c)

        first_game, first_level = tasks[0]
        first_model = models[0] if models else None
        first_cond = "nothing" if "nothing" in conds else conds[0]
        t1_result = self.run_task(
            chain,
            first_game,
            level=first_level,
            condition=first_cond,
            seed=seed,
            budget=budget,
            extract_knowledge=False,
            extract_conditions=conds if len(tasks) > 1 else (),
            store_chain=store_key,
            task_index=0,
            model_profile=first_model,
        )

        out: dict[str, list[RunResult]] = {c: [t1_result] for c in conds}
        for cond in conds:
            for idx in range(1, len(tasks)):
                game_id, level = tasks[idx]
                step_model = models[min(idx, len(models) - 1)] if models else None
                res = self.run_task(
                    chain,
                    game_id,
                    level=level,
                    condition=cond,
                    seed=seed,
                    budget=budget,
                    extract_knowledge=idx < len(tasks) - 1,
                    store_chain=store_key,
                    task_index=idx,
                    model_profile=step_model,
                )
                out[cond].append(res)
        return out
