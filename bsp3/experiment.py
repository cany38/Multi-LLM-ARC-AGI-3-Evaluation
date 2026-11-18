from collections.abc import Callable, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from bsp3.config import clean_slug, load_experiment_config, load_toml, resolve_path
from bsp3.oracle import oracle_from_config
from bsp3.runner import Runner

DEFAULT_CHAINS_PATH = "configs/chains.toml"


@dataclass(frozen=True)
class ChainSpec:
    name: str
    group: str
    description: str
    tasks: tuple[tuple[str, int], ...]
    models: tuple[str, ...] = ()


def load_chains_config(path: Path | str = DEFAULT_CHAINS_PATH) -> dict[str, Any]:
    return load_toml(path)


def _build_chain_spec(name: str, raw: dict[str, Any]) -> ChainSpec:
    raw_tasks = raw.get("tasks") or []
    tasks = tuple((str(item["game_id"]), int(item.get("level", 0))) for item in raw_tasks)
    if not tasks:
        raise ValueError(f"Chain {name!r} defines no tasks")
    raw_models = raw.get("models") or [item["model"] for item in raw_tasks if "model" in item]
    models = tuple(str(m).strip() for m in raw_models if str(m).strip())
    return ChainSpec(
        name=name,
        group=str(raw.get("group", "")),
        description=str(raw.get("description", "")),
        tasks=tasks,
        models=models,
    )


def chain_from_json(payload: str | dict[str, Any], *, name: str = "custom") -> ChainSpec:
    """Build a chain from an inline JSON spec, used by the monitor."""
    data = json.loads(payload) if isinstance(payload, str) else dict(payload)
    return _build_chain_spec(str(data.get("name") or name), data)


def resolve_chains(
    names: Sequence[str] | None = None,
    *,
    suite: str | None = None,
    config_path: Path | str = DEFAULT_CHAINS_PATH,
) -> list[ChainSpec]:
    cfg = load_chains_config(config_path)
    chains_map = cfg.get("chains", {})
    suites_map = cfg.get("suites", {})

    selected: list[str]
    if names:
        selected = list(names)
    else:
        suite_name = suite or str(cfg.get("defaults", {}).get("suite", "all"))
        if suite_name not in suites_map:
            raise ValueError(f"Unknown suite {suite_name!r}; expected {sorted(suites_map)}")
        selected = list(suites_map[suite_name])

    specs: list[ChainSpec] = []
    seen: set[str] = set()
    for name in selected:
        if name not in chains_map:
            raise ValueError(f"Unknown chain {name!r}; expected {sorted(chains_map)}")
        if name not in seen:
            seen.add(name)
            specs.append(_build_chain_spec(name, chains_map[name]))
    return specs


def _normalize_models(model_profile: str | Sequence[str] | None) -> tuple[str, ...]:
    if model_profile is None:
        return ()
    if isinstance(model_profile, str):
        return tuple(part.strip() for part in model_profile.split(",") if part.strip())
    return tuple(str(part).strip() for part in model_profile if str(part).strip())


def _resolve_step_models(
    chain: ChainSpec,
    model_profile: str | Sequence[str] | None,
    defaults: dict[str, Any],
) -> tuple[str, ...]:
    explicit = _normalize_models(model_profile)
    if explicit:
        return explicit
    if chain.models:
        return chain.models
    default_models = _normalize_models(defaults.get("models") or defaults.get("model_profile"))
    return default_models or ("minimax",)


def make_store_key(chain: str, model_profile: str | Sequence[str], seed: int) -> str:
    models = _normalize_models(model_profile) or ("minimax",)
    model_slug = "_".join(clean_slug(m) for m in models)
    return f"{clean_slug(chain)}__{model_slug}__s{seed}"


def _run_key(summary: dict[str, Any]) -> tuple[str, str, int, str, int, str, int]:
    model = str(summary.get("model_profile") or summary.get("exact_model") or "")
    return (
        str(summary.get("chain", "")),
        model,
        int(summary.get("seed", 0)),
        str(summary.get("condition", "")),
        int(summary.get("task_index", -1)),
        str(summary.get("game_id", "")),
        int(summary.get("level", 0)),
    )


def load_completed_keys(runs_dir: Path | str) -> set[tuple[str, str, int, str, int, str, int]]:
    base = resolve_path(runs_dir)
    if not base.exists():
        return set()
    keys: set[tuple[str, str, int, str, int, str, int]] = set()
    for path in sorted(base.glob("*/summary.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("complete") is True and "chain" in data:
            keys.add(_run_key(data))
    return keys


def _first_pending_followup(
    chain: ChainSpec,
    tasks: Sequence[tuple[str, int]],
    *,
    step_models: Sequence[str],
    seed: int,
    condition: str,
    done: set[tuple[str, str, int, str, int, str, int]],
) -> int | None:
    for idx in range(1, len(tasks)):
        game_id, level = tasks[idx]
        step_model = step_models[min(idx, len(step_models) - 1)]
        key = (chain.name, step_model, seed, condition, idx, game_id, level)
        if key not in done:
            return idx
    return None


def _run_single_task(
    runner: Runner,
    *,
    chain: ChainSpec,
    idx: int,
    game_id: str,
    level: int,
    condition: str,
    seed: int,
    budget: int | None,
    extract_knowledge: bool,
    extract_conditions: Sequence[str] | None = None,
    store_key: str,
    model_profile: str,
    continue_on_error: bool,
) -> tuple[dict[str, Any], bool]:
    base_row = {
        "chain": chain.name,
        "task_index": idx,
        "game_id": game_id,
        "level": level,
        "condition": condition,
        "seed": seed,
        "model_profile": model_profile,
        "store_chain": store_key,
    }
    try:
        result = runner.run_task(
            chain.name,
            game_id,
            level=level,
            condition=condition,
            seed=seed,
            budget=budget,
            extract_knowledge=extract_knowledge,
            extract_conditions=extract_conditions,
            store_chain=store_key,
            task_index=idx,
            model_profile=model_profile,
        )
    except Exception as exc:
        if not continue_on_error:
            raise
        return {**base_row, "status": "error", "error": str(exc)}, False

    return {
        **base_row,
        "status": "ok",
        "won": result.won,
        "outcome": result.outcome,
        "actions_taken": result.actions_taken,
        "model_calls": result.model_calls,
        "total_tokens": result.tokens.total_tokens,
        "elapsed_s": round(result.elapsed_s, 3),
        "exact_model": result.exact_model,
        "run_dir": str(result.run_dir) if result.run_dir else None,
    }, True


def run_experiment(
    chains: Sequence[ChainSpec],
    *,
    model_profile: str | Sequence[str] | None = None,
    knowledge_profile: str | None = None,
    judge_profile: str | None = None,
    conditions: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    budget: int | None = None,
    max_tasks: int | None = None,
    resume: bool = False,
    dry_run: bool = False,
    offline: bool = True,
    continue_on_error: bool = False,
    experiment_config: Path | str = "configs/experiment.toml",
    models_config: Path | str = "configs/models.toml",
    chains_config: Path | str = DEFAULT_CHAINS_PATH,
    runner_factory: Callable[[], Runner] | None = None,
) -> list[dict[str, Any]]:
    cfg = load_chains_config(chains_config)
    defaults = cfg.get("defaults", {})
    exp = load_experiment_config(experiment_config)
    conds = list(conditions or defaults.get("conditions", ("nothing", "trajectory", "notes", "ontology")))
    seed_list = [int(s) for s in (seeds if seeds is not None else defaults.get("seeds", [0]))]

    initial_models = _normalize_models(model_profile) or _normalize_models(
        defaults.get("models") or defaults.get("model_profile")
    ) or ("minimax",)

    runner = None
    if not dry_run:
        runner = (
            runner_factory()
            if runner_factory is not None
            else Runner.from_config(
                model_profile=initial_models[0],
                knowledge_profile=knowledge_profile,
                judge_profile=judge_profile,
                offline=offline,
                config_path=experiment_config,
                models_config_path=models_config,
            )
        )

    runs_dir = runner.runs_dir if runner is not None else exp.get("paths", {}).get("runs_dir", "runs")
    done = load_completed_keys(runs_dir) if resume else set()
    records: list[dict[str, Any]] = []

    for chain in chains:
        tasks = chain.tasks[:max_tasks] if max_tasks else chain.tasks
        if not tasks:
            continue
        step_models = _resolve_step_models(chain, model_profile, defaults)
        for seed in seed_list:
            store_key = make_store_key(chain.name, step_models, seed)
            first_game, first_level = tasks[0]
            first_model = step_models[0]
            first_cond = "nothing" if "nothing" in conds else conds[0]
            t1_key = (chain.name, first_model, seed, first_cond, 0, first_game, first_level)

            skip_t1 = False
            if resume and t1_key in done:
                if len(tasks) == 1 or runner is None:
                    skip_t1 = True
                else:
                    skip_t1 = all(
                        _first_pending_followup(
                            chain,
                            tasks,
                            step_models=step_models,
                            seed=seed,
                            condition=c,
                            done=done,
                        )
                        is None
                        or bool(runner.store.load(store_key, condition=c))
                        for c in conds
                        if c != "nothing"
                    )

            t1_ok = True
            if skip_t1:
                records.append(
                    {
                        "chain": chain.name,
                        "task_index": 0,
                        "game_id": first_game,
                        "level": first_level,
                        "condition": first_cond,
                        "seed": seed,
                        "model_profile": first_model,
                        "store_chain": store_key,
                        "status": "skipped",
                    }
                )
            elif dry_run:
                records.append(
                    {
                        "chain": chain.name,
                        "task_index": 0,
                        "game_id": first_game,
                        "level": first_level,
                        "condition": first_cond,
                        "seed": seed,
                        "model_profile": first_model,
                        "store_chain": store_key,
                        "status": "planned",
                    }
                )
            else:
                assert runner is not None
                for c in conds:
                    if c != "nothing":
                        runner.store.clear(store_key, condition=c)
                t1_row, t1_ok = _run_single_task(
                    runner,
                    chain=chain,
                    idx=0,
                    game_id=first_game,
                    level=first_level,
                    condition=first_cond,
                    seed=seed,
                    budget=budget,
                    extract_knowledge=False,
                    extract_conditions=conds if len(tasks) > 1 else (),
                    store_key=store_key,
                    model_profile=first_model,
                    continue_on_error=continue_on_error,
                )
                records.append(t1_row)

            if len(tasks) == 1:
                continue

            for condition in conds:
                if not t1_ok and condition != "nothing":
                    continue
                pending_idx = (
                    _first_pending_followup(
                        chain,
                        tasks,
                        step_models=step_models,
                        seed=seed,
                        condition=condition,
                        done=done,
                    )
                    if resume and skip_t1
                    else 1
                )

                for idx in range(1, len(tasks)):
                    game_id, level = tasks[idx]
                    step_model = step_models[min(idx, len(step_models) - 1)]
                    key = (chain.name, step_model, seed, condition, idx, game_id, level)
                    skip = resume and skip_t1 and (
                        key in done
                        if condition == "nothing"
                        else (pending_idx is None or idx < pending_idx)
                    )
                    if skip:
                        records.append(
                            {
                                "chain": chain.name,
                                "task_index": idx,
                                "game_id": game_id,
                                "level": level,
                                "condition": condition,
                                "seed": seed,
                                "model_profile": step_model,
                                "store_chain": store_key,
                                "status": "skipped",
                            }
                        )
                        continue

                    if dry_run:
                        records.append(
                            {
                                "chain": chain.name,
                                "task_index": idx,
                                "game_id": game_id,
                                "level": level,
                                "condition": condition,
                                "seed": seed,
                                "model_profile": step_model,
                                "store_chain": store_key,
                                "status": "planned",
                            }
                        )
                        continue

                    assert runner is not None
                    row, ok = _run_single_task(
                        runner,
                        chain=chain,
                        idx=idx,
                        game_id=game_id,
                        level=level,
                        condition=condition,
                        seed=seed,
                        budget=budget,
                        extract_knowledge=idx < len(tasks) - 1,
                        store_key=store_key,
                        model_profile=step_model,
                        continue_on_error=continue_on_error,
                    )
                    records.append(row)
                    if not ok and condition != "nothing":
                        break

    return records


def check_model_profiles(
    profiles: Sequence[str],
    *,
    config_path: Path | str = "configs/models.toml",
    oracle_factory: Callable[[str], Any] | None = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for profile in profiles:
        try:
            oracle = (
                oracle_factory(profile)
                if oracle_factory is not None
                else oracle_from_config(profile, config_path=config_path)
            )
            resp = oracle.query(
                "Reply with the single word READY.",
                temperature=0.0,
                max_tokens=1024,
            )
            results.append(
                {
                    "profile": profile,
                    "status": "ok",
                    "provider": resp.provider,
                    "exact_model": resp.model,
                    "latency_ms": round(resp.latency_ms, 1),
                    "total_tokens": resp.usage.total_tokens,
                    "reply": resp.text.strip()[:80],
                }
            )
        except Exception as exc:
            results.append({"profile": profile, "status": "error", "error": str(exc)})
    return results
