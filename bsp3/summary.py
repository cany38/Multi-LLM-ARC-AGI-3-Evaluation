import csv
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

from bsp3.config import resolve_path


def load_run_summaries(
    runs_dir: Path | str = "runs",
    *,
    chain: str | None = None,
    model: str | None = None,
    complete_only: bool = False,
) -> list[dict[str, Any]]:
    base = resolve_path(runs_dir)
    if not base.exists():
        return []
    summaries: list[dict[str, Any]] = []
    for path in sorted(base.glob("*/summary.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if complete_only and data.get("complete") is not True:
            continue
        if chain is not None and data.get("chain") != chain:
            continue
        run_model = data.get("model_profile") or data.get("exact_model")
        if model is not None and model not in (data.get("model_profile"), data.get("exact_model"), run_model):
            continue
        summaries.append(data)
    return summaries


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _aggregate_group(group_key: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    wins = sum(1 for row in rows if row.get("won"))
    actions = [float(row.get("actions_taken", 0)) for row in rows]
    calls = [float(row.get("model_calls", 0)) for row in rows]
    elapsed = [float(row.get("elapsed_s", 0.0)) for row in rows]
    prompt_toks = [float(row.get("tokens", {}).get("prompt_tokens", 0)) for row in rows]
    comp_toks = [float(row.get("tokens", {}).get("completion_tokens", 0)) for row in rows]
    thought_toks = [float(row.get("tokens", {}).get("thoughts_tokens", 0)) for row in rows]
    total_toks = [float(row.get("tokens", {}).get("total_tokens", 0)) for row in rows]

    return {
        **group_key,
        "runs": len(rows),
        "wins": wins,
        "win_rate": round(wins / len(rows), 4),
        "mean_actions": round(_mean(actions), 2),
        "mean_model_calls": round(_mean(calls), 2),
        "mean_prompt_tokens": round(_mean(prompt_toks), 1),
        "mean_completion_tokens": round(_mean(comp_toks), 1),
        "mean_thoughts_tokens": round(_mean(thought_toks), 1),
        "mean_total_tokens": round(_mean(total_toks), 1),
        "total_tokens_sum": int(sum(total_toks)),
        "mean_elapsed_s": round(_mean(elapsed), 2),
    }


def aggregate_by_condition(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in summaries:
        model = str(row.get("model_profile") or row.get("exact_model") or "unknown")
        condition = str(row.get("condition", "unknown"))
        grouped[(model, condition)].append(row)

    order = {"nothing": 0, "trajectory": 1, "notes": 2, "ontology": 3}
    items = sorted(grouped.items(), key=lambda kv: (kv[0][0], order.get(kv[0][1], 99), kv[0][1]))
    return [_aggregate_group({"model": model, "condition": cond}, rows) for (model, cond), rows in items]


def aggregate_by_task(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in summaries:
        chain = str(row.get("chain") or "(single)")
        task_index = int(row.get("task_index", -1))
        game_id = str(row.get("game_id", ""))
        level = int(row.get("level", 0))
        model = str(row.get("model_profile") or row.get("exact_model") or "unknown")
        condition = str(row.get("condition", "unknown"))
        grouped[(chain, task_index, game_id, level, model, condition)].append(row)

    order = {"nothing": 0, "trajectory": 1, "notes": 2, "ontology": 3}
    items = sorted(
        grouped.items(),
        key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], kv[0][3], kv[0][4], order.get(kv[0][5], 99)),
    )
    return [
        _aggregate_group(
            {
                "chain": chain,
                "task_index": idx,
                "game_id": game_id,
                "level": level,
                "model": model,
                "condition": cond,
            },
            rows,
        )
        for (chain, idx, game_id, level, model, cond), rows in items
    ]


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "No runs matched."
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(str(row.get(col, "")) for col in columns) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def format_summary_markdown(summaries: list[dict[str, Any]]) -> str:
    overall = aggregate_by_condition(summaries)
    per_task = aggregate_by_task(summaries)
    overall_cols = [
        "model",
        "condition",
        "runs",
        "wins",
        "win_rate",
        "mean_actions",
        "mean_model_calls",
        "mean_total_tokens",
        "mean_elapsed_s",
    ]
    task_cols = [
        "chain",
        "task_index",
        "game_id",
        "level",
        "model",
        "condition",
        "runs",
        "wins",
        "win_rate",
        "mean_actions",
        "mean_model_calls",
        "mean_total_tokens",
        "mean_elapsed_s",
    ]
    return (
        "# Experiment summary\n\n"
        f"Total matched runs: {len(summaries)}\n\n"
        "## Overall by condition\n\n"
        f"{_markdown_table(overall, overall_cols)}\n\n"
        "## Breakdown by chain and task\n\n"
        f"{_markdown_table(per_task, task_cols)}\n"
    )


def write_summary_files(
    summaries: list[dict[str, Any]],
    *,
    csv_path: Path | str = "runs/results_summary.csv",
    md_path: Path | str = "runs/results_summary.md",
) -> tuple[Path, Path]:
    per_task = aggregate_by_task(summaries)
    csv_out = resolve_path(csv_path)
    md_out = resolve_path(md_path)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "chain",
        "task_index",
        "game_id",
        "level",
        "model",
        "condition",
        "runs",
        "wins",
        "win_rate",
        "mean_actions",
        "mean_model_calls",
        "mean_prompt_tokens",
        "mean_completion_tokens",
        "mean_thoughts_tokens",
        "mean_total_tokens",
        "total_tokens_sum",
        "mean_elapsed_s",
    ]
    with csv_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_task)

    md_out.write_text(format_summary_markdown(summaries), encoding="utf-8")
    return csv_out, md_out
