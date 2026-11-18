import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bsp3.experiment import chain_from_json, check_model_profiles, resolve_chains, run_experiment


def _split_csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    items = [part.strip() for part in value.split(",") if part.strip()]
    return items or None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch ARC-AGI-3 transfer experiments or check model endpoints.")
    parser.add_argument("--chains", default=None, help="Comma-separated chain names from configs/chains.toml")
    parser.add_argument("--chain-json", default=None, help="Inline chain JSON with tasks (game_id, level, model)")
    parser.add_argument("--suite", default=None, help="Chain suite name from configs/chains.toml")
    parser.add_argument("--model", default=None, help="Model profile or comma-separated step models from configs/models.toml")
    parser.add_argument("--knowledge", default=None, help="Knowledge writer profile override")
    parser.add_argument("--judge", default=None, help="Judge profile override")
    parser.add_argument("--conditions", default=None, help="Comma-separated conditions (default from chains.toml)")
    parser.add_argument("--seeds", default=None, help="Comma-separated integer seeds (default from chains.toml)")
    parser.add_argument("--budget", type=int, default=None, help="Action budget override")
    parser.add_argument("--max-tasks", type=int, default=None, help="Limit tasks per chain (e.g. 2 for T1->T2)")
    parser.add_argument("--resume", action="store_true", help="Skip tasks already logged with complete=true")
    parser.add_argument("--dry-run", action="store_true", help="Print planned runs without calling any model or game")
    parser.add_argument("--online", action="store_true", help="Allow fetching games from ARC API")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue other runs after a task error")
    parser.add_argument("--check-models", default=None, help="Comma-separated model profiles to ping and exit")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.check_models:
        profiles = _split_csv(args.check_models) or _split_csv(args.model) or ["minimax"]
        report = check_model_profiles(profiles)
        print(json.dumps(report, indent=2))
        return 0 if all(item["status"] == "ok" for item in report) else 1

    chains = (
        [chain_from_json(args.chain_json)]
        if args.chain_json
        else resolve_chains(_split_csv(args.chains), suite=args.suite)
    )
    conditions = _split_csv(args.conditions)
    seeds_raw = _split_csv(args.seeds)
    seeds = [int(s) for s in seeds_raw] if seeds_raw is not None else None

    records = run_experiment(
        chains,
        model_profile=args.model,
        knowledge_profile=args.knowledge,
        judge_profile=args.judge,
        conditions=conditions,
        seeds=seeds,
        budget=args.budget,
        max_tasks=args.max_tasks,
        resume=args.resume,
        dry_run=args.dry_run,
        offline=not args.online,
        continue_on_error=args.continue_on_error,
    )
    print(json.dumps(records, indent=2))
    return 0 if all(item.get("status") != "error" for item in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
