import argparse
import json

from bsp3.runner import Runner


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an ARC-AGI-3 transfer experiment task.")
    parser.add_argument("--game", required=True, help="Game ID (e.g. ls20-9607627b)")
    parser.add_argument("--level", type=int, default=0, help="Starting level index")
    parser.add_argument("--chain", default="default", help="Knowledge chain name")
    parser.add_argument(
        "--condition",
        default="nothing",
        choices=["nothing", "trajectory", "notes", "ontology"],
        help="Transfer condition",
    )
    parser.add_argument("--model", default="minimax", help="Model profile from configs/models.toml")
    parser.add_argument("--judge", default=None, help="Judge profile override")
    parser.add_argument("--budget", type=int, default=None, help="Action budget override")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--online", action="store_true", help="Allow fetching games from API")
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Skip extracting knowledge after the run",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runner = Runner.from_config(
        model_profile=args.model,
        judge_profile=args.judge,
        offline=not args.online,
    )
    result = runner.run_task(
        args.chain,
        args.game,
        level=args.level,
        condition=args.condition,
        seed=args.seed,
        budget=args.budget,
        extract_knowledge=not args.no_extract,
    )
    summary = {
        "game_id": result.game_id,
        "level": result.level,
        "condition": result.condition,
        "won": result.won,
        "outcome": result.outcome,
        "actions_taken": result.actions_taken,
        "model_calls": result.model_calls,
        "total_tokens": result.tokens.total_tokens,
        "provider": result.provider,
        "exact_model": result.exact_model,
        "run_dir": str(result.run_dir) if result.run_dir else None,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
