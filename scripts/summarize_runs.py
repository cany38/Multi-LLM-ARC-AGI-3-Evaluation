import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bsp3.summary import format_summary_markdown, load_run_summaries, write_summary_files


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate run summaries into CSV and Markdown tables.")
    parser.add_argument("--runs-dir", default="runs", help="Directory containing per-run summary.json folders")
    parser.add_argument("--chain", default=None, help="Filter by chain name")
    parser.add_argument("--model", default=None, help="Filter by model profile or exact model name")
    parser.add_argument("--complete-only", action="store_true", help="Include only runs marked complete=true")
    parser.add_argument("--csv", default="runs/results_summary.csv", help="Output CSV path")
    parser.add_argument("--md", default="runs/results_summary.md", help="Output Markdown path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summaries = load_run_summaries(
        args.runs_dir,
        chain=args.chain,
        model=args.model,
        complete_only=args.complete_only,
    )
    csv_out, md_out = write_summary_files(summaries, csv_path=args.csv, md_path=args.md)
    print(format_summary_markdown(summaries))
    print(f"Wrote {csv_out} and {md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
