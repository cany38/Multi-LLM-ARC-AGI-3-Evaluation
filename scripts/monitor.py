import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bsp3.monitor import serve_monitor


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Live web monitor for BSP3 ARC-AGI-3 runs.")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="Open browser automatically")
    args = parser.parse_args(argv)
    serve_monitor(args.runs_dir, host=args.host, port=args.port, open_browser=args.open)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
