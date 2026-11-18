"""Start the project: sync dependencies and open the live monitor."""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> int:
    if shutil.which("uv") is None:
        print("uv is not installed. Get it from https://docs.astral.sh/uv/")
        return 1

    print("Syncing dependencies...")
    if subprocess.call(["uv", "sync"], cwd=ROOT) != 0:
        print("uv sync failed.")
        return 1

    print("Opening the live monitor at http://127.0.0.1:8765/  (Ctrl+C to stop)")
    return subprocess.call(["uv", "run", "python", "scripts/monitor.py", "--open"], cwd=ROOT)


if __name__ == "__main__":
    sys.exit(main())
