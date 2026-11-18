from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

from bsp3.config import clean_slug, load_experiment_config, resolve_path


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class Logger:
    """Writes per-step JSONL traces and final run summaries to runs/."""

    def __init__(
        self,
        run_dir: Path | str,
        *,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        base = resolve_path(run_dir)
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        self.run_dir = base if run_id is None and base.name == self.run_id else base / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = dict(metadata) if metadata else {}
        self.steps_path = self.run_dir / "steps.jsonl"
        self.summary_path = self.run_dir / "summary.json"
        self.events_path = self.run_dir / "events.jsonl"
        self.live_path = self.run_dir / "live.json"
        self._live_state: dict[str, Any] = {"run_id": self.run_id, **self.metadata}

    @classmethod
    def for_run(
        cls,
        *,
        game_id: str,
        level: int,
        condition: str,
        seed: int,
        config_path: Path | str = "configs/experiment.toml",
        runs_dir: Path | str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "Logger":
        if runs_dir is None:
            runs_dir = load_experiment_config(config_path).get("paths", {}).get("runs_dir", "runs")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        slug = clean_slug(game_id.split("-", 1)[0])
        run_id = f"{stamp}_{slug}_L{level}_{clean_slug(condition)}_s{seed}"
        return cls(runs_dir, run_id=run_id, metadata=metadata)

    def record_step(self, step: dict[str, Any]) -> None:
        with self.steps_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(step, ensure_ascii=False) + "\n")

    def record_event(self, event: dict[str, Any]) -> None:
        payload = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **event}
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def record_frame(self, fields: dict[str, Any]) -> None:
        self._live_state.update(fields)
        self._live_state["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        _write_atomic(self.live_path, json.dumps(self._live_state, ensure_ascii=False))
        _write_atomic(self.run_dir.parent / "latest.txt", self.run_id)

    def record_run(self, summary: dict[str, Any]) -> None:
        payload = {"run_id": self.run_id, **self.metadata, **summary}
        self.summary_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def update_summary(self, fields: dict[str, Any]) -> None:
        if not self.summary_path.exists():
            return
        payload = json.loads(self.summary_path.read_text(encoding="utf-8"))
        payload.update(fields)
        self.summary_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def read_steps(self) -> list[dict[str, Any]]:
        return _read_jsonl(self.steps_path)

    def read_events(self) -> list[dict[str, Any]]:
        return _read_jsonl(self.events_path)

    def read_summary(self) -> dict[str, Any] | None:
        if not self.summary_path.exists():
            return None
        return json.loads(self.summary_path.read_text(encoding="utf-8"))
