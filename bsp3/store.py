import json
from pathlib import Path
from typing import Any

from bsp3.config import clean_slug, load_experiment_config, resolve_path


class Store:
    """Stores and loads transferred knowledge across game chains."""

    def __init__(
        self,
        store_dir: Path | str = "data/knowledge",
        *,
        max_trajectory_steps: int = 30,
        max_trajectory_chars: int = 12000,
    ) -> None:
        self.store_dir = resolve_path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.max_trajectory_steps = max_trajectory_steps
        self.max_trajectory_chars = max_trajectory_chars

    @classmethod
    def from_config(cls, config_path: Path | str = "configs/experiment.toml") -> "Store":
        cfg = load_experiment_config(config_path)
        store_dir = cfg.get("paths", {}).get("store_dir", "data/knowledge")
        kcfg = cfg.get("knowledge", {})
        return cls(
            store_dir,
            max_trajectory_steps=int(kcfg.get("max_trajectory_steps", 30)),
            max_trajectory_chars=int(kcfg.get("max_trajectory_chars", 12000)),
        )

    def _entry_path(self, chain: str, condition: str) -> Path:
        chain_dir = self.store_dir / clean_slug(chain)
        chain_dir.mkdir(parents=True, exist_ok=True)
        return chain_dir / f"{clean_slug(condition)}.json"

    def add(
        self,
        chain: str,
        knowledge: Any,
        *,
        condition: str = "notes",
        game_id: str = "",
        level: int = 0,
    ) -> Path:
        path = self._entry_path(chain, condition)
        items = self.load(chain, condition=condition)
        items.append({"game_id": game_id, "level": level, "knowledge": knowledge})
        path.write_text(
            json.dumps(items, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        if condition == "ontology" and isinstance(knowledge, str):
            ttl_path = path.with_suffix(".ttl")
            ttl_blocks = [str(item["knowledge"]).strip() for item in items if item["knowledge"]]
            ttl_path.write_text("\n\n".join(ttl_blocks) + "\n", encoding="utf-8")
        return path

    def load(self, chain: str, *, condition: str = "notes") -> list[dict[str, Any]]:
        if condition == "nothing":
            return []
        path = self._entry_path(chain, condition)
        if not path.exists():
            return []
        return json.loads(path.read_text(encoding="utf-8"))

    def load_text(self, chain: str, *, condition: str = "notes") -> str:
        items = self.load(chain, condition=condition)
        if not items:
            return ""
        if condition == "trajectory":
            trimmed = []
            for item in items:
                steps = item.get("knowledge")
                if isinstance(steps, list) and self.max_trajectory_steps > 0 and len(steps) > self.max_trajectory_steps:
                    trimmed.append({**item, "knowledge": steps[-self.max_trajectory_steps :]})
                else:
                    trimmed.append(item)
            if self.max_trajectory_chars > 0:
                while len(json.dumps(trimmed, ensure_ascii=False)) > self.max_trajectory_chars:
                    largest = max(
                        (i for i in trimmed if isinstance(i.get("knowledge"), list) and i["knowledge"]),
                        key=lambda i: len(json.dumps(i, ensure_ascii=False)),
                        default=None,
                    )
                    if largest is None:
                        break
                    largest["knowledge"] = largest["knowledge"][1:]
            return json.dumps(trimmed, ensure_ascii=False)
        sections = []
        for item in items:
            header = f"[{item['game_id']} level {item['level']}]" if item.get("game_id") else "[prior run]"
            body = item["knowledge"]
            text = body if isinstance(body, str) else json.dumps(body, ensure_ascii=False)
            sections.append(f"{header}\n{text}")
        return "\n\n".join(sections)

    def clear(self, chain: str, *, condition: str | None = None) -> None:
        chain_dir = self.store_dir / clean_slug(chain)
        if not chain_dir.exists():
            return
        if condition is None:
            for path in chain_dir.iterdir():
                if path.is_file():
                    path.unlink()
            return
        json_path = chain_dir / f"{clean_slug(condition)}.json"
        json_path.unlink(missing_ok=True)
        json_path.with_suffix(".ttl").unlink(missing_ok=True)
