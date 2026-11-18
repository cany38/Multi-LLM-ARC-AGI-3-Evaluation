from pathlib import Path
import re
import tomllib
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def resolve_path(path: Path | str) -> Path:
    p = Path(path)
    if p.exists() or p.is_absolute():
        return p
    return ROOT_DIR / p


def clean_slug(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text)


def load_toml(path: Path | str) -> dict[str, Any]:
    resolved = resolve_path(path)
    with resolved.open("rb") as f:
        return tomllib.load(f)


def load_models_config(path: Path | str = "configs/models.toml") -> dict[str, Any]:
    return load_toml(path)


def load_experiment_config(path: Path | str = "configs/experiment.toml") -> dict[str, Any]:
    return load_toml(path)


def load_prompt(name: str, prompts_dir: Path | str = "prompts") -> str:
    filename = name if name.endswith(".txt") else f"{name}.txt"
    path = resolve_path(prompts_dir) / filename
    return path.read_text(encoding="utf-8").strip()


def render_prompt(
    name: str,
    prompts_dir: Path | str = "prompts",
    **values: Any,
) -> str:
    template = load_prompt(name, prompts_dir=prompts_dir)

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in values:
            raise KeyError(f"Missing prompt variable {key!r} for {name}")
        return str(values[key])

    return _PLACEHOLDER_RE.sub(replace, template)
