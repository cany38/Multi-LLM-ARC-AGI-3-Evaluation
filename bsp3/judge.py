from collections.abc import Sequence
import json
from pathlib import Path
from typing import Any

from bsp3.config import load_experiment_config, render_prompt
from bsp3.oracle import Oracle


class Judge:
    """Checks candidate facts against observations and detects solution leaks."""

    def __init__(
        self,
        oracle: Oracle,
        *,
        prompts_dir: Path | str = "prompts",
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> None:
        self._oracle = oracle
        self._prompts_dir = prompts_dir
        self._temperature = temperature
        self._max_tokens = max_tokens

    @classmethod
    def from_config(
        cls,
        oracle: Oracle,
        config_path: Path | str = "configs/experiment.toml",
    ) -> "Judge":
        exp = load_experiment_config(config_path)
        jcfg = exp.get("judge", {})
        prompts_dir = exp.get("paths", {}).get("prompts_dir", "prompts")
        return cls(
            oracle,
            prompts_dir=prompts_dir,
            temperature=float(jcfg.get("temperature", 0.0)),
            max_tokens=int(jcfg.get("max_tokens", 512)),
        )

    def check_fact(
        self, fact: dict[str, Any], trajectory: Sequence[dict[str, Any]]
    ) -> bool:
        prompt = render_prompt(
            "judge_fact",
            prompts_dir=self._prompts_dir,
            fact_json=json.dumps(fact, ensure_ascii=False),
            trajectory_json=json.dumps(list(trajectory), ensure_ascii=False),
        )
        verdict = self._oracle.choose(
            prompt,
            ["YES", "NO"],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        return verdict == "YES"

    def check_leak(self, text: str) -> bool:
        if not text.strip():
            return False
        prompt = render_prompt(
            "judge_leak",
            prompts_dir=self._prompts_dir,
            text=text,
        )
        verdict = self._oracle.choose(
            prompt,
            ["YES", "NO"],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        return verdict == "YES"

    def filter_facts(
        self,
        facts: Sequence[dict[str, Any]],
        trajectory: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return [fact for fact in facts if self.check_fact(fact, trajectory)]
