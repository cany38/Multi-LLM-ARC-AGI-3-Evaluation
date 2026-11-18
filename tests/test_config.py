import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from bsp3.config import (
    load_experiment_config,
    load_models_config,
    load_prompt,
    render_prompt,
)
from bsp3.oracle import MockOracle, OpenAICompatOracle, OracleError, oracle_from_config


class ConfigAndPromptsTests(unittest.TestCase):
    def test_loads_model_and_experiment_configs(self) -> None:
        models = load_models_config()
        self.assertIn("gemini", models["providers"])
        self.assertNotIn("openrouter", models["providers"])
        self.assertIn("nvidia", models["providers"])
        self.assertIn("hpc", models["providers"])
        self.assertEqual(
            [entry["model"] for entry in models["models"]["gemma"]], ["gemma-4-26b-a4b-it"]
        )
        self.assertEqual(
            models["models"]["minimax"][0]["extra_body"]["reasoning_effort"], "low"
        )
        self.assertEqual(set(models["models"]), {"gemma", "minimax", "qwen", "mock"})

        exp = load_experiment_config()
        self.assertEqual(exp["agent"]["plan_steps"], 5)
        self.assertEqual(
            exp["conditions"]["valid"],
            ["nothing", "trajectory", "notes", "ontology"],
        )

    def test_prompt_rendering_preserves_json_braces(self) -> None:
        rendered = render_prompt(
            "agent_step",
            game_id="ls20",
            level=0,
            budget_remaining=30,
            actions="ACTION1, ACTION2",
            knowledge_section="No notes.",
            recent_history="[]",
            state_json='{"objects": [{"id": 1}]}',
            plan_steps=5,
        )
        self.assertIn("ls20", rendered)
        self.assertIn('{"objects": [{"id": 1}]}', rendered)

        system = load_prompt("agent_system")
        self.assertIn('{"plan": [{"action": "ACTION1"}], "reason": "short hypothesis"}', system)

    def test_render_prompt_rejects_missing_placeholder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "custom.txt"
            p.write_text('{"literal": 1}\nHello {user_name}', encoding="utf-8")
            self.assertEqual(
                render_prompt("custom", prompts_dir=tmp, user_name="Can"),
                '{"literal": 1}\nHello Can',
            )
            with self.assertRaises(KeyError):
                render_prompt("custom", prompts_dir=tmp)

    def test_oracle_from_config_profiles(self) -> None:
        self.assertIsInstance(oracle_from_config("mock"), MockOracle)

        with patch("bsp3.oracle.load_dotenv"):
            with patch.dict(os.environ, {"QWEN_API_KEY": "q-key"}, clear=True):
                qwen = oracle_from_config("qwen", env_path="missing.env")
                self.assertIsInstance(qwen, OpenAICompatOracle)
                self.assertEqual(qwen.model, "qwen3.5-9b")

            with patch.dict(os.environ, {"NVIDIA_API_KEY": "nv-key"}, clear=True):
                minimax = oracle_from_config("minimax", env_path="missing.env")
                self.assertIsInstance(minimax, OpenAICompatOracle)
                self.assertEqual(minimax._extra_body, {"reasoning_effort": "low"})
                self.assertEqual(minimax._max_tokens, 8192)
                self.assertEqual(minimax._timeout_s, 300.0)

            with self.assertRaises(OracleError):
                oracle_from_config("does_not_exist", env_path="missing.env")


if __name__ == "__main__":
    unittest.main()
