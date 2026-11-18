import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bsp3.cli import main, parse_args
from bsp3.oracle import TokenUsage


class CliAndWorkflowTests(unittest.TestCase):
    def test_parse_args_and_main_output(self) -> None:
        args = parse_args(["--game", "ls20-9607627b", "--level", "1", "--condition", "notes"])
        self.assertEqual(args.game, "ls20-9607627b")
        self.assertEqual(args.level, 1)
        self.assertEqual(args.condition, "notes")
        self.assertEqual(args.model, "minimax")

        fake_result = SimpleNamespace(
            game_id="toy-1",
            level=0,
            condition="nothing",
            won=True,
            outcome="won",
            actions_taken=2,
            model_calls=1,
            tokens=TokenUsage(total_tokens=42),
            provider="mock",
            exact_model="mock",
            run_dir=Path("runs/test_run"),
        )
        buf = io.StringIO()
        with patch("bsp3.cli.Runner.from_config") as mock_from_cfg:
            mock_from_cfg.return_value.run_task.return_value = fake_result
            with patch("sys.stdout", buf):
                code = main(["--game", "toy-1", "--model", "mock", "--no-extract"])
        self.assertEqual(code, 0)
        parsed = json.loads(buf.getvalue())
        self.assertEqual(parsed["outcome"], "won")
        self.assertEqual(parsed["total_tokens"], 42)

    def test_ci_workflow_exists_with_matrix_and_coverage(self) -> None:
        wf = Path(".github/workflows/test.yml").read_text(encoding="utf-8")
        self.assertIn("ubuntu-latest", wf)
        self.assertIn("windows-latest", wf)
        self.assertIn("--cov=bsp3", wf)


if __name__ == "__main__":
    unittest.main()
