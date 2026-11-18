import json
import tempfile
import unittest

from bsp3.logger import Logger
from bsp3.store import Store


class LoggerAndStoreTests(unittest.TestCase):
    def test_logger_writes_and_reads_steps_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = Logger.for_run(
                game_id="ls20-9607627b",
                level=0,
                condition="nothing",
                seed=7,
                runs_dir=tmp,
            )
            logger.record_step({"step": 1, "action": "ACTION1", "tokens": 50})
            logger.record_step({"step": 2, "action": "ACTION2", "tokens": 0})
            logger.record_run(
                {
                    "game_id": "ls20-9607627b",
                    "level": 0,
                    "condition": "nothing",
                    "seed": 7,
                    "agent_config": {"plan_steps": 5},
                    "exact_model": "mock",
                    "provider": "mock",
                    "model_settings": {"temperature": 0.0},
                    "actions_taken": 2,
                    "model_calls": 1,
                    "tokens": {"prompt_tokens": 30, "completion_tokens": 20, "total_tokens": 50},
                    "outcome": "won",
                    "won": True,
                }
            )

            steps = logger.read_steps()
            self.assertEqual(len(steps), 2)
            self.assertEqual(steps[0]["action"], "ACTION1")
            summary = logger.read_summary()
            self.assertIsNotNone(summary)
            self.assertEqual(summary["seed"], 7)
            self.assertEqual(summary["exact_model"], "mock")
            self.assertTrue(summary["won"])

    def test_store_adds_loads_and_exports_ttl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(tmp)
            self.assertEqual(store.load("chainA", condition="nothing"), [])
            self.assertEqual(store.load_text("chainA", condition="notes"), "")

            store.add("chainA", "Blue block moves on ACTION1.", condition="notes", game_id="ls20", level=0)
            store.add("chainA", "White cross swaps color.", condition="notes", game_id="ls20", level=1)
            notes_text = store.load_text("chainA", condition="notes")
            self.assertIn("Blue block moves", notes_text)
            self.assertIn("[ls20 level 1]", notes_text)

            ttl = "@prefix arc: <http://bsp3.local/arc#> .\narc:Player a arc:Agent ."
            path = store.add("chainA", ttl, condition="ontology", game_id="ls20", level=0)
            self.assertTrue(path.with_suffix(".ttl").exists())
            self.assertIn("arc:Player", store.load_text("chainA", condition="ontology"))

    def test_store_bounds_trajectory_steps_and_chars(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(tmp, max_trajectory_steps=2, max_trajectory_chars=120)
            steps = [{"step": i, "action": "ACTION1", "diff": {"changed_pixels": i}} for i in range(1, 6)]
            store.add("chainT", steps, condition="trajectory", game_id="ls20", level=0)
            text = store.load_text("chainT", condition="trajectory")
            self.assertLessEqual(len(text), 120)
            self.assertNotIn('"step": 1', text)
            self.assertIn('"step": 5', text)

    def test_store_trajectory_keeps_every_part_within_char_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(tmp, max_trajectory_chars=12000)
            big = [{"step": i, "action": "ACTION1", "note": "x" * 580} for i in range(1, 31)]
            small = [{"step": 1, "action": "ACTION2", "note": "small"}]
            store.add("chainBig", big, condition="trajectory", game_id="big", level=0)
            store.add("chainBig", small, condition="trajectory", game_id="small", level=1)

            text = store.load_text("chainBig", condition="trajectory")
            items = json.loads(text)
            self.assertLessEqual(len(text), 12000)
            self.assertEqual([item["game_id"] for item in items], ["big", "small"])
            self.assertEqual(items[1]["knowledge"], small)
            self.assertEqual(items[0]["knowledge"][-1]["step"], 30)
            self.assertLess(len(items[0]["knowledge"]), 30)

    def test_store_trajectory_under_budget_is_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(tmp, max_trajectory_chars=12000)
            steps = [{"step": 1, "action": "ACTION1"}, {"step": 2, "action": "ACTION2"}]
            store.add("chainSmall", steps, condition="trajectory", game_id="ls20", level=0)
            expected = json.dumps([{"game_id": "ls20", "level": 0, "knowledge": steps}], ensure_ascii=False)
            self.assertEqual(store.load_text("chainSmall", condition="trajectory"), expected)


if __name__ == "__main__":
    unittest.main()
