import json
from pathlib import Path
import unittest

import numpy as np

from bsp3.games import Games
from bsp3.grounding import Grounding, describe, extract_objects


class GroundingTests(unittest.TestCase):
    def test_extracts_objects_and_connectivity(self) -> None:
        grid = np.zeros((5, 5), dtype=np.int8)
        grid[1, 1] = 3
        grid[2, 2] = 3

        objs_4, bg = extract_objects(grid, connectivity=4)
        self.assertEqual(bg, 0)
        self.assertEqual(len(objs_4), 2)

        objs_8, _ = extract_objects(grid, connectivity=8)
        self.assertEqual(len(objs_8), 1)
        self.assertEqual(objs_8[0]["area"], 2)
        self.assertEqual(objs_8[0]["bbox"], [1, 1, 2, 2])

    def test_diff_detects_moved_recolored_appeared_disappeared(self) -> None:
        prev = np.zeros((8, 8), dtype=np.int8)
        prev[1:3, 1:3] = 5  # will move right by 2
        prev[5, 5] = 8      # will recolor to 9
        prev[7, 0] = 2      # will disappear

        curr = np.zeros((8, 8), dtype=np.int8)
        curr[1:3, 3:5] = 5  # moved
        curr[5, 5] = 9      # recolored
        curr[0, 6] = 4      # appeared

        state = describe(curr, prev)
        json.dumps(state)

        self.assertEqual(state["shape"], [8, 8])
        self.assertEqual(state["background_color"], 0)
        diff = state["diff"]
        self.assertIsNotNone(diff)
        self.assertEqual(len(diff["moved"]), 1)
        self.assertEqual(diff["moved"][0]["dx"], 2.0)
        self.assertEqual(diff["moved"][0]["dy"], 0.0)
        self.assertEqual(len(diff["recolored"]), 1)
        self.assertEqual(diff["recolored"][0]["from_color"], 8)
        self.assertEqual(diff["recolored"][0]["to_color"], 9)
        self.assertEqual(len(diff["appeared"]), 1)
        self.assertEqual(diff["appeared"][0]["color"], 4)
        self.assertEqual(len(diff["disappeared"]), 1)
        self.assertEqual(diff["disappeared"][0]["color"], 2)
        self.assertGreater(diff["changed_pixels"], 0)

    def test_grounding_from_config_and_real_ls20_step(self) -> None:
        g = Grounding.from_config()
        self.assertEqual(g.connectivity, 4)
        self.assertEqual(g.max_objects, 64)

        if not Path("environment_files/ls20").exists():
            self.skipTest("offline environment_files not downloaded")
        games = Games(offline=True)
        obs0 = games.reset("ls20-9607627b", level=0, seed=0)
        obs1 = games.step("ACTION1")
        state = g.describe(obs1.frame, obs0.frame)
        encoded = json.dumps(state)
        self.assertIn("changed_pixels", encoded)
        self.assertGreater(len(state["objects"]), 0)
        self.assertGreater(state["diff"]["changed_pixels"], 0)


if __name__ == "__main__":
    unittest.main()
