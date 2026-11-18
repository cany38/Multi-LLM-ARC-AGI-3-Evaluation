from pathlib import Path
from types import SimpleNamespace
import unittest

from arcengine import GameAction, GameState
import numpy as np

from bsp3.games import GameError, Games


class _FakeEnv:
    def __init__(self) -> None:
        self.action_space = [GameAction.RESET, GameAction.ACTION1, GameAction.ACTION6]
        self.info = SimpleNamespace(baseline_actions=[10, 20])
        self._game = None
        self._step = 0

    def reset(self):
        self._step = 0
        grid = np.zeros((4, 4), dtype=np.int8)
        return SimpleNamespace(
            frame=[grid],
            levels_completed=0,
            win_levels=2,
            state=GameState.NOT_FINISHED,
            available_actions=[1, 6],
        )

    def step(self, action: GameAction, data=None):
        self._step += 1
        grid = np.zeros((4, 4), dtype=np.int8)
        grid[0, min(self._step, 3)] = 4
        return SimpleNamespace(
            frame=[grid],
            levels_completed=1 if action == GameAction.ACTION6 else 0,
            win_levels=2,
            state=GameState.NOT_FINISHED,
            available_actions=[1, 6],
        )


class _FakeArcade:
    def __init__(self, env: _FakeEnv | None = None) -> None:
        self._env = env

    def get_environments(self):
        return [SimpleNamespace(game_id="fake-001")]

    def make(self, game_id: str, seed: int = 0):
        return self._env if game_id == "fake-001" else None


class GamesTests(unittest.TestCase):
    def test_fake_reset_step_and_errors(self) -> None:
        games = Games(arcade=_FakeArcade(_FakeEnv()))
        self.assertEqual(games.list_games(), ["fake-001"])

        with self.assertRaises(GameError):
            games.step("ACTION1")
        with self.assertRaises(GameError):
            games.reset("unknown-game")

        obs0 = games.reset("fake-001", level=0, seed=3)
        self.assertEqual(obs0.level, 0)
        self.assertEqual(obs0.win_levels, 2)
        self.assertEqual(obs0.available_actions, ["ACTION1", "ACTION6"])
        self.assertEqual(games.baseline_for_level(0), 10)
        self.assertIsNone(games.baseline_for_level(9))

        with self.assertRaises(GameError):
            games.step("ACTION6")
        with self.assertRaises(GameError):
            games.step("BAD_ACTION")

        obs1 = games.step("ACTION1")
        self.assertEqual(int(obs1.frame[0, 1]), 4)

        obs2 = games.step("ACTION6", data={"x": 2, "y": 2})
        self.assertEqual(obs2.levels_completed, 1)
        self.assertEqual(obs2.level, 1)

    def test_offline_arc_game_if_downloaded(self) -> None:
        ls20_dir = Path("environment_files/ls20")
        if not ls20_dir.exists():
            self.skipTest("offline environment_files not downloaded")

        games = Games(offline=True)
        obs0 = games.reset("ls20-9607627b", level=0, seed=0)
        self.assertEqual(obs0.frame.shape, (64, 64))
        self.assertEqual(obs0.win_levels, 7)
        self.assertEqual(games.baseline_for_level(0), 22)

        obs_lvl1 = games.reset("ls20-9607627b", level=1, seed=0)
        self.assertEqual(obs_lvl1.level, 1)
        self.assertFalse(np.array_equal(obs0.frame, obs_lvl1.frame))


if __name__ == "__main__":
    unittest.main()
