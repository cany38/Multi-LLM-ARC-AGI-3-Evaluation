from dataclasses import dataclass
from typing import Any

import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction
import numpy as np


class GameError(RuntimeError):
    """Raised when an ARC-AGI-3 environment cannot be opened or stepped."""


@dataclass(frozen=True, slots=True)
class GameObservation:
    game_id: str
    level: int
    frame: np.ndarray
    frames: list[np.ndarray]
    levels_completed: int
    win_levels: int
    state: str
    available_actions: list[str]


def _action_names(env: Any, obs: Any) -> list[str]:
    raw_ids = getattr(obs, "available_actions", None) or []
    if raw_ids:
        actions = [
            item if isinstance(item, GameAction) else GameAction.from_id(int(item))
            for item in raw_ids
        ]
        return [a.name for a in actions if a != GameAction.RESET]
    space = getattr(env, "action_space", ()) or ()
    return [a.name for a in space if a != GameAction.RESET]


def _to_action(action: str | GameAction) -> GameAction:
    if isinstance(action, GameAction):
        return action
    key = action.strip().upper()
    if key not in GameAction.__members__:
        raise GameError(f"Unknown action: {action!r}")
    return GameAction[key]


class Games:
    """Thin wrapper around arc_agi.Arcade with level selection support."""

    def __init__(
        self,
        *,
        offline: bool = True,
        arcade: Any = None,
    ) -> None:
        if arcade is not None:
            self._arcade = arcade
        else:
            mode = OperationMode.OFFLINE if offline else OperationMode.NORMAL
            self._arcade = arc_agi.Arcade(operation_mode=mode)
        self._env: Any = None
        self._game_id: str = ""
        self._start_level: int = 0

    def list_games(self) -> list[str]:
        return [info.game_id for info in self._arcade.get_environments()]

    def reset(self, game_id: str, level: int = 0, *, seed: int = 0) -> GameObservation:
        if level < 0:
            raise ValueError("level must be >= 0")
        env = self._arcade.make(game_id, seed=seed)
        if env is None:
            raise GameError(f"Could not open game {game_id!r}")
        obs = env.reset()
        if obs is None or not obs.frame:
            raise GameError(f"Empty reset frame for {game_id!r}")

        self._env = env
        self._game_id = game_id
        self._start_level = level

        if level > 0:
            if level >= obs.win_levels:
                raise ValueError(f"level {level} out of range (win_levels={obs.win_levels})")
            game = getattr(env, "_game", None)
            if game is None:
                raise GameError(f"{game_id!r} does not support starting at level {level}")
            # EnvironmentWrapper has no public level-select method; LocalEnvironmentWrapper exposes _game.
            game.set_level(level)
            frame = np.array(
                game.camera.render(game.current_level.get_sprites()),
                dtype=np.int8,
            )
            frames = [frame]
        else:
            frames = [np.array(f, dtype=np.int8) for f in obs.frame]

        return GameObservation(
            game_id=game_id,
            level=self._current_level(obs),
            frame=frames[-1],
            frames=frames,
            levels_completed=int(obs.levels_completed),
            win_levels=int(obs.win_levels),
            state=obs.state.name if hasattr(obs.state, "name") else str(obs.state),
            available_actions=_action_names(env, obs),
        )

    def step(
        self,
        action: str | GameAction,
        data: dict[str, int] | None = None,
    ) -> GameObservation:
        if self._env is None:
            raise GameError("Call reset() before step()")
        act = _to_action(action)
        if act.is_complex() and (not data or "x" not in data or "y" not in data):
            raise GameError(f"{act.name} requires x and y coordinates")

        obs = self._env.step(act, data=data)
        if obs is None or not obs.frame:
            raise GameError(f"Empty observation after {act.name} in {self._game_id!r}")

        frames = [np.array(f, dtype=np.int8) for f in obs.frame]
        return GameObservation(
            game_id=self._game_id,
            level=self._current_level(obs),
            frame=frames[-1],
            frames=frames,
            levels_completed=int(obs.levels_completed),
            win_levels=int(obs.win_levels),
            state=obs.state.name if hasattr(obs.state, "name") else str(obs.state),
            available_actions=_action_names(self._env, obs),
        )

    def baseline_for_level(self, level: int) -> int | None:
        if self._env is None:
            return None
        info = getattr(self._env, "info", None)
        baselines = getattr(info, "baseline_actions", None) or []
        if 0 <= level < len(baselines):
            return int(baselines[level])
        return None

    def _current_level(self, obs: Any) -> int:
        game = getattr(self._env, "_game", None)
        if game is not None and hasattr(game, "level_index"):
            return int(game.level_index)
        return self._start_level + int(obs.levels_completed)
