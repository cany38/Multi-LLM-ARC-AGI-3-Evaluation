import csv
from pathlib import Path

import arc_agi
import numpy as np
from arc_agi import OperationMode
from arc_agi.rendering import frame_to_rgb_array
from arcengine import GameAction
from PIL import Image

SURVEY_DIR = Path("data/survey")
FRAMES_DIR = SURVEY_DIR / "frames"
CSV_PATH = SURVEY_DIR / "games.csv"
REPLAY_STEPS = 8


def click_probes(frame: np.ndarray) -> list[dict[str, int]]:
    h, w = frame.shape
    values, counts = np.unique(frame, return_counts=True)
    bg = int(values[int(np.argmax(counts))])
    probes = [{"x": w // 2, "y": h // 2}]
    for color in values:
        if int(color) == bg:
            continue
        ys, xs = np.where(frame == color)
        mid = len(xs) // 2
        probes.append({"x": int(xs[mid]), "y": int(ys[mid])})
    return probes


def default_data(action: GameAction, frame: np.ndarray) -> dict[str, int] | None:
    if not action.is_complex():
        return None
    h, w = frame.shape
    return {"x": w // 2, "y": h // 2}


def action_changes_frame(
    arcade: arc_agi.Arcade, game_id: str, action: GameAction, base_frame: np.ndarray
) -> bool:
    probes = click_probes(base_frame) if action.is_complex() else [None]
    for data in probes:
        env = arcade.make(game_id, seed=0)
        if env is None:
            return False
        env.reset()
        obs = env.step(action, data=data)
        if obs is not None and any(not np.array_equal(f, base_frame) for f in obs.frame):
            return True
    return False


def check_repeatable(
    arcade: arc_agi.Arcade, game_id: str, actions: list[GameAction]
) -> bool:
    if not actions:
        return True
    seq = [actions[i % len(actions)] for i in range(REPLAY_STEPS)]
    runs: list[list[np.ndarray]] = []
    for _ in range(2):
        env = arcade.make(game_id, seed=0)
        if env is None:
            return False
        obs = env.reset()
        if obs is None or not obs.frame:
            return False
        frames = [obs.frame[-1]]
        for action in seq:
            obs = env.step(action, data=default_data(action, frames[-1]))
            if obs is None or not obs.frame:
                return False
            frames.append(obs.frame[-1])
        runs.append(frames)
    return all(np.array_equal(a, b) for a, b in zip(runs[0], runs[1]))


def check_start_at_level(
    arcade: arc_agi.Arcade,
    game_id: str,
    levels: int,
    base_frame: np.ndarray,
    actions: list[GameAction],
) -> bool:
    if levels <= 1:
        return False
    env = arcade.make(game_id, seed=0)
    if env is None or not hasattr(env, "_game") or env._game is None:
        return False
    # EnvironmentWrapper has no public level-select method; LocalEnvironmentWrapper exposes _game.
    env._game.set_level(1)
    frame = env._game.camera.render(env._game.current_level.get_sprites())
    if env._game.level_index != 1 or np.array_equal(frame, base_frame):
        return False
    if actions:
        obs = env.step(actions[0], data=default_data(actions[0], frame))
        return obs is not None and bool(obs.frame)
    return True


def run_survey() -> None:
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    arcade = arc_agi.Arcade()
    envs = arcade.get_environments()

    rows: list[dict[str, str | int | bool]] = []
    for info in envs:
        env = arcade.make(info.game_id, seed=0)
        if env is None:
            continue
        obs = env.reset()
        if obs is None or not obs.frame:
            continue

        base_frame = obs.frame[0]
        h, w = base_frame.shape
        colors = sorted(int(c) for c in np.unique(base_frame))
        actions = [a for a in env.action_space if a != GameAction.RESET]
        changing = [
            a.name
            for a in actions
            if action_changes_frame(arcade, info.game_id, a, base_frame)
        ]

        slug = info.game_id.split("-", 1)[0]
        rgb = frame_to_rgb_array(0, base_frame, scale=4)
        Image.fromarray(rgb).save(FRAMES_DIR / f"{slug}.png")

        rows.append(
            {
                "game_id": info.game_id,
                "title": info.title,
                "tags": ";".join(info.tags),
                "levels": obs.win_levels,
                "actions": ";".join(a.name for a in actions),
                "frame_size": f"{w}x{h}",
                "colors_used": ";".join(str(c) for c in colors),
                "baseline_actions": ";".join(str(x) for x in (info.baseline_actions or [])),
                "changing_actions": ";".join(changing),
                "can_start_at_level": check_start_at_level(
                    arcade, info.game_id, obs.win_levels, base_frame, actions
                ),
                "repeatable": check_repeatable(arcade, info.game_id, actions),
                "works_offline": False,
            }
        )
        print(f"{info.game_id}: levels={obs.win_levels}, actions={len(actions)}, changing={len(changing)}")

    if not rows:
        return

    # Scan environment_files after all downloads finish so newly fetched games are included.
    offline = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)
    for row in rows:
        env = offline.make(str(row["game_id"]), seed=0)
        if env is None:
            continue
        obs = env.reset()
        if obs is None or not obs.frame:
            continue
        first_action = str(row["actions"]).split(";")[0] if row["actions"] else ""
        if first_action:
            act = GameAction[first_action]
            step_obs = env.step(act, data=default_data(act, obs.frame[-1]))
            row["works_offline"] = step_obs is not None and bool(step_obs.frame)
        else:
            row["works_offline"] = True

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} games to {CSV_PATH}")


if __name__ == "__main__":
    run_survey()
