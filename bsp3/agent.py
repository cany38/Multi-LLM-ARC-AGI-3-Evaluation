from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any

from bsp3.config import load_experiment_config, load_prompt, render_prompt
from bsp3.games import GameObservation, Games
from bsp3.grounding import Grounding
from bsp3.knowledge import parse_json_payload
from bsp3.logger import Logger
from bsp3.oracle import Oracle, OracleError, OracleResponse, TokenUsage


@dataclass(slots=True)
class PlannedAction:
    action: str
    data: dict[str, int] | None = None


@dataclass(frozen=True, slots=True)
class RunResult:
    game_id: str
    level: int
    condition: str
    seed: int
    won: bool
    outcome: str
    actions_taken: int
    model_calls: int
    tokens: TokenUsage
    elapsed_s: float
    trajectory: list[dict[str, Any]]
    exact_model: str
    provider: str
    run_dir: Path | None = None


def _parse_plan(
    raw: str,
    available_actions: Sequence[str],
    frame_shape: Sequence[int],
) -> tuple[list[PlannedAction], str]:
    payload = parse_json_payload(raw)
    if isinstance(payload, dict):
        items = payload.get("plan") or []
        reason = str(payload.get("reason") or "")
    elif isinstance(payload, list):
        items = payload
        reason = ""
    else:
        raise ValueError(f"Unexpected plan JSON type: {type(payload).__name__}")

    h, w = int(frame_shape[0]), int(frame_shape[1])
    valid_set = set(available_actions)
    plan: list[PlannedAction] = []
    for item in items:
        if isinstance(item, str):
            name = item.strip().upper()
            raw_x, raw_y = None, None
        elif isinstance(item, dict):
            name = str(item.get("action") or "").strip().upper()
            raw_x, raw_y = item.get("x"), item.get("y")
        else:
            continue

        if name not in valid_set:
            continue
        if name == "ACTION6":
            x = max(0, min(w - 1, int(raw_x))) if raw_x is not None else w // 2
            y = max(0, min(h - 1, int(raw_y))) if raw_y is not None else h // 2
            plan.append(PlannedAction(action=name, data={"x": x, "y": y}))
        else:
            plan.append(PlannedAction(action=name))
    return plan, reason


def _knowledge_block(condition: str, knowledge: str) -> str:
    if not knowledge or condition == "nothing":
        return "Transferred knowledge: none."
    return f"Transferred knowledge ({condition}):\n{knowledge}"


def _won_level(obs: GameObservation, start_level: int, start_completed: int) -> bool:
    return (
        obs.state == "WIN"
        or obs.level > start_level
        or obs.levels_completed > start_completed
    )


class Agent:
    """Grounded multi-step planning agent operating behind the Oracle interface."""

    def __init__(
        self,
        oracle: Oracle,
        *,
        grounding: Grounding | None = None,
        prompts_dir: Path | str = "prompts",
        plan_steps: int = 5,
        budget_multiplier: float = 3.0,
        default_max_steps: int = 100,
        max_history_steps: int = 15,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> None:
        self._oracle = oracle
        self._grounding = grounding or Grounding()
        self._prompts_dir = prompts_dir
        self.plan_steps = plan_steps
        self.budget_multiplier = budget_multiplier
        self.default_max_steps = default_max_steps
        self.max_history_steps = max_history_steps
        self.temperature = temperature
        self.max_tokens = max_tokens

    @classmethod
    def from_config(
        cls,
        oracle: Oracle,
        config_path: Path | str = "configs/experiment.toml",
    ) -> "Agent":
        exp = load_experiment_config(config_path)
        acfg = exp.get("agent", {})
        prompts_dir = exp.get("paths", {}).get("prompts_dir", "prompts")
        return cls(
            oracle,
            grounding=Grounding.from_config(config_path),
            prompts_dir=prompts_dir,
            plan_steps=int(acfg.get("plan_steps", 5)),
            budget_multiplier=float(acfg.get("budget_multiplier", 3.0)),
            default_max_steps=int(acfg.get("default_max_steps", 100)),
            max_history_steps=int(acfg.get("max_history_steps", 15)),
            temperature=float(acfg.get("temperature", 0.0)),
            max_tokens=int(acfg.get("max_tokens", 1024)),
        )

    def _ask_plan(
        self,
        *,
        game_id: str,
        level: int,
        budget_remaining: int,
        obs: GameObservation,
        state: dict[str, Any],
        condition: str,
        knowledge: str,
        trajectory: list[dict[str, Any]],
        logger: Logger | None = None,
    ) -> tuple[list[PlannedAction], str, OracleResponse]:
        system = load_prompt("agent_system", prompts_dir=self._prompts_dir)
        recent = [
            {
                "step": item["step"],
                "action": item["action"],
                "diff": item["state"].get("diff"),
            }
            for item in trajectory[-self.max_history_steps :]
        ]
        prompt = render_prompt(
            "agent_step",
            prompts_dir=self._prompts_dir,
            game_id=game_id,
            level=level,
            budget_remaining=budget_remaining,
            actions=", ".join(obs.available_actions),
            knowledge_section=_knowledge_block(condition, knowledge),
            recent_history=json.dumps(recent, ensure_ascii=False),
            state_json=json.dumps(state, ensure_ascii=False),
            plan_steps=self.plan_steps,
        )
        response: OracleResponse | None = None
        try:
            response = self._oracle.query(
                prompt,
                system=system,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            plan, reason = _parse_plan(
                response.text, obs.available_actions, state["shape"]
            )
        except OracleError as exc:
            if "empty text" not in str(exc):
                raise
            plan, reason = [], f"fallback after empty response ({exc})"
        except (ValueError, json.JSONDecodeError, KeyError):
            plan, reason = [], ""

        if not plan and obs.available_actions:
            if logger is not None:
                logger.record_event(
                    {
                        "event": "agent_fallback",
                        "step": len(trajectory) + 1,
                        "reason": reason or "invalid_plan_json",
                    }
                )
            fallback_action = self._oracle.choose(
                f"Pick one action for step {len(trajectory) + 1}:",
                obs.available_actions,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            response = self._oracle.last_response or response
            if fallback_action == "ACTION6":
                h, w = state["shape"]
                plan = [PlannedAction("ACTION6", {"x": w // 2, "y": h // 2})]
            else:
                plan = [PlannedAction(fallback_action)]
        assert response is not None
        return plan[: self.plan_steps], reason, response

    def play(
        self,
        games: Games,
        game_id: str,
        *,
        level: int = 0,
        knowledge: str = "",
        condition: str = "nothing",
        budget: int | None = None,
        seed: int = 0,
        logger: Logger | None = None,
    ) -> RunResult:
        t0 = time.monotonic()
        obs = games.reset(game_id, level=level, seed=seed)
        baseline = games.baseline_for_level(level)
        action_budget = (
            budget
            if budget is not None
            else (
                max(1, int(round(baseline * self.budget_multiplier)))
                if baseline is not None
                else self.default_max_steps
            )
        )

        start_level = obs.level
        start_completed = obs.levels_completed
        prev_frame = None
        state = self._grounding.describe(obs.frame, prev_frame)

        trajectory: list[dict[str, Any]] = []
        queue: deque[PlannedAction] = deque()
        plan_reason = ""
        model_calls = 0
        prompt_toks = completion_toks = thoughts_toks = total_toks = 0
        last_model = self._oracle.model
        last_provider = self._oracle.provider
        last_settings: dict[str, Any] = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        won = False
        outcome = "budget_exhausted"
        prev_hook = self._oracle.on_event
        if logger is not None:
            self._oracle.on_event = logger.record_event
            logger.record_event(
                {
                    "event": "run_start",
                    "game_id": game_id,
                    "level": level,
                    "condition": condition,
                    "seed": seed,
                    "budget": action_budget,
                    "provider": last_provider,
                    "model": last_model,
                }
            )
            logger.record_frame(
                {
                    "status": "running",
                    "game_id": game_id,
                    "level": level,
                    "condition": condition,
                    "seed": seed,
                    "step": 0,
                    "budget": action_budget,
                    "provider": last_provider,
                    "model": last_model,
                    "game_state": obs.state,
                    "available_actions": list(obs.available_actions),
                    "grid": obs.frame.tolist(),
                    "model_calls": 0,
                    "tokens": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "thoughts_tokens": 0,
                        "total_tokens": 0,
                    },
                }
            )

        try:
            for step_idx in range(1, action_budget + 1):
                model_call_info: dict[str, Any] | None = None
                if not queue:
                    planned, plan_reason, resp = self._ask_plan(
                        game_id=game_id,
                        level=level,
                        budget_remaining=action_budget - step_idx + 1,
                        obs=obs,
                        state=state,
                        condition=condition,
                        knowledge=knowledge,
                        trajectory=trajectory,
                        logger=logger,
                    )
                    queue.extend(planned)
                    model_calls += 1
                    prompt_toks += resp.usage.prompt_tokens
                    completion_toks += resp.usage.completion_tokens
                    thoughts_toks += resp.usage.thoughts_tokens
                    total_toks += resp.usage.total_tokens
                    last_model = resp.model
                    last_provider = resp.provider
                    last_settings = dict(resp.settings)
                    if logger is not None:
                        logger.record_event(
                            {
                                "event": "agent_plan",
                                "step": step_idx,
                                "provider": resp.provider,
                                "model": resp.model,
                                "actions": [p.action for p in planned],
                                "reason": plan_reason,
                            }
                        )
                    model_call_info = {
                        "provider": resp.provider,
                        "model": resp.model,
                        "settings": resp.settings,
                        "usage": {
                            "prompt_tokens": resp.usage.prompt_tokens,
                            "completion_tokens": resp.usage.completion_tokens,
                            "thoughts_tokens": resp.usage.thoughts_tokens,
                            "total_tokens": resp.usage.total_tokens,
                        },
                        "latency_ms": round(resp.latency_ms, 1),
                        "plan": [{"action": p.action, "data": p.data} for p in planned],
                        "reason": plan_reason,
                    }

                step_act = queue.popleft()
                prev_frame = obs.frame
                obs = games.step(step_act.action, data=step_act.data)
                state = self._grounding.describe(obs.frame, prev_frame)

                entry = {
                    "step": step_idx,
                    "action": step_act.action,
                    "action_data": step_act.data,
                    "state": state,
                    "game_state": obs.state,
                    "model_call": model_call_info,
                }
                trajectory.append(entry)
                if logger is not None:
                    diff = state["diff"] or {}
                    logger.record_step(entry)
                    logger.record_event(
                        {
                            "event": "step",
                            "step": step_idx,
                            "budget": action_budget,
                            "action": step_act.action,
                            "action_data": step_act.data,
                            "changed_pixels": diff.get("changed_pixels"),
                            "game_state": obs.state,
                        }
                    )
                    logger.record_frame(
                        {
                            "step": step_idx,
                            "budget": action_budget,
                            "action": step_act.action,
                            "action_data": step_act.data,
                            "reason": plan_reason,
                            "changed_pixels": diff.get("changed_pixels"),
                            "game_state": obs.state,
                            "provider": last_provider,
                            "model": last_model,
                            "model_calls": model_calls,
                            "tokens": {
                                "prompt_tokens": prompt_toks,
                                "completion_tokens": completion_toks,
                                "thoughts_tokens": thoughts_toks,
                                "total_tokens": total_toks,
                            },
                            "grid": obs.frame.tolist(),
                        }
                    )

                if _won_level(obs, start_level, start_completed):
                    won = True
                    outcome = "won"
                    break
                if obs.state == "GAME_OVER":
                    won = False
                    outcome = "game_over"
                    break
                # Stop multi-step replay and replan immediately if the action had no effect.
                if state["diff"] and state["diff"]["changed_pixels"] == 0 and queue:
                    queue.clear()
                    if logger is not None:
                        logger.record_event({"event": "noop_replan", "step": step_idx})
        except Exception as exc:
            if logger is not None:
                logger.record_event({"event": "error", "error": str(exc)})
                logger.record_frame({"status": "error", "error": str(exc)})
            raise
        finally:
            if logger is not None:
                self._oracle.on_event = prev_hook

        elapsed_s = time.monotonic() - t0
        usage = TokenUsage(
            prompt_tokens=prompt_toks,
            completion_tokens=completion_toks,
            thoughts_tokens=thoughts_toks,
            total_tokens=total_toks,
        )
        agent_config = {
            "plan_steps": self.plan_steps,
            "budget_multiplier": self.budget_multiplier,
            "action_budget": action_budget,
            "connectivity": self._grounding.connectivity,
        }
        if logger is not None:
            logger.record_run(
                {
                    "game_id": game_id,
                    "level": level,
                    "condition": condition,
                    "seed": seed,
                    "agent_config": agent_config,
                    "exact_model": last_model,
                    "provider": last_provider,
                    "model_settings": last_settings,
                    "actions_taken": len(trajectory),
                    "model_calls": model_calls,
                    "tokens": {
                        "prompt_tokens": usage.prompt_tokens,
                        "completion_tokens": usage.completion_tokens,
                        "thoughts_tokens": usage.thoughts_tokens,
                        "total_tokens": usage.total_tokens,
                    },
                    "elapsed_s": round(elapsed_s, 3),
                    "outcome": outcome,
                    "won": won,
                }
            )
            logger.record_event(
                {
                    "event": "run_end",
                    "outcome": outcome,
                    "won": won,
                    "actions_taken": len(trajectory),
                    "elapsed_s": round(elapsed_s, 3),
                    "total_tokens": total_toks,
                }
            )
            logger.record_frame(
                {
                    "status": "finished",
                    "outcome": outcome,
                    "won": won,
                    "elapsed_s": round(elapsed_s, 3),
                }
            )

        return RunResult(
            game_id=game_id,
            level=level,
            condition=condition,
            seed=seed,
            won=won,
            outcome=outcome,
            actions_taken=len(trajectory),
            model_calls=model_calls,
            tokens=usage,
            elapsed_s=elapsed_s,
            trajectory=trajectory,
            exact_model=last_model,
            provider=last_provider,
            run_dir=logger.run_dir if logger is not None else None,
        )
