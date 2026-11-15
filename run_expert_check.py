"""Phase 0B: Bounded sanity check of BabyAIBot on BabyAI-GoToLocal-v0 (seeds 0..19)."""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import gymnasium as gym
from minigrid.utils.baby_ai_bot import BabyAIBot

SEEDS = list(range(20))
STUDENT_ALLOWED_ACTIONS = {0, 1, 2}  # left, right, forward


def run_episode(seed: int) -> dict:
    env = gym.make("BabyAI-GoToLocal-v0")
    unwrapped = env.unwrapped
    obs, _ = env.reset(seed=seed)
    bot = BabyAIBot(env)

    action_names = {int(a.value): a.name for a in unwrapped.actions}
    room = unwrapped.get_room(0, 0)
    object_count = len(room.objs)
    target_desc = f"{unwrapped.instrs.desc.color} {unwrapped.instrs.desc.type}"
    matching_targets = len(unwrapped.instrs.desc.obj_set)
    max_steps = int(unwrapped.max_steps)

    suggested_actions: list[int] = []
    executed_actions: list[int] = []
    unexpected_actions: list[dict] = []
    terminated = False
    truncated = False
    total_reward = 0.0
    exception_info = None
    prev_action = None

    try:
        while not (terminated or truncated):
            action_obj = bot.replan(prev_action)
            action = int(action_obj)
            suggested_actions.append(action)

            if action not in STUDENT_ALLOWED_ACTIONS:
                unexpected_actions.append(
                    {
                        "step": len(executed_actions),
                        "action_id": action,
                        "action_name": action_names.get(action, "unknown"),
                    }
                )

            obs, reward, terminated, truncated, _ = env.step(action)
            executed_actions.append(action)
            total_reward += float(reward)
            prev_action = action_obj
    except Exception as exc:  # noqa: BLE001 - intentionally recorded per protocol
        exception_info = f"{type(exc).__name__}: {exc}"
    finally:
        env.close()

    action_counts = Counter(action_names.get(a, str(a)) for a in executed_actions)
    return {
        "seed": seed,
        "mission": obs["mission"],
        "target_desc": target_desc,
        "matching_targets_count": matching_targets,
        "object_count": object_count,
        "max_steps": max_steps,
        "episode_length": len(executed_actions),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "success": bool(terminated and total_reward > 0.0 and exception_info is None),
        "total_reward": round(total_reward, 6),
        "exception": exception_info,
        "suggested_actions": suggested_actions,
        "executed_actions": executed_actions,
        "action_counts": dict(action_counts),
        "unexpected_actions": unexpected_actions,
    }


def main() -> None:
    t0 = time.perf_counter()
    episodes = [run_episode(seed) for seed in SEEDS]
    elapsed_sec = time.perf_counter() - t0

    agg_action_counts: Counter[str] = Counter()
    for ep in episodes:
        agg_action_counts.update(ep["action_counts"])

    summary = {
        "env_id": "BabyAI-GoToLocal-v0",
        "seed_rule": "fixed pre-registered development seeds 0..19",
        "num_episodes": len(episodes),
        "success_count": sum(1 for ep in episodes if ep["success"]),
        "timeout_count": sum(1 for ep in episodes if ep["truncated"]),
        "exception_count": sum(1 for ep in episodes if ep["exception"] is not None),
        "episodes_with_unexpected_actions": sum(
            1 for ep in episodes if len(ep["unexpected_actions"]) > 0
        ),
        "mean_episode_length": round(
            sum(ep["episode_length"] for ep in episodes) / len(episodes), 3
        ),
        "min_episode_length": min(ep["episode_length"] for ep in episodes),
        "max_episode_length": max(ep["episode_length"] for ep in episodes),
        "aggregate_action_counts": dict(agg_action_counts),
        "wall_clock_seconds": round(elapsed_sec, 4),
    }

    output = {"summary": summary, "episodes": episodes}
    out_path = Path("phase0b_expert_results.json")
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
