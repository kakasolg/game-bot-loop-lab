"""Phase 0A: Inspect exact MiniGrid v3.2.0 BabyAI-GoToLocal-v0 environment contract."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import gymnasium as gym
import minigrid
from minigrid.core.constants import (
    COLOR_TO_IDX,
    DIR_TO_VEC,
    IDX_TO_COLOR,
    IDX_TO_OBJECT,
    OBJECT_TO_IDX,
    STATE_TO_IDX,
)


def describe_wrappers(env: gym.Env) -> list[str]:
    chain = []
    cur = env
    while cur is not None:
        chain.append(type(cur).__name__)
        cur = getattr(cur, "env", None)
    return chain


def main() -> None:
    env = gym.make("BabyAI-GoToLocal-v0")
    unwrapped = env.unwrapped

    pre_reset = {
        "minigrid_version": minigrid.__version__,
        "gymnasium_version": gym.__version__,
        "minigrid_file": str(Path(minigrid.__file__).resolve()),
        "env_class": f"{type(unwrapped).__module__}.{type(unwrapped).__name__}",
        "wrapper_chain": describe_wrappers(env),
        "spec_id": env.spec.id if env.spec else None,
        "spec_max_episode_steps": env.spec.max_episode_steps if env.spec else None,
        "max_steps_before_reset": unwrapped.max_steps,
        "fixed_max_steps": unwrapped.fixed_max_steps,
        "room_size": unwrapped.room_size,
        "num_rows": unwrapped.num_rows,
        "num_cols": unwrapped.num_cols,
        "num_dists_param": unwrapped.num_dists,
        "width": unwrapped.width,
        "height": unwrapped.height,
        "agent_view_size": unwrapped.agent_view_size,
        "action_space": str(env.action_space),
        "actions": {a.name: int(a.value) for a in unwrapped.actions},
        "babyai_done_actions_env": os.environ.get("BABYAI_DONE_ACTIONS", None),
    }

    obs, info = env.reset(seed=0)
    room = unwrapped.get_room(0, 0)
    objs = [
        {
            "type": obj.type,
            "color": obj.color,
            "pos": [int(obj.cur_pos[0]), int(obj.cur_pos[1])],
        }
        for obj in room.objs
    ]
    type_color_counts = Counter((o["type"], o["color"]) for o in objs)

    # Verify egocentric observation coordinate convention empirically
    img = obs["image"]
    agent_view_cell = [int(x) for x in img[3, 6]]
    front_world = unwrapped.front_pos
    front_world_cell = unwrapped.grid.get(*front_world)
    front_obs_cell = [int(x) for x in img[3, 5]]

    # Check override behavior
    env_override = gym.make("BabyAI-GoToLocal-v0", max_steps=123)
    pre_override_max = env_override.unwrapped.max_steps
    env_override.reset(seed=0)
    post_override_max = env_override.unwrapped.max_steps
    env_override.close()

    # Check whether any duplicate (type, color) occurs across seeds 0..19
    dup_seeds = []
    for s in range(20):
        env.reset(seed=s)
        r = unwrapped.get_room(0, 0)
        c = Counter((o.type, o.color) for o in r.objs)
        if any(v > 1 for v in c.values()):
            dup_seeds.append(s)

    # Reset back to seed 0 to report canonical seed-0 state
    obs, info = env.reset(seed=0)
    post_reset_seed0 = {
        "max_steps_after_reset": unwrapped.max_steps,
        "step_count_after_reset": unwrapped.step_count,
        "mission": obs["mission"],
        "direction_value": int(obs["direction"]),
        "direction_vec": [int(x) for x in unwrapped.dir_vec],
        "agent_pos": [int(x) for x in unwrapped.agent_pos],
        "obs_keys": sorted(list(obs.keys())),
        "obs_image_shape": list(img.shape),
        "obs_image_dtype": str(img.dtype),
        "obs_agent_cell_3_6": agent_view_cell,
        "obs_front_cell_3_5": front_obs_cell,
        "world_front_cell": (
            f"{front_world_cell.color} {front_world_cell.type}"
            if front_world_cell is not None
            else "empty"
        ),
        "room_object_count_seed0": len(objs),
        "room_objects_seed0": objs,
        "duplicate_type_color_in_seed0": any(v > 1 for v in type_color_counts.values()),
        "seeds_with_duplicates_in_0_19": dup_seeds,
        "override_max_steps_before_reset": pre_override_max,
        "override_max_steps_after_reset": post_override_max,
        "dir_to_vec": {i: [int(v[0]), int(v[1])] for i, v in enumerate(DIR_TO_VEC)},
        "object_to_idx": OBJECT_TO_IDX,
        "idx_to_object": IDX_TO_OBJECT,
        "color_to_idx": COLOR_TO_IDX,
        "idx_to_color": IDX_TO_COLOR,
        "state_to_idx": STATE_TO_IDX,
    }
    env.close()

    report = {"pre_reset": pre_reset, "post_reset_seed0": post_reset_seed0}
    out_path = Path("phase0a_env_inspection.json")
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
