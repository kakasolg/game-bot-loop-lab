"""Phase 2: Read-only diagnostic audit of the 7 failing dev episodes (3, 15, 29, 43, 50, 80, 94)."""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import gymnasium as gym
import numpy as np
from minigrid.core.constants import IDX_TO_COLOR, IDX_TO_OBJECT, OBJECT_TO_IDX
from minigrid.utils.baby_ai_bot import BabyAIBot

from run_phase1 import ACTION_NAMES, encode_observation, forward_mlp, hash_bytes

FAILED_SEEDS = [3, 15, 29, 43, 50, 80, 94]
PROB_TOL = 1e-6
WALL_CLOCK_CAP_SEC = 300.0
MAX_STEPS_PER_EP = 64


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def decode_cell(cell_arr: np.ndarray) -> dict:
    obj_idx, color_idx, state_idx = int(cell_arr[0]), int(cell_arr[1]), int(cell_arr[2])
    obj_name = IDX_TO_OBJECT.get(obj_idx, f"unknown_{obj_idx}")
    color_name = IDX_TO_COLOR.get(color_idx, f"unknown_{color_idx}")
    return {
        "object": obj_name,
        "color": color_name if obj_idx not in (0, 1, 2) else None,
        "state": state_idx,
        "raw": [obj_idx, color_idx, state_idx],
    }


def summarize_obs_grid(img: np.ndarray, target_color: str, target_type: str) -> dict:
    """Compact deterministic summary of visible non-empty cells and target locations."""
    target_locs = []
    non_empty = []
    unseen_count = 0
    for vx in range(7):
        for vy in range(7):
            c = decode_cell(img[vx, vy])
            if c["object"] == "unseen":
                unseen_count += 1
            elif c["object"] != "empty":
                entry = {"vx": vx, "vy": vy, "object": c["object"]}
                if c["color"] is not None:
                    entry["color"] = c["color"]
                non_empty.append(entry)
                if c["object"] == target_type and c["color"] == target_color:
                    target_locs.append([vx, vy])
    return {
        "unseen_cell_count": unseen_count,
        "visible_target_egocentric_coords": target_locs,
        "front_cell_3_5": decode_cell(img[3, 5]),
        "visible_non_empty_cells": non_empty,
    }


def replay_without_bot(
    seed: int,
    params: dict[str, np.ndarray],
    saved_ep: dict,
    deadline: float,
) -> dict:
    env = gym.make("BabyAI-GoToLocal-v0")
    obs, _ = env.reset(seed=seed)
    terminated = False
    truncated = False
    total_reward = 0.0
    steps = []
    divergence = None

    try:
        while not (terminated or truncated):
            if time.monotonic() >= deadline:
                raise TimeoutError("Wall-clock cap reached during baseline replay")
            if len(steps) >= MAX_STEPS_PER_EP:
                raise RuntimeError(f"Exceeded {MAX_STEPS_PER_EP} steps in seed={seed}")

            t = len(steps)
            feat, meta = encode_observation(obs)
            feat_hash = hash_bytes(feat.tobytes())
            raw_hash = hash_bytes(obs["image"].tobytes() + obs["mission"].encode("utf-8"))
            pos = [int(env.unwrapped.agent_pos[0]), int(env.unwrapped.agent_pos[1])]
            heading = int(env.unwrapped.agent_dir)

            _, _, probs = forward_mlp(feat[None, :], params)
            probs_1d = probs[0]
            action = int(np.argmax(probs_1d))

            # Verify exact match to saved Phase 1 rollout step
            if t >= len(saved_ep["steps"]):
                divergence = {"step": t, "reason": "step_count_exceeded_saved"}
                break
            ref = saved_ep["steps"][t]
            checks = [
                ("feat_hash", feat_hash, ref["feat_hash"]),
                ("raw_obs_hash", raw_hash, ref["raw_obs_hash"]),
                ("chosen_action", action, ref["chosen_action"]),
                ("agent_pos", pos, ref["agent_pos"]),
                ("agent_dir", heading, ref["agent_dir"]),
                ("front_blocked", meta["front_blocked"], ref["front_blocked"]),
                ("target_in_view", meta["target_in_view"], ref["target_in_view"]),
            ]
            for field, val, exp in checks:
                if val != exp:
                    divergence = {"step": t, "field": field, "got": val, "expected": exp}
                    break
            if divergence is None:
                max_prob_diff = float(
                    np.max(np.abs(probs_1d - np.array(ref["probs"], dtype=np.float64)))
                )
                if max_prob_diff > PROB_TOL:
                    divergence = {
                        "step": t,
                        "field": "probs",
                        "max_diff": max_prob_diff,
                        "got": [float(x) for x in probs_1d],
                        "expected": ref["probs"],
                    }
            if divergence is not None:
                break

            parts = obs["mission"].split()
            target_color, target_type = parts[-2], parts[-1]
            obs_summary = summarize_obs_grid(obs["image"], target_color, target_type)

            # World front cell check (for blocked-forward verification)
            world_front = env.unwrapped.grid.get(*env.unwrapped.front_pos)
            world_front_info = (
                {
                    "type": world_front.type,
                    "color": world_front.color,
                    "can_overlap": bool(world_front.can_overlap()),
                }
                if world_front is not None
                else None
            )

            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_pos = [int(env.unwrapped.agent_pos[0]), int(env.unwrapped.agent_pos[1])]
            next_heading = int(env.unwrapped.agent_dir)

            if next_pos != pos:
                transition_effect = "changed_position"
            elif next_heading != heading:
                transition_effect = "changed_heading"
            else:
                transition_effect = "neither"

            steps.append(
                {
                    "step": t,
                    "mission": obs["mission"],
                    "agent_pos": pos,
                    "agent_dir": heading,
                    "feat_vec": feat,
                    "feat_hash": feat_hash,
                    "raw_obs_hash": raw_hash,
                    "front_blocked": meta["front_blocked"],
                    "target_in_view": meta["target_in_view"],
                    "obs_summary": obs_summary,
                    "world_front_cell": world_front_info,
                    "probs": [round(float(x), 6) for x in probs_1d],
                    "chosen_action": action,
                    "chosen_action_name": ACTION_NAMES[action],
                    "transition_effect": transition_effect,
                    "next_pos": next_pos,
                    "next_dir": next_heading,
                }
            )
            obs = next_obs
            total_reward += float(reward)
    finally:
        env.close()

    outcome_match = (
        divergence is None
        and len(steps) == saved_ep["episode_length"]
        and bool(terminated) == saved_ep["terminated"]
        and bool(truncated) == saved_ep["truncated"]
        and bool(terminated and total_reward > 0.0) == saved_ep["success"]
    )
    return {
        "seed": seed,
        "mission": obs["mission"],
        "matched_saved_baseline": outcome_match,
        "divergence": divergence,
        "episode_length": len(steps),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "success": bool(terminated and total_reward > 0.0),
        "steps": steps,
    }


def extract_cycle_anatomy(replay: dict) -> dict:
    """Identify first cycle entry, first recurrence, and every distinct state in the cycle."""
    steps = replay["steps"]
    seen_poses: dict[tuple[int, int, int], int] = {}
    start_step = None
    repeat_step = None
    for s in steps:
        pose = (s["agent_pos"][0], s["agent_pos"][1], s["agent_dir"])
        if pose in seen_poses:
            start_step = seen_poses[pose]
            repeat_step = s["step"]
            break
        seen_poses[pose] = s["step"]

    assert start_step is not None and repeat_step is not None
    cycle_len = repeat_step - start_step
    cycle_steps = steps[start_step:repeat_step]

    # Verify periodic recurrence through the rest of the episode
    periodic_exact = True
    recurrence_counts = Counter()
    for s in steps[start_step:]:
        offset = (s["step"] - start_step) % cycle_len
        template = cycle_steps[offset]
        if (
            s["agent_pos"] != template["agent_pos"]
            or s["agent_dir"] != template["agent_dir"]
            or s["feat_hash"] != template["feat_hash"]
            or s["chosen_action"] != template["chosen_action"]
            or s["probs"] != template["probs"]
        ):
            periodic_exact = False
        recurrence_counts[offset] += 1

    distinct_states = []
    for idx, s in enumerate(cycle_steps):
        distinct_states.append(
            {
                "cycle_offset": idx,
                "first_entry_step": s["step"],
                "visits_in_episode": recurrence_counts[idx],
                "agent_pos": s["agent_pos"],
                "agent_dir": s["agent_dir"],
                "mission": s["mission"],
                "feat_hash": s["feat_hash"],
                "raw_obs_hash": s["raw_obs_hash"],
                "visible_target_egocentric_coords": s["obs_summary"][
                    "visible_target_egocentric_coords"
                ],
                "front_cell_3_5": s["obs_summary"]["front_cell_3_5"],
                "world_front_cell": s["world_front_cell"],
                "visible_non_empty_cells": s["obs_summary"]["visible_non_empty_cells"],
                "unseen_cell_count": s["obs_summary"]["unseen_cell_count"],
                "front_blocked": s["front_blocked"],
                "target_in_view": s["target_in_view"],
                "probs": s["probs"],
                "chosen_action": s["chosen_action"],
                "chosen_action_name": s["chosen_action_name"],
                "transition_effect": s["transition_effect"],
                "feat_vec": s["feat_vec"],
            }
        )

    pre_cycle_prefix = [
        {
            "step": s["step"],
            "agent_pos": s["agent_pos"],
            "agent_dir": s["agent_dir"],
            "front_blocked": s["front_blocked"],
            "target_in_view": s["target_in_view"],
            "probs": s["probs"],
            "chosen_action_name": s["chosen_action_name"],
            "transition_effect": s["transition_effect"],
        }
        for s in steps[:start_step]
    ]

    return {
        "seed": replay["seed"],
        "mission": replay["mission"],
        "first_entry_step": start_step,
        "first_complete_recurrence_step": repeat_step,
        "cycle_length": cycle_len,
        "periodic_exact_through_timeout": periodic_exact,
        "pre_cycle_prefix": pre_cycle_prefix,
        "distinct_cycle_states": distinct_states,
    }


def compute_dataset_support(
    feat_vec: np.ndarray,
    student_action: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> dict:
    tr_mask = np.all(X_train == feat_vec, axis=1)
    val_mask = np.all(X_val == feat_vec, axis=1)

    tr_counts = Counter(ACTION_NAMES[int(a)] for a in y_train[tr_mask])
    val_counts = Counter(ACTION_NAMES[int(a)] for a in y_val[val_mask])
    student_name = ACTION_NAMES[student_action]

    return {
        "train_exact_matches": int(np.sum(tr_mask)),
        "train_action_counts": {k: tr_counts.get(k, 0) for k in ACTION_NAMES},
        "student_action_has_exact_train_support": tr_counts.get(student_name, 0) > 0,
        "val_exact_matches": int(np.sum(val_mask)),
        "val_action_counts": {k: val_counts.get(k, 0) for k in ACTION_NAMES},
    }


def summarize_blocked_front_split(
    X: np.ndarray,
    y: np.ndarray,
    front_blocked: np.ndarray,
    target_in_view: np.ndarray,
) -> dict:
    blk_mask = front_blocked
    total_blk = int(np.sum(blk_mask))
    blk_actions = Counter(ACTION_NAMES[int(a)] for a in y[blk_mask])

    wall_idx = 2  # channel 2 = wall
    non_tgt_idx = 4  # channel 4 = non_target_object
    front_wall_mask = blk_mask & (X[:, (3 * 7 + 5) * 5 + wall_idx] == 1.0)
    front_obj_mask = blk_mask & (X[:, (3 * 7 + 5) * 5 + non_tgt_idx] == 1.0)
    blk_vis_mask = blk_mask & target_in_view
    blk_hid_mask = blk_mask & (~target_in_view)

    def action_dict(mask: np.ndarray) -> dict:
        c = Counter(ACTION_NAMES[int(a)] for a in y[mask])
        return {"count": int(np.sum(mask)), "actions": {k: c.get(k, 0) for k in ACTION_NAMES}}

    return {
        "total_blocked_front_samples": total_blk,
        "demonstrated_action_counts": {k: blk_actions.get(k, 0) for k in ACTION_NAMES},
        "by_front_obstacle_type": {
            "wall": action_dict(front_wall_mask),
            "non_target_object": action_dict(front_obj_mask),
        },
        "by_target_visibility": {
            "target_in_view": action_dict(blk_vis_mask),
            "target_hidden": action_dict(blk_hid_mask),
        },
    }


def run_advisory_replay(
    seed: int,
    params: dict[str, np.ndarray],
    baseline_replay: dict,
    deadline: float,
) -> dict:
    env = gym.make("BabyAI-GoToLocal-v0")
    obs, _ = env.reset(seed=seed)
    bot = BabyAIBot(env)
    all_action_names = {int(a.value): a.name for a in env.unwrapped.actions}

    terminated = False
    truncated = False
    prev_student_action = None
    trajectory_identical = True
    divergence = None
    advisory_steps = []
    expert_exceptions = []
    expert_non_student_actions = []

    try:
        while not (terminated or truncated):
            if time.monotonic() >= deadline:
                raise TimeoutError("Wall-clock cap reached during advisory replay")

            t = len(advisory_steps)
            feat, meta = encode_observation(obs)
            feat_hash = hash_bytes(feat.tobytes())
            pos = [int(env.unwrapped.agent_pos[0]), int(env.unwrapped.agent_pos[1])]
            heading = int(env.unwrapped.agent_dir)

            _, _, probs = forward_mlp(feat[None, :], params)
            probs_1d = probs[0]
            student_action = int(np.argmax(probs_1d))

            # Verify identical student step against baseline replay
            ref = baseline_replay["steps"][t]
            if (
                pos != ref["agent_pos"]
                or heading != ref["agent_dir"]
                or feat_hash != ref["feat_hash"]
                or student_action != ref["chosen_action"]
            ):
                trajectory_identical = False
                divergence = {"step": t, "pos": pos, "ref_pos": ref["agent_pos"]}
                break

            expert_action = None
            expert_name = None
            expert_exc = None
            try:
                expert_action_obj = bot.replan(prev_student_action)
                expert_action = int(expert_action_obj)
                expert_name = all_action_names.get(expert_action, f"unknown_{expert_action}")
                if expert_action not in (0, 1, 2):
                    expert_non_student_actions.append(
                        {"step": t, "action_id": expert_action, "action_name": expert_name}
                    )
            except Exception as exc:  # noqa: BLE001
                expert_exc = f"{type(exc).__name__}: {exc}"
                expert_exceptions.append({"step": t, "error": expert_exc})

            advisory_steps.append(
                {
                    "step": t,
                    "feat_hash": feat_hash,
                    "agent_pos": pos,
                    "agent_dir": heading,
                    "student_action": student_action,
                    "student_action_name": ACTION_NAMES[student_action],
                    "student_probs": [round(float(x), 6) for x in probs_1d],
                    "expert_action": expert_action,
                    "expert_action_name": expert_name,
                    "agrees": (expert_action == student_action)
                    if expert_action is not None
                    else None,
                    "bot_stack_repr": [repr(sg) for sg in bot.stack],
                    "bot_vis_mask_true_cells": int(np.sum(bot.vis_mask)),
                    "expert_exception": expert_exc,
                }
            )

            obs, reward, terminated, truncated, _ = env.step(student_action)
            prev_student_action = student_action
    finally:
        env.close()

    if len(advisory_steps) != baseline_replay["episode_length"]:
        trajectory_identical = False

    # Check whether expert recommendation changed across repeated visits to identical student feat_hash
    by_feat: dict[str, list[dict]] = {}
    for s in advisory_steps:
        by_feat.setdefault(s["feat_hash"], []).append(s)
    changed_across_repeats = []
    for fh, visits in by_feat.items():
        if len(visits) > 1:
            recs = [v["expert_action_name"] for v in visits]
            if len(set(recs)) > 1:
                changed_across_repeats.append(
                    {
                        "feat_hash": fh,
                        "visits": [
                            {"step": v["step"], "expert_action": v["expert_action_name"]}
                            for v in visits
                        ],
                    }
                )

    return {
        "seed": seed,
        "trajectory_identical_to_baseline": trajectory_identical,
        "divergence": divergence,
        "expert_exceptions": expert_exceptions,
        "expert_non_student_actions": expert_non_student_actions,
        "expert_changed_across_repeated_student_inputs": changed_across_repeats,
        "advisory_steps": advisory_steps,
    }


def main() -> None:
    t0 = time.monotonic()
    deadline = t0 + WALL_CLOCK_CAP_SEC

    weights_path = Path("phase1_weights.npz")
    weights_sha256 = sha256_file(weights_path)
    weights_npz = np.load(weights_path)
    params = {k: weights_npz[k] for k in weights_npz.files}

    cfg = json.loads(Path("phase1_config.json").read_text(encoding="utf-8"))
    amendment = json.loads(
        Path("phase1_config_amendment.json").read_text(encoding="utf-8")
    )
    train_meta = json.loads(Path("phase1_training.json").read_text(encoding="utf-8"))
    saved_eval = json.loads(Path("phase1_eval_results.json").read_text(encoding="utf-8"))
    saved_by_seed = {ep["seed"]: ep for ep in saved_eval["episodes"]}

    demos = np.load("phase1_demos_filtered.npz")
    X_train, y_train = demos["X_train"], demos["y_train"]
    fb_train, tv_train = demos["front_blocked_train"], demos["target_in_view_train"]
    X_val, y_val = demos["X_val"], demos["y_val"]
    fb_val, tv_val = demos["front_blocked_val"], demos["target_in_view_val"]

    # 1. Baseline replay verification (without advisory bot)
    replays = {}
    all_replays_matched = True
    for seed in FAILED_SEEDS:
        rep = replay_without_bot(seed, params, saved_by_seed[seed], deadline)
        replays[seed] = rep
        if not rep["matched_saved_baseline"]:
            all_replays_matched = False
            break

    if not all_replays_matched:
        out = {
            "status": "halted_baseline_replay_mismatch",
            "weights_sha256": weights_sha256,
            "replays": {
                s: {k: v for k, v in r.items() if k != "steps"}
                for s, r in replays.items()
            },
        }
        Path("phase2_audit_results.json").write_text(
            json.dumps(out, indent=2), encoding="utf-8"
        )
        print(json.dumps(out, indent=2))
        return

    # 2. Question A & Question B: Cycle anatomy + Dataset support + Blocked-forward checks
    cycles = {}
    blocked_forward_checks = {}

    for seed in FAILED_SEEDS:
        anatomy = extract_cycle_anatomy(replays[seed])
        for st in anatomy["distinct_cycle_states"]:
            fv = st.pop("feat_vec")
            st["dataset_support"] = compute_dataset_support(
                fv, st["chosen_action"], X_train, y_train, X_val, y_val
            )
        cycles[str(seed)] = anatomy

        if seed in (3, 50):
            rep_steps = replays[seed]["steps"]
            c_step = anatomy["distinct_cycle_states"][0]["first_entry_step"]
            s0 = rep_steps[c_step]
            recur_steps = rep_steps[c_step:]
            all_unchanged_state = all(
                rs["transition_effect"] == "neither"
                and rs["agent_pos"] == s0["agent_pos"]
                and rs["agent_dir"] == s0["agent_dir"]
                and rs["feat_hash"] == s0["feat_hash"]
                and rs["probs"] == s0["probs"]
                and rs["chosen_action"] == 2
                for rs in recur_steps
            )
            front_raw = s0["obs_summary"]["front_cell_3_5"]
            world_front = s0["world_front_cell"]
            blocked_forward_checks[str(seed)] = {
                "cycle_step_start": c_step,
                "num_consecutive_blocked_forward_steps": len(recur_steps),
                "raw_front_cell_3_5": front_raw,
                "world_front_cell": world_front,
                "raw_and_world_front_cell_consistent": (
                    world_front is not None
                    and front_raw["object"] == world_front["type"]
                    and (
                        world_front["type"] == "wall"
                        or front_raw["color"] == world_front["color"]
                    )
                ),
                "world_front_impassable": (
                    world_front is not None and not world_front["can_overlap"]
                ),
                "feature_front_blocked": s0["front_blocked"],
                "chosen_action_name": s0["chosen_action_name"],
                "probs": s0["probs"],
                "all_recurrences_identical_pos_dir_feat_probs": all_unchanged_state,
            }

    blocked_front_dataset_summary = {
        "train_filtered": summarize_blocked_front_split(
            X_train, y_train, fb_train, tv_train
        ),
        "val": summarize_blocked_front_split(X_val, y_val, fb_val, tv_val),
    }

    # 3. Question C: Advisory BabyAIBot comparison on identical student trajectories
    advisory_results = {}
    for seed in FAILED_SEEDS:
        adv = run_advisory_replay(seed, params, replays[seed], deadline)
        assert adv["trajectory_identical_to_baseline"], (
            f"Advisory bot altered trajectory on seed {seed}: {adv['divergence']}"
        )
        # Attach advisory recommendations aligned to distinct cycle states
        anatomy = cycles[str(seed)]
        start_step = anatomy["first_entry_step"]
        cycle_len = anatomy["cycle_length"]
        adv_by_offset = []
        for offset in range(cycle_len):
            visit_steps = [
                adv["advisory_steps"][t]
                for t in range(start_step + offset, len(adv["advisory_steps"]), cycle_len)
            ]
            adv_by_offset.append(
                {
                    "cycle_offset": offset,
                    "student_action": visit_steps[0]["student_action_name"],
                    "student_probs": visit_steps[0]["student_probs"],
                    "expert_actions_across_visits": [
                        {
                            "step": v["step"],
                            "expert_action": v["expert_action_name"],
                            "agrees": v["agrees"],
                            "bot_vis_mask_true_cells": v["bot_vis_mask_true_cells"],
                            "bot_stack_top": (
                                v["bot_stack_repr"][-1] if v["bot_stack_repr"] else None
                            ),
                        }
                        for v in visit_steps
                    ],
                    "unique_expert_actions": sorted(
                        set(v["expert_action_name"] for v in visit_steps)
                    ),
                }
            )
        adv["cycle_offset_summary"] = adv_by_offset
        advisory_results[str(seed)] = adv

    elapsed = round(time.monotonic() - t0, 4)

    audit = {
        "baseline_metadata": {
            "phase1_weights_sha256": weights_sha256,
            "selected_epoch": train_meta["selected_epoch"],
            "selection_reason": train_meta["selection_reason"],
            "config_env_id": cfg["env_id"],
            "config_architecture": cfg["architecture"],
            "excluded_training_seeds": amendment["training_seed_exclusion"][
                "excluded_seeds"
            ],
            "audited_dev_seeds": FAILED_SEEDS,
            "probability_tolerance": PROB_TOL,
        },
        "baseline_replay_verification": {
            "all_7_episodes_exact_match": True,
            "per_seed": {
                str(s): {
                    "matched": replays[s]["matched_saved_baseline"],
                    "episode_length": replays[s]["episode_length"],
                    "truncated": replays[s]["truncated"],
                    "success": replays[s]["success"],
                }
                for s in FAILED_SEEDS
            },
        },
        "question_a_and_b_cycles": cycles,
        "question_b_blocked_forward_checks": blocked_forward_checks,
        "question_b_blocked_front_dataset_summary": blocked_front_dataset_summary,
        "question_c_advisory_bot": advisory_results,
        "runtime": {
            "cpu_only": True,
            "wall_clock_seconds": elapsed,
            "wall_clock_cap_seconds": WALL_CLOCK_CAP_SEC,
            "completed_within_cap": elapsed < WALL_CLOCK_CAP_SEC,
        },
    }

    Path("phase2_audit_results.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )

    # Print concise summary for terminal inspection
    compact = {
        "weights_sha256": weights_sha256,
        "baseline_replay_all_matched": True,
        "wall_clock_seconds": elapsed,
        "blocked_front_dataset_summary": blocked_front_dataset_summary,
        "blocked_forward_checks": blocked_forward_checks,
        "cycles_summary": {
            s: {
                "mission": c["mission"],
                "entry": c["first_entry_step"],
                "recurrence": c["first_complete_recurrence_step"],
                "length": c["cycle_length"],
                "periodic_exact": c["periodic_exact_through_timeout"],
                "distinct_states_support": [
                    {
                        "offset": st["cycle_offset"],
                        "pose": [*st["agent_pos"], st["agent_dir"]],
                        "front": st["front_cell_3_5"]["object"],
                        "tgt_coords": st["visible_target_egocentric_coords"],
                        "action": st["chosen_action_name"],
                        "probs": st["probs"],
                        "effect": st["transition_effect"],
                        "train_matches": st["dataset_support"]["train_exact_matches"],
                        "train_actions": st["dataset_support"]["train_action_counts"],
                        "student_has_train_support": st["dataset_support"][
                            "student_action_has_exact_train_support"
                        ],
                        "val_matches": st["dataset_support"]["val_exact_matches"],
                    }
                    for st in c["distinct_cycle_states"]
                ],
                "advisory": {
                    "identical_traj": advisory_results[s][
                        "trajectory_identical_to_baseline"
                    ],
                    "non_student_actions": advisory_results[s][
                        "expert_non_student_actions"
                    ],
                    "exceptions": advisory_results[s]["expert_exceptions"],
                    "changed_across_repeats": advisory_results[s][
                        "expert_changed_across_repeated_student_inputs"
                    ],
                    "per_offset_expert": [
                        {
                            "offset": o["cycle_offset"],
                            "student": o["student_action"],
                            "expert_unique": o["unique_expert_actions"],
                        }
                        for o in advisory_results[s]["cycle_offset_summary"]
                    ],
                },
            }
            for s, c in cycles.items()
        },
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
