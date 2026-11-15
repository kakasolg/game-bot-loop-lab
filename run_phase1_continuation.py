"""Phase 1 continuation: filter seed=10462 with provenance check, train once, evaluate frozen argmax."""

from __future__ import annotations

import base64
import json
import time
from collections import Counter
from pathlib import Path

import gymnasium as gym
import numpy as np
from minigrid.utils.baby_ai_bot import BabyAIBot

from run_phase1 import (
    ACTION_NAMES,
    ALLOWED_ACTIONS,
    encode_observation,
    evaluate_dataset,
    forward_mlp,
    hash_bytes,
    train_mlp,
    verify_feature_assertions,
)

EXCLUDED_SEEDS = {10462}
EXCLUSION_REASON = (
    "The expert proposed pickup, which is outside the student's allowed "
    "action set {left, right, forward}."
)


def replay_split_with_seeds(
    split_name: str,
    seeds: list[int],
    deadline: float,
) -> tuple[dict, bool]:
    """Deterministically replay expert collection tracking per-sample (seed, step)."""
    features: list[np.ndarray] = []
    actions: list[int] = []
    front_blocked_flags: list[bool] = []
    target_visible_flags: list[bool] = []
    sample_seeds: list[int] = []
    sample_steps: list[int] = []
    episode_records: list[dict] = []

    for seed in seeds:
        if time.monotonic() >= deadline:
            return {}, True

        env = gym.make("BabyAI-GoToLocal-v0")
        obs, _ = env.reset(seed=seed)
        bot = BabyAIBot(env)
        terminated = False
        truncated = False
        total_reward = 0.0
        prev_action = None
        ep_len = 0
        unexpected_action = None

        try:
            while not (terminated or truncated):
                action_obj = bot.replan(prev_action)
                action = int(action_obj)
                if action not in ALLOWED_ACTIONS:
                    unexpected_action = action
                    break

                feat, meta = encode_observation(obs)
                features.append(feat)
                actions.append(action)
                front_blocked_flags.append(meta["front_blocked"])
                target_visible_flags.append(meta["target_in_view"])
                sample_seeds.append(seed)
                sample_steps.append(ep_len)

                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                prev_action = action_obj
                ep_len += 1
        finally:
            env.close()

        episode_records.append(
            {
                "seed": seed,
                "steps_collected": ep_len,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "unexpected_action": unexpected_action,
                "success": bool(
                    terminated and total_reward > 0.0 and unexpected_action is None
                ),
            }
        )

    return (
        {
            "split": split_name,
            "X": np.stack(features),
            "y": np.array(actions, dtype=np.int64),
            "front_blocked": np.array(front_blocked_flags, dtype=bool),
            "target_in_view": np.array(target_visible_flags, dtype=bool),
            "sample_seeds": np.array(sample_seeds, dtype=np.int64),
            "sample_steps": np.array(sample_steps, dtype=np.int64),
            "episode_records": episode_records,
        },
        False,
    )


def subset_cross_entropy(
    X: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    params: dict[str, np.ndarray],
) -> float | None:
    if not np.any(mask):
        return None
    _, _, probs = forward_mlp(X[mask], params)
    ym = y[mask]
    ce = float(-np.mean(np.log(np.clip(probs[np.arange(len(ym)), ym], 1e-12, 1.0))))
    return round(ce, 6)


def evaluate_frozen_argmax(
    seeds: list[int],
    params: dict[str, np.ndarray],
    deadline: float,
) -> tuple[dict, bool]:
    episodes = []
    timed_out_wall_clock = False

    for seed in seeds:
        if time.monotonic() >= deadline:
            timed_out_wall_clock = True
            break

        env = gym.make("BabyAI-GoToLocal-v0")
        obs, _ = env.reset(seed=seed)
        terminated = False
        truncated = False
        total_reward = 0.0
        exc_info = None
        steps = []

        seen_feat_hashes: set[str] = set()
        seen_raw_hashes: set[str] = set()
        seen_poses: dict[tuple[int, int, int], int] = {}

        repeated_feat_count = 0
        repeated_raw_count = 0
        repeated_pose_count = 0
        obs_alias_without_pose_repeat = 0
        blocked_fwd_steps = 0
        cur_blocked_fwd_run = 0
        max_blocked_fwd_run = 0
        first_pose_cycle: dict | None = None

        try:
            while not (terminated or truncated):
                feat, meta = encode_observation(obs)
                feat_hash = hash_bytes(feat.tobytes())
                raw_hash = hash_bytes(
                    obs["image"].tobytes() + obs["mission"].encode("utf-8")
                )
                obs_image_b64 = base64.b64encode(obs["image"].tobytes()).decode("ascii")

                pos_x = int(env.unwrapped.agent_pos[0])
                pos_y = int(env.unwrapped.agent_pos[1])
                heading = int(env.unwrapped.agent_dir)
                pose = (pos_x, pos_y, heading)

                _, _, probs = forward_mlp(feat[None, :], params)
                probs_1d = probs[0]
                if not np.all(np.isfinite(probs_1d)):
                    raise FloatingPointError(f"Non-finite policy probs at seed={seed}")
                action = int(np.argmax(probs_1d))

                is_feat_rep = feat_hash in seen_feat_hashes
                is_raw_rep = raw_hash in seen_raw_hashes
                is_pose_rep = pose in seen_poses

                if is_feat_rep:
                    repeated_feat_count += 1
                    if not is_pose_rep:
                        obs_alias_without_pose_repeat += 1
                if is_raw_rep:
                    repeated_raw_count += 1
                if is_pose_rep:
                    repeated_pose_count += 1
                    if first_pose_cycle is None:
                        prev_step = seen_poses[pose]
                        cycle_len = len(steps) - prev_step
                        first_pose_cycle = {
                            "start_step": prev_step,
                            "repeat_step": len(steps),
                            "cycle_length": cycle_len,
                            "pose": [pos_x, pos_y, heading],
                            "cycle_poses": [
                                [*s["agent_pos"], s["agent_dir"]]
                                for s in steps[prev_step:]
                            ],
                            "cycle_actions": [
                                s["chosen_action_name"] for s in steps[prev_step:]
                            ],
                            "cycle_probs": [s["probs"] for s in steps[prev_step:]],
                            "cycle_feat_hashes": [
                                s["feat_hash"] for s in steps[prev_step:]
                            ],
                            "cycle_target_in_view_flags": [
                                s["target_in_view"] for s in steps[prev_step:]
                            ],
                            "cycle_front_blocked_flags": [
                                s["front_blocked"] for s in steps[prev_step:]
                            ],
                        }

                seen_feat_hashes.add(feat_hash)
                seen_raw_hashes.add(raw_hash)
                if pose not in seen_poses:
                    seen_poses[pose] = len(steps)

                is_blocked_fwd = bool(meta["front_blocked"] and action == 2)
                if is_blocked_fwd:
                    blocked_fwd_steps += 1
                    cur_blocked_fwd_run += 1
                    if cur_blocked_fwd_run > max_blocked_fwd_run:
                        max_blocked_fwd_run = cur_blocked_fwd_run
                else:
                    cur_blocked_fwd_run = 0

                steps.append(
                    {
                        "step": len(steps),
                        "obs_image_b64": obs_image_b64,
                        "feat_hash": feat_hash,
                        "raw_obs_hash": raw_hash,
                        "probs": [round(float(p), 6) for p in probs_1d],
                        "chosen_action": action,
                        "chosen_action_name": ACTION_NAMES[action],
                        "agent_pos": [pos_x, pos_y],
                        "agent_dir": heading,
                        "front_blocked": meta["front_blocked"],
                        "target_in_view": meta["target_in_view"],
                        "blocked_forward": is_blocked_fwd,
                    }
                )

                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
        except Exception as exc:  # noqa: BLE001
            exc_info = f"{type(exc).__name__}: {exc}"
        finally:
            env.close()

        episodes.append(
            {
                "seed": seed,
                "mission": obs["mission"],
                "success": bool(terminated and total_reward > 0.0 and exc_info is None),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "exception": exc_info,
                "episode_length": len(steps),
                "total_reward": round(total_reward, 6),
                "repeated_feat_obs_steps": repeated_feat_count,
                "repeated_raw_obs_steps": repeated_raw_count,
                "repeated_pose_steps": repeated_pose_count,
                "obs_alias_without_pose_repeat_steps": obs_alias_without_pose_repeat,
                "blocked_forward_steps": blocked_fwd_steps,
                "max_consecutive_blocked_forward": max_blocked_fwd_run,
                "first_pose_cycle": first_pose_cycle,
                "steps": steps,
            }
        )

    n_eps = len(episodes)
    cycle_lengths = Counter(
        e["first_pose_cycle"]["cycle_length"]
        for e in episodes
        if e["first_pose_cycle"] is not None
    )
    summary = {
        "num_episodes_completed": n_eps,
        "stopped_by_wall_clock_cap": timed_out_wall_clock,
        "success_count": sum(1 for e in episodes if e["success"]),
        "timeout_count": sum(1 for e in episodes if e["truncated"]),
        "exception_count": sum(1 for e in episodes if e["exception"] is not None),
        "mean_episode_length": (
            round(sum(e["episode_length"] for e in episodes) / n_eps, 4)
            if n_eps
            else 0.0
        ),
        "episodes_with_repeated_feature_obs": sum(
            1 for e in episodes if e["repeated_feat_obs_steps"] > 0
        ),
        "episodes_with_repeated_raw_obs": sum(
            1 for e in episodes if e["repeated_raw_obs_steps"] > 0
        ),
        "episodes_with_obs_alias_without_pose_repeat": sum(
            1 for e in episodes if e["obs_alias_without_pose_repeat_steps"] > 0
        ),
        "episodes_with_repeated_pose": sum(
            1 for e in episodes if e["repeated_pose_steps"] > 0
        ),
        "episodes_with_any_blocked_forward": sum(
            1 for e in episodes if e["blocked_forward_steps"] > 0
        ),
        "episodes_with_consecutive_blocked_forward_ge2": sum(
            1 for e in episodes if e["max_consecutive_blocked_forward"] >= 2
        ),
        "total_blocked_forward_steps": sum(e["blocked_forward_steps"] for e in episodes),
        "episodes_with_pose_cycle_length_1": sum(
            1
            for e in episodes
            if e["first_pose_cycle"] and e["first_pose_cycle"]["cycle_length"] == 1
        ),
        "episodes_with_pose_cycle_length_2": sum(
            1
            for e in episodes
            if e["first_pose_cycle"] and e["first_pose_cycle"]["cycle_length"] == 2
        ),
        "episodes_with_pose_cycle_length_ge3": sum(
            1
            for e in episodes
            if e["first_pose_cycle"] and e["first_pose_cycle"]["cycle_length"] >= 3
        ),
        "first_pose_cycle_length_histogram": {
            str(k): v for k, v in sorted(cycle_lengths.items())
        },
        "pose_cycle_episodes_target_never_visible_in_cycle": sum(
            1
            for e in episodes
            if e["first_pose_cycle"]
            and not any(e["first_pose_cycle"]["cycle_target_in_view_flags"])
        ),
        "pose_cycle_episodes_target_visible_in_cycle": sum(
            1
            for e in episodes
            if e["first_pose_cycle"]
            and any(e["first_pose_cycle"]["cycle_target_in_view_flags"])
        ),
    }
    return {"summary": summary, "episodes": episodes}, timed_out_wall_clock


def main() -> None:
    t0 = time.monotonic()
    cfg = json.loads(Path("phase1_config.json").read_text(encoding="utf-8"))
    amendment = json.loads(
        Path("phase1_config_amendment.json").read_text(encoding="utf-8")
    )
    cap = float(amendment["continuation_wall_clock_cap_seconds"])
    deadline = t0 + cap

    verify_feature_assertions()

    # Load original preserved archive to verify exact provenance before filtering
    orig = np.load("phase1_demos.npz")

    tr_cfg = cfg["seeds"]["train"]
    val_cfg = cfg["seeds"]["val"]
    train_seeds = list(range(tr_cfg["start"], tr_cfg["end_inclusive"] + 1))
    val_seeds = list(range(val_cfg["start"], val_cfg["end_inclusive"] + 1))

    replayed_tr, cap_r1 = replay_split_with_seeds("train", train_seeds, deadline)
    replayed_val, cap_r2 = replay_split_with_seeds("val", val_seeds, deadline)
    if cap_r1 or cap_r2:
        print(json.dumps({"status": "halted_wall_clock_during_provenance_replay"}))
        return

    assert np.array_equal(replayed_tr["X"], orig["X_train"]), "X_train mismatch"
    assert np.array_equal(replayed_tr["y"], orig["y_train"]), "y_train mismatch"
    assert np.array_equal(
        replayed_tr["front_blocked"], orig["front_blocked_train"]
    ), "front_blocked_train mismatch"
    assert np.array_equal(
        replayed_tr["target_in_view"], orig["target_in_view_train"]
    ), "target_in_view_train mismatch"
    assert np.array_equal(replayed_val["X"], orig["X_val"]), "X_val mismatch"
    assert np.array_equal(replayed_val["y"], orig["y_val"]), "y_val mismatch"

    keep_mask = ~np.isin(replayed_tr["sample_seeds"], list(EXCLUDED_SEEDS))
    removed_indices = np.where(~keep_mask)[0]
    assert len(removed_indices) == 1, f"Expected 1 removed sample, got {len(removed_indices)}"
    removed_seed = int(replayed_tr["sample_seeds"][removed_indices[0]])
    removed_step = int(replayed_tr["sample_steps"][removed_indices[0]])
    assert removed_seed == 10462 and removed_step == 0

    filtered_train = {
        "X": replayed_tr["X"][keep_mask],
        "y": replayed_tr["y"][keep_mask],
        "front_blocked": replayed_tr["front_blocked"][keep_mask],
        "target_in_view": replayed_tr["target_in_view"][keep_mask],
        "sample_seeds": replayed_tr["sample_seeds"][keep_mask],
        "sample_steps": replayed_tr["sample_steps"][keep_mask],
    }
    assert not np.any(np.isin(filtered_train["sample_seeds"], list(EXCLUDED_SEEDS)))

    val_data = {
        "X": orig["X_val"],
        "y": orig["y_val"],
        "front_blocked": orig["front_blocked_val"],
        "target_in_view": orig["target_in_view_val"],
    }

    np.savez_compressed(
        "phase1_demos_filtered.npz",
        X_train=filtered_train["X"],
        y_train=filtered_train["y"],
        front_blocked_train=filtered_train["front_blocked"],
        target_in_view_train=filtered_train["target_in_view"],
        sample_seeds_train=filtered_train["sample_seeds"],
        sample_steps_train=filtered_train["sample_steps"],
        X_val=val_data["X"],
        y_val=val_data["y"],
        front_blocked_val=val_data["front_blocked"],
        target_in_view_val=val_data["target_in_view"],
        sample_seeds_val=replayed_val["sample_seeds"],
        sample_steps_val=replayed_val["sample_steps"],
    )

    tr_action_counts = Counter(ACTION_NAMES[int(a)] for a in filtered_train["y"])
    val_action_counts = Counter(ACTION_NAMES[int(a)] for a in val_data["y"])
    filtering_manifest = {
        "original_archive": "phase1_demos.npz",
        "filtered_dataset": "phase1_demos_filtered.npz",
        "provenance_verified_exact_match_to_original": True,
        "excluded_episodes": [
            {
                "seed": 10462,
                "reason": EXCLUSION_REASON,
                "removed_sample_count": int(len(removed_indices)),
                "removed_sample_steps": [removed_step],
            }
        ],
        "train": {
            "original_episodes_attempted": 500,
            "retained_episodes": len(
                set(int(s) for s in filtered_train["sample_seeds"].tolist())
            ),
            "original_sample_count": int(len(orig["y_train"])),
            "removed_sample_count": int(len(removed_indices)),
            "filtered_sample_count": int(len(filtered_train["y"])),
            "remaining_excluded_samples": int(
                np.sum(np.isin(filtered_train["sample_seeds"], list(EXCLUDED_SEEDS)))
            ),
            "action_counts": dict(tr_action_counts),
            "front_blocked_samples": int(np.sum(filtered_train["front_blocked"])),
            "target_in_view_samples": int(np.sum(filtered_train["target_in_view"])),
        },
        "val": {
            "retained_episodes": 100,
            "sample_count": int(len(val_data["y"])),
            "unchanged_from_original": True,
            "action_counts": dict(val_action_counts),
            "front_blocked_samples": int(np.sum(val_data["front_blocked"])),
            "target_in_view_samples": int(np.sum(val_data["target_in_view"])),
        },
    }
    Path("phase1_filtering_manifest.json").write_text(
        json.dumps(filtering_manifest, indent=2), encoding="utf-8"
    )

    # Train single model once on filtered dataset
    t_train_start = time.monotonic()
    best_params, train_report, cap_train = train_mlp(
        filtered_train, val_data, cfg, deadline
    )
    t_train_sec = time.monotonic() - t_train_start

    # Check finite weights and add visible/hidden/blocked cross-entropy details
    for k, arr in best_params.items():
        if not np.all(np.isfinite(arr)):
            raise FloatingPointError(f"Non-finite parameter tensor {k}")
    for ep_row in train_report["epoch_history"]:
        if not (np.isfinite(ep_row["train_ce"]) and np.isfinite(ep_row["val_ce"])):
            raise FloatingPointError(f"Non-finite loss at epoch {ep_row['epoch']}")

    for split_key, dset in (("train_metrics_at_selected", filtered_train), ("val_metrics_at_selected", val_data)):
        m = train_report[split_key]
        m["visible_target"]["cross_entropy"] = subset_cross_entropy(
            dset["X"], dset["y"], dset["target_in_view"], best_params
        )
        m["hidden_target"]["cross_entropy"] = subset_cross_entropy(
            dset["X"], dset["y"], ~dset["target_in_view"], best_params
        )
        m["blocked_front"]["cross_entropy"] = subset_cross_entropy(
            dset["X"], dset["y"], dset["front_blocked"], best_params
        )

    train_report["selection_reason"] = (
        f"Epoch {train_report['selected_epoch']} selected by lowest val_cross_entropy "
        f"({train_report['val_metrics_at_selected']['cross_entropy']}) under "
        f"{cfg['training']['checkpoint_selection_rule']}."
    )
    train_report["training_wall_clock_seconds"] = round(t_train_sec, 4)

    # Save weights and training report BEFORE rollout evaluation
    np.savez("phase1_weights.npz", **best_params)
    Path("phase1_training.json").write_text(
        json.dumps(train_report, indent=2), encoding="utf-8"
    )

    if cap_train:
        print(
            json.dumps(
                {
                    "status": "halted_after_training_cap",
                    "filtering_manifest": filtering_manifest,
                    "training": {
                        k: v for k, v in train_report.items() if k != "epoch_history"
                    },
                },
                indent=2,
            )
        )
        return

    # Evaluate frozen deterministic argmax policy on dev seeds 0..99
    dev_cfg = cfg["seeds"]["dev_eval"]
    dev_seeds = list(range(dev_cfg["start"], dev_cfg["end_inclusive"] + 1))

    t_eval_start = time.monotonic()
    eval_report, cap_eval = evaluate_frozen_argmax(dev_seeds, best_params, deadline)
    t_eval_sec = time.monotonic() - t_eval_start
    total_sec = time.monotonic() - t0

    eval_report["summary"]["eval_wall_clock_seconds"] = round(t_eval_sec, 4)
    eval_report["summary"]["total_continuation_wall_clock_seconds"] = round(total_sec, 4)
    eval_report["summary"]["completed_within_cap"] = not cap_eval

    Path("phase1_eval_results.json").write_text(
        json.dumps(eval_report, indent=2), encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "status": "completed" if not cap_eval else "halted_during_eval_cap",
                "filtering_manifest": filtering_manifest,
                "training_selected": {
                    k: v for k, v in train_report.items() if k != "epoch_history"
                },
                "dev_eval_summary": eval_report["summary"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
