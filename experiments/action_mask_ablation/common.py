"""Shared evaluation, statistical, and timing helpers for action-mask ablation."""

from __future__ import annotations

import hashlib
import statistics
import time
from collections import Counter
from pathlib import Path

import gymnasium as gym
import numpy as np

from experiments.action_mask_ablation.mask_selector import (
    select_action_with_blocked_forward_mask,
)
from external_wrapper import ExternalObservationCountWrapper
from phase5_stats import exact_mcnemar_test, newcombe_paired_method10_ci
from run_phase1 import ACTION_NAMES, encode_observation, forward_mlp, hash_bytes


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_single_episode(
    seed: int,
    params: dict[str, np.ndarray],
    arm: str,
    wrapper: ExternalObservationCountWrapper | None = None,
    record_e2e_ns: bool = False,
    capture_raw_obs_for_replay: bool = False,
    capture_step_traces: bool = False,
) -> dict:
    """Execute one episode on BabyAI-GoToLocal-v0 for Arm 'A', 'B', or 'C'."""
    if arm not in ("A", "B", "C"):
        raise ValueError(f"Unknown arm {arm!r}")
    if arm == "C" and wrapper is None:
        raise ValueError("Arm 'C' requires an ExternalObservationCountWrapper instance")

    env = gym.make("BabyAI-GoToLocal-v0")
    obs, _ = env.reset(seed=seed)
    mission = str(obs["mission"])

    reset_ns: int | None = None
    if arm == "C":
        tr0 = time.perf_counter_ns()
        wrapper.reset()
        reset_ns = time.perf_counter_ns() - tr0

    terminated = False
    truncated = False
    total_reward = 0.0
    exc_info: str | None = None
    step_count = 0

    seen_feat_hashes: set[str] = set()
    seen_raw_keys: set[tuple[bytes, str]] = set()
    seen_poses: dict[tuple[int, int, int], int] = {}

    repeated_feat_count = 0
    repeated_raw_count = 0
    repeated_pose_count = 0
    obs_alias_without_pose_repeat = 0
    blocked_fwd_steps = 0
    cur_blocked_fwd_run = 0
    max_blocked_fwd_run = 0
    first_pose_cycle_length: int | None = None

    # Arm-specific intervention/edge-case counters
    override_steps = 0
    mask_exclusions = 0
    mask_action_changes = 0
    empty_mask_fallbacks = 0
    unmasked_prob_ties = 0
    masked_zero_prob_underflows = 0

    e2e_ns_samples: list[int] = []
    replay_obs_seq: list[tuple[np.ndarray, str]] = []
    step_traces: list[dict] = []

    try:
        while not (terminated or truncated):
            t0 = time.perf_counter_ns() if record_e2e_ns else 0
            feat, meta = encode_observation(obs)
            _, _, probs = forward_mlp(feat[None, :], params)
            probs_1d = probs[0]
            if not np.all(np.isfinite(probs_1d)):
                raise FloatingPointError(
                    f"Non-finite policy probs at arm={arm} seed={seed} step={step_count}"
                )

            if arm == "A":
                action = int(np.argmax(probs_1d))
                sel_meta = None
            elif arm == "B":
                action, sel_meta = select_action_with_blocked_forward_mask(
                    obs["image"], obs["mission"], probs_1d
                )
            else:
                action, sel_meta = wrapper.select_action(
                    obs["image"], obs["mission"], probs_1d
                )
            t1 = time.perf_counter_ns() if record_e2e_ns else 0

            if record_e2e_ns:
                e2e_ns_samples.append(t1 - t0)

            if capture_raw_obs_for_replay:
                replay_obs_seq.append((obs["image"].copy(), str(obs["mission"])))

            if arm == "B" and sel_meta is not None:
                if sel_meta["forward_masked"]:
                    mask_exclusions += 1
                    if sel_meta["all_allowed_zero_prob"]:
                        masked_zero_prob_underflows += 1
                if sel_meta["action_changed"]:
                    mask_action_changes += 1
                if sel_meta["fallback_empty_mask"]:
                    empty_mask_fallbacks += 1
                if sel_meta["unmasked_prob_tie"]:
                    unmasked_prob_ties += 1
            elif arm == "C" and sel_meta is not None:
                if sel_meta["overridden"]:
                    override_steps += 1

            # Offline trajectory diagnostics (never exposed to policy/mask/wrapper)
            feat_hash = hash_bytes(feat.tobytes())
            raw_key = (obs["image"].tobytes(), str(obs["mission"]))
            pos_x = int(env.unwrapped.agent_pos[0])
            pos_y = int(env.unwrapped.agent_pos[1])
            heading = int(env.unwrapped.agent_dir)
            pose = (pos_x, pos_y, heading)

            is_feat_rep = feat_hash in seen_feat_hashes
            is_raw_rep = raw_key in seen_raw_keys
            is_pose_rep = pose in seen_poses

            if is_feat_rep:
                repeated_feat_count += 1
                if not is_pose_rep:
                    obs_alias_without_pose_repeat += 1
            if is_raw_rep:
                repeated_raw_count += 1
            if is_pose_rep:
                repeated_pose_count += 1
                if first_pose_cycle_length is None:
                    first_pose_cycle_length = step_count - seen_poses[pose]

            seen_feat_hashes.add(feat_hash)
            seen_raw_keys.add(raw_key)
            if pose not in seen_poses:
                seen_poses[pose] = step_count

            is_blocked_fwd = bool(meta["front_blocked"] and action == 2)
            if is_blocked_fwd:
                blocked_fwd_steps += 1
                cur_blocked_fwd_run += 1
                if cur_blocked_fwd_run > max_blocked_fwd_run:
                    max_blocked_fwd_run = cur_blocked_fwd_run
            else:
                cur_blocked_fwd_run = 0

            if capture_step_traces:
                trace_item = {
                    "step": step_count,
                    "pose": [pos_x, pos_y, heading],
                    "front_cell_raw": [int(x) for x in obs["image"][3, 5]],
                    "front_blocked_feat": meta["front_blocked"],
                    "target_in_view": meta["target_in_view"],
                    "probs": [round(float(p), 6) for p in probs_1d],
                    "base_argmax": int(np.argmax(probs_1d)),
                    "chosen_action": action,
                    "chosen_action_name": ACTION_NAMES[action],
                }
                if arm == "B" and sel_meta is not None:
                    trace_item.update(
                        {
                            "forward_masked": sel_meta["forward_masked"],
                            "action_changed": sel_meta["action_changed"],
                            "allowed_actions": sel_meta["allowed_actions"],
                            "front_is_mission_target": sel_meta["front_is_mission_target"],
                        }
                    )
                elif arm == "C" and sel_meta is not None:
                    trace_item.update(
                        {
                            "overridden": sel_meta["overridden"],
                            "prior_counts": sel_meta["prior_counts"],
                            "post_counts": sel_meta["post_counts"],
                        }
                    )
                step_traces.append(trace_item)

            step_count += 1
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
    except Exception as exc:  # noqa: BLE001
        exc_info = f"{type(exc).__name__}: {exc}"
    finally:
        env.close()

    success = bool(terminated and total_reward > 0.0 and exc_info is None)
    if exc_info is not None:
        outcome = "exception"
    elif success:
        outcome = "success"
    elif truncated and not terminated:
        outcome = "timeout"
    else:
        outcome = "terminated_non_positive_reward"

    record = {
        "seed": seed,
        "arm": arm,
        "mission": mission,
        "outcome": outcome,
        "success": success,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "exception": exc_info,
        "episode_length": step_count,
        "total_reward": round(total_reward, 6),
        "repeated_raw_obs_steps": repeated_raw_count,
        "repeated_feat_obs_steps": repeated_feat_count,
        "repeated_pose_steps": repeated_pose_count,
        "obs_alias_without_pose_repeat_steps": obs_alias_without_pose_repeat,
        "blocked_forward_steps": blocked_fwd_steps,
        "max_consecutive_blocked_forward": max_blocked_fwd_run,
        "first_pose_cycle_length": first_pose_cycle_length,
    }
    if arm == "B":
        record.update(
            {
                "mask_exclusions": mask_exclusions,
                "mask_action_changes": mask_action_changes,
                "empty_mask_fallbacks": empty_mask_fallbacks,
                "unmasked_prob_ties": unmasked_prob_ties,
                "masked_zero_prob_underflows": masked_zero_prob_underflows,
            }
        )
    elif arm == "C":
        w_stats = wrapper.episode_stats()
        record.update(
            {
                "override_steps": override_steps,
                "retained_obs_keys": w_stats["retained_obs_keys"],
                "post_full_coverage_decisions": w_stats["post_full_coverage_decisions"],
                "distinct_keys_with_post_full_coverage_decision": w_stats[
                    "distinct_keys_with_post_full_coverage_decision"
                ],
                "distinct_keys_reaching_all_three_counts_ge_1_by_end": w_stats[
                    "distinct_keys_reaching_all_three_counts_ge_1_by_end"
                ],
            }
        )

    return {
        "record": record,
        "e2e_ns_samples": e2e_ns_samples,
        "wrapper_reset_ns": reset_ns,
        "replay_obs_seq": replay_obs_seq,
        "step_traces": step_traces,
    }


def length_stats(lengths: list[int]) -> dict:
    if not lengths:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(lengths),
        "mean": round(sum(lengths) / len(lengths), 4),
        "median": round(float(statistics.median(lengths)), 4),
        "min": min(lengths),
        "max": max(lengths),
    }


def summarize_arm_records(records: list[dict], arm: str) -> dict:
    n = len(records)
    succ = sum(1 for r in records if r["outcome"] == "success")
    out = {
        "episodes_completed": n,
        "success_count": succ,
        "success_rate": round(succ / n, 6) if n else None,
        "timeout_count": sum(1 for r in records if r["outcome"] == "timeout"),
        "exception_count": sum(1 for r in records if r["outcome"] == "exception"),
        "terminated_non_positive_count": sum(
            1 for r in records if r["outcome"] == "terminated_non_positive_reward"
        ),
        "total_decisions": sum(r["episode_length"] for r in records),
        "all_episode_lengths": length_stats([r["episode_length"] for r in records]),
        "successful_episode_lengths": length_stats(
            [r["episode_length"] for r in records if r["success"]]
        ),
        "episodes_with_repeated_raw_obs": sum(
            1 for r in records if r["repeated_raw_obs_steps"] > 0
        ),
        "episodes_with_repeated_feat_obs": sum(
            1 for r in records if r["repeated_feat_obs_steps"] > 0
        ),
        "episodes_with_obs_alias_without_pose_repeat": sum(
            1 for r in records if r["obs_alias_without_pose_repeat_steps"] > 0
        ),
        "episodes_with_repeated_pose": sum(
            1 for r in records if r["repeated_pose_steps"] > 0
        ),
        "episodes_with_any_blocked_forward": sum(
            1 for r in records if r["blocked_forward_steps"] > 0
        ),
        "episodes_with_consecutive_blocked_forward_ge2": sum(
            1 for r in records if r["max_consecutive_blocked_forward"] >= 2
        ),
        "total_blocked_forward_steps": sum(
            r["blocked_forward_steps"] for r in records
        ),
        "first_pose_cycle_length_histogram": {
            str(k): int(v)
            for k, v in sorted(
                Counter(
                    r["first_pose_cycle_length"]
                    for r in records
                    if r["first_pose_cycle_length"] is not None
                ).items()
            )
        },
    }
    if arm == "B":
        out["mask_interventions"] = {
            "episodes_with_mask_exclusion": sum(
                1 for r in records if r["mask_exclusions"] > 0
            ),
            "total_mask_exclusions": sum(r["mask_exclusions"] for r in records),
            "episodes_with_mask_action_change": sum(
                1 for r in records if r["mask_action_changes"] > 0
            ),
            "total_mask_action_changes": sum(
                r["mask_action_changes"] for r in records
            ),
            "empty_mask_fallbacks_total": sum(
                r["empty_mask_fallbacks"] for r in records
            ),
            "unmasked_prob_ties_total": sum(
                r["unmasked_prob_ties"] for r in records
            ),
            "masked_zero_prob_underflows_total": sum(
                r["masked_zero_prob_underflows"] for r in records
            ),
        }
    elif arm == "C":
        out["wrapper_interventions"] = {
            "episodes_with_at_least_one_override": sum(
                1 for r in records if r["override_steps"] > 0
            ),
            "total_overridden_actions": sum(r["override_steps"] for r in records),
            "mean_retained_obs_keys_per_episode": (
                round(sum(r["retained_obs_keys"] for r in records) / n, 4) if n else 0.0
            ),
            "peak_retained_obs_keys_in_episode": max(
                (r["retained_obs_keys"] for r in records), default=0
            ),
            "peak_retained_obs_keys_seed": (
                max(records, key=lambda r: r["retained_obs_keys"])["seed"]
                if records
                else None
            ),
            "post_full_coverage_decisions_total": sum(
                r["post_full_coverage_decisions"] for r in records
            ),
            "distinct_obs_keys_with_post_full_coverage_decision_total": sum(
                r["distinct_keys_with_post_full_coverage_decision"] for r in records
            ),
            "distinct_obs_keys_reaching_all_three_ge_1_by_end_total": sum(
                r["distinct_keys_reaching_all_three_counts_ge_1_by_end"]
                for r in records
            ),
        }
    return out


def compute_paired_contrast(
    seeds: list[int],
    recs_first: list[dict],
    recs_second: list[dict],
    label_first: str,
    label_second: str,
) -> dict:
    """Compute paired comparison (second - first) using Newcombe Method 10 & exact McNemar."""
    both_succ_seeds: list[int] = []
    first_only_seeds: list[int] = []
    second_only_seeds: list[int] = []
    both_fail_seeds: list[int] = []

    for seed, r1, r2 in zip(seeds, recs_first, recs_second, strict=True):
        s1 = bool(r1["success"])
        s2 = bool(r2["success"])
        if s1 and s2:
            both_succ_seeds.append(seed)
        elif s1 and not s2:
            first_only_seeds.append(seed)
        elif not s1 and s2:
            second_only_seeds.append(seed)
        else:
            both_fail_seeds.append(seed)

    ci = newcombe_paired_method10_ci(
        len(both_succ_seeds),
        len(first_only_seeds),
        len(second_only_seeds),
        len(both_fail_seeds),
    )
    mcnemar = exact_mcnemar_test(len(first_only_seeds), len(second_only_seeds))
    return {
        "contrast": f"{label_second}_minus_{label_first}",
        "first_arm": label_first,
        "second_arm": label_second,
        "n_total": len(seeds),
        "table_counts": {
            "both_succeed": len(both_succ_seeds),
            f"{label_first}_only_success": len(first_only_seeds),
            f"{label_second}_only_success": len(second_only_seeds),
            "both_fail": len(both_fail_seeds),
        },
        f"{label_first}_only_seeds": first_only_seeds,
        f"{label_second}_only_seeds": second_only_seeds,
        "delta_second_minus_first": ci["delta_wrapped_minus_baseline"],
        "first_rate": ci["baseline_rate"],
        "second_rate": ci["wrapped_rate"],
        "newcombe_method10_paired_95_ci": {
            "ci_lower": ci["ci_lower"],
            "ci_upper": ci["ci_upper"],
            "ci_includes_zero": ci["ci_includes_zero"],
            "phi_hat": ci["phi_hat"],
            "degenerate_margin": ci["degenerate_margin"],
        },
        "exact_mcnemar_test": mcnemar,
        "matched_lengths": {
            "both_succeed": {
                label_first: length_stats(
                    [r1["episode_length"] for r1, r2 in zip(recs_first, recs_second, strict=True) if r1["success"] and r2["success"]]
                ),
                label_second: length_stats(
                    [r2["episode_length"] for r1, r2 in zip(recs_first, recs_second, strict=True) if r1["success"] and r2["success"]]
                ),
            },
            f"{label_second}_only_success": {
                label_first: length_stats(
                    [r1["episode_length"] for r1, r2 in zip(recs_first, recs_second, strict=True) if not r1["success"] and r2["success"]]
                ),
                label_second: length_stats(
                    [r2["episode_length"] for r1, r2 in zip(recs_first, recs_second, strict=True) if not r1["success"] and r2["success"]]
                ),
            },
            f"{label_first}_only_success": {
                label_first: length_stats(
                    [r1["episode_length"] for r1, r2 in zip(recs_first, recs_second, strict=True) if r1["success"] and not r2["success"]]
                ),
                label_second: length_stats(
                    [r2["episode_length"] for r1, r2 in zip(recs_first, recs_second, strict=True) if r1["success"] and not r2["success"]]
                ),
            },
            "both_fail": {
                label_first: length_stats(
                    [r1["episode_length"] for r1, r2 in zip(recs_first, recs_second, strict=True) if not r1["success"] and not r2["success"]]
                ),
                label_second: length_stats(
                    [r2["episode_length"] for r1, r2 in zip(recs_first, recs_second, strict=True) if not r1["success"] and not r2["success"]]
                ),
            },
        },
    }


def holm_bonferroni_adjust(p_items: list[tuple[str, float]]) -> dict[str, dict]:
    """Step-down Holm-Bonferroni family-wise adjustment across named contrasts."""
    m = len(p_items)
    ordered = sorted(p_items, key=lambda x: x[1])
    running_max = 0.0
    adjusted: dict[str, dict] = {}
    for rank_idx, (name, raw_p) in enumerate(ordered):
        mult = m - rank_idx
        step_p = min(1.0, raw_p * mult)
        running_max = max(running_max, step_p)
        adjusted[name] = {
            "raw_p_value": raw_p,
            "holm_multiplier": mult,
            "holm_adjusted_p_value": round(running_max, 10),
        }
    return adjusted


def summarize_ns_distribution(samples_ns: list[int]) -> dict:
    """Summarize nanosecond latency samples in microseconds (min, p50, p95, p99, max, mean)."""
    if not samples_ns:
        return {
            "sample_count": 0,
            "min_us": None,
            "p50_us": None,
            "p95_us": None,
            "p99_us": None,
            "max_us": None,
            "mean_us": None,
        }
    arr_us = np.array(samples_ns, dtype=np.float64) / 1000.0
    return {
        "sample_count": int(arr_us.size),
        "min_us": round(float(np.min(arr_us)), 3),
        "p50_us": round(float(np.percentile(arr_us, 50)), 3),
        "p95_us": round(float(np.percentile(arr_us, 95)), 3),
        "p99_us": round(float(np.percentile(arr_us, 99)), 3),
        "max_us": round(float(np.max(arr_us)), 3),
        "mean_us": round(float(np.mean(arr_us)), 3),
    }
