"""Frozen paired confirmation evaluation on reserved seeds 30000..30199."""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import gymnasium as gym
import minigrid
import numpy as np

from external_wrapper import ExternalObservationCountWrapper, run_wrapper_unit_tests
from phase5_stats import (
    exact_mcnemar_test,
    newcombe_paired_method10_ci,
    run_statistical_unit_tests,
)
from run_phase1 import ACTION_NAMES, encode_observation, forward_mlp, hash_bytes, verify_feature_assertions


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_locked_plan(plan: dict) -> tuple[bool, dict]:
    """Verify exact SHA-256 hashes and package versions against frozen confirmation plan."""
    expected_hashes: dict[str, str] = plan["locked_artifacts_sha256"]
    observed_hashes: dict[str, str] = {}
    hash_mismatches: list[dict] = []

    for fname, exp_sha in expected_hashes.items():
        p = Path(fname)
        if not p.exists():
            hash_mismatches.append({"file": fname, "expected": exp_sha, "observed": "MISSING"})
            continue
        obs_sha = sha256_file(p)
        observed_hashes[fname] = obs_sha
        if obs_sha != exp_sha:
            hash_mismatches.append({"file": fname, "expected": exp_sha, "observed": obs_sha})

    expected_versions: dict[str, str] = plan["locked_versions"]
    observed_versions = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "gymnasium": gym.__version__,
        "minigrid": minigrid.__version__,
    }
    version_mismatches = [
        {"package": k, "expected": v, "observed": observed_versions.get(k)}
        for k, v in expected_versions.items()
        if observed_versions.get(k) != v
    ]

    ok = len(hash_mismatches) == 0 and len(version_mismatches) == 0
    return ok, {
        "all_verified": ok,
        "observed_hashes": observed_hashes,
        "hash_mismatches": hash_mismatches,
        "observed_versions": observed_versions,
        "version_mismatches": version_mismatches,
    }


def run_single_episode(
    seed: int,
    params: dict[str, np.ndarray],
    wrapper: ExternalObservationCountWrapper | None,
) -> tuple[dict, float]:
    """Execute one episode under frozen baseline or frozen+wrapper; return summary & wrapper time."""
    env = gym.make("BabyAI-GoToLocal-v0")
    obs, _ = env.reset(seed=seed)
    mission = str(obs["mission"])

    wrapper_sec = 0.0
    if wrapper is not None:
        tw0 = time.perf_counter()
        wrapper.reset()
        wrapper_sec += time.perf_counter() - tw0

    terminated = False
    truncated = False
    total_reward = 0.0
    exc_info = None
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
    override_steps = 0

    try:
        while not (terminated or truncated):
            feat, meta = encode_observation(obs)
            feat_hash = hash_bytes(feat.tobytes())
            raw_key = (obs["image"].tobytes(), str(obs["mission"]))

            # Diagnostics only (never passed to policy or wrapper)
            pos_x = int(env.unwrapped.agent_pos[0])
            pos_y = int(env.unwrapped.agent_pos[1])
            heading = int(env.unwrapped.agent_dir)
            pose = (pos_x, pos_y, heading)

            _, _, probs = forward_mlp(feat[None, :], params)
            probs_1d = probs[0]
            if not np.all(np.isfinite(probs_1d)):
                raise FloatingPointError(f"Non-finite policy probs at seed={seed} step={step_count}")

            if wrapper is None:
                action = int(np.argmax(probs_1d))
            else:
                tw0 = time.perf_counter()
                action, w_meta = wrapper.select_action(
                    obs["image"], obs["mission"], probs_1d
                )
                wrapper_sec += time.perf_counter() - tw0
                if w_meta["overridden"]:
                    override_steps += 1

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

            if meta["front_blocked"] and action == 2:
                blocked_fwd_steps += 1
                cur_blocked_fwd_run += 1
                if cur_blocked_fwd_run > max_blocked_fwd_run:
                    max_blocked_fwd_run = cur_blocked_fwd_run
            else:
                cur_blocked_fwd_run = 0

            step_count += 1
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
    except Exception as exc:  # noqa: BLE001
        exc_info = f"{type(exc).__name__}: {exc}"
    finally:
        env.close()

    # Exact classification: success requires terminated, positive reward, and no exception
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
    if wrapper is not None:
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
    return record, wrapper_sec


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


def summarize_condition(records: list[dict], is_wrapped: bool) -> dict:
    n = len(records)
    out = {
        "episodes_completed": n,
        "success_count": sum(1 for r in records if r["outcome"] == "success"),
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
    if is_wrapped:
        out["intervention_and_memory"] = {
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


def main() -> None:
    t0 = time.monotonic()
    plan_path = Path("phase5_confirmation_plan.json")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    cap_sec = float(plan["runtime_cap_seconds"])
    deadline = t0 + cap_sec

    lock_ok, lock_report = verify_locked_plan(plan)
    if not lock_ok:
        err_out = {
            "status": "halted_lock_verification_failed",
            "lock_verification": lock_report,
        }
        Path("phase5_confirmation_results.json").write_text(
            json.dumps(err_out, indent=2), encoding="utf-8"
        )
        print(json.dumps(err_out, indent=2))
        raise SystemExit(1)

    wrapper_tests = run_wrapper_unit_tests()
    stats_tests = run_statistical_unit_tests()
    verify_feature_assertions()

    p1_npz = np.load("phase1_weights.npz")
    p1_params = {k: p1_npz[k] for k in p1_npz.files}
    wrapper = ExternalObservationCountWrapper()

    seed_start = int(plan["seeds"]["start"])
    seed_end = int(plan["seeds"]["end_inclusive"])
    planned_seeds = list(range(seed_start, seed_end + 1))

    paired_records: list[dict] = []
    base_records: list[dict] = []
    wrap_records: list[dict] = []

    base_eval_sec = 0.0
    wrap_eval_sec = 0.0
    wrapper_accum_sec = 0.0

    stopped_reason: str | None = None
    stopped_seed: int | None = None

    jsonl_path = Path("phase5_confirmation_episodes.jsonl")
    with jsonl_path.open("w", encoding="utf-8") as jf:
        for seed in planned_seeds:
            if time.monotonic() >= deadline:
                stopped_reason = "wall_clock_cap_reached_before_seed"
                stopped_seed = seed
                break

            tb0 = time.perf_counter()
            base_rec, _ = run_single_episode(seed, p1_params, None)
            base_eval_sec += time.perf_counter() - tb0

            if time.monotonic() >= deadline:
                # Preserve partial unpaired baseline record visibly before halting
                base_records.append(base_rec)
                partial_pair = {
                    "seed": seed,
                    "baseline": base_rec,
                    "wrapped": None,
                    "paired_category": "incomplete_wall_clock_cap",
                }
                paired_records.append(partial_pair)
                jf.write(json.dumps(partial_pair) + "\n")
                jf.flush()
                stopped_reason = "wall_clock_cap_reached_mid_pair"
                stopped_seed = seed
                break

            tw0 = time.perf_counter()
            wrap_rec, w_sec = run_single_episode(seed, p1_params, wrapper)
            wrap_eval_sec += time.perf_counter() - tw0
            wrapper_accum_sec += w_sec

            if base_rec["success"] and wrap_rec["success"]:
                cat = "both_succeed"
            elif base_rec["success"] and not wrap_rec["success"]:
                cat = "baseline_only_success"
            elif not base_rec["success"] and wrap_rec["success"]:
                cat = "wrapper_only_success"
            else:
                cat = "both_fail"

            pair_rec = {
                "seed": seed,
                "mission": base_rec["mission"],
                "paired_category": cat,
                "baseline": base_rec,
                "wrapped": wrap_rec,
            }
            base_records.append(base_rec)
            wrap_records.append(wrap_rec)
            paired_records.append(pair_rec)
            jf.write(json.dumps(pair_rec) + "\n")
            jf.flush()

            if base_rec["exception"] is not None or wrap_rec["exception"] is not None:
                stopped_reason = "unexpected_episode_exception"
                stopped_seed = seed
                break

    total_sec = round(time.monotonic() - t0, 4)

    if stopped_reason is not None:
        partial_out = {
            "status": f"halted_{stopped_reason}",
            "complete": False,
            "stopped_at_seed": stopped_seed,
            "stopped_reason": stopped_reason,
            "planned_seed_count": len(planned_seeds),
            "completed_paired_count": min(len(base_records), len(wrap_records)),
            "lock_verification": lock_report,
            "pre_run_tests": {
                "wrapper_unit_tests": wrapper_tests,
                "statistical_unit_tests": stats_tests,
            },
            "runtime": {
                "cpu_only": True,
                "wall_clock_cap_seconds": cap_sec,
                "base_eval_seconds": round(base_eval_sec, 4),
                "wrapped_eval_seconds": round(wrap_eval_sec, 4),
                "wrapper_accumulated_call_seconds": round(wrapper_accum_sec, 6),
                "total_seconds": total_sec,
            },
            "partial_paired_records_jsonl": str(jsonl_path),
            "paired_records": paired_records,
        }
        Path("phase5_confirmation_results.json").write_text(
            json.dumps(partial_out, indent=2), encoding="utf-8"
        )
        print(json.dumps(partial_out, indent=2))
        raise SystemExit(1)

    # All 200 pairs completed cleanly
    both_succeed = [p for p in paired_records if p["paired_category"] == "both_succeed"]
    base_only = [p for p in paired_records if p["paired_category"] == "baseline_only_success"]
    wrap_only = [p for p in paired_records if p["paired_category"] == "wrapper_only_success"]
    both_fail = [p for p in paired_records if p["paired_category"] == "both_fail"]

    newcombe_ci = newcombe_paired_method10_ci(
        len(both_succeed), len(base_only), len(wrap_only), len(both_fail)
    )
    mcnemar = exact_mcnemar_test(len(base_only), len(wrap_only))

    baseline_success_pairs = [p for p in paired_records if p["baseline"]["success"]]
    base_succ_with_repeat = [
        p for p in baseline_success_pairs if p["baseline"]["repeated_raw_obs_steps"] > 0
    ]
    base_succ_with_override = [
        p for p in baseline_success_pairs if p["wrapped"]["override_steps"] > 0
    ]
    base_succ_override_preserved = [
        p for p in base_succ_with_override if p["wrapped"]["success"]
    ]
    base_succ_override_regressed = [
        p for p in base_succ_with_override if not p["wrapped"]["success"]
    ]
    base_succ_no_repeat = [
        p for p in baseline_success_pairs if p["baseline"]["repeated_raw_obs_steps"] == 0
    ]
    base_succ_no_repeat_diverged = [
        p["seed"]
        for p in base_succ_no_repeat
        if (
            not p["wrapped"]["success"]
            or p["wrapped"]["override_steps"] != 0
            or p["wrapped"]["episode_length"] != p["baseline"]["episode_length"]
        )
    ]

    results = {
        "status": "completed",
        "complete": True,
        "seed_range": {"start": seed_start, "end_inclusive": seed_end, "count": len(planned_seeds)},
        "lock_verification": lock_report,
        "pre_run_tests": {
            "wrapper_unit_tests": wrapper_tests,
            "statistical_unit_tests": stats_tests,
        },
        "condition_summaries": {
            "baseline_phase1_frozen": summarize_condition(base_records, is_wrapped=False),
            "wrapped_phase1_plus_phase4": summarize_condition(wrap_records, is_wrapped=True),
        },
        "paired_2x2_table": {
            "n_total": len(paired_records),
            "both_succeed": len(both_succeed),
            "baseline_only_success": len(base_only),
            "wrapper_only_success": len(wrap_only),
            "both_fail": len(both_fail),
            "rescue_seeds": [p["seed"] for p in wrap_only],
            "regression_seeds": [p["seed"] for p in base_only],
            "both_fail_seeds": [p["seed"] for p in both_fail],
        },
        "paired_statistical_inference": {
            "newcombe_method10_paired_95_ci": newcombe_ci,
            "exact_mcnemar_test": mcnemar,
        },
        "matched_group_episode_lengths": {
            "both_succeed": {
                "count": len(both_succeed),
                "baseline": length_stats([p["baseline"]["episode_length"] for p in both_succeed]),
                "wrapped": length_stats([p["wrapped"]["episode_length"] for p in both_succeed]),
            },
            "wrapper_only_success_rescued": {
                "count": len(wrap_only),
                "baseline_timeouts": length_stats([p["baseline"]["episode_length"] for p in wrap_only]),
                "wrapped_successes": length_stats([p["wrapped"]["episode_length"] for p in wrap_only]),
            },
            "baseline_only_success_regressions": {
                "count": len(base_only),
                "baseline_successes": length_stats([p["baseline"]["episode_length"] for p in base_only]),
                "wrapped_failures": length_stats([p["wrapped"]["episode_length"] for p in base_only]),
            },
            "both_fail": {
                "count": len(both_fail),
                "baseline": length_stats([p["baseline"]["episode_length"] for p in both_fail]),
                "wrapped": length_stats([p["wrapped"]["episode_length"] for p in both_fail]),
            },
        },
        "baseline_successful_revisit_audit": {
            "baseline_successful_total": len(baseline_success_pairs),
            "baseline_successful_with_repeated_raw_obs_count": len(base_succ_with_repeat),
            "baseline_successful_with_repeated_raw_obs_seeds": [
                p["seed"] for p in base_succ_with_repeat
            ],
            "baseline_successful_with_wrapper_override_count": len(base_succ_with_override),
            "baseline_successful_with_wrapper_override_seeds": [
                p["seed"] for p in base_succ_with_override
            ],
            "override_preserved_success_count": len(base_succ_override_preserved),
            "override_preserved_success_seeds": [
                p["seed"] for p in base_succ_override_preserved
            ],
            "override_regressed_count": len(base_succ_override_regressed),
            "override_regressed_seeds": [
                p["seed"] for p in base_succ_override_regressed
            ],
            "non_repeating_baseline_success_count": len(base_succ_no_repeat),
            "non_repeating_baseline_success_divergences": base_succ_no_repeat_diverged,
        },
        "runtime": {
            "cpu_only": True,
            "wall_clock_cap_seconds": cap_sec,
            "timing_procedure": "Pairwise synchronous perf_counter() timing Baseline and Wrapped loops per seed on seeds 30000..30199, plus accumulated perf_counter() inside wrapper.reset() and wrapper.select_action(). Shorter total wrapped runtime reflects fewer steps on rescued episodes, not negative wrapper overhead.",
            "base_eval_seconds": round(base_eval_sec, 4),
            "wrapped_eval_seconds": round(wrap_eval_sec, 4),
            "wrapper_accumulated_call_seconds": round(wrapper_accum_sec, 6),
            "total_seconds": total_sec,
            "completed_within_cap": total_sec < cap_sec,
        },
        "paired_records": paired_records,
    }

    Path("phase5_confirmation_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    # Print compact summary (exclude verbose 200-item paired_records array from stdout)
    compact = {k: v for k, v in results.items() if k != "paired_records"}
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
