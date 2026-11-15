"""Phase 4: Evaluate frozen Phase 1 policy wrapped with ExternalObservationCountWrapper on dev 0..99."""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import gymnasium as gym
import numpy as np

from external_wrapper import ExternalObservationCountWrapper, run_wrapper_unit_tests
from run_phase1 import ACTION_NAMES, encode_observation, forward_mlp, hash_bytes, verify_feature_assertions
from run_phase1_continuation import evaluate_frozen_argmax

EXPECTED_PHASE1_WEIGHTS_SHA256 = (
    "e31f335039c27dc2b41a937a82d8a52e062828e15c3e049f8f470c614a6a19fd"
)
WALL_CLOCK_CAP_SEC = 300.0
PROB_TOL = 1e-6
FAILED_PHASE1_SEEDS = [3, 15, 29, 43, 50, 80, 94]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_wrapped_rollouts(
    seeds: list[int],
    params: dict[str, np.ndarray],
    deadline: float,
) -> tuple[dict, float, bool]:
    """Run dev seeds 0..99 with strict separation between policy, wrapper, and evaluator."""
    wrapper = ExternalObservationCountWrapper()
    episodes = []
    timed_out_wall_clock = False
    wrapper_accum_sec = 0.0

    for seed in seeds:
        if time.monotonic() >= deadline:
            timed_out_wall_clock = True
            break

        env = gym.make("BabyAI-GoToLocal-v0")
        obs, _ = env.reset(seed=seed)

        tw0 = time.perf_counter()
        wrapper.reset()
        wrapper_accum_sec += time.perf_counter() - tw0

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
        override_steps = 0

        try:
            while not (terminated or truncated):
                feat, meta = encode_observation(obs)
                feat_hash = hash_bytes(feat.tobytes())
                raw_hash = hash_bytes(
                    obs["image"].tobytes() + obs["mission"].encode("utf-8")
                )

                pos_x = int(env.unwrapped.agent_pos[0])
                pos_y = int(env.unwrapped.agent_pos[1])
                heading = int(env.unwrapped.agent_dir)
                pose = (pos_x, pos_y, heading)

                _, _, probs = forward_mlp(feat[None, :], params)
                probs_1d = probs[0]
                if not np.all(np.isfinite(probs_1d)):
                    raise FloatingPointError(f"Non-finite policy probs at seed={seed}")

                # Narrow information boundary: wrapper sees ONLY obs['image'], obs['mission'], probs
                tw0 = time.perf_counter()
                action, w_meta = wrapper.select_action(
                    obs["image"], obs["mission"], probs_1d
                )
                wrapper_accum_sec += time.perf_counter() - tw0

                if w_meta["overridden"]:
                    override_steps += 1

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
                            "cycle_poses": [
                                [*s["agent_pos"], s["agent_dir"]]
                                for s in steps[prev_step:]
                            ],
                            "cycle_actions": [
                                s["chosen_action_name"] for s in steps[prev_step:]
                            ],
                            "cycle_base_argmax_actions": [
                                ACTION_NAMES[s["base_argmax"]] for s in steps[prev_step:]
                            ],
                            "cycle_target_in_view_flags": [
                                s["target_in_view"] for s in steps[prev_step:]
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
                        "feat_hash": feat_hash,
                        "raw_obs_hash": raw_hash,
                        "probs": [round(float(p), 6) for p in probs_1d],
                        "base_argmax": w_meta["base_argmax"],
                        "chosen_action": action,
                        "chosen_action_name": ACTION_NAMES[action],
                        "overridden": w_meta["overridden"],
                        "prior_counts": w_meta["prior_counts"],
                        "all_three_ge_one_before": w_meta[
                            "all_three_ge_one_before_decision"
                        ],
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

        w_stats = wrapper.episode_stats()
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
                "override_steps": override_steps,
                "retained_obs_keys": w_stats["retained_obs_keys"],
                "post_full_coverage_decisions": w_stats["post_full_coverage_decisions"],
                "distinct_keys_with_post_full_coverage_decision": w_stats[
                    "distinct_keys_with_post_full_coverage_decision"
                ],
                "distinct_keys_reaching_all_three_counts_ge_1_by_end": w_stats[
                    "distinct_keys_reaching_all_three_counts_ge_1_by_end"
                ],
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
        "total_blocked_forward_steps": sum(
            e["blocked_forward_steps"] for e in episodes
        ),
        "first_pose_cycle_length_histogram": {
            str(k): int(v) for k, v in sorted(cycle_lengths.items())
        },
        "pose_cycle_episodes_target_never_visible_in_cycle": sum(
            1
            for e in episodes
            if e["first_pose_cycle"] is not None
            and not any(e["first_pose_cycle"]["cycle_target_in_view_flags"])
        ),
        "pose_cycle_episodes_target_visible_in_cycle": sum(
            1
            for e in episodes
            if e["first_pose_cycle"] is not None
            and any(e["first_pose_cycle"]["cycle_target_in_view_flags"])
        ),
        "intervention_burden": {
            "episodes_with_at_least_one_override": sum(
                1 for e in episodes if e["override_steps"] > 0
            ),
            "episodes_with_override_seeds": [
                e["seed"] for e in episodes if e["override_steps"] > 0
            ],
            "total_overridden_actions": sum(e["override_steps"] for e in episodes),
            "total_decisions_all_episodes": sum(
                e["episode_length"] for e in episodes
            ),
            "mean_retained_obs_keys_per_episode": (
                round(sum(e["retained_obs_keys"] for e in episodes) / n_eps, 4)
                if n_eps
                else 0.0
            ),
            "peak_retained_obs_keys_in_episode": max(
                (e["retained_obs_keys"] for e in episodes), default=0
            ),
            "peak_retained_obs_keys_seed": (
                max(episodes, key=lambda e: e["retained_obs_keys"])["seed"]
                if episodes
                else None
            ),
            "post_full_coverage_decisions_total": sum(
                e["post_full_coverage_decisions"] for e in episodes
            ),
            "distinct_obs_keys_with_post_full_coverage_decision_total": sum(
                e["distinct_keys_with_post_full_coverage_decision"] for e in episodes
            ),
            "distinct_obs_keys_reaching_all_three_ge_1_by_end_total": sum(
                e["distinct_keys_reaching_all_three_counts_ge_1_by_end"]
                for e in episodes
            ),
        },
    }
    return {"summary": summary, "episodes": episodes}, wrapper_accum_sec, timed_out_wall_clock


def verify_baseline_success_invariant(
    wrapped_episodes: list[dict], saved_by_seed: dict[int, dict]
) -> dict:
    """Check step-for-step trajectory invariance on all 93 Phase 1 baseline successes."""
    baseline_success_seeds = sorted(
        s for s, ep in saved_by_seed.items() if ep["success"]
    )
    wrapped_by_seed = {ep["seed"]: ep for ep in wrapped_episodes}
    divergences = []

    for s in baseline_success_seeds:
        ref = saved_by_seed[s]
        cur = wrapped_by_seed[s]
        first_div = None
        if cur["override_steps"] != 0:
            first_div = {
                "reason": "non_zero_override_steps",
                "override_steps": cur["override_steps"],
            }
        elif not cur["success"] or cur["episode_length"] != ref["episode_length"]:
            first_div = {
                "reason": "outcome_or_length_mismatch",
                "cur_success": cur["success"],
                "ref_length": ref["episode_length"],
                "cur_length": cur["episode_length"],
            }
        else:
            for st_cur, st_ref in zip(cur["steps"], ref["steps"]):
                prob_diff = max(
                    abs(a - b) for a, b in zip(st_cur["probs"], st_ref["probs"])
                )
                if (
                    st_cur["agent_pos"] != st_ref["agent_pos"]
                    or st_cur["agent_dir"] != st_ref["agent_dir"]
                    or st_cur["chosen_action"] != st_ref["chosen_action"]
                    or prob_diff > PROB_TOL
                ):
                    first_div = {
                        "reason": "step_mismatch",
                        "step": st_cur["step"],
                        "cur_pose": [*st_cur["agent_pos"], st_cur["agent_dir"]],
                        "ref_pose": [*st_ref["agent_pos"], st_ref["agent_dir"]],
                        "cur_action": st_cur["chosen_action"],
                        "ref_action": st_ref["chosen_action"],
                        "prob_diff": prob_diff,
                    }
                    break
        if first_div is not None:
            divergences.append({"seed": s, **first_div})

    return {
        "baseline_success_count": len(baseline_success_seeds),
        "all_baseline_successes_trajectory_identical": len(divergences) == 0,
        "divergence_count": len(divergences),
        "divergences": divergences,
    }


def main() -> None:
    t0 = time.monotonic()
    deadline = t0 + WALL_CLOCK_CAP_SEC

    unit_tests = run_wrapper_unit_tests()
    verify_feature_assertions()

    weights_path = Path("phase1_weights.npz")
    weights_sha = sha256_file(weights_path)
    if weights_sha != EXPECTED_PHASE1_WEIGHTS_SHA256:
        raise RuntimeError(
            f"Unexpected phase1_weights.npz sha256: {weights_sha} != {EXPECTED_PHASE1_WEIGHTS_SHA256}"
        )

    p1_npz = np.load(weights_path)
    p1_params = {k: p1_npz[k] for k in p1_npz.files}
    saved_phase1_eval = json.loads(
        Path("phase1_eval_results.json").read_text(encoding="utf-8")
    )
    saved_by_seed = {ep["seed"]: ep for ep in saved_phase1_eval["episodes"]}
    dev_seeds = list(range(0, 100))

    t_base_start = time.perf_counter()
    base_eval, base_timeout = evaluate_frozen_argmax(dev_seeds, p1_params, deadline)
    t_base_sec = round(time.perf_counter() - t_base_start, 4)

    t_wrap_start = time.perf_counter()
    wrap_eval, wrapper_accum_sec, wrap_timeout = evaluate_wrapped_rollouts(
        dev_seeds, p1_params, deadline
    )
    t_wrap_sec = round(time.perf_counter() - t_wrap_start, 4)

    if base_timeout or wrap_timeout:
        partial = {
            "status": "halted_by_wall_clock_cap",
            "unit_tests": unit_tests,
            "base_eval_seconds": t_base_sec,
            "wrapped_eval_seconds": t_wrap_sec,
            "episodes_completed_wrapped": len(wrap_eval["episodes"]),
        }
        Path("phase4_results.json").write_text(
            json.dumps(partial, indent=2), encoding="utf-8"
        )
        print(json.dumps(partial, indent=2))
        raise SystemExit(1)

    baseline_invariant = verify_baseline_success_invariant(
        wrap_eval["episodes"], saved_by_seed
    )

    wrapped_eps = wrap_eval["episodes"]
    rescued = sorted(
        e["seed"]
        for e in wrapped_eps
        if e["success"] and not saved_by_seed[e["seed"]]["success"]
    )
    harmed = sorted(
        e["seed"]
        for e in wrapped_eps
        if not e["success"] and saved_by_seed[e["seed"]]["success"]
    )

    base_succ_lens = [e["episode_length"] for e in base_eval["episodes"] if e["success"]]
    wrap_succ_lens = [e["episode_length"] for e in wrapped_eps if e["success"]]
    common_success_seeds = sorted(
        e["seed"]
        for e in wrapped_eps
        if e["success"] and saved_by_seed[e["seed"]]["success"]
    )
    common_set = set(common_success_seeds)

    # Compact per-episode details for the 7 original Phase 1 failure seeds + any harmed
    focus_seeds = sorted(set(FAILED_PHASE1_SEEDS) | set(harmed))
    focus_outcomes = {}
    wrap_by_seed = {ep["seed"]: ep for ep in wrapped_eps}
    for s in focus_seeds:
        ew = wrap_by_seed[s]
        eb = saved_by_seed[s]
        focus_outcomes[str(s)] = {
            "seed": s,
            "mission": ew["mission"],
            "phase1_success": eb["success"],
            "phase1_length": eb["episode_length"],
            "phase1_cycle_length": (
                eb["first_pose_cycle"]["cycle_length"]
                if eb["first_pose_cycle"] is not None
                else None
            ),
            "phase1_blocked_forward_steps": eb["blocked_forward_steps"],
            "wrapped_success": ew["success"],
            "wrapped_truncated": ew["truncated"],
            "wrapped_exception": ew["exception"],
            "wrapped_length": ew["episode_length"],
            "wrapped_override_steps": ew["override_steps"],
            "wrapped_retained_obs_keys": ew["retained_obs_keys"],
            "wrapped_post_full_coverage_decisions": ew[
                "post_full_coverage_decisions"
            ],
            "wrapped_distinct_keys_with_post_full_coverage": ew[
                "distinct_keys_with_post_full_coverage_decision"
            ],
            "wrapped_distinct_keys_reaching_all_three_by_end": ew[
                "distinct_keys_reaching_all_three_counts_ge_1_by_end"
            ],
            "wrapped_blocked_forward_steps": ew["blocked_forward_steps"],
            "wrapped_first_pose_cycle": ew["first_pose_cycle"],
            "wrapped_first_16_steps": [
                {
                    "step": st["step"],
                    "pose": [*st["agent_pos"], st["agent_dir"]],
                    "base": ACTION_NAMES[st["base_argmax"]],
                    "chosen": st["chosen_action_name"],
                    "overridden": st["overridden"],
                    "prior_counts": st["prior_counts"],
                    "front_blocked": st["front_blocked"],
                    "target_in_view": st["target_in_view"],
                }
                for st in ew["steps"][:16]
            ],
        }

    total_sec = round(time.monotonic() - t0, 4)

    results = {
        "status": "completed",
        "unit_tests": unit_tests,
        "baseline_verification": {
            "phase1_weights_sha256": weights_sha,
            "baseline_93_success_invariant": baseline_invariant,
        },
        "phase1_baseline_summary": {
            "success_count": base_eval["summary"]["success_count"],
            "timeout_count": base_eval["summary"]["timeout_count"],
            "exception_count": base_eval["summary"]["exception_count"],
            "mean_episode_length_all": base_eval["summary"]["mean_episode_length"],
            "mean_episode_length_successful_only": round(
                sum(base_succ_lens) / len(base_succ_lens), 4
            ),
            "episodes_with_repeated_feature_obs": base_eval["summary"][
                "episodes_with_repeated_feature_obs"
            ],
            "episodes_with_repeated_raw_obs": base_eval["summary"][
                "episodes_with_repeated_raw_obs"
            ],
            "episodes_with_repeated_pose": base_eval["summary"][
                "episodes_with_repeated_pose"
            ],
            "episodes_with_any_blocked_forward": base_eval["summary"][
                "episodes_with_any_blocked_forward"
            ],
            "total_blocked_forward_steps": base_eval["summary"][
                "total_blocked_forward_steps"
            ],
        },
        "phase4_wrapped_summary": {
            **wrap_eval["summary"],
            "mean_episode_length_successful_only": (
                round(sum(wrap_succ_lens) / len(wrap_succ_lens), 4)
                if wrap_succ_lens
                else None
            ),
            "rescued_vs_phase1_count": len(rescued),
            "rescued_vs_phase1_seeds": rescued,
            "harmed_vs_phase1_count": len(harmed),
            "harmed_vs_phase1_seeds": harmed,
            "remaining_failure_seeds": sorted(
                e["seed"] for e in wrapped_eps if not e["success"]
            ),
        },
        "matched_length_comparisons": {
            "common_success_count": len(common_success_seeds),
            "phase1_mean_length_on_common": round(
                sum(
                    saved_by_seed[s]["episode_length"] for s in common_success_seeds
                )
                / len(common_success_seeds),
                4,
            ),
            "wrapped_mean_length_on_common": round(
                sum(wrap_by_seed[s]["episode_length"] for s in common_success_seeds)
                / len(common_success_seeds),
                4,
            ),
            "rescued_episode_lengths_wrapped": {
                str(s): wrap_by_seed[s]["episode_length"] for s in rescued
            },
        },
        "focus_seeds_outcomes_and_traces": focus_outcomes,
        "runtime": {
            "cpu_only": True,
            "wall_clock_cap_seconds": WALL_CLOCK_CAP_SEC,
            "timing_procedure": "Single synchronous run measuring wall-clock perf_counter() around unwrapped Phase 1 dev 0..99 loop, wrapped dev 0..99 loop, and accumulated time inside wrapper.reset()/wrapper.select_action().",
            "base_eval_seconds": t_base_sec,
            "wrapped_eval_seconds": t_wrap_sec,
            "wrapper_accumulated_call_seconds": round(wrapper_accum_sec, 6),
            "total_seconds": total_sec,
            "completed_within_cap": total_sec < WALL_CLOCK_CAP_SEC,
        },
    }

    Path("phase4_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
