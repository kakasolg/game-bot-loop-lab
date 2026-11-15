"""Single-shot held-out confirmation runner for action-mask ablation (seeds 50000..50499)."""

from __future__ import annotations

import os

for _env_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_env_var, "1")

import gc
import json
import platform
import sys
import time
from pathlib import Path

import gymnasium as gym
import minigrid
import numpy as np

from experiments.action_mask_ablation.common import (
    compute_paired_contrast,
    holm_bonferroni_adjust,
    length_stats,
    run_single_episode,
    sha256_file,
    summarize_arm_records,
    summarize_ns_distribution,
)
from experiments.action_mask_ablation.mask_selector import (
    select_action_with_blocked_forward_mask,
)
from experiments.action_mask_ablation.validate_dev import (
    run_mask_unit_tests,
    verify_live_front_coordinate_alignment,
)
from external_wrapper import ExternalObservationCountWrapper, run_wrapper_unit_tests
from phase5_stats import run_statistical_unit_tests
from run_phase1 import encode_observation, forward_mlp, verify_feature_assertions

EXP_DIR = Path("experiments/action_mask_ablation")
OUT_DIR = EXP_DIR / "outputs"
ARM_ROTATIONS = (
    ("A", "B", "C"),
    ("B", "C", "A"),
    ("C", "A", "B"),
)


def verify_freeze_lock() -> tuple[bool, dict]:
    """Verify frozen_plan.json, policy weights, wrapper, stats, and comparator hashes before confirmation."""
    lock = json.loads((EXP_DIR / "freeze_lock.json").read_text(encoding="utf-8"))
    expected_hashes: dict[str, str] = lock["locked_sha256"]
    observed_hashes: dict[str, str] = {}
    hash_mismatches: list[dict] = []

    for rel_path, exp_sha in expected_hashes.items():
        p = Path(rel_path)
        if not p.exists():
            hash_mismatches.append({"file": rel_path, "expected": exp_sha, "observed": "MISSING"})
            continue
        obs_sha = sha256_file(p)
        observed_hashes[rel_path] = obs_sha
        if obs_sha != exp_sha:
            hash_mismatches.append({"file": rel_path, "expected": exp_sha, "observed": obs_sha})

    expected_versions: dict[str, str] = lock["locked_versions"]
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
        "freeze_lock_sha256": sha256_file(EXP_DIR / "freeze_lock.json"),
        "observed_hashes": observed_hashes,
        "hash_mismatches": hash_mismatches,
        "observed_versions": observed_versions,
        "version_mismatches": version_mismatches,
    }


def build_eight_cell_table(
    seeds: list[int],
    recs_a: list[dict],
    recs_b: list[dict],
    recs_c: list[dict],
) -> dict:
    """Build full 2x2x2 (A, B, C) outcome table and matched episode length stats."""
    cell_keys = [
        "000", "001", "010", "011",
        "100", "101", "110", "111",
    ]
    buckets: dict[str, list[tuple[int, dict, dict, dict]]] = {k: [] for k in cell_keys}
    for seed, ra, rb, rc in zip(seeds, recs_a, recs_b, recs_c, strict=True):
        key = f"{int(ra['success'])}{int(rb['success'])}{int(rc['success'])}"
        buckets[key].append((seed, ra, rb, rc))

    cells: dict[str, dict] = {}
    for key in cell_keys:
        items = buckets[key]
        cells[key] = {
            "count": len(items),
            "pattern_A_B_C": [int(key[0]), int(key[1]), int(key[2])],
            "seeds": [x[0] for x in items] if key != "111" else f"({len(items)} all-succeed seeds omitted for brevity; see jsonl)",
            "episode_lengths": {
                "A": length_stats([x[1]["episode_length"] for x in items]),
                "B": length_stats([x[2]["episode_length"] for x in items]),
                "C": length_stats([x[3]["episode_length"] for x in items]),
            },
        }
    return cells


def select_diagnostic_trace_seeds(
    seeds: list[int],
    recs_a: list[dict],
    recs_b: list[dict],
    recs_c: list[dict],
    max_per_stratum: int = 3,
) -> list[dict]:
    """Prespecified <=15-seed inspection rule in ascending seed order."""
    strata: dict[str, list[int]] = {
        "regressions": [],
        "A0_B1_C1": [],
        "A0_B0_C1": [],
        "A0_B1_C0": [],
        "A0_B0_C0": [],
    }
    for seed, ra, rb, rc in zip(seeds, recs_a, recs_b, recs_c, strict=True):
        sa, sb, sc = bool(ra["success"]), bool(rb["success"]), bool(rc["success"])
        if sa and (not sb or not sc):
            strata["regressions"].append(seed)
        elif not sa and sb and sc:
            strata["A0_B1_C1"].append(seed)
        elif not sa and not sb and sc:
            strata["A0_B0_C1"].append(seed)
        elif not sa and sb and not sc:
            strata["A0_B1_C0"].append(seed)
        elif not sa and not sb and not sc:
            strata["A0_B0_C0"].append(seed)

    selected: list[dict] = []
    seen_seeds: set[int] = set()
    for stratum_name in ("regressions", "A0_B1_C1", "A0_B0_C1", "A0_B1_C0", "A0_B0_C0"):
        for seed in sorted(strata[stratum_name])[:max_per_stratum]:
            if seed not in seen_seeds:
                seen_seeds.add(seed)
                selected.append({"stratum": stratum_name, "seed": seed})
    return selected


def run_off_policy_fixed_trace_timing(
    recorded_arm_a_episodes: list[list[tuple[np.ndarray, str]]],
    params: dict[str, np.ndarray],
    wrapper: ExternalObservationCountWrapper,
    warmup_reps: int = 1,
    timed_reps: int = 5,
) -> dict:
    """Measure isolated fixed-input e2e and execution-layer latency over recorded Arm A traces.

    Explicitly off-policy fixed-input cost measurement only (NOT a valid Arm C rollout).
    Arm C's wrapper.reset() is called at each replayed episode boundary so within-episode
    memory evolves consistently across each episode during cost measurement.
    """
    total_steps = sum(len(ep) for ep in recorded_arm_a_episodes)
    n_episodes = len(recorded_arm_a_episodes)

    # Precompute probs outside timed execution-layer loops so Pass 2b times execution layer alone
    precomputed_episodes: list[list[tuple[np.ndarray, str, np.ndarray]]] = []
    for ep in recorded_arm_a_episodes:
        ep_pre: list[tuple[np.ndarray, str, np.ndarray]] = []
        for img, mission in ep:
            feat, _ = encode_observation({"image": img, "mission": mission})
            _, _, probs = forward_mlp(feat[None, :], params)
            probs_1d = probs[0].copy()
            ep_pre.append((img, mission, probs_1d))
        precomputed_episodes.append(ep_pre)

    # Pass 2a: Fixed-trace end-to-end decision latency (encode_observation + forward_mlp + selection)
    e2e_ns: dict[str, list[int]] = {"A": [], "B": [], "C": []}
    e2e_reset_ns_c: list[int] = []

    for rep in range(-warmup_reps, timed_reps):
        collect = rep >= 0
        for arm in ("A", "B", "C"):
            for ep in recorded_arm_a_episodes:
                if arm == "C":
                    tr0 = time.perf_counter_ns()
                    wrapper.reset()
                    tr1 = time.perf_counter_ns()
                    if collect:
                        e2e_reset_ns_c.append(tr1 - tr0)

                for img, mission in ep:
                    t0 = time.perf_counter_ns()
                    feat, _ = encode_observation({"image": img, "mission": mission})
                    _, _, probs = forward_mlp(feat[None, :], params)
                    probs_1d = probs[0]
                    if arm == "A":
                        _ = int(np.argmax(probs_1d))
                    elif arm == "B":
                        _, _ = select_action_with_blocked_forward_mask(img, mission, probs_1d)
                    else:
                        _, _ = wrapper.select_action(img, mission, probs_1d)
                    t1 = time.perf_counter_ns()
                    if collect:
                        e2e_ns[arm].append(t1 - t0)

    # Pass 2b: Fixed-trace isolated execution-layer latency (separate pass, no nested timers)
    exec_ns: dict[str, list[int]] = {"A": [], "B": [], "C": []}
    exec_reset_ns_c: list[int] = []

    for rep in range(-warmup_reps, timed_reps):
        collect = rep >= 0
        for arm in ("A", "B", "C"):
            for ep_pre in precomputed_episodes:
                if arm == "C":
                    tr0 = time.perf_counter_ns()
                    wrapper.reset()
                    tr1 = time.perf_counter_ns()
                    if collect:
                        exec_reset_ns_c.append(tr1 - tr0)

                for img, mission, probs_1d in ep_pre:
                    t0 = time.perf_counter_ns()
                    if arm == "A":
                        _ = int(np.argmax(probs_1d))
                    elif arm == "B":
                        _, _ = select_action_with_blocked_forward_mask(img, mission, probs_1d)
                    else:
                        _, _ = wrapper.select_action(img, mission, probs_1d)
                    t1 = time.perf_counter_ns()
                    if collect:
                        exec_ns[arm].append(t1 - t0)

    return {
        "measurement_type": "off_policy_fixed_input_cost_measurement_NOT_valid_arm_c_rollout",
        "trace_source": "recorded Arm A episodes from seeds 50000..50499",
        "trace_episodes": n_episodes,
        "trace_steps_per_pass": total_steps,
        "warmup_repetitions": warmup_reps,
        "timed_repetitions": timed_reps,
        "achieved_samples_per_arm": total_steps * timed_reps,
        "pass2a_fixed_trace_decision_e2e_us": {
            arm: summarize_ns_distribution(e2e_ns[arm]) for arm in ("A", "B", "C")
        },
        "pass2a_wrapper_reset_us": summarize_ns_distribution(e2e_reset_ns_c),
        "pass2b_fixed_trace_exec_layer_only_us": {
            arm: summarize_ns_distribution(exec_ns[arm]) for arm in ("A", "B", "C")
        },
        "pass2b_wrapper_reset_us": summarize_ns_distribution(exec_reset_ns_c),
    }


def main() -> None:
    t_start = time.monotonic()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plan = json.loads((EXP_DIR / "frozen_plan.json").read_text(encoding="utf-8"))
    lock_ok, lock_report = verify_freeze_lock()
    if not lock_ok:
        err_out = {
            "status": "halted_freeze_lock_verification_failed",
            "lock_verification": lock_report,
        }
        (OUT_DIR / "confirmation_results.json").write_text(
            json.dumps(err_out, indent=2), encoding="utf-8"
        )
        print(json.dumps(err_out, indent=2))
        raise SystemExit(1)

    # Pre-confirmation unit checks
    verify_feature_assertions()
    wrap_tests = run_wrapper_unit_tests()
    stat_tests = run_statistical_unit_tests()
    mask_tests = run_mask_unit_tests()
    coord_check = verify_live_front_coordinate_alignment()

    p1_npz = np.load("phase1_weights.npz")
    params = {k: p1_npz[k] for k in p1_npz.files}
    wrapper = ExternalObservationCountWrapper()

    # Untimed warm-up sweep on dev seeds 0..99 before timed measurements
    dev_cfg = plan["seeds"]["dev_validation"]
    for dev_seed in range(int(dev_cfg["start"]), int(dev_cfg["end_inclusive"]) + 1):
        for arm in ("A", "B", "C"):
            _ = run_single_episode(
                dev_seed,
                params,
                arm,
                wrapper=wrapper if arm == "C" else None,
                record_e2e_ns=False,
            )

    conf_cfg = plan["seeds"]["confirmation"]
    seed_start = int(conf_cfg["start"])
    seed_end = int(conf_cfg["end_inclusive"])
    planned_seeds = list(range(seed_start, seed_end + 1))
    cap_sec = float(plan["statistical_protocol"]["safety_wall_clock_cap_seconds"])
    deadline = t_start + cap_sec

    recs_by_arm: dict[str, list[dict]] = {"A": [], "B": [], "C": []}
    live_e2e_ns: dict[str, list[int]] = {"A": [], "B": [], "C": []}
    live_reset_ns_c: list[int] = []
    recorded_arm_a_episodes: list[list[tuple[np.ndarray, str]]] = []
    step_traces_by_seed: dict[int, dict[str, list[dict]]] = {}
    paired_jsonl_rows: list[dict] = []

    stopped_reason: str | None = None
    stopped_seed: int | None = None

    jsonl_path = OUT_DIR / "confirmation_episodes.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as jf:
        for idx, seed in enumerate(planned_seeds):
            if time.monotonic() >= deadline:
                stopped_reason = "safety_wall_clock_cap_reached_before_seed"
                stopped_seed = seed
                break

            arm_order = ARM_ROTATIONS[idx % 3]
            per_seed_arm_records: dict[str, dict] = {}
            per_seed_arm_traces: dict[str, list[dict]] = {}
            seed_exc = False

            for arm in arm_order:
                res = run_single_episode(
                    seed=seed,
                    params=params,
                    arm=arm,
                    wrapper=wrapper if arm == "C" else None,
                    record_e2e_ns=True,
                    capture_raw_obs_for_replay=(arm == "A"),
                    capture_step_traces=True,
                )
                rec = res["record"]
                per_seed_arm_records[arm] = rec
                per_seed_arm_traces[arm] = res["step_traces"]
                live_e2e_ns[arm].extend(res["e2e_ns_samples"])
                if arm == "C" and res["wrapper_reset_ns"] is not None:
                    live_reset_ns_c.append(res["wrapper_reset_ns"])
                if arm == "A":
                    recorded_arm_a_episodes.append(res["replay_obs_seq"])
                if rec["exception"] is not None:
                    seed_exc = True

            pair_row = {
                "seed": seed,
                "arm_execution_order": list(arm_order),
                "mission": per_seed_arm_records["A"]["mission"],
                "outcomes_A_B_C": {
                    "A": per_seed_arm_records["A"]["outcome"],
                    "B": per_seed_arm_records["B"]["outcome"],
                    "C": per_seed_arm_records["C"]["outcome"],
                },
                "success_A_B_C": {
                    "A": per_seed_arm_records["A"]["success"],
                    "B": per_seed_arm_records["B"]["success"],
                    "C": per_seed_arm_records["C"]["success"],
                },
                "lengths_A_B_C": {
                    "A": per_seed_arm_records["A"]["episode_length"],
                    "B": per_seed_arm_records["B"]["episode_length"],
                    "C": per_seed_arm_records["C"]["episode_length"],
                },
                "arms": {
                    "A": per_seed_arm_records["A"],
                    "B": per_seed_arm_records["B"],
                    "C": per_seed_arm_records["C"],
                },
            }
            jf.write(json.dumps(pair_row) + "\n")
            jf.flush()
            paired_jsonl_rows.append(pair_row)

            for arm in ("A", "B", "C"):
                recs_by_arm[arm].append(per_seed_arm_records[arm])
            step_traces_by_seed[seed] = per_seed_arm_traces

            if seed_exc:
                stopped_reason = "unexpected_episode_exception"
                stopped_seed = seed
                break

    if stopped_reason is not None:
        partial_out = {
            "status": f"halted_{stopped_reason}",
            "complete": False,
            "stopped_at_seed": stopped_seed,
            "completed_seeds": len(recs_by_arm["A"]),
            "lock_verification": lock_report,
            "partial_jsonl_path": str(jsonl_path),
            "total_elapsed_seconds": round(time.monotonic() - t_start, 4),
        }
        (OUT_DIR / "confirmation_results.json").write_text(
            json.dumps(partial_out, indent=2), encoding="utf-8"
        )
        print(json.dumps(partial_out, indent=2))
        raise SystemExit(1)

    pass1_elapsed_sec = round(time.monotonic() - t_start, 4)

    # Pass 2: Off-policy fixed-trace replay timing sweeps
    fixed_trace_timing = run_off_policy_fixed_trace_timing(
        recorded_arm_a_episodes=recorded_arm_a_episodes,
        params=params,
        wrapper=wrapper,
        warmup_reps=int(plan["timing_protocol"]["pass2_off_policy_fixed_trace"]["warmup_repetitions"]),
        timed_reps=int(plan["timing_protocol"]["pass2_off_policy_fixed_trace"]["timed_repetitions"]),
    )

    total_elapsed_sec = round(time.monotonic() - t_start, 4)

    # Statistical summaries & paired inferences
    recs_a, recs_b, recs_c = recs_by_arm["A"], recs_by_arm["B"], recs_by_arm["C"]
    primary_c_vs_b = compute_paired_contrast(planned_seeds, recs_b, recs_c, "B", "C")
    secondary_b_vs_a = compute_paired_contrast(planned_seeds, recs_a, recs_b, "A", "B")
    secondary_c_vs_a = compute_paired_contrast(planned_seeds, recs_a, recs_c, "A", "C")

    supplementary_holm = holm_bonferroni_adjust(
        [
            ("C_vs_B_primary", primary_c_vs_b["exact_mcnemar_test"]["p_value"]),
            ("B_vs_A_secondary", secondary_b_vs_a["exact_mcnemar_test"]["p_value"]),
            ("C_vs_A_secondary", secondary_c_vs_a["exact_mcnemar_test"]["p_value"]),
        ]
    )

    eight_cell = build_eight_cell_table(planned_seeds, recs_a, recs_b, recs_c)
    selected_trace_specs = select_diagnostic_trace_seeds(
        planned_seeds,
        recs_a,
        recs_b,
        recs_c,
        max_per_stratum=int(plan["trace_selection_rule"]["max_per_stratum"]),
    )
    diagnostic_traces = []
    for item in selected_trace_specs:
        s = item["seed"]
        idx_s = s - seed_start
        diagnostic_traces.append(
            {
                "stratum": item["stratum"],
                "seed": s,
                "mission": recs_a[idx_s]["mission"],
                "summary_A_B_C": {
                    "A": {k: recs_a[idx_s][k] for k in ("outcome", "episode_length", "blocked_forward_steps", "first_pose_cycle_length")},
                    "B": {k: recs_b[idx_s][k] for k in ("outcome", "episode_length", "blocked_forward_steps", "mask_exclusions", "mask_action_changes", "first_pose_cycle_length")},
                    "C": {k: recs_c[idx_s][k] for k in ("outcome", "episode_length", "blocked_forward_steps", "override_steps", "retained_obs_keys", "first_pose_cycle_length")},
                },
                "trajectories": step_traces_by_seed[s],
            }
        )

    clock_info = time.get_clock_info("perf_counter")
    results = {
        "status": "completed",
        "complete": True,
        "plan_version": plan["plan_version"],
        "seed_range": {"start": seed_start, "end_inclusive": seed_end, "count": len(planned_seeds)},
        "lock_verification": lock_report,
        "pre_run_unit_tests": {
            "feature_assertions_passed": True,
            "wrapper_unit_tests": wrap_tests,
            "statistical_unit_tests": stat_tests,
            "mask_unit_tests": mask_tests,
            "coordinate_alignment": coord_check,
        },
        "arm_summaries": {
            "A_frozen_baseline": summarize_arm_records(recs_a, "A"),
            "B_blocked_forward_mask": summarize_arm_records(recs_b, "B"),
            "C_action_count_wrapper": summarize_arm_records(recs_c, "C"),
        },
        "primary_comparison_C_vs_B": primary_c_vs_b,
        "secondary_exploratory_comparisons": {
            "B_vs_A": secondary_b_vs_a,
            "C_vs_A": secondary_c_vs_a,
        },
        "supplementary_familywise_holm_bonferroni": {
            "note": "Supplementary 3-contrast Holm-Bonferroni adjustment separated from prespecified primary C_vs_B test.",
            "adjusted_p_values": supplementary_holm,
        },
        "joint_eight_cell_table_A_B_C": eight_cell,
        "timing_live_rollouts": {
            "measurement_type": "live_rollout_e2e_observation_to_selected_action_excluding_env_step",
            "arm_order_rotation": "seed_offset % 3: [(A,B,C), (B,C,A), (C,A,B)]",
            "warmup_completed": "dev 0..99 untimed",
            "gc_enabled": gc.isenabled(),
            "per_arm_t_decision_e2e_us": {
                arm: summarize_ns_distribution(live_e2e_ns[arm]) for arm in ("A", "B", "C")
            },
            "arm_c_wrapper_reset_us": summarize_ns_distribution(live_reset_ns_c),
        },
        "timing_off_policy_fixed_trace": fixed_trace_timing,
        "runtime_environment": {
            "os": f"{platform.system()} {platform.release()} ({platform.version()})",
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count_logical": os.cpu_count(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "gymnasium": gym.__version__,
            "minigrid": minigrid.__version__,
            "blas_thread_env": {
                k: os.environ.get(k)
                for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
            },
            "perf_counter_clock_info": {
                "implementation": clock_info.implementation,
                "resolution_sec": clock_info.resolution,
                "monotonic": clock_info.monotonic,
            },
            "pass1_live_elapsed_seconds": pass1_elapsed_sec,
            "total_script_elapsed_seconds": total_elapsed_sec,
            "provisional_15s_note": "Any <15s expectation was provisional only; full run executed without premature truncation.",
        },
        "diagnostic_traces_prespecified_subset": diagnostic_traces,
        "raw_per_seed_jsonl": str(jsonl_path),
    }

    out_path = OUT_DIR / "confirmation_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    compact = {k: v for k, v in results.items() if k != "diagnostic_traces_prespecified_subset"}
    compact["diagnostic_trace_seeds"] = [
        {"stratum": d["stratum"], "seed": d["seed"]} for d in diagnostic_traces
    ]
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
