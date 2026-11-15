"""Read-only post-run audit and full-precision Holm-Bonferroni correction."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from experiments.action_mask_ablation.common import (
    compute_paired_contrast,
    summarize_arm_records,
)

EXP_DIR = Path("experiments/action_mask_ablation")
OUT_DIR = EXP_DIR / "outputs"
EVIDENCE_DIR = EXP_DIR / "evidence"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_mcnemar_full_precision(n_first_only: int, n_second_only: int) -> dict:
    """Two-sided exact binomial McNemar test preserving full IEEE-754 float precision."""
    if n_first_only < 0 or n_second_only < 0:
        raise ValueError("Discordant pair counts must be non-negative")
    m = n_first_only + n_second_only
    if m == 0:
        return {
            "method": "exact_two_sided_mcnemar_binomial_full_precision",
            "discordant_pairs_total": 0,
            "first_only_success": 0,
            "second_only_success": 0,
            "p_value": 1.0,
        }
    k = min(n_first_only, n_second_only)
    one_tail = math.fsum(math.comb(m, i) for i in range(k + 1)) * math.ldexp(1.0, -m)
    p_value = min(1.0, 2.0 * one_tail)
    return {
        "method": "exact_two_sided_mcnemar_binomial_full_precision",
        "discordant_pairs_total": m,
        "first_only_success": n_first_only,
        "second_only_success": n_second_only,
        "p_value": p_value,
    }


def holm_bonferroni_full_precision(p_items: list[tuple[str, float]]) -> dict[str, dict]:
    """Step-down Holm-Bonferroni adjustment retaining full float precision."""
    if not p_items:
        raise ValueError("Expected non-empty contrast list")
    for name, p in p_items:
        if not isinstance(p, (int, float)) or not math.isfinite(p) or not (0.0 <= p <= 1.0):
            raise ValueError(f"Invalid p-value for {name!r}: {p!r}")

    m = len(p_items)
    ordered = sorted(p_items, key=lambda x: (x[1], x[0]))
    running_max = 0.0
    adjusted: dict[str, dict] = {}
    for rank_idx, (name, raw_p) in enumerate(ordered):
        mult = m - rank_idx
        step_p = min(1.0, raw_p * mult)
        running_max = max(running_max, step_p)
        adjusted[name] = {
            "rank": rank_idx + 1,
            "raw_p_value": raw_p,
            "raw_p_sci": f"{raw_p:.6e}",
            "holm_multiplier": mult,
            "holm_adjusted_p_value": running_max,
            "holm_adjusted_p_sci": f"{running_max:.6e}",
        }
    return adjusted


def run_audit() -> dict:
    """Audit saved JSONL/JSON artifacts without running any episodes."""
    lock = json.loads((EXP_DIR / "freeze_lock.json").read_text(encoding="utf-8"))
    saved_res = json.loads((OUT_DIR / "confirmation_results.json").read_text(encoding="utf-8"))
    jsonl_lines = (OUT_DIR / "confirmation_episodes.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in jsonl_lines]

    assert len(rows) == 500
    seeds = [r["seed"] for r in rows]
    assert seeds == list(range(50000, 50500))

    # 1. Verify post-run artifact hashes against pre-run lock without modifying freeze_lock.json
    post_run_hash_checks: dict[str, dict] = {}
    hash_mismatches: list[dict] = []
    for rel_path, expected_sha in lock["locked_sha256"].items():
        observed_sha = sha256_file(Path(rel_path))
        matches = observed_sha == expected_sha
        post_run_hash_checks[rel_path] = {
            "expected_sha256": expected_sha,
            "observed_sha256": observed_sha,
            "matches": matches,
        }
        if not matches:
            hash_mismatches.append({"file": rel_path, "expected": expected_sha, "observed": observed_sha})

    # 2. Recompute arm summaries, pairwise tables, and 8-cell table directly from saved JSONL
    recs_a = [r["arms"]["A"] for r in rows]
    recs_b = [r["arms"]["B"] for r in rows]
    recs_c = [r["arms"]["C"] for r in rows]

    recomputed_arms = {
        "A_frozen_baseline": summarize_arm_records(recs_a, "A"),
        "B_blocked_forward_mask": summarize_arm_records(recs_b, "B"),
        "C_action_count_wrapper": summarize_arm_records(recs_c, "C"),
    }
    assert recomputed_arms == saved_res["arm_summaries"], "Arm summary mismatch vs JSONL!"

    recomputed_c_vs_b = compute_paired_contrast(seeds, recs_b, recs_c, "B", "C")
    recomputed_b_vs_a = compute_paired_contrast(seeds, recs_a, recs_b, "A", "B")
    recomputed_c_vs_a = compute_paired_contrast(seeds, recs_a, recs_c, "A", "C")
    assert recomputed_c_vs_b == saved_res["primary_comparison_C_vs_B"]
    assert recomputed_b_vs_a == saved_res["secondary_exploratory_comparisons"]["B_vs_A"]
    assert recomputed_c_vs_a == saved_res["secondary_exploratory_comparisons"]["C_vs_A"]

    eight_cell_counts: dict[str, int] = {f"{a}{b}{c}": 0 for a in (0, 1) for b in (0, 1) for c in (0, 1)}
    for r in rows:
        k = f"{int(r['arms']['A']['success'])}{int(r['arms']['B']['success'])}{int(r['arms']['C']['success'])}"
        eight_cell_counts[k] += 1
    for k, cnt in eight_cell_counts.items():
        assert cnt == saved_res["joint_eight_cell_table_A_B_C"][k]["count"]

    # 3. Full-precision McNemar & corrected Holm-Bonferroni adjustment
    fp_c_vs_b = exact_mcnemar_full_precision(
        len(recomputed_c_vs_b["B_only_seeds"]), len(recomputed_c_vs_b["C_only_seeds"])
    )
    fp_b_vs_a = exact_mcnemar_full_precision(
        len(recomputed_b_vs_a["A_only_seeds"]), len(recomputed_b_vs_a["B_only_seeds"])
    )
    fp_c_vs_a = exact_mcnemar_full_precision(
        len(recomputed_c_vs_a["A_only_seeds"]), len(recomputed_c_vs_a["C_only_seeds"])
    )
    corrected_holm = holm_bonferroni_full_precision(
        [
            ("C_vs_B_primary", fp_c_vs_b["p_value"]),
            ("B_vs_A_secondary", fp_b_vs_a["p_value"]),
            ("C_vs_A_secondary", fp_c_vs_a["p_value"]),
        ]
    )

    # 4. Verify full-batch diagnostic breakdowns from JSONL (vs 9-trace subset)
    a_win_rows = [r for r in rows if r["arms"]["A"]["success"]]
    a_win_divergences = [
        r["seed"]
        for r in a_win_rows
        if (
            not r["arms"]["B"]["success"]
            or not r["arms"]["C"]["success"]
            or r["arms"]["B"]["mask_action_changes"] > 0
            or r["arms"]["C"]["override_steps"] > 0
            or r["arms"]["B"]["episode_length"] != r["arms"]["A"]["episode_length"]
            or r["arms"]["C"]["episode_length"] != r["arms"]["A"]["episode_length"]
        )
    ]

    a_fail_rows = [r for r in rows if not r["arms"]["A"]["success"]]
    blocked_fwd_rows = [r for r in a_fail_rows if r["arms"]["A"]["first_pose_cycle_length"] == 1]
    unblocked_cyc_rows = [r for r in a_fail_rows if r["arms"]["A"]["first_pose_cycle_length"] != 1]

    blk_b_rescued = [r["seed"] for r in blocked_fwd_rows if r["arms"]["B"]["success"]]
    blk_b_failed = [r["seed"] for r in blocked_fwd_rows if not r["arms"]["B"]["success"]]
    blk_b_fail_cycle_lens = [r["arms"]["B"]["first_pose_cycle_length"] for r in blocked_fwd_rows if not r["arms"]["B"]["success"]]
    blk_c_rescued = [r["seed"] for r in blocked_fwd_rows if r["arms"]["C"]["success"]]
    blk_c_failed = [r["seed"] for r in blocked_fwd_rows if not r["arms"]["C"]["success"]]

    nonblk_b_action_changed = [r["seed"] for r in unblocked_cyc_rows if r["arms"]["B"]["mask_action_changes"] > 0]
    nonblk_b_rescued = [r["seed"] for r in unblocked_cyc_rows if r["arms"]["B"]["success"]]
    nonblk_c_rescued = [r["seed"] for r in unblocked_cyc_rows if r["arms"]["C"]["success"]]
    nonblk_c_failed = [r["seed"] for r in unblocked_cyc_rows if not r["arms"]["C"]["success"]]

    saved_traces = saved_res["diagnostic_traces_prespecified_subset"]
    saved_trace_summary = [
        {"stratum": t["stratum"], "seed": t["seed"], "mission": t["mission"], "summary_A_B_C": t["summary_A_B_C"]}
        for t in saved_traces
    ]

    # 5. Confirm whether raw timing sample arrays exist in outputs
    timing_raw_samples_saved = any(
        isinstance(v, list)
        for section in (
            saved_res["timing_live_rollouts"]["per_arm_t_decision_e2e_us"],
            saved_res["timing_off_policy_fixed_trace"]["pass2a_fixed_trace_decision_e2e_us"],
            saved_res["timing_off_policy_fixed_trace"]["pass2b_fixed_trace_exec_layer_only_us"],
        )
        for v in section.values()
    )

    audit_report = {
        "audit_status": "verified_with_statistical_correction",
        "correction_note": {
            "root_cause_location": "Both analysis code/saved summary (confirmation_results.json) and prior narrative summary",
            "mechanism": (
                "phase5_stats.exact_mcnemar_test() rounded p_value to 10 decimal places, truncating "
                "2/2^36 (~2.910e-11 for C vs B) and 2/2^42 (~4.547e-13 for C vs A) to 0.0. "
                "common.holm_bonferroni_adjust() sorted on those rounded 0.0 values in insertion order "
                "(reversing the Holm multipliers for C_vs_B_primary [assigned 3] and C_vs_A_secondary [assigned 2]) "
                "and also rounded adjusted p-values to 10 decimals (producing 0.0 in confirmation_results.json)."
            ),
            "remediation": (
                "Original outputs/confirmation_results.json, outputs/confirmation_episodes.jsonl, frozen_plan.json, "
                "freeze_lock.json, and phase5_stats.py are preserved unchanged. Full-precision exact McNemar "
                "p-values and corrected Holm-Bonferroni adjustments are computed and recorded here."
            ),
        },
        "post_run_freeze_lock_verification": {
            "freeze_lock_sha256": sha256_file(EXP_DIR / "freeze_lock.json"),
            "all_pre_run_locked_files_match": len(hash_mismatches) == 0,
            "hash_mismatches": hash_mismatches,
            "checked_files": post_run_hash_checks,
        },
        "recomputed_from_jsonl_verification": {
            "episodes_checked": len(rows),
            "arm_summaries_exact_match": True,
            "pairwise_tables_exact_match": True,
            "eight_cell_counts_exact_match": True,
            "eight_cell_counts": eight_cell_counts,
        },
        "full_precision_statistical_results": {
            "primary_C_vs_B": {
                "B_successes": recomputed_arms["B_blocked_forward_mask"]["success_count"],
                "C_successes": recomputed_arms["C_action_count_wrapper"]["success_count"],
                "n_total": 500,
                "C_only_success": len(recomputed_c_vs_b["C_only_seeds"]),
                "B_only_success": len(recomputed_c_vs_b["B_only_seeds"]),
                "delta_C_minus_B": recomputed_c_vs_b["delta_second_minus_first"],
                "newcombe_method10_95_ci": recomputed_c_vs_b["newcombe_method10_paired_95_ci"],
                "exact_mcnemar_full_precision": fp_c_vs_b,
                "exact_mcnemar_p_sci": f"{fp_c_vs_b['p_value']:.6e}",
            },
            "secondary_exploratory": {
                "B_vs_A": {
                    "A_successes": recomputed_arms["A_frozen_baseline"]["success_count"],
                    "B_successes": recomputed_arms["B_blocked_forward_mask"]["success_count"],
                    "B_only_success": len(recomputed_b_vs_a["B_only_seeds"]),
                    "A_only_success": len(recomputed_b_vs_a["A_only_seeds"]),
                    "delta_B_minus_A": recomputed_b_vs_a["delta_second_minus_first"],
                    "newcombe_method10_95_ci": recomputed_b_vs_a["newcombe_method10_paired_95_ci"],
                    "exact_mcnemar_full_precision": fp_b_vs_a,
                    "exact_mcnemar_p_sci": f"{fp_b_vs_a['p_value']:.6e}",
                },
                "C_vs_A": {
                    "A_successes": recomputed_arms["A_frozen_baseline"]["success_count"],
                    "C_successes": recomputed_arms["C_action_count_wrapper"]["success_count"],
                    "C_only_success": len(recomputed_c_vs_a["C_only_seeds"]),
                    "A_only_success": len(recomputed_c_vs_a["A_only_seeds"]),
                    "delta_C_minus_A": recomputed_c_vs_a["delta_second_minus_first"],
                    "newcombe_method10_95_ci": recomputed_c_vs_a["newcombe_method10_paired_95_ci"],
                    "exact_mcnemar_full_precision": fp_c_vs_a,
                    "exact_mcnemar_p_sci": f"{fp_c_vs_a['p_value']:.6e}",
                },
            },
            "supplementary_holm_bonferroni_corrected": {
                "note": "Supplementary 3-contrast Holm-Bonferroni adjustment sorted by full-precision raw p-value, clearly separated from prespecified primary C_vs_B test.",
                "contrasts": corrected_holm,
            },
        },
        "full_batch_diagnostic_audit": {
            "baseline_454_wins": {
                "count": len(a_win_rows),
                "divergences_in_B_or_C": a_win_divergences,
                "zero_regressions_and_zero_interventions_on_all_baseline_wins": len(a_win_divergences) == 0,
            },
            "baseline_46_timeouts_partition_by_first_pose_cycle_length": {
                "cycle_len_1_blocked_forward_count": len(blocked_fwd_rows),
                "cycle_len_1_all_had_B_action_change": all(r["arms"]["B"]["mask_action_changes"] > 0 for r in blocked_fwd_rows),
                "cycle_len_1_B_rescued_count": len(blk_b_rescued),
                "cycle_len_1_B_rescued_seeds": blk_b_rescued,
                "cycle_len_1_B_failed_count": len(blk_b_failed),
                "cycle_len_1_B_failed_seeds": blk_b_failed,
                "cycle_len_1_B_failed_new_cycle_lengths": blk_b_fail_cycle_lens,
                "cycle_len_1_C_rescued_count": len(blk_c_rescued),
                "cycle_len_1_C_failed_seeds": blk_c_failed,
                "unblocked_cycle_count": len(unblocked_cyc_rows),
                "unblocked_cycle_B_action_changed_count": len(nonblk_b_action_changed),
                "unblocked_cycle_B_rescued_count": len(nonblk_b_rescued),
                "unblocked_cycle_C_rescued_count": len(nonblk_c_rescued),
                "unblocked_cycle_C_failed_seeds": nonblk_c_failed,
            },
        },
        "prespecified_trace_subset_audit": {
            "count": len(saved_trace_summary),
            "note": "9 seeds selected by max_per_stratum=3 ascending (50024, 50042, 50050; 50012, 50019, 50032; 50006, 50095, 50123). Triple-timeout seed 50270 has per-seed JSONL summary only (no saved step trace).",
            "selected_traces_summary": saved_trace_summary,
        },
        "timing_evidence_audit": {
            "raw_per_decision_ns_arrays_saved_on_disk": timing_raw_samples_saved,
            "limitation_note": (
                "Only aggregate latency distribution summaries (sample_count, min_us, p50_us, p95_us, p99_us, "
                "max_us, mean_us) were persisted in confirmation_results.json; raw per-call nanosecond sample "
                "arrays were not written to disk."
            ),
            "timing_live_rollouts": saved_res["timing_live_rollouts"],
            "timing_off_policy_fixed_trace": saved_res["timing_off_policy_fixed_trace"],
        },
    }
    return audit_report


if __name__ == "__main__":
    print(json.dumps(run_audit(), indent=2))
