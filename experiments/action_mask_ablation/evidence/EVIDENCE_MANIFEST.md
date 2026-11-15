# Evidence Bundle Manifest: Action-Mask Ablation (`50000..50499`)

This directory preserves an immutable, Git-trackable copy of the frozen configuration, pre-run lock, development validation output, confirmation summary (with diagnostic traces and timing summaries), raw per-seed confirmation JSONL log, and post-run statistical audit/correction while leaving the original files under `experiments/action_mask_ablation/outputs/` unmodified.

## Statistical Correction Note

- **Issue identified:** Both the original analysis code/saved summary ([`confirmation_results.json`](confirmation_results.json)) and the initial summary report exhibited reversed Holm-Bonferroni multipliers for `C vs. B` and `C vs. A`. Specifically, `phase5_stats.exact_mcnemar_test()` rounds `p_value` to 10 decimal places (`0.0` for both `2 / 2^36 ≈ 2.910383e-11` and `2 / 2^42 ≈ 4.547474e-13`), so sorting on those rounded values preserved insertion order (`C_vs_B` rank 1 `×3`, `C_vs_A` rank 2 `×2`), and `common.holm_bonferroni_adjust()` rounded adjusted p-values to 10 decimal places (`0.0`).
- **Resolution:** Original locked artifacts (`confirmation_results.json`, `confirmation_episodes.jsonl`, `phase5_stats.py`, `common.py`, `run_confirmation.py`, `frozen_plan.json`, `freeze_lock.json`) are preserved unchanged so all 11 pre-run lock hashes match (`11/11` verified post-run). Full-precision exact McNemar p-values and corrected Holm-Bonferroni values are recorded separately in [`confirmation_audit_and_correction.json`](confirmation_audit_and_correction.json), tested by [`../test_holm_correction.py`](../test_holm_correction.py).

## Timing Evidence Definitions & Limitation

- **Pass 1 (`timing_live_rollouts`):** Outer `time.perf_counter_ns()` around `encode_observation(obs)` + `forward_mlp` + arm selection on each arm's actual `500`-seed trajectory (excluding `gym.make`/`env.step`; Arm C `wrapper.reset()` timed separately; arm order rotated `seed_offset % 3` after dev `0..99` warm-up).
- **Pass 2 (`timing_off_policy_fixed_trace`):** Separate un-nested replay passes (`R = 5` timed passes after `1` warm-up) over the recorded `500` Arm A episodes (`5,333` steps/pass = `26,665` calls/arm), reporting Pass 2a (`t_decision_e2e`) and Pass 2b (`t_exec_layer` only on precomputed `probs`). Explicitly an off-policy fixed-input cost measurement, **not** a valid Arm C rollout.
- **Raw-sample limitation:** Only aggregate distribution summaries (`sample_count`, `min_us`, `p50_us`, `p95_us`, `p99_us`, `max_us`, `mean_us`) were persisted in `confirmation_results.json`; raw per-decision nanosecond sample arrays were **not** saved to disk.

## Reproduction & Audit Commands

Distinguish two workflows:

1. **Audit & Correction on Saved Evidence (0 episodes run):** Runs the Holm regression tests and recomputes full-precision statistics and diagnostics from `experiments/action_mask_ablation/outputs/` (`confirmation_results.json`, `confirmation_episodes.jsonl`). On a fresh checkout where git-ignored `outputs/` does not exist yet, copy those two files from `experiments/action_mask_ablation/evidence/` into `experiments/action_mask_ablation/outputs/` first (do not overwrite `confirmation_audit_and_correction.json` unless intentionally refreshing the derived output).

```powershell
# Focused regression tests for 3-contrast Holm ordering and cumulative-max adjustment (2 tests)
.venv\Scripts\python.exe -m unittest -v experiments.action_mask_ablation.test_holm_correction

# Read-only post-run audit & full-precision correction against saved outputs/ & freeze_lock.json (0 episodes run)
.venv\Scripts\python.exe -m experiments.action_mask_ablation.audit_confirmation
```

2. **Reproducing Original Experiment Outputs (runs episodes):** Executes dev validation (`0..99`) and the single-shot confirmation run (`50000..50499`) into `experiments/action_mask_ablation/outputs/`. Note that `run_confirmation` reproduces the original locked output format, which stores rounded `p_value = 0.0` for `C_vs_B` / `C_vs_A` and the incorrect Holm adjustment; always run `audit_confirmation` afterward for the corrected statistical interpretation.

```powershell
# Development-only validation (constructed tests + dev seeds 0..99)
.venv\Scripts\python.exe -m experiments.action_mask_ablation.validate_dev

# Single-shot confirmation runner (writes original outputs to experiments/action_mask_ablation/outputs/)
.venv\Scripts\python.exe -m experiments.action_mask_ablation.run_confirmation

# Follow with audit_confirmation to compute corrected full-precision p-values and Holm adjustments
.venv\Scripts\python.exe -m experiments.action_mask_ablation.audit_confirmation
```

## Artifact Hash Manifest

| File | Bytes | SHA-256 | Role |
| :--- | ---: | :--- | :--- |
| [`frozen_plan.json`](frozen_plan.json) | 3,680 | `e539fcf2a339770d8a4043e90defd5334899aaf1297191e1e2918fa9cdf81b70` | Frozen machine-readable plan |
| [`freeze_lock.json`](freeze_lock.json) | 1,367 | `7dc28eece7e53766bbac76c30285b141751991823bbb159740f76dcbea7d9878` | Pre-run SHA-256 & version lock |
| [`dev_validation_results.json`](dev_validation_results.json) | 13,514 | `65f0c631d5bda84c410217c6aa4d3aac2269439c89591917cebae1bbcb592578` | Dev `0..99` & unit validation log |
| [`confirmation_results.json`](confirmation_results.json) | 869,803 | `a781fe0cd1502c5b5b0d9aa58f55cf0a91e503780655e7235b11ec16d0bfa643` | Original confirmation summary, 9 traces & timing summaries |
| [`confirmation_episodes.jsonl`](confirmation_episodes.jsonl) | 951,660 | `0d3001d0a2963e9bfe2c6072b85914e2f19fd53091573e40a8c3dfa4b382ed96` | Raw per-seed (`50000..50499`) paired A/B/C log (`500` lines) |
| [`confirmation_audit_and_correction.json`](confirmation_audit_and_correction.json) | 21,197 | `46d5521a8a802cb0de19036fec7c9562799f98d8946e17f12ebdd3b71f7c3d7a` | Recomputed JSONL audit, full-precision stats & lock check |
