# Experiment Report: Blocked-Forward Action Mask vs. Observation-Count Wrapper

## Conclusion

For the frozen Phase 1 policy (`phase1_weights.npz`, `247 -> 64 ReLU -> 3`) on `BabyAI-GoToLocal-v0` (`max_steps = 64`), the specified stateless blocked-forward mask did not account for the benefit of the existing action-count wrapper (`ExternalObservationCountWrapper`).

This comparison evaluates one frozen policy on one deterministic discrete task and does not establish superiority over action masking in general, nor does it isolate episode-local memory as the sole causal factor (Arm B and Arm C also differ on first-visit vs. revisit timing and non-blocked cycle handling).

---

## 1. Setup, Pre-Run Lock, and Development Validation

- **Plan & Lock:** Frozen in [`action_mask_experiment_plan.md`](../action_mask_experiment_plan.md), [`frozen_plan.json`](../../experiments/action_mask_ablation/frozen_plan.json), and [`freeze_lock.json`](../../experiments/action_mask_ablation/freeze_lock.json). All `11` pre-run locked artifact SHA-256 hashes (including [`phase1_weights.npz`](../../phase1_weights.npz) and [`external_wrapper.py`](../../external_wrapper.py)) and pinned runtime versions (`python 3.14.3`, `numpy 2.5.4`, `gymnasium 1.4.0`, `minigrid 3.2.0`) verified `11/11` matching both pre-run and post-run (`hash_mismatches = []`).
- **Arm definitions (`{0: left, 1: right, 2: forward}`):**
  - **Arm A (Baseline):** Unmodified frozen policy `argmax` over `{0, 1, 2}`.
  - **Arm B (Blocked-Forward Mask, Option 1):** Reads only `obs["image"][3, 5, :]`, `obs["mission"]`, and finite `probs`; excludes `forward` (`2`) iff the front cell is impassable and does not match the mission `(color, type)`; selects `min(allowed_actions, key=lambda a: (-float(probs[a]), a))` (no division/renormalization). Turning (`0`, `1`) is never excluded.
  - **Arm C (Wrapper):** Unchanged `ExternalObservationCountWrapper` with per-episode `reset()`, keying by `(image.tobytes(), str(mission))`.
- **Dev validation (`0..99`, non-confirmatory):** Constructed unit tests, front-coordinate alignment across headings `[0, 1, 2, 3]`, and historical dev `0..99` checks (`Arm A = 93/100`, `Arm B = 94/100`, `Arm C = 98/100`) all passed ([`dev_validation_results.json`](../../experiments/action_mask_ablation/evidence/dev_validation_results.json)).

---

## 2. Held-Out Confirmation Outcomes (`50000..50499`, `N = 500`)

### Per-Arm Summaries

| Arm | Successes / `500` | Rate | Timeouts | Exceptions | Total Decisions | All-Episode Length (`mean / med / min–max`) | Success Length (`mean`) |
| :--- | ---: | ---: | ---: | ---: | ---: | :---: | ---: |
| **A: Frozen Baseline** | **`454 / 500`** | `90.80%` | `46` | `0` | `5,333` | `10.666 / 5.0 / 1–64` | `5.2621` |
| **B: Blocked-Forward Mask** | **`460 / 500`** | `92.00%` | `40` | `0` | `5,016` | `10.032 / 5.0 / 1–64` | `5.3391` |
| **C: Action-Count Wrapper** | **`496 / 500`** | `99.20%` | `4` | `0` | `3,386` | `6.772 / 5.0 / 1–64` | `6.3105` |

### Prespecified Primary Comparison (`Arm C vs. Arm B`)

- **Paired outcome counts:** `460` both succeed, **`36` `C_only_success` (gains)**, **`0` `B_only_success` (losses)**, `4` both fail (`50006, 50095, 50123, 50270`).
- **Effect size (`delta = rate(C) - rate(B)`):** **`+0.072`** (`+7.2` percentage points, `99.20% - 92.00%`).
- **Newcombe Method 10 paired 95% CI:** **`[+0.050857, +0.098247]`** (`ci_includes_zero = false`, `phi_hat = 0.263158`).
- **Exact two-sided McNemar `p`-value (full precision):** **`2.910383e-11`** (`2 / 2^36`).
- **Matched lengths (`C vs. B`):** `both_succeed` (`460`): mean `5.3391` (`B`) vs. `5.3522` (`C`); `C_only_success` (`36`): `64.0` (`B`) vs. mean `18.5556`, median `17.5`, range `5–45` (`C`).

### Joint 8-Cell Table `(A, B, C)` and Interventions

- **`(1, 1, 1)`:** `454` seeds (identical length mean `5.2621` across A, B, and C; zero divergences or interventions across all `454` baseline wins).
- **`(0, 1, 1)`:** `6` seeds (`50024, 50042, 50050, 50242, 50411, 50463`; mean length `11.1667` in B vs. `12.1667` in C).
- **`(0, 0, 1)`:** `36` seeds (mean length `18.5556` in C).
- **`(0, 0, 0)`:** `4` seeds (`50006, 50095, 50123, 50270`).
- All remaining 4 cells `(0,1,0)`, `(1,0,0)`, `(1,0,1)`, `(1,1,0)` have count `0`.
- **Intervention counts (distinct definitions):** Arm B `mask_exclusions` (`forward` hidden regardless of `base_argmax`) = `1,103` steps across `301` episodes; Arm B `mask_action_changes` (`chosen_action != base_argmax`) = `189` steps across `13` episodes (`0` empty-mask fallbacks, `0` ties, `0` underflows); Arm C `wrapper_overrides` = `203` steps across `46` episodes (`33` post-full-coverage decisions across `26` keys).

---

## 3. Secondary Exploratory Contrasts & Statistical Correction Note

- **Secondary exploratory paired contrasts (unadjusted):**
  - **`B vs. A`:** `+6` `B_only`, `0` `A_only`, `delta = +0.012` (`+1.2` pp), Newcombe 95% CI `[+0.000712, +0.024785]`, exact McNemar `p = 3.125000e-02` (`0.03125`).
  - **`C vs. A`:** `+42` `C_only`, `0` `A_only`, `delta = +0.084` (`+8.4` pp), Newcombe 95% CI `[+0.061263, +0.111758]`, exact McNemar `p = 4.547474e-13` (`2 / 2^42`).
- **Supplementary 3-contrast Holm-Bonferroni adjustment (full precision, separated from primary inference):**
  1. `C vs. A` (secondary, rank 1, `×3`): raw `4.547474e-13` → Holm-adjusted **`1.364242e-12`**
  2. `C vs. B` (primary, rank 2, `×2`): raw `2.910383e-11` → Holm-adjusted **`5.820766e-11`**
  3. `B vs. A` (secondary, rank 3, `×1`): raw `3.125000e-02` → Holm-adjusted **`3.125000e-02`**
- **Correction note:** Original [`confirmation_results.json`](../../experiments/action_mask_ablation/evidence/confirmation_results.json) stored `p_value = 0.0` (from `phase5_stats.exact_mcnemar_test`'s 10-decimal `round`) for both `C vs. B` and `C vs. A`, reversing their Holm multipliers (`C_vs_B ×3`, `C_vs_A ×2`) and rounding adjusted values to `0.0`. Original files were left unchanged and corrected full-precision numbers are preserved in [`confirmation_audit_and_correction.json`](../../experiments/action_mask_ablation/evidence/confirmation_audit_and_correction.json).

---

## 4. Latency Results (Live-Rollout vs. Fixed-Input Kept Separate)

CPU-only, single-thread BLAS, Windows 11 `AMD64`, `QueryPerformanceCounter()` (`0.1 µs` resolution), GC enabled. **Limitation:** only summary percentiles were stored (`confirmation_results.json`); raw per-call nanosecond arrays were not persisted to disk.

| Measurement Mode | Arm | `sample_count` | `p50` (`µs`) | `p95` (`µs`) | `p99` (`µs`) | `mean` (`µs`) |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: |
| **Live-rollout `t_decision_e2e`** *(actual trajectories, rotated `seed % 3`)* | **A** | `5,333` | `28.6` | `63.2` | `109.6` | `34.636` |
| **Live-rollout `t_decision_e2e`** | **B** | `5,016` | `36.5` | `78.05` | `135.615` | `43.400` |
| **Live-rollout `t_decision_e2e`** | **C** | `3,386` | `34.1` | `74.475` | `135.31` | `41.374` |
| **Off-policy fixed-trace `t_decision_e2e`** *(`5×` Arm A steps; NOT valid C rollout)* | **A / B / C** | `26,665` each | `19.6 / 24.7 / 22.5` | `27.48 / 38.88 / 30.0` | `74.9 / 98.108 / 85.8` | `22.412 / 28.232 / 25.055` |
| **Off-policy fixed-trace `t_exec_layer`** *(selection only on precomputed probs)* | **A / B / C** | `26,665` each | `0.7 / 5.1 / 3.3` | `0.9 / 6.2 / 3.9` | `1.9 / 15.8 / 9.536` | `0.753 / 5.603 / 3.605` |

Arm C `wrapper.reset()` `p50 / p95 / p99` was `1.4 / 3.8 / 6.1 µs` live (`500` calls) and `0.4 / 1.2 / 2.0 µs` (Pass 2a) / `0.3 / 0.6 / 1.401 µs` (Pass 2b) in replay (`2,500` calls).

---

## 5. Observed Failure Patterns & Remaining Limitations

- **Full-batch JSONL breakdown (`N = 500`, `46` Arm A timeouts):**
  - **`13` blocked-forward `cycle_length = 1` timeouts in Arm A:** Arm B had `mask_action_changes > 0` on all `13`, rescuing `6` (`50024, 50042, 50050, 50242, 50411, 50463`) while the remaining `7` transitioned into unblocked cycles (`6` `len=2`, `1` `len=8` on `50270`). Arm C rescued `12/13` and also timed out on `50270` (`41` retained keys, `21` overrides).
  - **`33` unblocked multi-step cycles in Arm A (`len = 2, 6, 8, 10, 14`):** Arm B made `0` action changes (`0/33` rescued); Arm C rescued `30/33` and timed out on `3` (`50006, 50095, 50123`).
- **Prespecified `9`-trace inspection (`3` per populated stratum: `50024, 50042, 50050`; `50012, 50019, 50032`; `50006, 50095, 50123`):** Directly confirms Arm B bypassing blocked-forward stalls 1 step earlier than Arm C on `A0_B1_C1` seeds (`7/13/8` vs. `8/14/9`), Arm B following Arm A identically on non-blocked cycles `50012` (`len=6`), `50019` (`len=2`), `50032` (`len=2`) where Arm C escaped via revisit overrides, and Arm C exhausting `64` steps despite `25–28` overrides on `50006, 50095, 50123` (seed `50270` was audited via its per-seed JSONL summary only; no step trace was saved for `50270`).
- **Limitations:** Single frozen `247 -> 64 -> 3` MLP on noiseless symbolic `(7, 7, 3)` inputs in single-room `BabyAI-GoToLocal-v0`; `0` regressions reflects that zero baseline wins revisited exact observations (`0` overrides/changes on `454` baseline wins), leaving legitimate-revisit harm ([`contributions/02_harmful_legitimate_revisit.md`](../../contributions/02_harmful_legitimate_revisit.md)), observation noise, partial aliasing, and external real-time latency deadlines untested.

## 6. Reproduction vs. Audit/Correction Paths

The locked confirmation artifact [`confirmation_results.json`](../../experiments/action_mask_ablation/evidence/confirmation_results.json) preserves the exact original output schema, which rounded `p_value` to `0.0` (10 decimals) for `C_vs_B` and `C_vs_A` and stored an incorrect Holm adjustment. The separate audit and correction file [`confirmation_audit_and_correction.json`](../../experiments/action_mask_ablation/evidence/confirmation_audit_and_correction.json) supplies the corrected full-precision McNemar and Holm-Bonferroni statistics without altering the original locked outputs. See [`EVIDENCE_MANIFEST.md`](../../experiments/action_mask_ablation/evidence/EVIDENCE_MANIFEST.md) for artifact SHA-256 hashes.

- **Audit/correction on saved results (0 episodes run):** First copy [`confirmation_results.json`](../../experiments/action_mask_ablation/evidence/confirmation_results.json) and [`confirmation_episodes.jsonl`](../../experiments/action_mask_ablation/evidence/confirmation_episodes.jsonl) into `experiments/action_mask_ablation/outputs/` if running from a clean checkout without git-ignored `outputs/`, then run:
  ```powershell
  .venv\Scripts\python.exe -m unittest -v experiments.action_mask_ablation.test_holm_correction
  .venv\Scripts\python.exe -m experiments.action_mask_ablation.audit_confirmation
  ```
- **Reproducing original experiment outputs (runs episodes):** Regenerates `experiments/action_mask_ablation/outputs/` (with the original rounded `p_value = 0.0` and unadjusted Holm fields), followed by the audit/correction step for the corrected statistical summary:
  ```powershell
  .venv\Scripts\python.exe -m experiments.action_mask_ablation.validate_dev
  .venv\Scripts\python.exe -m experiments.action_mask_ablation.run_confirmation
  .venv\Scripts\python.exe -m experiments.action_mask_ablation.audit_confirmation
  ```
