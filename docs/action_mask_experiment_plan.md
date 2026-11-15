# Experiment Plan: Observation-Based Action Masking vs. Observation-Count Wrapper

*Status: Approved with corrections — ready for development validation, machine-readable freeze, and single-shot held-out execution.*

---

## 1. Existing Setup Inspected

- **Policy artifact (`phase1_weights.npz`, SHA-256 `e31f335039c27dc2b41a937a82d8a52e062828e15c3e049f8f470c614a6a19fd`):** Frozen `247 -> 64 (ReLU) -> 3` NumPy MLP (`forward_mlp` in [`run_phase1.py`](../run_phase1.py)), producing finite `float64` softmax probabilities `probs` over 3 actions.
- **Environment & episode limits:** `gym.make("BabyAI-GoToLocal-v0")` (`minigrid==3.2.0`, `gymnasium==1.4.0`), `8×8` room (`6×6` interior), `num_dists=8`, deterministic `env.reset(seed=seed)`, fixed `max_steps = 64` (set during `reset()`). Success requires `terminated and total_reward > 0.0 and exception is None` (per `verifier.py`, `GoToInstr` terminates with success when `env.front_pos` equals a mission-matching object's position after `env.step(action)`). Timeout is `truncated and not terminated and exception is None` at 64 steps.
- **Public observation interface (`encode_observation` & `ExternalObservationCountWrapper`):** Environment `obs` exposes `{"image", "direction", "mission"}`. Both [`encode_observation`](../run_phase1.py) and [`ExternalObservationCountWrapper`](../external_wrapper.py) consume **only** `obs["image"]` (`uint8`, shape `(7, 7, 3)` encoding `(OBJECT_IDX, COLOR_IDX, STATE)` with agent at `(3, 6)` and immediate front cell at `(3, 5)`) and `obs["mission"]` (`str`, `"go to (a|the) <color> <type>"`). Neither reads `obs["direction"]`, global pose, internal grid, or `BabyAIBot`.
- **Action space & selection rules:** Native MiniGrid action space is `Discrete(7)`, while the student operates on `{0: left, 1: right, 2: forward}`.
  - Frozen baseline ([`run_phase5_confirmation.py`](../run_phase5_confirmation.py)): `int(np.argmax(probs_1d))` (earliest index `0 < 1 < 2` breaks ties).
  - Existing wrapper ([`external_wrapper.py`](../external_wrapper.py)): `wrapper.reset()` at episode start; per step, keys by `(obs["image"].tobytes(), str(obs["mission"]))` and selects `min((0, 1, 2), key=lambda a: (counts[a], -float(probs[a]), a))`, incrementing `counts[chosen] += 1`. On first visit (`[0, 0, 0]`), this matches `np.argmax(probs)` including exact ties.

---

## 2. Research Question and Comparison Arms

**Question:** On `BabyAI-GoToLocal-v0` with the frozen Phase 1 policy, can a stateless current-observation **blocked-forward mask** explain the completion improvement of [`ExternalObservationCountWrapper`](../external_wrapper.py), or does episode-local observation-count memory provide additional benefit beyond masking?

*(Narrow empirical scope: evaluates one frozen MLP on one discrete MiniGrid task. It does not establish general method superiority, research novelty, real-time DSR suitability, or behavior on legitimate-revisit tasks.)*

| Property | Arm A: Frozen Policy Baseline | Arm B: Observation-Based Masking *(Blocked-Forward Mask)* | Arm C: Existing Action-Count Wrapper |
| :--- | :--- | :--- | :--- |
| **Policy & preprocessing** | `phase1_weights.npz` + `encode_observation` | Same (frozen) | Same (frozen) |
| **Execution-layer input boundary** | `probs` (`shape=(3,)`) | `obs["image"]`, `obs["mission"]`, `probs` (`shape=(3,)`) | `obs["image"]`, `obs["mission"]`, `probs`, `reset()` |
| **Within-episode memory** | None (stateless) | None (stateless) | Episode-local `counts[(image.tobytes(), mission)]` |
| **Decision rule** | `argmax` over `{0, 1, 2}` | `min_{a in allowed} (-float(probs[a]), a)` (excludes `2: forward` when front-blocked non-target rule holds) | `min_{a in {0,1,2}} (counts[a], -float(probs[a]), a)` |
| **First-visit behavior at blocked non-target front** | Executes `forward` (`2`) if `base_argmax == 2` | Immediately suppresses `forward` (`2`) on first blocked step | Executes `forward` (`2`) once if `base_argmax == 2`; overrides on revisit |
| **Behavior on unblocked `len >= 2` cycles** | Repeats cycle until timeout (`64` steps) | Identical to Arm A (cannot suppress allowed turns/moves) | Forces untried actions when exact `(image, mission)` repeats |

**Unavoidable structural differences:** Even though all three arms share identical seeds, environment resets, weights, and `encode_observation`, Arm B diverges on the *first* blocked non-target step whenever `base_argmax == 2` (avoiding the initial blocked no-op step), whereas Arm C follows `base_argmax` on every first visit and diverges only on exact `(image, mission)` revisits.

---

## 3. Exact Proposed Observation-Based Masking Rule

### Environment Action Semantics & Comparator Naming
In MiniGrid `MiniGridEnv.step(action)`:
- `0` (`left`) and `1` (`right`) always rotate `agent_dir` by `±90° mod 4` and are never blocked or no-ops.
- `2` (`forward`) is **not prohibited** by the environment (`Discrete(7)` accepts `2` on every state); when `fwd_cell` has `can_overlap() == False` (`wall`, `key`, `ball`, `box`, `agent`, or non-open `door`), `env.step(2)` is an **allowed action that produces no movement or heading change**.
- Accordingly, Arm B is accurately named a **blocked-forward mask** (or *front-obstacle forward-exclusion mask*), **not** an "invalid-action mask". It never suppresses turning (`left`/`right`) or revisiting merely because a move might appear unproductive.

### Mask Specification (Strict Public Boundary)
1. **Public fields & input validation (fail fast):**
   - Reads only `obs["image"]` (must be `np.ndarray` with `shape == (7, 7, 3)` and integer dtype; fails clearly with `ValueError` otherwise), specifically `obs["image"][3, 5, :]` (`(obj_idx, color_idx, state_idx)`), `obs["mission"]` (parsed strictly via `parse_mission(mission)` -> `(target_color_idx, target_type_idx)`, raising `ValueError` on unrecognized format), and `probs` (`shape == (3,)`, finite, raising `ValueError` / `FloatingPointError` otherwise). Zero `obs["direction"]`, coordinates, internal grid, or expert APIs.
2. **Impassable front-cell check (MiniGrid v3.2.0 `world_object.py` semantics):**
   - Passable or unknown in observation: `obj_idx in {empty (1), unseen (0), floor (3), goal (8), lava (9)}` or `(obj_idx == door (4) and state_idx == open (0))`.
   - Impassable: `obj_idx in {wall (2), key (5), ball (6), box (7), agent (10)}` or `(obj_idx == door (4) and state_idx != open (0))`.
3. **Target-facing exemption (Option 1 — adopted):**
   - In `BabyAI-GoToLocal-v0`, `GoToInstr` verifies `front_pos == target_pos` *after* `env.step(action)`. If the agent starts facing a matching target object (`obj_idx == target_type_idx and color_idx == target_color_idx` at `(3, 5)`), `forward` (`2`) preserves `front_pos` and wins in 1 step, whereas turning moves `front_pos` off target.
   - **Rule:** Exclude `action = 2` (`forward`) iff `front_is_impassable` **and not** `front_is_mission_target`. Actions `0` (`left`) and `1` (`right`) are never excluded.
4. **Action selection & numerical edge cases:**
   - Select from remaining `allowed_actions` (`(0, 1)` when masked, `(0, 1, 2)` when unmasked) via:
     `chosen = min(allowed_actions, key=lambda a: (-float(probs[a]), a))`
   - **Why no probability renormalization:** Zeroing `probs[2]` and dividing by `probs[0] + probs[1]` risks `0/0 -> NaN` if extreme logits cause float underflow (`probs[0] == probs[1] == 0.0`). Lexicographic selection avoids division entirely, breaks exact float ties in favor of lowest `action_id` (`0 < 1 < 2`), and matches `np.argmax(probs)` bit-for-bit whenever unmasked.
5. **Empty-mask guard:** By construction, `0` and `1` always remain (`len(allowed_actions) >= 2`). As a defensive invariant check, if `not allowed_actions`, increment `fallback_empty_mask_count` and fall back to `int(np.argmax(probs))` (unit-tested never to trigger).
6. **Application cadence:** The mask predicate is evaluated at **every decision step** (`step 0 .. episode_length - 1`).

---

## 4. Development Validation, Seed Separation, and Analysis Plan

### Development-Only Validation Gate (Before Confirmation)
Using constructed inputs and dev seeds `0..99` only (never confirmation seeds), verify:
1. Actual configured observation shape `(7, 7, 3)` and front-cell coordinate `(3, 5)` alignment against live rotated orientations in `BabyAI-GoToLocal-v0` on dev `0..99`.
2. Object/state decoding (`wall`, `key`, `ball`, `box`, `empty`, `unseen`, `open`/`closed`/`locked` door) and strict error raising on bad image shapes or unrecognized mission strings.
3. Target-facing exemption (`forward` kept when front cell matches mission target) vs. blocked non-target (`forward` masked) vs. unblocked (`empty` / `unseen` front -> `forward` kept).
4. Exact float ties and finite probability vectors with zero-probability allowed actions (e.g., `[0.0, 0.0, 1.0]` when `forward` is masked -> selects `0`, no `NaN`).
5. No-op masking invariant: when no action is masked, Arm B equals Arm A (`np.argmax(probs)`).
6. Unchanged `ExternalObservationCountWrapper` unit tests including episode-local `reset()`.
7. Pipeline sanity check on dev seeds `0..99` (`N = 100`), confirming exact Arm A (`93/100`) and Arm C (`98/100`) match against [`phase1_eval_results.json`](../phase1_eval_results.json) and [`phase4_results.json`](../phase4_results.json). *Do not run a reference replay on `30000..30199`.*

### Holdout & Stage Separation
| Stage | Seed Range | Episodes (`N`) | Purpose |
| :--- | :--- | ---: | :--- |
| **Previously exposed (excluded from confirmation)** | `0..99`, `10000..10499`, `20000..20099`, `30000..30199` | — | Dev, train, val, and published Phase 5 reproduction set |
| **Stage 1: Development & pipeline validation** | Dev `0..99` only | `100` | Verify unit invariants, Arm A/C historical match on `0..99`, logging, and timing instrumentation before freezing |
| **Stage 2: Frozen held-out confirmation** | Fresh `50000..50499` *(untouched; disjoint from `30200..30399` and `40000..40499` briefs)* | **`500`** | Single-shot paired evaluation of frozen Arms A, B, C after locking code, hashes, and analysis plan |

- **Justification for `N = 500`:** `N = 500` is a fixed, budget-conscious confirmation sample (`1,500` paired rollouts) chosen before inspecting held-out outcomes. Because paired-test power and confidence-interval precision depend on the **discordant outcome structure** (`n_01`, `n_10`) rather than marginal failure rate or marginal effect size alone, smaller differences between Arms B and C (or between A and B) may remain statistically unresolved at `N = 500`. We will not add an adaptive power search or change `N` after observing outcomes.
- **No early stopping & interruption rule:** All `500` seeds run to completion without early stopping. A generous `300.0 s` safety ceiling guards only against hangs (no valid run will be truncated to force a `15 s` target). If interrupted or if an exception occurs, partial JSONL/JSON records are preserved and reported as incomplete without silent replacement.

### Statistical Protocol (Pre-Specified Before Stage 2)
- **Primary outcome:** Binary episode success (`terminated and total_reward > 0.0 and exception is None`).
- **Primary prespecified comparison:** **`Arm C vs. Arm B`** (`delta_C_vs_B = rate(C) - rate(B)`), evaluated using [`phase5_stats.py`](../phase5_stats.py):
  - `newcombe_paired_method10_ci` (95% paired Wilson CI), and
  - `exact_mcnemar_test` (two-sided exact binomial test on discordant pairs `(B=1, C=0)` vs. `(B=0, C=1)`, `alpha = 0.05`).
- **Secondary exploratory comparisons:** **`Arm B vs. Arm A`** and **`Arm C vs. Arm A`**, reported clearly as secondary exploratory contrasts with raw exact McNemar p-values and Newcombe 95% CIs.
- **Supplementary family-wise adjustment (clearly separated from primary test):** Report Holm-Bonferroni adjusted p-values across the 3 pairwise contrasts (`C vs. B`, `B vs. A`, `C vs. A`) in a clearly labeled supplementary table.
- **Required interpretation & edge-case safeguards:**
  - *Identical success/failure outcomes on all seeds* (`n_01 == n_10 == 0`) does **not** mean identical trajectories or equivalent methods.
  - *A non-significant result* (`p >= 0.05` or CI containing `0`) does **not** establish equivalence; report as inconclusive / no statistically significant difference detected at `N = 500`.
  - If the primary exact McNemar test (`p < 0.05`) and Newcombe 95% CI (`excludes 0`) yield different threshold-based conclusions, report **both** verbatim without selectively emphasizing either.

---

## 5. Outcomes, Interventions, and Failure-Analysis Plan

All per-seed paired records will be preserved in JSONL and JSON under `experiments/action_mask_ablation/outputs/` alongside frozen hashes and run metadata.

1. **Task outcomes:**
   - Per-arm `success`, `timeout`, `exception` counts and rates.
   - All three pairwise `2×2` contingency tables (`C vs. B`, `B vs. A`, `C vs. A`) **plus** the complete `2×2×2` (`8`-cell) joint outcome table `(A, B, C) in {0, 1}^3`.
   - Baseline failures rescued (`A=0, arm=1`) and baseline successes lost (`A=1, arm=0`) for B and C.
   - Episode length summaries (`count, mean, median, min, max`) for all `500` episodes, all successful episodes per arm, pairwise common-success sets, and every populated `(A, B, C)` outcome cell.
2. **Interventions (distinct definitions — never equated):**
   - `mask_exclusions` (Arm B): steps/episodes where `forward` was excluded (`allowed_actions == (0, 1)`), regardless of `base_argmax`.
   - `mask_action_changes` (Arm B): steps/episodes where `chosen_action != base_argmax` (`forward` excluded **and** `base_argmax == 2`).
   - `wrapper_overrides` (Arm C): steps/episodes where `ExternalObservationCountWrapper` returned `overridden == True` (`chosen_action != base_argmax` due to observation-key counts), plus `retained_obs_keys` and `post_full_coverage_decisions`.
   - Edge-case counters: `empty_mask_fallbacks`, `unmasked_prob_ties`, `masked_zero_prob_underflows`, `non_finite_exceptions`.
3. **Failure analysis (prespecified trajectory selection):**
   - Inspect up to **`15` seeds total** (first `3` seeds in ascending order from each populated stratum):
     1. Regressions (`A=1` with `B=0` or `C=0`)
     2. Joint rescues (`A=0, B=1, C=1`)
     3. Wrapper-only rescues over mask (`A=0, B=0, C=1`)
     4. Mask-only rescues over wrapper (`A=0, B=1, C=0`)
     5. Triple timeouts (`A=0, B=0, C=0`)
   - Separate **observed trajectory facts** (poses, `obs["image"][3, 5]` cell contents, cycle lengths, blocked-forward counts, intervention steps) from **inferred causes**.

---

## 6. Timing Protocol and Its Limits

Two measurement modes are kept strictly separate and **never** averaged together:

1. **Pass 1 — Live-rollout `t_decision_e2e` (no nested inner timers):**
   - Measured on each arm's actual `500`-seed held-out trajectories (`50000..50499`, arm execution rotated by `seed % 3`: `A,B,C` / `B,C,A` / `C,A,B`), following `1` untimed warm-up pass on dev `0..99`.
   - Single outer `time.perf_counter_ns()` window per step covering `encode_observation(obs)` + `forward_mlp(feat, params)` + the arm's selection call (`env.step`/`gym.make` excluded). In Arm C, `wrapper.reset()` is timed separately per episode.
   - Sample count equals the actual number of live decisions made per arm (`sample_count` reported verbatim per arm).
2. **Pass 2 — Off-policy fixed-input matched microbenchmark (isolated cost only):**
   - Replays the exact recorded Arm A observation episodes (`(obs["image"], obs["mission"])`, with precomputed `probs` for `t_exec_layer` and raw `obs` for `t_decision_e2e`) across **`R = 5` fixed timed passes per arm** after `1` untimed warm-up pass (`achieved sample_count = 5 × Arm_A_total_steps` reported verbatim — no pre-promised call count).
   - **Arm C boundary consistency:** Within each replayed episode, `wrapper.reset()` is called at episode start and `wrapper.select_action(...)` updates Arm C's memory cleanly step-by-step across that replayed episode.
   - **Strict label:** Off-policy, fixed-input cost measurement only — **not** a valid Arm C rollout and never used as task-performance evidence. Separate inner/outer passes ensure nested `perf_counter_ns()` does not distort end-to-end measurements.

- **Context, statistics, and limits:** Pin single-process CPU execution (`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`), leave GC active with state noted, log OS, CPU, Python/NumPy versions, and `time.get_clock_info("perf_counter").resolution`. Report `sample_count`, `min`, `p50`, `p95`, `p99`, `max`, `mean`. Acknowledge that Python/Windows `p95`/`p99` tails reflect OS scheduling/GC jitter over finite samples, toy symbolic-grid timings do not transfer to DSR/pixel/external-engine loops, and no external real-time deadline exists (no synthetic deadline invented). The `<15 s` runtime expectation is provisional, not a hard runtime guarantee.

---

## 7. Scope Boundaries

- **Excluded from this experiment:** No 4th arm combining masking + wrapper; no retraining, DAgger, or `BabyAIBot` queries; no new legitimate-revisit benchmarks ([`contributions/02_harmful_legitimate_revisit.md`](../contributions/02_harmful_legitimate_revisit.md)); no observation noise or Unity/Unreal/DSR integration; no edits to [`external_wrapper.py`](../external_wrapper.py), [`phase1_weights.npz`](../phase1_weights.npz), Phase 5 artifacts, or `README.md`; no Git commits or pushes.
