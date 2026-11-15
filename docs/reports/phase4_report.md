> **Public documentation adaptation:** This file is a sanitized public copy of the local Phase 4 report. Machine-local `file:///` links have been converted to repository-relative paths (`../../...`); substantive numbers, tables, and caveats are preserved unchanged, and this file is not the byte-identical original.

# Phase 4 Report: External-Integrator Per-Observation Least-Tried Action Wrapper

## 1. Saved Plan, Boundary Verification & Unit Tests

- **Saved pre-execution plan**: [phase4_plan.json](../../phase4_plan.json)
- **Wrapper & runner scripts**: [external_wrapper.py](../../external_wrapper.py), [run_phase4_wrapper.py](../../run_phase4_wrapper.py)
- **Machine-readable results**: [phase4_results.json](../../phase4_results.json)
- **Base policy preserved**: `phase1_weights.npz` SHA-256 verified as `e31f335039c27dc2b41a937a82d8a52e062828e15c3e049f8f470c614a6a19fd` (`0` retraining, `0` expert calls, original `64`-step limit, reserved confirmation seeds `30000..30199` untouched).

### Implementation Tests (`external_wrapper.py`)
All unit tests passed prior to rollout evaluation (`all_unit_tests_passed = true`):
1. First visits reproduce `np.argmax(probs)` including exact float ties (`[0.2, 0.4, 0.4] -> 1`, `[1/3, 1/3, 1/3] -> 0`).
2. Counts increment only for the action actually issued.
3. All three actions `{0, 1, 2}` are selected once before any receives a second selection at the same observation (`(obs['image'].tobytes(), obs['mission'])`), with post-full-coverage counters evaluated before the decision (`min(prior_counts) >= 1`).
4. Different missions with identical `obs['image']` map to distinct keys.
5. `reset()` clears all retained counts and counters.

### Information Boundary & System Memory Disclosure
- `ExternalObservationCountWrapper` receives **only** `obs['image']`, `obs['mission']`, base policy `probs`, and `reset()` — no `env`, seed, pose, grid, target location, `BabyAIBot`, Phase 3 weights, or evaluator results.
- The frozen MLP (`247 -> 64 ReLU -> 3`) operates on its `247`-d engineered task-relative features, whereas the wrapper keys observations by the exact raw public tuple `(obs['image'].tobytes(), obs['mission'])`. These are distinct representations derived from permitted public observations.
- **The complete wrapped system has episode-local memory** (`action_counts` retained across steps within each episode) and must **not** be described as memoryless.

### Baseline-Success Trajectory Invariant (`93` Episodes)
- `all_baseline_successes_trajectory_identical = true` (`divergence_count = 0` across all `93` Phase 1 successes on pose, action, length, and probabilities within `1e-6`).
- **Baseline successes were preserved as expected because the wrapper never intervened on their trajectories** (`0` repeated raw observations in all `93`). This must **not** be presented as independent evidence that the wrapper is generally non-regressive: successful episodes containing legitimate repeated observations remain an untested risk outside this development set.

---

## 2. Effectiveness & Burden Results (Dev Seeds `0..99`)

| Metric | Phase 1 Frozen Baseline | Phase 4 Wrapped (`Phase 1 + Wrapper`) | Phase 3 Separate Reference *(Expert-Assisted Retrain)* |
| :--- | ---: | ---: | ---: |
| **Access / Supervision Model** | Stateless policy | External interface only (`0` labels, `0` retrain, ep-local memory) | `BabyAIBot` (`3,502` queries) + `5,040` updates |
| **Successes / Timeouts / Exceptions** | **`93` / `7` / `0`** | **`98` / `2` / `0`** | `95` / `5` / `0` |
| **Rescued vs. Phase 1** | `0` | **`5`** (`seeds 3, 15, 29, 43, 80`) | `3` (`seeds 3, 29, 80`) |
| **Harmed vs. Phase 1** | `0` | **`0`** *(see invariant caveat)* | `1` (`seed 88`) |
| Remaining failure seeds | `3, 15, 29, 43, 50, 80, 94` | `50, 94` | `15, 43, 50, 88, 94` |
| Mean length (all `100` eps) | `9.19` | `7.41` | `8.00` |
| Mean length (successful only) | `5.0645` (`93` eps) | `6.2551` (`98` eps) | `5.0526` (`95` eps) |
| Mean length (`93` common baseline successes) | `5.0645` | `5.0645` | — |
| Episodes with repeated `raw_obs` / `feat_hash` / pose | `7` / `7` / `7` | `7` / `7` / `7` *(first repeat triggers wrapper)* | `5` / `5` / `5` |
| Episodes with `>= 1` blocked `forward` | `2` (`seeds 3, 50`) | `5` (`seeds 3, 15, 43, 50, 94`) | `0` |
| Episodes with `>= 2` consecutive blocked `forward` | `2` | **`0`** | `0` |
| Total blocked-`forward` steps | `118` | `10` | `0` |

### Outcomes & Changed Patterns on the Seven Original Phase 1 Failures

| Seed | Mission | Phase 1 Outcome & Length | Wrapped Outcome & Length | Overrides | Retained Keys | Post-Full-Cov Decisions (`Keys`) | End Keys `>=1` All 3 | Blocked-Fwd (`P1 -> P4`) |
| ---: | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| `3` | `go to the red key` | Timeout `64` (`cyc=1`) | **Success `11`** | `1` | `10` | `0` (`0`) | `0` | `62 -> 1` |
| `15` | `go to the grey box` | Timeout `64` (`cyc=2`) | **Success `36`** | `9` | `27` | `0` (`0`) | `2` | `0 -> 1` |
| `29` | `go to the green key` | Timeout `64` (`cyc=8`) | **Success `39`** | `8` | `30` | `1` (`1`) | `1` | `0 -> 0` |
| `43` | `go to the grey ball` | Timeout `64` (`cyc=6`) | **Success `38`** | `12` | `25` | `1` (`1`) | `2` | `0 -> 1` |
| `50` | `go to the green box` | Timeout `64` (`cyc=1`) | **Timeout `64`** | `21` | `42` | `2` (`1`) | `5` | `56 -> 3` |
| `80` | `go to the blue ball` | Timeout `64` (`cyc=8`) | **Success `18`** | `3` | `15` | `0` (`0`) | `0` | `0 -> 0` |
| `94` | `go to the grey ball` | Timeout `64` (`cyc=8`) | **Timeout `64`** | `23` | `34` | `11` (`5`) | `9` | `0 -> 4` |

- Note on rescued episode lengths: `seeds 3, 15, 29, 43, 80` solved in `11, 36, 39, 38, 18` steps (lifting `mean_episode_length_successful_only` from `5.0645` to `6.2551`, while all-100 mean length fell from `9.19` to `7.41`).
- Changed pattern on remaining failures: on `seed 50`, breaking the step-`8` fixed point led to visiting `42` distinct observations (`21` overrides, `3` non-consecutive blocked-forward steps, `2` post-full-coverage decisions at `1` key) before timing out at `64`; on `seed 94`, the agent visited `34` observations with `23` overrides and `11` post-full-coverage decisions across `5` keys before timing out at `64`. These observations do not by themselves establish the root cause of why `50` and `94` timed out.

---

## 3. Intervention Burden & Timing

- **Episodes with `>= 1` override**: **`7 / 100`** (`seeds 3, 15, 29, 43, 50, 80, 94`; `0` among the `93` baseline successes).
- **Total actions differing from base `argmax`**: **`77`** out of `741` total decisions across `100` episodes.
- **Retained exact observation keys per episode**: Mean **`6.54`**; peak **`42`** (`seed 50`).
- **Post-full-coverage metrics** *(measured where all three action counts were already `>= 1` before the decision)*:
  - Total post-full-coverage decisions: **`15`** (`seed 29`: `1`, `seed 43`: `1`, `seed 50`: `2`, `seed 94`: `11`).
  - Sum of per-episode distinct keys with a post-full-coverage decision: **`8`** (`29`: `1`, `43`: `1`, `50`: `1`, `94`: `5`).
  - Sum of per-episode distinct keys reaching all three counts `>= 1` by episode end: **`19`**.
- **Explicit aborts / exceptions**: **`0`** (count cycling wraps cleanly when all 3 counts reach `>= 1`).
- **Timing**: Synchronous `perf_counter()` measured **`0.1828 s`** for the unwrapped Phase 1 dev `0..99` loop vs. **`0.1650 s`** for the wrapped dev `0..99` loop (`0.3584 s` total script runtime, within `300.0 s` cap). Total accumulated time inside `wrapper.reset()` + `wrapper.select_action()` across all `100` episodes (`741` decisions) was **`0.005408 s`** (`~5.4 ms`). The wrapped loop finished slightly faster overall solely because `5` rescued episodes exited in `11–39` steps instead of running `64` steps; per-call wrapper overhead is microsecond-scale and too small relative to single-run Python/Gymnasium loop jitter to report as an overhead ratio.

---

## 4. Remaining Limitations

1. **Exact observation matching**: Using `(obs['image'].tobytes(), obs['mission'])` assumes discrete, noiseless, deterministic observations where revisiting a situation yields byte-identical inputs. It does not directly transfer to continuous, noisy, animated, or non-stationary external games.
2. **Legitimate revisits & aliasing**: The wrapper uses repeated observations as an empirical trigger without distinguishing stalled loops from legitimate backtracking (e.g., dead-end exits) or aliased partial views across distinct physical poses; in environments where passing episodes naturally revisit states, overriding `argmax` remains an untested regression risk.
3. **Boundary conveniences**: Relies on explicit `reset()` episode boundaries and a `3`-action discrete space small enough to cycle locally.
4. **Scope limits**: Results on dev `0..99` remain exploratory; confirmation seeds `30000..30199` remain untouched; this does not rank execution wrappers against retraining in general and is not a Laya evaluation.
