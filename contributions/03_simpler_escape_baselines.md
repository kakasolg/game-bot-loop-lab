# Is a simpler escape rule enough?

## Why This Question Matters

[`ExternalObservationCountWrapper`](../external_wrapper.py) stores a per-episode dictionary mapping exact raw observation-and-mission keys to `[left, right, forward]` action counts (`peak = 39` retained keys on confirmation seed `30004`). That works on deterministic discrete tiles, **but** exact key matching breaks as soon as observations vary slightly (e.g., animations, pixels, noise). Before keeping exact-observation count memory, a practical bot integrator should ask: **Could an even simpler stateless or small-counter escape rule rescue the same stalled policies without building or hashing a per-observation dictionary?**

## What Is Already Known

- On confirmation seeds `30000..30199`, the wrapper rescued `15/19` baseline timeouts (`181/200 -> 196/200`), triggered overrides only on the `19` baseline timeout episodes (`143` total overrides), and left `4` timeouts unresolved (`30000, 30004, 30095, 30193`).
- Earlier Phase 2 diagnostics ([Public Phase 2 Report](../docs/reports/phase2_report.md)) found two baseline failure modes on dev `0..99`: `len=1` fixed points (blocked `forward` steps) and `len=2..8` multi-step spatial/heading cycles.
- Random escape and fixed action cycling have **not** yet been evaluated in this project—they are candidate comparisons, not established baselines.

## What Remains Unknown

- How much of the `181 -> 196` gain comes from tracking counts *per exact observation* versus simply perturbing repeated actions or running a simple stateless/small-window escape rule:
  - **Candidate comparison A (random escape):** with a small fixed probability $\epsilon$ (or only after $k$ repeated `(action)` or consecutive same-action triggers), pick uniformly among non-argmax discrete actions `{0, 1, 2}`.
  - **Candidate comparison B (fixed action cycling / last-$k$ loop breaker):** cycle `[left, right, forward]` deterministically whenever the exact same action repeats $k$ consecutive times or whenever the single immediately preceding observation repeats (`k = 1` memory instead of an unbounded per-episode table).
- Whether simpler rules introduce **regressions** (`baseline_only_success > 0`) on the episodes where [`ExternalObservationCountWrapper`](../external_wrapper.py) had `0` regressions, or fail to escape longer multi-step spatial cycles (`cycle_length = 8` or `10`).

## Bounded Task

Implement one cleanly separated comparison script in `experiments/<name>/` keeping [`phase1_weights.npz`](../phase1_weights.npz) and [`encode_observation`](../run_phase1.py) fixed:

1. Pre-declare **1–2 concrete simpler candidate rules** (e.g., seeded $\epsilon$-random escape on `0..99` to choose one reasonable hyperparameter, or zero-hyperparameter deterministic fixed cycling on repeated consecutive `forward`/last-observation) and document every design or tuning choice using **dev seeds `0..99` only**.
2. Once frozen, evaluate Baseline (`phase1_weights.npz`), [`ExternalObservationCountWrapper`](../external_wrapper.py), and your frozen simpler candidate rule(s) side-by-side on a **fresh held-out seed range not used for tuning** (for example `30200..30399`, keeping `30000..30199` as a reference check only—never tune on `30000..30199` or `30200..30399` and claim confirmation).
3. Compare completion rate, paired rescue/regression counts vs. Baseline and vs. `ExternalObservationCountWrapper`, episode lengths, and memory/randomness requirements.

## Minimum Evidence Needed

- Exact algorithm pseudocode/code for the candidate baseline(s), plus RNG seed if stochastic (`random escape`).
- Transparent seed split disclosure: what was tuned or inspected on dev `0..99` (or `30000..30199`), and exact fresh held-out seeds (`e.g. 30200..30399`) used for final paired comparison.
- Full paired contingency tables comparing Candidate vs. Frozen Baseline and Candidate vs. `ExternalObservationCountWrapper` (rescues, regressions, episode lengths, total decisions).

## How to Report Negative, Divergent, or Partial Results

- **Simpler rule ties or beats `ExternalObservationCountWrapper`:** great—shows per-observation count tables were unnecessary for this task! Report any trade-off in episode length or stochastic variance honestly.
- **Simpler rule fails to rescue multi-step loops or causes regressions on passing runs:** equally valuable—shows *why* per-observation history matters here rather than simple action perturbation. Report the paired failure counts and seed examples.

## Stopping Point

Stop after running the single pre-declared held-out comparison and reporting the paired tables and failure modes. Do not iteratively retune $\epsilon$, cycle length, or trigger thresholds against the held-out evaluation seeds after inspecting results.
