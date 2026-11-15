# Can you find a legitimate revisit that this wrapper harms?

## Why This Question Matters

Across both dev (`0..99`, `93` baseline successes) and confirmation (`30000..30199`, `181` baseline successes), [`ExternalObservationCountWrapper`](../external_wrapper.py) produced **zero** regressions (`0/274` combined) strictly because zero baseline-successful trajectories ever repeated a raw `(bytes(obs['image']), obs['mission'])` key—meaning the wrapper never intervened on a winning run. If an agent legitimately revisits an observation while making valid progress (for example, backing out of a dead end, exploring around an obstacle, or observing unchanged views during wait/turn steps), this count-based override rule will force a non-argmax action anyway. Finding concrete cases where that override breaks success establishes the wrapper's failure boundary.

## What Is Already Known

- On `BabyAI-GoToLocal-v0` dev `0..99` and confirmation `30000..30199`, `non_repeating_baseline_success_divergences = []` and `baseline_successful_with_repeated_raw_obs_count = 0`.
- When a raw `(image, mission)` key repeats once, the wrapper forces the second-highest-probability action instead of the base `argmax`, regardless of whether the base action was actually stalled or simply re-entering a previously seen view cleanly.
- The wrapper has no notion of physical progress, dead-end topology, or multi-step goals.

## What Remains Unknown

- Whether any unused seed range of `BabyAI-GoToLocal-v0` contains baseline-successful episodes with a repeated raw observation where the wrapper overrides `argmax` into a timeout or longer detour.
- How often regressions occur on related discrete BabyAI/MiniGrid tasks with more natural backtracking (e.g., two-room layouts, locked/blocked corridors, or stochastic/noisy transitions where observations legitimately repeat).
- Whether adding simple policy noise, partial observation aliasing, or mild sensory jitter triggers premature overrides on winning trajectories.

## Bounded Task

Design one **read-only counterexample hunt** in a separate directory (leaving all published repo artifacts untouched):

1. Evaluate the frozen [`phase1_weights.npz`](../phase1_weights.npz) baseline vs. [`ExternalObservationCountWrapper`](../external_wrapper.py) on a **pre-declared, previously unused** seed range in `BabyAI-GoToLocal-v0` (for example `40000..40499`, `N = 500`), **or** on a clearly specified discrete MiniGrid variant/perturbation (stating all seed rules and task changes upfront).
2. Log every seed where the frozen baseline succeeds (`success=True`) **and** visits at least one raw `(obs['image'], obs['mission'])` key more than once.
3. Inspect those paired runs for regressions (`baseline_only_success`) or trajectory lengthening (`wrapped_length > baseline_length`).

## Minimum Evidence Needed

- Exact script/environment definition, task name, `max_steps`, seed range chosen *before* inspecting outcomes, and SHA-256 verification showing [`phase1_weights.npz`](../phase1_weights.npz) and [`external_wrapper.py`](../external_wrapper.py) were unmodified.
- Summary counts across the tested slice: `both_succeed`, `wrapper_only_success`, **`baseline_only_success`**, `both_fail`, plus how many baseline-successful runs contained `repeated_raw_obs_steps > 0`.
- For at least one concrete counterexample seed (or up to 5 if multiple occur): mission, step-by-step paired trace showing which observation repeated, base probabilities, the forced override action, and how it caused timeout or regression.
- Full disclosure if you searched multiple tasks or seed blocks before finding a counterexample.

## How to Report Negative, Divergent, or Partial Results

- **Finding zero regressions even after searching a fresh 500-seed `BabyAI-GoToLocal-v0` slice** is itself a valuable empirical result: report whether any baseline-successful run in that slice even had `repeated_raw_obs_steps > 0` (`override_preserved_success` vs. never triggered).
- Report total seeds screened honestly so readers do not mistake a cherry-picked failure rate for an unbiased population rate.
- Do not modify [`ExternalObservationCountWrapper`](../external_wrapper.py) to "fix" discovered counterexamples inside the same confirmation benchmark.

## Stopping Point

Stop once you have reported the pre-declared slice's full paired 2×2 table, revisit-override counts, and step trace(s) for any discovered `baseline_only_success` counterexample (or confirmed `0` regressions found on that pre-declared slice). Any proposed wrapper fix requires a separate issue/experiment proposal on fresh seeds.
