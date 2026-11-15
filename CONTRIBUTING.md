# Contributing

Thank you for looking closely at this project. Replications, counterexamples, simpler baselines, and corrections to claims or documentation are all valuable contributions.

## Ground Rules for Frozen Evidence

- **Do not overwrite published artifacts.** Leave frozen weights ([`phase1_weights.npz`](phase1_weights.npz)), [`external_wrapper.py`](external_wrapper.py), [`phase5_confirmation_plan.json`](phase5_confirmation_plan.json), [`phase5_confirmation_results.json`](phase5_confirmation_results.json), [`phase5_confirmation_episodes.jsonl`](phase5_confirmation_episodes.jsonl), and existing phase reports untouched in pull requests unless a PR explicitly targets a documented addendum or bug report.
- **Keep new work isolated.** Put new scripts, alternative wrappers, baselines, or ablation outputs in a separate directory (for example, `experiments/<short_name>/`) so published results stay intact.
- **Keep holdout sets clean.** Seeds `0..99` are exploratory development seeds and seeds `30000..30199` are the now-public Phase 5 reproduction set. Any newly designed or tuned method requires a fresh, non-overlapping seed range for confirmation claims.
- **No unverified claims or fabricated outputs.** Report negative, null, and inconclusive outcomes accurately alongside positive ones.

---

## Open Questions

These questions remain open and unestablished here (they are practical gaps to test, not claims of novelty):

1. **Is action masking enough?**
   Would excluding observably blocked actions achieve similar gains to the action-count wrapper?
   This comparison has not been evaluated. See [Issue Brief 3](contributions/03_simpler_escape_baselines.md) and the [deferred note in STATUS.md](STATUS.md#deferred-research-question-execution-layer-safeguards-vs-reaction-latency).
2. **When is revisiting useful?**
   Can an execution layer distinguish an ineffective loop from legitimate backtracking without privileged state?
   Successful repeated-observation trajectories remain untested. See [Issue Brief 2](contributions/02_harmful_legitimate_revisit.md).
3. **Does it fit a real-time action budget?**
   Can the benefit be retained within an observation-to-action latency budget, including tail latency and missed deadlines?
   The existing toy-environment timings do not answer this. See the [deferred latency note in STATUS.md](STATUS.md#deferred-research-question-execution-layer-safeguards-vs-reaction-latency).

---

## 1. Reporting a Reproduction

Use [Issue Brief 1](contributions/01_reproduce_published_result.md) as a guide and include:

- **Platform & environment versions:** OS, architecture, Python version, `numpy`, `gymnasium`, `minigrid`, and whether `.venv` used [`requirements.txt`](requirements.txt) or another install path.
- **Verification checks:** Whether [`run_phase5_confirmation.py`](run_phase5_confirmation.py) lock verification (`all_verified`) and unit tests passed when run in an isolated copy.
- **Observed paired summary:** `both_succeed`, `wrapper_only_success`, `baseline_only_success`, `both_fail`, `delta`, 95% CI, McNemar `p`-value, and any exact differences from [`phase5_confirmation_results.json`](phase5_confirmation_results.json).

## 2. Submitting a Counterexample

Use [Issue Brief 2](contributions/02_harmful_legitimate_revisit.md) as a guide and include:

- **Exact environment setup and seed** (including `env_id`, `max_steps`, and any observation/action modifications if outside vanilla `BabyAI-GoToLocal-v0`).
- **Frozen policy & wrapper hashes** (`sha256` of checkpoint and wrapper used).
- **Paired step trace** showing baseline action selection, wrapper observation-count trigger/override step, and how the override changed an otherwise successful or shorter trajectory into failure or longer detour.

## 3. Proposing a New Experiment Before Running or Integrating

Before integrating new wrappers, retraining runs, or alternate benchmarks into the repo, open a short proposal covering:

1. **Concrete question** (for example: testing a simpler baseline in [Issue Brief 3](contributions/03_simpler_escape_baselines.md)).
2. **Comparison arms and frozen components** (what stays fixed vs. what varies).
3. **Seed split discipline** (distinguishing tuning/dev seeds from fresh evaluation seeds).
4. **Metrics and negative-result criteria** (what outcome would count against your hypothesis).

## 4. Submitting Small Corrections

Documentation clarifications, wording precision fixes, cross-platform setup notes, and statistical/audit corrections are welcome without proposing a full experiment—just link directly to the relevant file and line numbers.
