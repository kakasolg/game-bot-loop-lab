# Can you reproduce the published result?

## Why This Question Matters

The Phase 5 result (`181/200` baseline vs. `196/200` wrapped on `BabyAI-GoToLocal-v0` seeds `30000..30199`) was produced once on a single Windows machine (`Python 3.14.3`, `numpy 2.5.4`, `gymnasium 1.4.0`, `minigrid 3.2.0`). Confirming whether an independent environment reproduces exact counts, confidence intervals, and runtime—or uncovers version sensitivity or setup friction—validates the public baseline before anyone builds on it.

## What Is Already Known

- [`phase5_confirmation_plan.json`](../phase5_confirmation_plan.json) records the locked artifact SHA-256 checksums and package versions verified before the published run completed.
- [`phase5_confirmation_results.json`](../phase5_confirmation_results.json) reports `181` both-succeed, `15` wrapper-only successes (`30005, 30018, 30021, 30027, 30049, 30061, 30090, 30093, 30111, 30125, 30144, 30161, 30177, 30178, 30179`), `0` baseline-only successes, and `4` both-fail (`30000, 30004, 30095, 30193`).
- [Public Phase 5 Report](../docs/reports/phase5_confirmation_report.md) documents that an initial invocation aborted pre-flight on `NameError` (`gymnasium.__version__`) before touching any reserved seed, after which a 1-line fix (`gym.__version__`) enabled the completed run.

## What Remains Unknown

- Whether [`requirements.txt`](../requirements.txt) (compiled on Windows `Python 3.14.3`) installs cleanly on Linux, macOS, or other Python versions (`3.11`–`3.13`).
- Whether different BLAS/NumPy builds or MiniGrid/Gymnasium minor versions change floating-point logits enough to shift any boundary decision across seeds `30000..30199`.

## Bounded Task

1. Create a fresh environment, install dependencies (starting with [`requirements.txt`](../requirements.txt), or recording exact substitutions if your OS/Python version cannot match those pins), and run the non-episode unit checks in [`phase5_stats.py`](../phase5_stats.py), [`external_wrapper.py`](../external_wrapper.py), and [`tests/test_reproduce_confirmation.py`](../tests/test_reproduce_confirmation.py).
2. Run [`reproduce_confirmation.py`](../reproduce_confirmation.py) targeting a dedicated new directory (for example `python reproduce_confirmation.py --dest repro_runs/my_repro`) so the unchanged [`run_phase5_confirmation.py`](../run_phase5_confirmation.py) executes inside an isolated folder without overwriting published JSON/JSONL files. Note that `30000..30199` are public reproduction seeds, not a new confirmation holdout.
3. Compare your generated `phase5_confirmation_results.json` against the published file (noting any `lock_verification` version differences if your OS required different versions).

## Minimum Evidence Needed

- OS, CPU/architecture, exact `python`, `numpy`, `gymnasium`, and `minigrid` versions used.
- Whether `lock_verification`, `wrapper_unit_tests`, and `statistical_unit_tests` passed.
- Your observed `paired_2x2_table` (`both_succeed`, `wrapper_only_success`, `baseline_only_success`, `both_fail`), `delta`, Newcombe Method 10 95% CI, exact McNemar `p`-value, and any seed list differences vs. [`phase5_confirmation_results.json`](../phase5_confirmation_results.json).
- If [`run_phase5_confirmation.py`](../run_phase5_confirmation.py) fails on install or hash/version checks, paste the exact dependency install log or lock-verification error output.

## How to Report Negative, Divergent, or Partial Results

- **Divergent counts or seeds** are high-value bug/sensitivity findings: report every differing seed, environment version, and step length difference rather than retuning or patching around it.
- **Failed setup or platform incompatibility** is also a useful reproduction outcome: share the exact OS, Python version, install command, and error message so dependency or documentation blockers can be tracked honestly in [`STATUS.md`](../STATUS.md).
- Disclose any modifications made to run on a different OS or package set; do not describe a modified run as an exact locked reproduction.

## Stopping Point

Stop once you have compared one clean run against [`phase5_confirmation_results.json`](../phase5_confirmation_results.json) (or documented the exact installation/verification blocker) and reported the environment details and outcome diff. Do not tune the wrapper or expand onto new seed ranges inside this task.
