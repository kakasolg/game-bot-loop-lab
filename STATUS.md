# Project Status: What We Know and What Remains Open

This document summarizes the current verified state of the project so future verified replications, counterexamples, and baseline comparisons can visibly update it.

---

## What We Know (Observed in Saved Artifacts)

- **Frozen Phase 1 baseline (`247 -> 64 ReLU -> 3` NumPy MLP):**
  - Trained via behavioral cloning on `499` `BabyAIBot` episodes (`2,656` samples, seeds `10000..10499` excluding `10462`), selected at epoch `21` on validation seeds `20000..20099` (`514` samples). See [Public Phase 1 Report](docs/reports/phase1_report.md).
  - Dev evaluation (`0..99`): `93/100` successes, `7` deterministic pose-and-observation cycles (`3, 15, 29, 43, 50, 80, 94`), `0` repeated raw observations in the `93` successes ([Public Phase 2 Report](docs/reports/phase2_report.md)).
- **Bounded DAgger-1 retrain reference (dev `0..99` only):**
  - `95/100` dev successes (`3,502` extra `BabyAIBot` queries, `5,040` optimizer steps; rescued `3, 29, 80`, harmed `88`, remaining `15, 43, 50, 88, 94`). Excluded from Phase 5 confirmation ([Public Phase 3 Report](docs/reports/phase3_report.md)).
- **Frozen Phase 1 + [`ExternalObservationCountWrapper`](external_wrapper.py):**
  - Dev evaluation (`0..99`): `98/100` (`5` rescued: `3, 15, 29, 43, 80`; `0` harmed; `2` remaining: `50, 94`). Zero divergence on all `93` baseline successes ([Public Phase 4 Report](docs/reports/phase4_report.md)).
  - Reserved evaluation (`30000..30199`, `N = 200`, `max_steps = 64`): **`181/200` (`90.50%`) baseline vs. `196/200` (`98.00%`) wrapped** (`15` wrapper-only successes, `0` baseline-only successes, `4` both-fail; `delta = +0.075`, Newcombe Method 10 95% CI `[+0.039486, +0.120391]`, exact McNemar `p = 6.10352e-05`). Conducted as a single completed evaluation following one aborted pre-flight `NameError` fixed via `gym.__version__` ([Public Phase 5 Report](docs/reports/phase5_confirmation_report.md)).
  - In both dev (`0..99`) and confirmation (`30000..30199`), zero baseline-successful episodes contained repeated raw observations, so zero overrides occurred on baseline successes.
- **Public documentation, safe launcher & seed-3 demo:**
  - Public-safe adapted reports live under [`docs/reports/`](docs/reports); [`reproduce_confirmation.py`](reproduce_confirmation.py) isolates Phase 5 public-seed reproductions in a dedicated directory; and [`demo/seed3_baseline_vs_wrapper.gif`](demo/seed3_baseline_vs_wrapper.gif) illustrates dev seed `3` (`64`-step baseline timeout vs. `11`-step wrapped success with `1` override at 0-indexed step `3`).

---

## What Remains Open

- **Cross-platform / independent reproduction:** [`requirements.txt`](requirements.txt) and `README` verification were performed on Windows (`Python 3.14.3`); independent reproductions on Linux/macOS or clean environments via [`reproduce_confirmation.py`](reproduce_confirmation.py) have not yet been reported ([Issue Brief 1](contributions/01_reproduce_published_result.md)).
- **Harm during legitimate state revisits:** Whether the wrapper degrades performance when a policy succeeds only after revisiting exact raw observations remains untested ([Issue Brief 2](contributions/02_harmful_legitimate_revisit.md)).
- **Simpler execution-side escape baselines:** Random escape or fixed action-cycling rules have not been tested against exact observation-count tracking on a clean evaluation split ([Issue Brief 3](contributions/03_simpler_escape_baselines.md)).
- **Noisy, continuous, or partial-alias environments (including Laya / DSR):** Exact `(bytes(obs['image']), mission)` hashing relies on noiseless discrete inputs; transfer to continuous, animated, or raw-pixel game interfaces is unestablished.
- **Remaining cycles (`30000, 30004, 30095, 30193`) & human cost measurement:** Why the 4 both-fail confirmation seeds timed out and what total human engineering cost compares across approaches remain unmeasured.
- **Genuine owner decisions before release:** No `LICENSE` file has been selected yet, and optional historical archives in Phases 0–4 await final keep/omit review in [`PUBLICATION_MANIFEST.md`](PUBLICATION_MANIFEST.md) (original private-path reports and `phase0a_env_inspection.json` are excluded via [`.gitignore`](.gitignore)).

### Deferred Research Question: Execution-Layer Safeguards vs. Reaction Latency
*Status: Deferred — not implemented or evaluated.*

- **Practical question:** Beyond whether execution safeguards improve task completion, do they delay action delivery enough to degrade real-time gameplay?
- **Candidate approaches to compare:** Observation-derived action masking, the existing observation-count wrapper ([`ExternalObservationCountWrapper`](external_wrapper.py)), and a lightweight repetition/recovery harness.
- **Latency mechanisms & boundary:** Simple local O(1)/in-memory checks differ in cost from safeguards requiring extra model calls, planning, or network requests, and post-inference masking does not by itself reduce the base model's inference time. Existing BabyAI wrapper timings measure only isolated Python/NumPy calls in a turn-based toy environment and do not establish end-to-end latency or real-time DSR suitability.
- **Future evaluation requirements:** Any future evaluation should jointly measure task outcomes and observation-to-action latency—including tail latency and missed action deadlines—and any new method developed using the now-public `30000..30199` seeds requires a fresh held-out seed set for confirmation claims.
