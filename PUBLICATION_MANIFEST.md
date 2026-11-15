# Public File Manifest

This manifest classifies every file and directory in the workspace for public GitHub preparation. No files or intermediate checkpoints were deleted, and **no Git repository has been initialized** (`has_dot_git = false`), so no files are currently Git-tracked.

> [!IMPORTANT]
> **Preventing accidental commits:** [`.gitignore`](.gitignore) enforces exclusion of local-only preservation reports (`phase1_report.md`–`phase5_confirmation_report.md` containing machine-local paths), `phase0a_env_inspection.json`, optional large/regenerable `.npz` binary archives (other than required `phase1_weights.npz`), `.venv/`, caches, and reproduction outputs so a future `git add .` cannot accidentally stage them.

---

## 1. Publication Preparation Status

### Completed Publication-Preparation Work
- **License & third-party notices:** Standard unmodified [`LICENSE`](LICENSE) (`MIT License`, `Copyright (c) 2026 Mark Sol`) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) added, with license scope and non-binding courtesy note documented in [`README.md`](README.md).
- **Public documentation & positioning:** [`README.md`](README.md), [`CONTRIBUTING.md`](CONTRIBUTING.md), [`STATUS.md`](STATUS.md), and three contribution briefs under [`contributions/`](contributions).
- **Public-safe adapted reports ([`docs/reports/`](docs/reports)):** Sanitized copies of the Phase 1–5 reports (`phase1_report.md`, `phase2_report.md`, `phase3_report.md`, `phase4_report.md`, `phase5_confirmation_report.md`) with repository-relative links, explicit public-adaptation notices, and the private local execution-log link removed from Section 7 while preserving all substantive results, limitations, and provenance meaning. Original reports at the workspace root remain unmodified locally and ignored by `.gitignore`.
- **Isolated non-destructive reproduction launcher:** [`reproduce_confirmation.py`](reproduce_confirmation.py) and its non-episode test suite [`tests/test_reproduce_confirmation.py`](tests/test_reproduce_confirmation.py), copying only the `8` required frozen inputs into a validated new/empty directory, running the unchanged [`run_phase5_confirmation.py`](run_phase5_confirmation.py) there, and preserving all hash and version checks.
- **Illustrative dev-seed-3 real-frame demo ([`demo/`](demo)):** [`run_seed3_demo.py`](run_seed3_demo.py) executed strictly `2` episodes on dev seed `3` (`Baseline` timeout at `64` steps; `Wrapped` success at `11` steps with `1` override at `0`-indexed step `3`), verified exact agreement against saved Phase 2/4 traces, and saved [`demo/seed3_baseline_vs_wrapper.gif`](demo/seed3_baseline_vs_wrapper.gif), [`demo/seed3_keyframes.png`](demo/seed3_keyframes.png), and [`demo/seed3_demo_metadata.json`](demo/seed3_demo_metadata.json) labeled verbatim and integrated into [`README.md`](README.md).
- **Privacy, link, and frozen-artifact verification:** Required raw JSON/JSONL files audited (`0` private paths/usernames/credentials); all public Markdown links resolve within the intended `Core + Group A` set (`49` files including license files); locked Phase 5 SHA-256 hashes remain unchanged.

### Approved Work Still Pending
- None from the current publication-preparation scope (`Core + Group A` plus [`LICENSE`](LICENSE) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) = `49` files in the proposed initial publication set; `Group B` kept excluded).

### Concrete Technical Blockers & Environment Constraints
- **Platform dependency lock:** [`requirements.txt`](requirements.txt) (`Python 3.14.3`, `minigrid==3.2.0`, `gymnasium==1.4.0`, `numpy==2.5.4`, `pygame-ce==2.5.8`) was resolved on Windows `x86_64` and has not yet been tested on Linux or macOS.
- **Optional imaging packages absent in `.venv`:** `PIL` (`Pillow`), `imageio`, `matplotlib`, and `cv2` are not installed; [`run_seed3_demo.py`](run_seed3_demo.py) works without them via `minigrid` (`rgb_array`), `pygame-ce`, `numpy`, and a pure-Python GIF89a LZW encoder.
- **Original root reports & `phase0a_env_inspection.json` not directly publishable:** They remain preserved locally at the workspace root and are blocked from Git staging in [`.gitignore`](.gitignore) due to machine-local `file:///`, transcript, or `.venv` paths.
- **Third-party rights nuances:** Upstream `minigrid==3.2.0` shows a metadata/file discrepancy (`MIT` vs. `Apache-2.0`), `pygame-ce` bundles no `dist-info` license text (`LGPL v2.1`), and attribution alone does not settle legal edge cases for `BabyAIBot`-trained weights (`phase1_weights.npz`) or rendered frames (`THIRD_PARTY_NOTICES.md`).

### Genuine Owner Decisions Required Before Release
1. **Final repository name & Git/GitHub release authorization:** Confirm repository name (provisional: `game-bot-loop-lab`) and authorize local Git init/commit and remote push.
2. **Optional Linux/macOS environment verification:** Deferred; decide whether to verify cross-platform installs later.

---

## 2. Required Core Public File Set (Ready for Publication)

| Path | Size | Role |
| :--- | ---: | :--- |
| [`README.md`](README.md) | — | Public entry point, summary, demo visual, limitations, and safe reproduction steps |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | — | Rules for reproductions, counterexamples, proposals, and doc fixes |
| [`STATUS.md`](STATUS.md) | — | Compact summary of verified knowledge vs. open questions |
| [`PUBLICATION_MANIFEST.md`](PUBLICATION_MANIFEST.md) | — | This classification, status summary, and pre-release checklist |
| [`LICENSE`](LICENSE) | — | Standard unmodified MIT License (`Copyright (c) 2026 Mark Sol`) |
| [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) | — | Verified upstream dependency licenses, notices, and provenance uncertainties |
| [`.gitignore`](.gitignore) | — | Enforced exclusions for private originals, large `.npz` archives, venvs, and outputs |
| [`requirements.txt`](requirements.txt) | 303 B | Pinned Windows Python `3.14.3` dependencies |
| [`docs/reports/phase1_report.md`](docs/reports/phase1_report.md) | — | Public-safe adapted Phase 1 report |
| [`docs/reports/phase2_report.md`](docs/reports/phase2_report.md) | — | Public-safe adapted Phase 2 report |
| [`docs/reports/phase3_report.md`](docs/reports/phase3_report.md) | — | Public-safe adapted Phase 3 report |
| [`docs/reports/phase4_report.md`](docs/reports/phase4_report.md) | — | Public-safe adapted Phase 4 report |
| [`docs/reports/phase5_confirmation_report.md`](docs/reports/phase5_confirmation_report.md) | — | Public-safe adapted Phase 5 report + Section 7 provenance addendum |
| [`contributions/01_reproduce_published_result.md`](contributions/01_reproduce_published_result.md) | — | Contribution brief 1: independent reproduction |
| [`contributions/02_harmful_legitimate_revisit.md`](contributions/02_harmful_legitimate_revisit.md) | — | Contribution brief 2: counterexamples during legitimate revisits |
| [`contributions/03_simpler_escape_baselines.md`](contributions/03_simpler_escape_baselines.md) | — | Contribution brief 3: simpler escape-rule baselines |
| [`phase1_weights.npz`](phase1_weights.npz) | 65,234 B | **Required frozen base checkpoint** (SHA-256 locked by confirmation plan) |
| [`phase1_config.json`](phase1_config.json) | 1,650 B | Phase 1 architecture/hyperparameter configuration (hash-verified; privacy clean) |
| [`run_phase1.py`](run_phase1.py) | 28,475 B | Feature encoder (`encode_observation`), `forward_mlp`, assertions (hash-verified) |
| [`run_phase1_continuation.py`](run_phase1_continuation.py) | 21,478 B | Phase 1 training & dev evaluator (hash-verified) |
| [`external_wrapper.py`](external_wrapper.py) | 6,010 B | Episode-local observation-count wrapper & unit tests (hash-verified) |
| [`phase5_stats.py`](phase5_stats.py) | 8,598 B | Newcombe Method 10 CI, exact McNemar test, and unit tests (hash-verified) |
| [`run_phase5_confirmation.py`](run_phase5_confirmation.py) | 22,616 B | Frozen confirmation runner for seeds `30000..30199` (hash-verified) |
| [`reproduce_confirmation.py`](reproduce_confirmation.py) | — | Isolated non-destructive launcher (`--dest`) preserving frozen runner checks |
| [`tests/test_reproduce_confirmation.py`](tests/test_reproduce_confirmation.py) | — | Non-episode unit tests for launcher safety, staging, and pre-flight checks |
| [`run_seed3_demo.py`](run_seed3_demo.py) | — | Dev-seed-`3` comparison visual generator & dependency checker |
| [`demo/seed3_baseline_vs_wrapper.gif`](demo/seed3_baseline_vs_wrapper.gif) | 153,200 B | Illustrative dev-seed-`3` animated GIF comparison |
| [`demo/seed3_keyframes.png`](demo/seed3_keyframes.png) | 33,563 B | Illustrative dev-seed-`3` static 4-keyframe progression strip |
| [`demo/seed3_demo_metadata.json`](demo/seed3_demo_metadata.json) | — | Machine-readable trace & media metadata for dev seed `3` |
| [`phase5_confirmation_plan.json`](phase5_confirmation_plan.json) | 2,723 B | Locked SHA-256 checksums, runtime cap, and statistical protocol (privacy clean) |
| [`phase5_confirmation_results.json`](phase5_confirmation_results.json) | 316,408 B | Published consolidated results across `30000..30199` (privacy clean) |
| [`phase5_confirmation_episodes.jsonl`](phase5_confirmation_episodes.jsonl) | 228,327 B | Published per-seed paired episode summaries for `30000..30199` (privacy clean) |

---

## 3. Optional Historical & Training Artifacts (Phases 0–4)

### 3A. Lightweight Audit Trail — **Included in Proposed `Core + Group A` Publication Set** (`17` files, `1,512,974` B / `~1.44` MiB total)
Included so every audit script and JSON reference cited in [`docs/reports/`](docs/reports) resolves cleanly inside the repository:

- **Phase 0:** [`inspect_env.py`](inspect_env.py), [`run_expert_check.py`](run_expert_check.py), [`phase0b_expert_results.json`](phase0b_expert_results.json), [`phase0_report.md`](phase0_report.md).
- **Phase 1:** [`phase1_collection.json`](phase1_collection.json), [`phase1_config_amendment.json`](phase1_config_amendment.json), [`phase1_filtering_manifest.json`](phase1_filtering_manifest.json), [`phase1_training.json`](phase1_training.json), [`phase1_eval_results.json`](phase1_eval_results.json) (`749,633` B).
- **Phase 2:** [`audit_phase2.py`](audit_phase2.py), [`phase2_audit_results.json`](phase2_audit_results.json) (`551,008` B).
- **Phase 3 (Reference Metadata):** [`run_phase3_dagger.py`](run_phase3_dagger.py), [`phase3_plan.json`](phase3_plan.json) *(mentions generic `.venv` string only; no local user/drive path)*, [`phase3_results.json`](phase3_results.json).
- **Phase 4 (Dev `0..99`):** [`run_phase4_wrapper.py`](run_phase4_wrapper.py), [`phase4_plan.json`](phase4_plan.json) *(mentions generic `.venv` string only; no local user/drive path)*, [`phase4_results.json`](phase4_results.json).

### 3B. Optional Binary Archives — **Excluded via `.gitignore`** (`5` files, `8,315,578` B / `~7.93` MiB total)
Preserved locally on disk and excluded in [`.gitignore`](.gitignore). Reproducing the headline `181 -> 196` confirmation result does not require them, whereas reproducing historical Phase 1 or Phase 3 training/retraining scripts may require regenerating these archives:

- `phase1_demos.npz` (`76,453` B) and `phase1_demos_filtered.npz` (`79,662` B) — regenerable via `run_phase1.py` / `run_phase1_continuation.py`.
- `phase3_demos_dagger1.npz` (`8,028,995` B), `phase3_weights_control.npz` (`65,234` B), and `phase3_weights_dagger.npz` (`65,234` B) — regenerable via `run_phase3_dagger.py`.

---

## 4. Local Preservation & Strictly Excluded Files (Enforced in `.gitignore`)

Do **not** include these in a public Git commit; all root-workspace items below remain intact locally and are listed in [`.gitignore`](.gitignore):

- **Local preservation report originals (NOT publication-ready due to machine-local paths/URIs):** `phase1_report.md`, `phase2_report.md`, `phase3_report.md`, `phase4_report.md`, and `phase5_confirmation_report.md` (which also references a private local session log URI). Public-safe adaptations live in [`docs/reports/`](docs/reports).
- **Local environment inspection output:** `phase0a_env_inspection.json` — contains an absolute Windows `.venv` filesystem path (`minigrid_file`).
- **Environments, caches, logs & temporary folders:** `.venv/`, `__pycache__/`, `.gemini/`, external session logs (`transcript*.jsonl`, `*session*.jsonl`), `*.log`, `repro_runs/`, `reproduction_outputs/`, `experiments/*/outputs/`, and `experiments/*/runs/`.
