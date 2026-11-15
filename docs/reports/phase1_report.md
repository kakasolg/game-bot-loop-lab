> **Public documentation adaptation:** This file is a sanitized public copy of the local Phase 1 report. Machine-local `file:///` links have been converted to repository-relative paths (`../../...`); substantive numbers, counts, and caveats are preserved unchanged, and this file is not the byte-identical original.

# Phase 1 Report: Independent CPU-Only NumPy BC Baseline

## 1. Configuration & Amendment

- Original configuration (saved prior to collection): [phase1_config.json](../../phase1_config.json)
  - Engineered, current-observation-only `247`-dim task-relative features (`7×7×5` cell one-hot `{unseen, empty, wall, target_match, non_target_object}` + `front_blocked` + `target_in_view`; no `direction`, history, global coordinates, or expert map).
  - Pure-NumPy MLP: `247 -> 64 (ReLU) -> 3`, `init_seed=42`, Adam (`lr=3e-3, l2=1e-4, batch_size=64, shuffle_seed=42, max_epochs=60`), checkpoint selected by lowest `val_cross_entropy`.
- Amendment: [phase1_config_amendment.json](../../phase1_config_amendment.json)
  - Excluded `seed=10462` completely (including its partial `step=0` sample) because the expert proposed `pickup` (`3`), outside `{left, right, forward}`. Original [phase1_collection.json](../../phase1_collection.json) and `phase1_demos.npz` were preserved unchanged.

## 2. Verified Dataset Counts ([phase1_filtering_manifest.json](../../phase1_filtering_manifest.json))

- **Provenance verification**: Replaying `10000..10499` and `20000..20099` matched `phase1_demos.npz` identically.
- **Filtered train (`phase1_demos_filtered.npz`)**: `499` complete episodes (`10000..10499` excluding `10462`), `2,656` samples (`2,657 - 1`, `0` samples from `10462` remaining).
  - Action counts: `forward: 1,616`, `right: 553`, `left: 487`.
  - `front_blocked`: `436` | `target_in_view`: `2,235` | hidden target: `421`.
- **Validation (unchanged)**: `100` episodes (`20000..20099`), `514` samples.
  - Action counts: `forward: 311`, `right: 109`, `left: 94`.
  - `front_blocked`: `75` | `target_in_view`: `426` | hidden target: `88`.

## 3. Training & Validation Results ([phase1_training.json](../../phase1_training.json))

- **Selected checkpoint**: **Epoch 21** (lowest `val_cross_entropy = 0.235611`), saved to `phase1_weights.npz` prior to rollout evaluation. All 60 epochs finished with finite losses (`0.6319 s`).
- **Train at Epoch 21 (`2,656` steps)**: CE `0.056288`, overall agreement `0.983057` (`left`: `487` / `0.979466`, `right`: `553` / `0.969259`, `forward`: `1,616` / `0.988861`; visible target `2,235`: `0.991499` / CE `0.032952`; hidden target `421`: `0.938242` / CE `0.180172`; blocked front `436`: `0.995413` / CE `0.029328`, predicted `forward=0`).
- **Validation at Epoch 21 (`514` steps)**: CE `0.235611`, overall agreement `0.920233` (`left`: `94` / `0.882979`, `right`: `109` / `0.853211`, `forward`: `311` / `0.954984`; visible target `426`: `0.955399` / CE `0.130651`; hidden target `88`: `0.750000` / CE `0.743711`; blocked front `75`: `0.933333` / CE `0.148781`, predicted `left=40, right=35, forward=0`).

## 4. Development Rollouts (`0..99`) & Repetition Diagnostics ([phase1_eval_results.json](../../phase1_eval_results.json))

- **Outcomes**: `93 / 100` succeeded, `7 / 100` timed out at `64` steps, `0` exceptions; mean episode length `9.19` steps (`0.189 s` eval, `2.3052 s` total continuation time within `300 s` cap).
- **Separated repetition metrics**:
  - Episodes with repeated 247-d policy input (`feat_hash`): `7` (same `7` timeouts; `0` among the `93` successes).
  - Episodes with repeated raw `(image, mission)`: `7`.
  - Episodes with observation aliasing without pose repeat: `0`.
  - Episodes with repeated true `(pos, dir)` pose: `7`.
  - Episodes with consecutive blocked-`forward` (`>= 2`): `2` (`total_blocked_forward_steps = 118`: `62` in seed `3`, `56` in seed `50`).
  - Pose-cycle lengths: `2` of length `1`, `1` of length `2`, `4` of length `>= 3` (`1` of length `6`, `3` of length `8`).
  - Target visibility in cycles: `7 / 7` cycle episodes had `target_in_view=True` during at least part of the cycle (`0` never-visible cycles).

### Representative Traces
1. **Seed 50 (`"go to the green box"`, cycle length 1, blocked `forward`)**: Reaches `(5, 1, dir=2)` at step `8` with `front_blocked=True, target_in_view=True`; predicts `forward` (`p=0.99249`) at step `8` and repeats it for `56` consecutive steps until timeout at step `64`. (Seed `3`, `"go to the red key"`, similarly loops at `(2, 1, dir=1)` from step `2` for `62` consecutive blocked `forward` steps with `p=0.602771`.)
2. **Seed 15 (`"go to the grey box"`, cycle length 2, `right <-> left`)**: Enters `(4, 4, dir=2)` at step `5` (`front_blocked=True, target_in_view=True`), selects `right` (`p=0.974255`) to `(4, 4, dir=3)` (`front_blocked=True, target_in_view=True`), then selects `left` (`p=0.638094`) back to `(4, 4, dir=2)`, repeating every 2 steps until step `64`.
3. **Seed 29 (`"go to the green key"`, cycle length 8)**: Cycles from step `0` through poses `(4,4,0) -> (5,4,0) -> (6,4,0) -> (6,4,1) -> (6,4,2) -> (5,4,2) -> (4,4,2) -> (4,4,3)` with actions `[forward, forward, right, right, forward, forward, right, right]` (`target_in_view` is `True` on steps `0..4` and `False` on steps `5..7`).
