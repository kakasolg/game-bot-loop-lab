> **Public documentation adaptation:** This file is a sanitized public copy of the local Phase 2 report. Machine-local `file:///` links have been converted to repository-relative paths (`../../...`); substantive numbers, traces, and caveats are preserved unchanged, and this file is not the byte-identical original.

# Phase 2 Diagnostic Audit Report: Seven Failing Development Episodes

## 1. Baseline Preservation and Exact Replay Verification

All Phase 1 artifacts remain unmodified. The Phase 2 audit ([audit_phase2.py](../../audit_phase2.py)) loaded the frozen Phase 1 weights and configuration on CPU (`BabyAI-GoToLocal-v0`, `247 -> 64 ReLU -> 3`, deterministic `argmax`, 64-step episode limit, filtered training dataset excluding training seed `10462` with `2,656` samples across 499 episodes, validation dataset with `514` samples across 100 episodes):

- **`phase1_weights.npz` SHA-256**: `e31f335039c27dc2b41a937a82d8a52e062828e15c3e049f8f470c614a6a19fd`
- **Selected checkpoint**: Epoch `21` (`val_cross_entropy = 0.235611`, selected under `lowest_val_cross_entropy_tiebreak_val_agreement_then_earliest_epoch`)
- **Audited dev seeds**: `3, 15, 29, 43, 50, 80, 94` (`30000..30199` untouched)
- **Baseline replay without advisory bot (`PROB_TOL = 1e-6`)**: `all_7_episodes_exact_match = true` — all 7 seeds matched [phase1_eval_results.json](../../phase1_eval_results.json) identically on discrete poses, discrete actions, action probabilities, cycle detection, and `64`-step timeout outcomes (`success = false`, `truncated = true`).

All machine-readable outputs are saved in [phase2_audit_results.json](../../phase2_audit_results.json). Execution ran strictly CPU-only in `0.3142 s` (within the `300.0 s` cap).

---

## 2. Question A: Cycle Structure, Traces, and Entry vs. Recurrence

Because the policy is a frozen deterministic mapping of the current 247-d observation and the environment transitions are deterministic, once `(agent_pos, agent_dir)` and the feature vector repeat (with static world objects untouched), the trajectory becomes strictly periodic through step `63` (`periodic_exact_through_timeout = true` for all 7 seeds).

We explicitly separate:
- **First observed mistake**: Reported only where the move is unambiguously invalid from local physics (`forward` into an impassable cell causing neither position nor heading change, i.e., `seeds 3` and `50`), rather than equating any disagreement with the heuristic `BabyAIBot` to proof of an error.
- **First cycle entry (`first_entry_step`)**: First step visiting a state belonging to the eventual periodic loop.
- **First complete recurrence (`first_complete_recurrence_step`)**: First step returning to the exact entry state (`entry + cycle_length`).

| Seed | Mission | Entry Step | Recurrence Step | Cycle Len | Distinct States | Pre-Cycle Prefix (Student Actions & Notes) | Cycle Action Sequence (Offsets `0..L-1`) | Unambiguous First Mistake? |
| ---: | :--- | ---: | ---: | ---: | ---: | :--- | :--- | :--- |
| `3` | `go to the red key` | `2` | `3` | `1` | `1` | `0: forward` (`[1,1],0`), `1: right` (`[2,1],0`, blocked front) | `[forward]` (`[2,1],1` facing `red box`) | **Step 2** (`forward` into impassable `red box`, entering 1-step loop; step 0 merely differs from bot) |
| `15` | `go to the grey box` | `5` | `7` | `2` | `2` | `0: left`, `1: forward`, `2: forward`, `3: left` (`[5,4],3` blocked front, bot=`right`), `4: forward` | `[right, left]` (`[4,4],2 <-> [4,4],3`) | Not established pre-cycle (step 3 `left` diverges from bot `right`, step 6 `left` oscillates back) |
| `29` | `go to the green key` | `0` | `8` | `8` | `8` | *(empty; episode starts inside cycle)* | `[forward, forward, right, right, forward, forward, right, right]` (`[4,4]<->[6,4]`, dirs `0,1,2,3`) | Not isolated to a single step (step 1 diverges from bot `right`) |
| `43` | `go to the grey ball` | `4` | `10` | `6` | `6` | `0: right`, `1: right`, `2: forward`, `3: left` (`[4,4],3` blocked front, bot=`right`) | `[forward, right, right, forward, right, right]` (`[3,4]<->[4,4]`) | Not established pre-cycle (step 3 `left` diverges from bot `right`) |
| `50` | `go to the green box` | `8` | `9` | `1` | `1` | `0: right`, `1..5: forward` (`[6,6]->[6,1],3`), `6: left` (blocked front), `7: forward` (`[6,1]->[5,1],2`) | `[forward]` (`[5,1],2` facing `yellow box`) | **Step 8** (`forward` into impassable `yellow box`, entering 1-step loop; steps 1..5 & 7 diverge from bot `left`) |
| `80` | `go to the blue ball` | `0` | `8` | `8` | `8` | *(empty; episode starts inside cycle)* | `[right, forward, forward, right, right, forward, forward, right]` (`[3,4]<->[3,6]`) | Not isolated to a single step (step 2 diverges from bot `right`) |
| `94` | `go to the grey ball` | `4` | `12` | `8` | `8` | `0: forward`, `1: left`, `2: left` (`[2,6],0` blocked front, bot=`pickup`), `3: forward` (`[2,6]->[2,5],3`) | `[forward, right, forward, right, forward, right, forward, right]` (`[2,5]<->[3,5]`) | Step 2 diverges from bot `pickup` (outside student action set); student enters 8-step box loop at step 4 |

### Compact Cycle Mechanics by Failure Pattern

1. **Blocked-Forward Fixed Points (`seeds 3, 50`)**:
   - **Seed `3`**: Enters `[2,1], dir=1` at step 2 with `red key` visible at egocentric `(1,4)` and `red box` directly ahead at `(3,5)` (`front_blocked = true`). Outputs `forward` (`p = 0.602771`, vs `right: 0.294098`, `left: 0.103131`) for all remaining 62 steps.
   - **Seed `50`**: Enters `[5,1], dir=2` at step 8 with `green box` visible at egocentric `(3,4)` and `yellow box` directly ahead at `(3,5)` (`front_blocked = true`). Outputs `forward` (`p = 0.992490`, vs `left: 0.006108`, `right: 0.001401`) for all remaining 56 steps.
2. **2-Step Heading Ping-Pong (`seed 15`)**:
   - Alternates at `[4,4]` between `dir=2` (front `yellow key`, target at `(5,5)`, `right`, `p = 0.974255`) and `dir=3` (front `yellow ball`, target at `(2,4)`, `left`, `p = 0.638094` vs `right: 0.340462`). Neither pose steps forward nor completes a 180-degree turn.
3. **Closed Spatial Patrol Loops (`seeds 29, 43, 80, 94`)**:
   - **Seed `29` (length 8)**: Shuttles back and forth along row `y=4` between `x=4` and `x=6`, turning `right` twice at each end (`[4,4] -> [5,4] -> [6,4] (right, right) -> [5,4] -> [4,4] (right, right)`).
   - **Seed `43` (length 6)**: Shuttles 1 cell along row `y=4` between `[3,4]` and `[4,4]`, making two `right` turns at each end.
   - **Seed `80` (length 8)**: Shuttles along column `x=3` between `y=4` and `y=6`, turning `right` twice at `[3,6]` (`wall` ahead then `green key` ahead) and `right` at `[3,4], dir=3` (instead of turning around toward the target).
   - **Seed `94` (length 8)**: Traces a clockwise 2x2 perimeter `[2,5] -> [2,4] -> [3,4] -> [3,5] -> [2,5]` via alternating `[forward, right]` pairs.

---

## 3. Question B: Exact Training/Validation Support & Blocked-Forward Checks

Exact `247`-dimensional feature-vector matching against the filtered training set (`2,656` samples) and validation set (`514` samples), kept strictly separate:

- **Zero training coverage on 33 of 34 distinct cycle states**: Across all 7 cycles (`1 + 2 + 8 + 6 + 1 + 8 + 8 = 34` distinct cycle states), **33/34 states have `0` exact matches in `train_filtered`**.
- **Single exact training match (`seed 94`, offset 7)**: Only `[2,5], dir=2` (where target is not in the 7x7 view) matched `train_filtered`, with `2` samples whose expert labels were split (`right: 1, forward: 1, left: 0`). The student selected `right` (`p = 0.894151`), so `student_action_has_exact_train_support = true` for that single state (`1/34` total).
- **Zero validation coverage on all 34 cycle states**: `34/34` distinct cycle states have `0` exact matches in `val`.

### Blocked-Forward Verification (`seeds 3, 50`) vs. Dataset Support

Direct inspection of `raw obs['image'][3, 5]`, `env.grid.get(*front_pos)`, and the `247`-d `front_blocked` bit in both fixed-point cycles confirms there is **no encoding bug or grid inconsistency**:

| Check | Seed `3` (step `2..63`, 62 steps) | Seed `50` (step `8..63`, 56 steps) |
| :--- | :--- | :--- |
| Raw `(3, 5)` cell | `box`, `red`, `state=0`, raw `[7, 0, 0]` | `box`, `yellow`, `state=0`, raw `[7, 4, 0]` |
| World `front_pos` cell | `box`, `red`, `can_overlap=false` | `box`, `yellow`, `can_overlap=false` |
| `raw_and_world_front_cell_consistent` | `true` | `true` |
| `world_front_impassable` | `true` | `true` |
| Feature `front_blocked` bit (`dim 245`) | `true` (`1.0`) | `true` (`1.0`) |
| Chosen action & probs `[L, R, F]` | `forward`: `[0.103131, 0.294098, 0.602771]` | `forward`: `[0.006108, 0.001401, 0.992490]` |
| `train_exact_matches` / `val_exact_matches` | `0` / `0` | `0` / `0` |

Comparison against all blocked-front samples (`front_blocked = true`) in the datasets:

- **`train_filtered` (`436` blocked-front samples)**: `left = 194`, `right = 242`, **`forward = 0`**
  - Front obstacle `wall` (`219`): `left = 98, right = 121, forward = 0`
  - Front obstacle `non_target_object` (`217`): `left = 96, right = 121, forward = 0`
  - `target_in_view = true` (`326`): `left = 150, right = 176, forward = 0`
  - `target_in_view = false` (`110`): `left = 44, right = 66, forward = 0`
- **`val` (`75` blocked-front samples)**: `left = 39`, `right = 36`, **`forward = 0`**
  - Front obstacle `wall` (`32`): `left = 15, right = 17, forward = 0`
  - Front obstacle `non_target_object` (`43`): `left = 24, right = 19, forward = 0`
  - `target_in_view = true` (`61`): `left = 35, right = 26, forward = 0`
  - `target_in_view = false` (`14`): `left = 4, right = 10, forward = 0`

Although `0 / 436` training examples (and `0 / 75` validation examples) with `front_blocked = true` ever label `forward`, the MLP is not hard-constrained to suppress `forward`, and at these unseen feature vectors in `seeds 3` and `50`, the learned linear/ReLU combination assigns highest logit to `forward`.

---

## 4. Question C: `BabyAIBot` Advisory API Verification and Replay Results

### Verified MiniGrid `v3.2.0` Advisory API

From installed `minigrid/utils/baby_ai_bot.py` (`BabyAIBot.replan`, `PickupSubgoal`, `GoNextToSubgoal`, `ExploreSubgoal`):
1. `bot = BabyAIBot(env)` tracks mission subgoals on a stack and updates an internal `vis_mask` from a read-only `env.gen_obs_grid()` copy without mutating `env`.
2. Calling `bot.replan(previous_actual_action)` (passing `None` at step `0`, then the student's executed action at each subsequent step) is the supported advisory/off-policy interface: each call updates subgoal progress using `previous_actual_action`, marks newly visible cells in `vis_mask`, pops any exploratory subgoal (`ExploreSubgoal`), and dynamically replans shortest paths.
3. **Verified source-level edge case**: When `GoNextToSubgoal` attempts to clear a blocking object, it pushes `PickupSubgoal` onto the stack. `PickupSubgoal.replan_after_action` only pops/handles `None`, `pickup`, `left`, or `right` — if the student takes another action such as `forward` before turning away, `PickupSubgoal` can remain active and recommend `pickup` (action `3`), which is outside the student's 3-action set `{left, right, forward}`.

### Advisory Replay Results

- **Trajectory preservation**: For all 7 seeds, `trajectory_identical_to_baseline = true` (zero divergence in poses, features, student actions, or probabilities) and `expert_exceptions = []`.
- **Recommendation consistency across repeated cycle visits**: For all 7 seeds, `expert_changed_across_repeated_student_inputs = []` — even as `bot.vis_mask` expanded over initial cycle passes (e.g., `28 -> 63` visible cells in `seed 29`, `35 -> 63` in `seed 80`, `58 -> 62` in `seed 94`), the advisory expert returned the exact same unique action on every revisit to each cycle offset.
- **Disagreement at cycle states (`student` vs. `advisory`)**:
  - `seed 3` (len 1): `[forward]` vs `['right']`
  - `seed 15` (len 2): `[right, left]` vs `['right', 'right']` (offset 1 `left` reverses offset 0 `right`)
  - `seed 29` (len 8): agrees at offsets `0, 2, 3, 4` (`[forward, right, right, forward]`), disagrees at offsets `1, 5, 6, 7` (student `[forward, forward, right, right]` vs bot `['right', 'left', 'left', 'right']`)
  - `seed 43` (len 6): agrees at offsets `1, 2, 3` (`[right, right, forward]`), disagrees at offsets `0, 4, 5` (student `[forward, right, right]` vs bot `['right', 'forward', 'left']`)
  - `seed 50` (len 1): `[forward]` vs `['left']`
  - `seed 80` (len 8): agrees at offsets `0, 1, 3, 4, 5` (`[right, forward, right, right, forward]`), disagrees at offsets `2, 6, 7` (student `[forward, forward, right]` vs bot `['right', 'left', 'left']`)
  - `seed 94` (len 8): agrees at offsets `1, 2, 3, 4` (`[right, forward, right, forward]`), disagrees at offsets `0, 5, 6, 7` (student `[forward, right, forward, right]` vs bot `['right', 'pickup', 'left', 'left']`)
- **`seed 94` non-student `pickup` observation**: `BabyAIBot` proposed `pickup` (`action_id = 3`) 8 times in `seed 94` — at pre-cycle step `2` (`[2,6], dir=0`, when `GoNextToSubgoal` pushed `DropSubgoal`/`GoNextTo((1,6))`/`PickupSubgoal` to clear a blocker) and on every visit to cycle offset `5` (`steps 9, 17, 25, 33, 41, 49, 57`, pose `[3,5], dir=1`, where `GoNextTo([3,6])` completed and exposed `PickupSubgoal` on top of the stack; the student's `right` action then popped `PickupSubgoal` at the next step). Note that even on step `2`, a blocker-clearing `pickup` outside `{left, right, forward}` appeared on the advisory stack.

---

## 5. Narrow Conclusions and Unresolved Hypotheses

### What Is Established
1. All 7 Phase 1 failures are deterministic exact observation-and-pose cycles (`length` 1 to 8) under the frozen `247 -> 64 ReLU -> 3` policy.
2. Nearly all distinct cycle states (`33/34` vs `train_filtered`, `34/34` vs `val`) have zero exact feature-vector matches in the training and validation sets.
3. On `seeds 3` and `50`, `front_blocked` is encoded accurately (`true`) and matches the impassable `box` cell in front of the agent, yet the MLP outputs `forward` with high probability (`0.602771` and `0.992490`) despite `0/436` `forward` labels among blocked-front training examples.
4. `BabyAIBot.replan(prev_action)` works stably alongside frozen rollouts, gives consistent recommendations across repeated visits to each cycle state (0 changes), and cleanly provides `left`/`right`/`forward` labels across `6/7` failing seeds (`3, 15, 29, 43, 50, 80`), while revealing a blocker-clearing `pickup` proposal on `seed 94` (`step 2` and `offset 5`).

### What Remains Unresolved (Do Not Overclaim)
- **Why generalization failed at unseen cycle states**: We observed zero exact training matches on `33/34` cycle states, but this does not prove general "covariate shift" alone caused all 7 failures, nor does it show that adding on-policy labels would fix them.
- **Role of memory vs. single-step features**: Even with a single-step feature vector, `seed 94` offset `7` exhibits two identical `247`-d training observations labeled differently (`right: 1, forward: 1`) and an advisory label of `left` in `seed 94`. We have not yet tested whether single-step aliasing, lack of `247`-d expressivity (e.g., missing non-target color/type distinctions or heading history), hidden-layer capacity (`64` units), or non-student `pickup` detours prevent zero-loop convergence even with on-policy supervision.
- **Original Laya experiment**: Nothing here addresses or explains the unavailable original Laya model or setup.

---

## 6. Proposed Next Experiment (Not Yet Executed)

- **Proposal**: Run a single bounded **on-policy advisory relabeling (DAgger-style) retraining pass** on CPU using the exact same `247 -> 64 ReLU -> 3` architecture, training hyperparameters, training seeds (`10000..10499`), and evaluation set (`0..99`, keeping `30000..30199` untouched). Roll out the frozen Phase 1 student on training seeds (`10000..10499`) with `BabyAIBot.replan` advising on visited states (retaining `{left, right, forward}` labels and recording/excluding any `pickup`/non-student proposals transparently via the existing filtering convention), combine with the filtered Phase 1 training set, train one new `247 -> 64 ReLU -> 3` checkpoint, and inspect the 7 dev seeds `3, 15, 29, 43, 50, 80, 94` alongside overall dev rate (`0..99`).
- **What it would distinguish**: Whether these failures—especially the `front_blocked=True -> forward` extrapolations (`seeds 3, 50`) and short loops (`seeds 15, 29, 43, 80`)—persist when the training distribution includes expert labels on the exact student-visited states, vs. being resolved by on-policy coverage without changing the model class or adding memory/recovery wrappers.
- **What it would NOT establish**:
  - Solving some or all of the 7 dev seeds would **not** prove a memoryless policy solves `100%` of `BabyAI-GoToLocal-v0` or generalize cleanly to the unseen confirmation set (`30000..30199`).
  - Persistent failures (for example on `seed 94` where `pickup` appears on the advisory stack or on aliased hidden-target observations) would **not** prove memory is strictly required rather than richer current-observation features or larger capacity.
