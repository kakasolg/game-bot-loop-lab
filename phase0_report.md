# Phase 0 Report: MiniGrid `v3.2.0` `BabyAI-GoToLocal-v0` & `BabyAIBot`

## 1. Installed Package & Source Locations

- `minigrid==3.2.0` (`gymnasium==1.4.0`), installed in `.venv` at:
  - `minigrid/__init__.py`
  - `minigrid/envs/babyai/goto.py`
  - `minigrid/envs/babyai/core/roomgrid_level.py`
  - `minigrid/envs/babyai/core/verifier.py`
  - `minigrid/utils/baby_ai_bot.py`
  - `minigrid/core/roomgrid.py`, `minigrid/minigrid_env.py`, `minigrid/core/grid.py`, `minigrid/core/constants.py`, `minigrid/core/actions.py`

## 2. Phase 0A: Exact Environment Contract (`BabyAI-GoToLocal-v0`)

Verified from installed `v3.2.0` source and runtime execution of `inspect_env.py` (`phase0a_env_inspection.json`):

- **Registered IDs (`minigrid/__init__.py` L604–679)**: `BabyAI-GoToLocal-v0` is registered with no default `kwargs` and no `max_episode_steps` wrapper limit (`env.spec.max_episode_steps is None`). Registered variants are `S5N2`, `S6N2`–`S6N4`, `S7N4`–`S7N5`, and `S8N2`–`S8N7` (no `S8N8` entry exists).
- **Wrapper stack**: `OrderEnforcing -> PassiveEnvChecker -> GoToLocal` (no `TimeLimit` wrapper).
- **Room geometry**: `num_rows=1, num_cols=1, room_size=8`, giving an `8×8` grid (`width=8, height=8`) including outer boundary walls (`6×6` interior).
- **Object count & duplicates (`goto.py` L329–338, `roomgrid.py` L396–438)**: `GoToLocal.__init__` defaults to `room_size=8, num_dists=8`. `gen_mission()` calls `objs = self.add_distractors(num_distractors=8, all_unique=False)` (creating **8 objects total**, not 8 distractors plus a 9th target) and then selects the target from `objs` (`obj = self._rand_elem(objs)`). Because `all_unique=False`, duplicate `(type, color)` pairs can occur (observed in `20/20` seeds `0..19` for room objects, and `8/20` seeds had 2 objects matching the mission target).
- **Step limit (`roomgrid_level.py` L54–85)**:
  - Before `reset()` (when `max_steps=None`): `fixed_max_steps=False`, `unwrapped.max_steps == 0`.
  - After `reset()`: `nav_time_room = room_size**2 = 64`, `nav_time_maze = 64 * 1 * 1 = 64`, `num_navs = num_navs_needed(GoToInstr) = 1`, yielding `max_steps = 64`.
  - With a user-supplied override (e.g., `max_steps=123`): `fixed_max_steps=True` and `max_steps` stays `123` both before and after `reset()`.
- **Native action space (`actions.py` L7–20)**: `Discrete(7)` (`0: left`, `1: right`, `2: forward`, `3: pickup`, `4: drop`, `5: toggle`, `6: done`).
- **Observation space & coordinates (`minigrid_env.py` L70–85, L426–451, L597–650; `constants.py` L49–58)**:
  - Keys: `direction` (`Discrete(4)`), `image` (`uint8`, shape `(7, 7, 3)`), `mission` (`str`).
  - Direction encoding (`DIR_TO_VEC`): `0: (+1, 0)` right, `1: (0, +1)` down, `2: (-1, 0)` left, `3: (0, -1)` up.
  - `image[vx, vy, :]` stores `(OBJECT_IDX, COLOR_IDX, STATE)` where `vx in 0..6` runs left-to-right relative to the agent (`vx=3` is straight ahead), `vy in 0..6` runs far-to-near (`vy=5` is 1 cell ahead / `front_pos`, `vy=6` is the agent's own cell), and occluded cells are `unseen` (`OBJECT_IDX=0`).
- **Success verification (`verifier.py` L233–235, L290–316)**: With `BABYAI_DONE_ACTIONS` unset, `GoToInstr.verify_action` returns `"success"` immediately after any step whose resulting `env.front_pos` equals `pos` for any position in `self.desc.obj_poss` (standing orthogonally adjacent to a matching object and facing it).
- **Solvability with `{left, right, forward}` (`roomgrid_level.py` L250–302, `goto.py` L336)**: `gen_mission()` runs `self.check_objs_reachable()`, which verifies via 4-neighbor BFS over empty cells that every non-wall object is adjacent to a cell reachable from `agent_pos` without moving any object. Because `GoToLocal` has a single room with no doors and `GoToInstr` completes on `front_pos == target_pos`, every valid generated episode is solvable using only `{left, right, forward}` without moving objects.

## 3. Phase 0B: `BabyAIBot` Sanity Check (Dev Seeds `0..19`)

Run via `run_expert_check.py` (`phase0b_expert_results.json`):

- **Protocol**: Fresh env + fresh `BabyAIBot(env)` after `reset(seed=seed)` for fixed pre-selected development seeds `0..19`, passing `prev_action` to `bot.replan(prev_action)` (`None` on step 0) and stopping immediately on `terminated` or `truncated`.
- **Outcomes**:
  - Episodes: `20`
  - Successes: `20/20` (`terminated=True`, `reward > 0`)
  - Timeouts (`truncated`): `0`
  - Exceptions: `0`
  - Unexpected actions (outside `{0: left, 1: right, 2: forward}`): `0`
  - Episode length: mean `4.55` steps, min `1`, max `11` (total `91` steps)
  - Aggregate action counts: `forward: 56`, `right: 22`, `left: 13`
  - Wall-clock runtime: `0.0395` s (CPU, headless)
- **Bot state & constraints (`baby_ai_bot.py` L577–825)**: `BabyAIBot` is stateful across steps (maintaining cumulative `self.vis_mask` and subgoal `self.stack`), restricts BFS path expansion and target lookup to `self.vis_mask`, and reads `unwrapped` environment attributes for local attribute updates and frontier checks.
