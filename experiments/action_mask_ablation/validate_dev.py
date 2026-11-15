"""Development-only validation gate (constructed inputs + dev seeds 0..99 only)."""

from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX, STATE_TO_IDX

from experiments.action_mask_ablation.common import (
    compute_paired_contrast,
    holm_bonferroni_adjust,
    run_single_episode,
    sha256_file,
    summarize_arm_records,
)
from experiments.action_mask_ablation.mask_selector import (
    select_action_with_blocked_forward_mask,
)
from external_wrapper import ExternalObservationCountWrapper, run_wrapper_unit_tests
from phase5_stats import run_statistical_unit_tests
from run_phase1 import verify_feature_assertions

DEV_SEEDS = list(range(100))
EXPECTED_PHASE1_WEIGHTS_SHA256 = (
    "e31f335039c27dc2b41a937a82d8a52e062828e15c3e049f8f470c614a6a19fd"
)
EXPECTED_WRAPPER_SHA256 = (
    "956571f84345d98b936a83186ada71558e3a6868336dbe323bd1c4b636485bec"
)


def _make_blank_image() -> np.ndarray:
    img = np.zeros((7, 7, 3), dtype=np.uint8)
    img[:, :, 0] = OBJECT_TO_IDX["empty"]
    return img


def run_mask_unit_tests() -> dict:
    """Verify decoding, mission parsing, edge cases, ties, and no-op invariance."""
    mission = "go to the red key"

    # 1. Unblocked forward (empty & unseen) -> allows {0, 1, 2}, matches argmax
    for obj_name in ("empty", "unseen", "floor", "goal", "lava"):
        img = _make_blank_image()
        img[3, 5, 0] = OBJECT_TO_IDX[obj_name]
        probs = np.array([0.1, 0.2, 0.7], dtype=np.float64)
        a, meta = select_action_with_blocked_forward_mask(img, mission, probs)
        assert a == 2 == int(np.argmax(probs))
        assert not meta["forward_masked"] and not meta["action_changed"]
        assert meta["allowed_actions"] == [0, 1, 2]

    # 2. Door states: open -> unblocked, closed / locked -> blocked non-target
    img_door_open = _make_blank_image()
    img_door_open[3, 5] = [OBJECT_TO_IDX["door"], COLOR_TO_IDX["blue"], STATE_TO_IDX["open"]]
    a, meta = select_action_with_blocked_forward_mask(
        img_door_open, mission, np.array([0.1, 0.2, 0.7], dtype=np.float64)
    )
    assert a == 2 and not meta["forward_masked"]

    for door_state in ("closed", "locked"):
        img_d = _make_blank_image()
        img_d[3, 5] = [OBJECT_TO_IDX["door"], COLOR_TO_IDX["blue"], STATE_TO_IDX[door_state]]
        a, meta = select_action_with_blocked_forward_mask(
            img_d, mission, np.array([0.15, 0.25, 0.60], dtype=np.float64)
        )
        assert a == 1 and meta["forward_masked"] and meta["action_changed"]
        assert meta["allowed_actions"] == [0, 1]

    # 3. Blocked non-target forward (wall, wrong-color key, wrong-type ball, box, agent)
    blocked_cases = [
        (OBJECT_TO_IDX["wall"], COLOR_TO_IDX["grey"]),
        (OBJECT_TO_IDX["key"], COLOR_TO_IDX["blue"]),
        (OBJECT_TO_IDX["ball"], COLOR_TO_IDX["red"]),
        (OBJECT_TO_IDX["box"], COLOR_TO_IDX["yellow"]),
        (OBJECT_TO_IDX["agent"], COLOR_TO_IDX["red"]),
    ]
    for obj_id, color_id in blocked_cases:
        img = _make_blank_image()
        img[3, 5] = [obj_id, color_id, 0]
        probs = np.array([0.35, 0.10, 0.55], dtype=np.float64)
        a, meta = select_action_with_blocked_forward_mask(img, mission, probs)
        assert a == 0 and meta["forward_masked"] and meta["action_changed"]
        assert not meta["front_is_mission_target"]

    # 4. Target-facing exemption (Option 1): matching target key in front keeps forward
    img_target = _make_blank_image()
    img_target[3, 5] = [OBJECT_TO_IDX["key"], COLOR_TO_IDX["red"], 0]
    a, meta = select_action_with_blocked_forward_mask(
        img_target, "go to a red key", np.array([0.1, 0.2, 0.7], dtype=np.float64)
    )
    assert a == 2 and meta["front_is_impassable"] and meta["front_is_mission_target"]
    assert not meta["forward_masked"] and not meta["action_changed"]

    # 5. Exact ties and zero-probability allowed actions (must not divide by zero -> NaN)
    img_wall = _make_blank_image()
    img_wall[3, 5] = [OBJECT_TO_IDX["wall"], COLOR_TO_IDX["grey"], 0]

    a_tie, m_tie = select_action_with_blocked_forward_mask(
        img_wall, mission, np.array([0.2, 0.2, 0.6], dtype=np.float64)
    )
    assert a_tie == 0 and m_tie["unmasked_prob_tie"] and m_tie["action_changed"]

    a_zero, m_zero = select_action_with_blocked_forward_mask(
        img_wall, mission, np.array([0.0, 0.0, 1.0], dtype=np.float64)
    )
    assert a_zero == 0 and m_zero["all_allowed_zero_prob"] and m_zero["unmasked_prob_tie"]

    # 6. No-op masking: unblocked cases reproduce np.argmax exact float ties
    img_empty = _make_blank_image()
    for p_vec in (
        np.array([0.2, 0.4, 0.4], dtype=np.float64),
        np.array([1.0 / 3, 1.0 / 3, 1.0 / 3], dtype=np.float64),
        np.array([0.0, 0.0, 0.0], dtype=np.float64),
    ):
        a_noop, m_noop = select_action_with_blocked_forward_mask(img_empty, mission, p_vec)
        assert a_noop == int(np.argmax(p_vec)) and not m_noop["forward_masked"]

    # 7. Strict failure on unsupported shapes, dtypes, object/state IDs, missions, probs
    bad_cases = [
        lambda: select_action_with_blocked_forward_mask(np.zeros((5, 5, 3), dtype=np.uint8), mission, np.array([0.2, 0.3, 0.5])),
        lambda: select_action_with_blocked_forward_mask(np.zeros((7, 7, 3), dtype=np.float32), mission, np.array([0.2, 0.3, 0.5])),
        lambda: select_action_with_blocked_forward_mask(img_empty, "pick up the red key", np.array([0.2, 0.3, 0.5])),
        lambda: select_action_with_blocked_forward_mask(img_empty, "go to the cyan key", np.array([0.2, 0.3, 0.5])),
        lambda: select_action_with_blocked_forward_mask(img_empty, mission, np.array([0.5, 0.5])),
        lambda: select_action_with_blocked_forward_mask(img_empty, mission, np.array([np.nan, 0.5, 0.5])),
    ]
    img_bad_obj = _make_blank_image()
    img_bad_obj[3, 5, 0] = 99
    bad_cases.append(lambda: select_action_with_blocked_forward_mask(img_bad_obj, mission, np.array([0.2, 0.3, 0.5])))
    img_bad_door = _make_blank_image()
    img_bad_door[3, 5] = [OBJECT_TO_IDX["door"], 0, 99]
    bad_cases.append(lambda: select_action_with_blocked_forward_mask(img_bad_door, mission, np.array([0.2, 0.3, 0.5])))

    for idx, fn in enumerate(bad_cases):
        try:
            fn()
        except (ValueError, KeyError, FloatingPointError):
            continue
        raise AssertionError(f"Expected validation failure on invalid input case {idx}")

    return {
        "mask_unit_tests_passed": True,
        "cases_verified": [
            "unblocked_object_types",
            "door_open_closed_locked_states",
            "blocked_non_target_objects",
            "target_facing_exemption_option1",
            "exact_ties_and_zero_prob_allowed_actions",
            "no_op_masking_matches_argmax",
            "strict_rejection_of_invalid_inputs",
        ],
    }


def verify_live_front_coordinate_alignment() -> dict:
    """Check obs['image'][3, 5] matches env.grid.get(*env.front_pos) across all 4 orientations on dev seeds."""
    env = gym.make("BabyAI-GoToLocal-v0")
    orientations_checked = set()
    non_empty_front_matches = 0
    empty_front_matches = 0
    try:
        for seed in range(25):
            obs, _ = env.reset(seed=seed)
            for rot in range(4):
                assert obs["image"].shape == (7, 7, 3)
                assert obs["image"].dtype == np.uint8
                u = env.unwrapped
                orientations_checked.add(int(u.agent_dir))
                world_cell = u.grid.get(*u.front_pos)
                obs_front = [int(x) for x in obs["image"][3, 5]]
                if world_cell is None:
                    assert obs_front[0] == OBJECT_TO_IDX["empty"]
                    empty_front_matches += 1
                else:
                    enc = list(world_cell.encode())
                    assert obs_front == enc, f"Mismatch seed={seed} dir={u.agent_dir}: {obs_front} != {enc}"
                    non_empty_front_matches += 1
                obs, _, _, _, _ = env.step(1)  # turn right through all 4 headings
    finally:
        env.close()

    assert orientations_checked == {0, 1, 2, 3}
    assert non_empty_front_matches > 0 and empty_front_matches > 0
    return {
        "front_coordinate_alignment_passed": True,
        "orientations_verified": sorted(orientations_checked),
        "empty_front_checks": empty_front_matches,
        "non_empty_front_checks": non_empty_front_matches,
    }


def main() -> None:
    assert sha256_file(Path("phase1_weights.npz")) == EXPECTED_PHASE1_WEIGHTS_SHA256
    assert sha256_file(Path("external_wrapper.py")) == EXPECTED_WRAPPER_SHA256

    verify_feature_assertions()
    wrap_tests = run_wrapper_unit_tests()
    stat_tests = run_statistical_unit_tests()
    mask_tests = run_mask_unit_tests()
    coord_check = verify_live_front_coordinate_alignment()

    p1_npz = np.load("phase1_weights.npz")
    params = {k: p1_npz[k] for k in p1_npz.files}
    wrapper = ExternalObservationCountWrapper()

    arm_a_records: list[dict] = []
    arm_b_records: list[dict] = []
    arm_c_records: list[dict] = []

    for seed in DEV_SEEDS:
        res_a = run_single_episode(seed, params, "A")
        res_b = run_single_episode(seed, params, "B")
        res_c = run_single_episode(seed, params, "C", wrapper=wrapper)
        arm_a_records.append(res_a["record"])
        arm_b_records.append(res_b["record"])
        arm_c_records.append(res_c["record"])

    # Verify exact reproduction on dev 0..99 against Phase 1 and Phase 4 saved artifacts
    p1_saved = json.loads(Path("phase1_eval_results.json").read_text(encoding="utf-8"))
    for r_new, r_old in zip(arm_a_records, p1_saved["episodes"], strict=True):
        assert r_new["seed"] == r_old["seed"]
        assert r_new["success"] == r_old["success"]
        assert r_new["episode_length"] == r_old["episode_length"]

    p4_saved = json.loads(Path("phase4_results.json").read_text(encoding="utf-8"))
    p4_sum = p4_saved["phase4_wrapped_summary"]
    sum_c = summarize_arm_records(arm_c_records, "C")
    assert sum_c["success_count"] == p4_sum["success_count"] == 98
    assert sum_c["total_decisions"] == p4_sum["intervention_burden"]["total_decisions_all_episodes"] == 741
    assert sum_c["wrapper_interventions"]["total_overridden_actions"] == p4_sum["intervention_burden"]["total_overridden_actions"] == 77

    sum_a = summarize_arm_records(arm_a_records, "A")
    sum_b = summarize_arm_records(arm_b_records, "B")

    c_vs_b = compute_paired_contrast(DEV_SEEDS, arm_b_records, arm_c_records, "B", "C")
    b_vs_a = compute_paired_contrast(DEV_SEEDS, arm_a_records, arm_b_records, "A", "B")
    c_vs_a = compute_paired_contrast(DEV_SEEDS, arm_a_records, arm_c_records, "A", "C")
    holm = holm_bonferroni_adjust(
        [
            ("C_vs_B", c_vs_b["exact_mcnemar_test"]["p_value"]),
            ("B_vs_A", b_vs_a["exact_mcnemar_test"]["p_value"]),
            ("C_vs_A", c_vs_a["exact_mcnemar_test"]["p_value"]),
        ]
    )

    out_dir = Path("experiments/action_mask_ablation/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "status": "dev_validation_passed",
        "scope_note": "Development-only validation on constructed cases and seeds 0..99 (NOT held-out confirmation).",
        "unit_checks": {
            "feature_assertions_passed": True,
            "wrapper_unit_tests": wrap_tests,
            "statistical_unit_tests": stat_tests,
            "mask_unit_tests": mask_tests,
            "coordinate_alignment": coord_check,
        },
        "historical_dev_reproduction_verified": {
            "arm_a_matches_phase1_eval_results": True,
            "arm_c_matches_phase4_results": True,
        },
        "dev_arm_summaries": {"A": sum_a, "B": sum_b, "C": sum_c},
        "dev_paired_contrasts": {
            "primary_C_vs_B": c_vs_b,
            "secondary_B_vs_A": b_vs_a,
            "secondary_C_vs_A": c_vs_a,
            "supplementary_holm": holm,
        },
    }
    (out_dir / "dev_validation_results.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
