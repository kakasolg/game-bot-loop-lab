"""Blocked-forward observation-based action mask comparator (Arm B)."""

from __future__ import annotations

import numpy as np
from minigrid.core.constants import OBJECT_TO_IDX, STATE_TO_IDX

from run_phase1 import parse_mission

PASSABLE_OR_UNKNOWN_OBJ_IDS = frozenset(
    {
        OBJECT_TO_IDX["unseen"],
        OBJECT_TO_IDX["empty"],
        OBJECT_TO_IDX["floor"],
        OBJECT_TO_IDX["goal"],
        OBJECT_TO_IDX["lava"],
    }
)
IMPASSABLE_OBJ_IDS = frozenset(
    {
        OBJECT_TO_IDX["wall"],
        OBJECT_TO_IDX["key"],
        OBJECT_TO_IDX["ball"],
        OBJECT_TO_IDX["box"],
        OBJECT_TO_IDX["agent"],
    }
)
DOOR_OBJ_ID = OBJECT_TO_IDX["door"]
VALID_DOOR_STATES = frozenset(STATE_TO_IDX.values())
OPEN_DOOR_STATE = STATE_TO_IDX["open"]
FRONT_VX = 3
FRONT_VY = 5
EXPECTED_IMAGE_SHAPE = (7, 7, 3)


def select_action_with_blocked_forward_mask(
    image: np.ndarray,
    mission: str,
    probs: np.ndarray,
) -> tuple[int, dict]:
    """Stateless blocked-forward mask using ONLY obs['image'], obs['mission'], and probs.

    Excludes action 2 ('forward') iff obs['image'][3, 5] encodes an impassable
    cell that does not match the mission's target (color, type).
    """
    if not isinstance(image, np.ndarray) or image.shape != EXPECTED_IMAGE_SHAPE:
        shape = getattr(image, "shape", type(image).__name__)
        raise ValueError(
            f"Expected obs['image'] np.ndarray shape {EXPECTED_IMAGE_SHAPE}, got {shape}"
        )
    if not np.issubdtype(image.dtype, np.integer):
        raise ValueError(
            f"Expected integer dtype for obs['image'], got {image.dtype}"
        )
    if not isinstance(mission, str):
        raise ValueError(f"Expected str obs['mission'], got {type(mission).__name__}")
    if not isinstance(probs, np.ndarray) or probs.shape != (3,):
        shape = getattr(probs, "shape", type(probs).__name__)
        raise ValueError(f"Expected probs shape (3,), got {shape}")
    if not np.all(np.isfinite(probs)):
        raise FloatingPointError("Non-finite policy probabilities passed to mask selector")

    target_color_idx, target_type_idx = parse_mission(mission)

    front_obj_idx = int(image[FRONT_VX, FRONT_VY, 0])
    front_color_idx = int(image[FRONT_VX, FRONT_VY, 1])
    front_state_idx = int(image[FRONT_VX, FRONT_VY, 2])

    if front_obj_idx in PASSABLE_OR_UNKNOWN_OBJ_IDS:
        front_is_impassable = False
    elif front_obj_idx in IMPASSABLE_OBJ_IDS:
        front_is_impassable = True
    elif front_obj_idx == DOOR_OBJ_ID:
        if front_state_idx not in VALID_DOOR_STATES:
            raise ValueError(f"Unrecognized door state index at front cell: {front_state_idx}")
        front_is_impassable = front_state_idx != OPEN_DOOR_STATE
    else:
        raise ValueError(f"Unrecognized MiniGrid object index at front cell: {front_obj_idx}")

    front_is_mission_target = bool(
        front_obj_idx == target_type_idx and front_color_idx == target_color_idx
    )
    forward_masked = bool(front_is_impassable and not front_is_mission_target)

    allowed_actions = (0, 1) if forward_masked else (0, 1, 2)
    base_argmax = int(np.argmax(probs))

    fallback_empty_mask = False
    if not allowed_actions:
        fallback_empty_mask = True
        chosen = base_argmax
    else:
        chosen = min(allowed_actions, key=lambda a: (-float(probs[a]), a))

    chosen_prob = float(probs[chosen])
    tie_count = sum(1 for a in allowed_actions if float(probs[a]) == chosen_prob)

    return chosen, {
        "base_argmax": base_argmax,
        "forward_masked": forward_masked,
        "action_changed": bool(chosen != base_argmax),
        "allowed_actions": list(allowed_actions),
        "front_obj_idx": front_obj_idx,
        "front_color_idx": front_color_idx,
        "front_state_idx": front_state_idx,
        "front_is_impassable": front_is_impassable,
        "front_is_mission_target": front_is_mission_target,
        "fallback_empty_mask": fallback_empty_mask,
        "unmasked_prob_tie": bool(tie_count > 1),
        "all_allowed_zero_prob": bool(
            all(float(probs[a]) == 0.0 for a in allowed_actions)
        ),
    }
