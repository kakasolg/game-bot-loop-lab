"""Isolated external-integrator wrapper using only permitted public observations."""

from __future__ import annotations

import numpy as np


class ExternalObservationCountWrapper:
    """Episode-local deterministic per-observation least-tried action selector.

    Information boundary:
    - Receives ONLY raw obs['image'], obs['mission'], base policy action probabilities,
      and explicit episode reset() calls.
    - Retains within-episode per-observation issued-action counts (episode-local memory).
    """

    def __init__(self) -> None:
        self._counts: dict[tuple[bytes, str], list[int]] = {}
        self._post_full_coverage_decisions: int = 0
        self._keys_with_post_full_coverage_decision: set[tuple[bytes, str]] = set()

    def reset(self) -> None:
        self._counts.clear()
        self._post_full_coverage_decisions = 0
        self._keys_with_post_full_coverage_decision.clear()

    def select_action(
        self,
        image: np.ndarray,
        mission: str,
        probs: np.ndarray,
    ) -> tuple[int, dict]:
        """Apply single lexicographic rule: (lowest issued count, -prob, action id)."""
        if probs.shape != (3,):
            raise ValueError(f"Expected probs shape (3,), got {probs.shape}")
        if not np.all(np.isfinite(probs)):
            raise FloatingPointError("Non-finite policy probabilities passed to wrapper")

        obs_key = (image.tobytes(), str(mission))
        counts = self._counts.get(obs_key)
        if counts is None:
            counts = [0, 0, 0]
            self._counts[obs_key] = counts

        prior_counts = tuple(counts)
        all_three_ge_one_before = min(prior_counts) >= 1
        if all_three_ge_one_before:
            self._post_full_coverage_decisions += 1
            self._keys_with_post_full_coverage_decision.add(obs_key)

        base_argmax = int(np.argmax(probs))
        chosen = min(
            (0, 1, 2),
            key=lambda a: (counts[a], -float(probs[a]), a),
        )
        counts[chosen] += 1

        return chosen, {
            "base_argmax": base_argmax,
            "overridden": chosen != base_argmax,
            "prior_counts": list(prior_counts),
            "post_counts": list(counts),
            "all_three_ge_one_before_decision": all_three_ge_one_before,
        }

    def episode_stats(self) -> dict:
        keys_ge1_after = sum(1 for c in self._counts.values() if min(c) >= 1)
        return {
            "retained_obs_keys": len(self._counts),
            "post_full_coverage_decisions": self._post_full_coverage_decisions,
            "distinct_keys_with_post_full_coverage_decision": len(
                self._keys_with_post_full_coverage_decision
            ),
            "distinct_keys_reaching_all_three_counts_ge_1_by_end": keys_ge1_after,
        }


def run_wrapper_unit_tests() -> dict:
    """Verify authoritative lexicographic rule, tie-breaking, counts, and reset."""
    w = ExternalObservationCountWrapper()
    img_a = np.zeros((7, 7, 3), dtype=np.uint8)
    img_b = np.ones((7, 7, 3), dtype=np.uint8)

    # 1. First visits reproduce base argmax, including exact float ties (lowest ID wins)
    probs_tie = np.array([0.2, 0.4, 0.4], dtype=np.float32)  # tie between 1 and 2 -> 1
    a1, m1 = w.select_action(img_a, "go to the red key", probs_tie)
    assert a1 == int(np.argmax(probs_tie)) == 1
    assert not m1["overridden"]
    assert m1["prior_counts"] == [0, 0, 0] and m1["post_counts"] == [0, 1, 0]

    probs_all_tie = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3], dtype=np.float32)
    b1, bm1 = w.select_action(img_b, "go to the red key", probs_all_tie)
    assert b1 == int(np.argmax(probs_all_tie)) == 0
    assert not bm1["overridden"]

    # 2 & 3. Counts increment only for issued action, and all 3 actions tried once before 2nd
    probs_pref = np.array([0.1, 0.7, 0.2], dtype=np.float32)
    a2, m2 = w.select_action(img_a, "go to the red key", probs_pref)
    assert a2 == 2  # count[1]==1, count[2]==0 (0.2 > 0.1)
    assert m2["overridden"] and m2["post_counts"] == [0, 1, 1]
    assert not m2["all_three_ge_one_before_decision"]

    a3, m3 = w.select_action(img_a, "go to the red key", probs_pref)
    assert a3 == 0  # only action 0 has count 0
    assert m3["overridden"] and m3["post_counts"] == [1, 1, 1]
    assert not m3["all_three_ge_one_before_decision"]
    assert {a1, a2, a3} == {0, 1, 2}

    # 4th decision at img_a: all three counts already >= 1 before decision -> returns to highest prob (1)
    a4, m4 = w.select_action(img_a, "go to the red key", probs_pref)
    assert a4 == 1
    assert not m4["overridden"]
    assert m4["all_three_ge_one_before_decision"]
    assert m4["post_counts"] == [1, 2, 1]

    stats = w.episode_stats()
    assert stats["retained_obs_keys"] == 2
    assert stats["post_full_coverage_decisions"] == 1
    assert stats["distinct_keys_with_post_full_coverage_decision"] == 1

    # Different mission with same image uses a distinct key
    c1, cm1 = w.select_action(img_a, "go to the blue box", probs_pref)
    assert c1 == 1 and not cm1["overridden"]

    # 4. Episode reset clears all retained state and counters
    w.reset()
    assert w.episode_stats() == {
        "retained_obs_keys": 0,
        "post_full_coverage_decisions": 0,
        "distinct_keys_with_post_full_coverage_decision": 0,
        "distinct_keys_reaching_all_three_counts_ge_1_by_end": 0,
    }
    a_after_reset, m_after_reset = w.select_action(img_a, "go to the red key", probs_pref)
    assert a_after_reset == 1 and not m_after_reset["overridden"]

    return {
        "all_unit_tests_passed": True,
        "tests_verified": [
            "first_visit_reproduces_base_argmax_and_float_ties",
            "counts_increment_only_for_issued_action",
            "all_three_actions_selected_once_before_second_selection",
            "post_full_coverage_counts_tracked_before_decision",
            "mission_change_uses_distinct_key",
            "episode_reset_clears_all_retained_state",
        ],
    }
