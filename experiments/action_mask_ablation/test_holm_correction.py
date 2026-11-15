"""Focused regression tests for 3-contrast Holm-Bonferroni ordering and McNemar precision."""

from __future__ import annotations

import math
import unittest

from experiments.action_mask_ablation.audit_confirmation import (
    exact_mcnemar_full_precision,
    holm_bonferroni_full_precision,
)


class TestHolmCorrection(unittest.TestCase):
    def test_tiny_nonzero_p_values_preserve_ordering(self) -> None:
        p_cb = exact_mcnemar_full_precision(0, 36)["p_value"]
        p_ca = exact_mcnemar_full_precision(0, 42)["p_value"]
        p_ba = exact_mcnemar_full_precision(0, 6)["p_value"]

        self.assertGreater(p_ca, 0.0)
        self.assertGreater(p_cb, 0.0)
        self.assertLess(p_ca, p_cb)
        self.assertLess(p_cb, p_ba)

        # Insertion order deliberately puts C_vs_B before C_vs_A (the original bug trigger)
        adj = holm_bonferroni_full_precision(
            [
                ("C_vs_B_primary", p_cb),
                ("B_vs_A_secondary", p_ba),
                ("C_vs_A_secondary", p_ca),
            ]
        )

        self.assertEqual(adj["C_vs_A_secondary"]["rank"], 1)
        self.assertEqual(adj["C_vs_A_secondary"]["holm_multiplier"], 3)
        self.assertEqual(adj["C_vs_B_primary"]["rank"], 2)
        self.assertEqual(adj["C_vs_B_primary"]["holm_multiplier"], 2)
        self.assertEqual(adj["B_vs_A_secondary"]["rank"], 3)
        self.assertEqual(adj["B_vs_A_secondary"]["holm_multiplier"], 1)

        self.assertTrue(math.isclose(adj["C_vs_A_secondary"]["holm_adjusted_p_value"], 3.0 * (2.0 / 2**42), rel_tol=1e-12))
        self.assertTrue(math.isclose(adj["C_vs_B_primary"]["holm_adjusted_p_value"], 2.0 * (2.0 / 2**36), rel_tol=1e-12))
        self.assertAlmostEqual(adj["B_vs_A_secondary"]["holm_adjusted_p_value"], 0.03125, places=12)

    def test_cumulative_maximum_enforced(self) -> None:
        adj = holm_bonferroni_full_precision(
            [
                ("p1", 0.012),
                ("p2", 0.015),
                ("p3", 0.020),
            ]
        )
        # 3 * 0.012 = 0.036; 2 * 0.015 = 0.030 -> cumulative max must stay 0.036
        self.assertAlmostEqual(adj["p1"]["holm_adjusted_p_value"], 0.036, places=12)
        self.assertAlmostEqual(adj["p2"]["holm_adjusted_p_value"], 0.036, places=12)
        self.assertAlmostEqual(adj["p3"]["holm_adjusted_p_value"], 0.036, places=12)


if __name__ == "__main__":
    unittest.main()
