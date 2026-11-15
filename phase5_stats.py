"""Statistical helpers for paired binary outcomes (Newcombe Method 10 + exact McNemar)."""

from __future__ import annotations

import math
from statistics import NormalDist


def wilson_interval(k: int, n: int, z: float) -> tuple[float, float]:
    """Uncorrected Wilson score confidence interval (l, u) for proportion k / n."""
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if not (0 <= k <= n):
        raise ValueError(f"k must satisfy 0 <= k <= n, got k={k}, n={n}")
    p_hat = k / n
    z2 = z * z
    denom = 2.0 * (n + z2)
    center = 2.0 * k + z2
    half = z * math.sqrt(z2 + 4.0 * k * (1.0 - p_hat))
    lower = max(0.0, (center - half) / denom)
    upper = min(1.0, (center + half) / denom)
    return lower, upper


def newcombe_paired_method10_ci(
    n_both_succeed: int,
    n_baseline_only: int,
    n_wrapped_only: int,
    n_both_fail: int,
    alpha: float = 0.05,
) -> dict:
    """Newcombe (1998) Method 10 95% CI for paired difference delta = p_wrapped - p_baseline.

    Paired 2x2 table:
      a = n_both_succeed   (A=1, B=1)
      b = n_baseline_only  (A=1, B=0)
      c = n_wrapped_only   (A=0, B=1)
      d = n_both_fail      (A=0, B=0)
    Marginals:
      e = a + b  (baseline successes), f = c + d  (baseline failures)
      g = a + c  (wrapped successes),  h = b + d  (wrapped failures)
      N = a + b + c + d
    Correlation adjustment phi_hat (Newcombe 1998, p. 2639; Altman et al. 2000, pp. 53-54):
      If e*f*g*h == 0: phi_hat = 0.0
      Else if (a*d - b*c) > 0: phi_hat = max(a*d - b*c - N/2, 0.0) / sqrt(e*f*g*h)
      Else: phi_hat = (a*d - b*c) / sqrt(e*f*g*h)
    Interval for delta = p_wrapped - p_baseline = (c - b) / N:
      dl = sqrt((p_B - l_B)^2 - 2 * phi_hat * (p_B - l_B) * (u_A - p_A) + (u_A - p_A)^2)
      du = sqrt((u_B - p_B)^2 - 2 * phi_hat * (u_B - p_B) * (p_A - l_A) + (p_A - l_A)^2)
      CI = (delta - dl, delta + du) clamped to [-1.0, 1.0].
    """
    a = n_both_succeed
    b = n_baseline_only
    c = n_wrapped_only
    d = n_both_fail
    if min(a, b, c, d) < 0:
        raise ValueError("Paired table counts must be non-negative")
    n = a + b + c + d
    if n <= 0:
        raise ValueError("Total count N must be positive")

    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    e = a + b
    f = c + d
    g = a + c
    h = b + d

    p_a = e / n
    p_b = g / n
    delta = (c - b) / n

    l_a, u_a = wilson_interval(e, n, z)
    l_b, u_b = wilson_interval(g, n, z)

    denom_prod = e * f * g * h
    t_cross = a * d - b * c
    if denom_prod == 0:
        phi_hat = 0.0
    elif t_cross > 0:
        phi_hat = max(t_cross - n / 2.0, 0.0) / math.sqrt(denom_prod)
    else:
        phi_hat = t_cross / math.sqrt(denom_prod)

    # Guard tiny float rounding inside sqrt (algebraically >= (x - y)^2 >= 0 since |phi_hat| <= 1)
    rad_l = (p_b - l_b) ** 2 - 2.0 * phi_hat * (p_b - l_b) * (u_a - p_a) + (u_a - p_a) ** 2
    rad_u = (u_b - p_b) ** 2 - 2.0 * phi_hat * (u_b - p_b) * (p_a - l_a) + (p_a - l_a) ** 2
    dl = math.sqrt(max(0.0, rad_l))
    du = math.sqrt(max(0.0, rad_u))

    ci_low = max(-1.0, delta - dl)
    ci_high = min(1.0, delta + du)
    return {
        "method": "newcombe_1998_method10_paired_wilson",
        "alpha": alpha,
        "z": round(z, 6),
        "n": n,
        "baseline_rate": round(p_a, 6),
        "wrapped_rate": round(p_b, 6),
        "delta_wrapped_minus_baseline": round(delta, 6),
        "phi_hat": round(phi_hat, 6),
        "degenerate_margin": denom_prod == 0,
        "ci_lower": round(ci_low, 6),
        "ci_upper": round(ci_high, 6),
        "ci_includes_zero": bool(ci_low <= 0.0 <= ci_high),
    }


def exact_mcnemar_test(n_baseline_only: int, n_wrapped_only: int) -> dict:
    """Two-sided exact binomial McNemar test on discordant pairs (b=n_10, c=n_01)."""
    if n_baseline_only < 0 or n_wrapped_only < 0:
        raise ValueError("Discordant pair counts must be non-negative")
    m = n_baseline_only + n_wrapped_only
    if m == 0:
        return {
            "method": "exact_two_sided_mcnemar_binomial",
            "discordant_pairs_total": 0,
            "baseline_only_success": 0,
            "wrapped_only_success": 0,
            "p_value": 1.0,
        }
    k = min(n_baseline_only, n_wrapped_only)
    one_tail = math.fsum(math.comb(m, i) for i in range(k + 1)) * (0.5 ** m)
    p_value = min(1.0, 2.0 * one_tail)
    return {
        "method": "exact_two_sided_mcnemar_binomial",
        "discordant_pairs_total": m,
        "baseline_only_success": n_baseline_only,
        "wrapped_only_success": n_wrapped_only,
        "p_value": round(p_value, 10),
    }


def run_statistical_unit_tests() -> dict:
    """Validate paired orientation, edge cases, Newcombe Method 10, and symmetry on synthetic tables."""
    # 1. Table orientation and sign: delta = wrapped - baseline = (c - b) / N
    res_pos = newcombe_paired_method10_ci(90, 1, 6, 3)  # N=100, A=91, B=96 -> delta=+0.05
    assert abs(res_pos["baseline_rate"] - 0.91) < 1e-9
    assert abs(res_pos["wrapped_rate"] - 0.96) < 1e-9
    assert abs(res_pos["delta_wrapped_minus_baseline"] - 0.05) < 1e-9
    assert res_pos["ci_lower"] < 0.05 < res_pos["ci_upper"]

    # 2. Symmetry under exchanging baseline <-> wrapped labels (b <-> c)
    res_swapped = newcombe_paired_method10_ci(90, 6, 1, 3)
    assert abs(res_swapped["delta_wrapped_minus_baseline"] + 0.05) < 1e-9
    assert abs(res_swapped["phi_hat"] - res_pos["phi_hat"]) < 1e-9
    assert abs(res_swapped["ci_lower"] + res_pos["ci_upper"]) < 1e-6
    assert abs(res_swapped["ci_upper"] + res_pos["ci_lower"]) < 1e-6
    mc_pos = exact_mcnemar_test(1, 6)
    mc_swapped = exact_mcnemar_test(6, 1)
    assert abs(mc_pos["p_value"] - mc_swapped["p_value"]) < 1e-12
    assert abs(mc_pos["p_value"] - 2.0 * (1 + 7) / 128.0) < 1e-10  # 2*(C(7,0)+C(7,1))/2^7 = 16/128 = 0.125

    # 3. Zero discordant pairs (b = c = 0): McNemar p == 1.0, CI strictly non-zero around 0
    mc_zero = exact_mcnemar_test(0, 0)
    assert mc_zero["p_value"] == 1.0
    ci_zero = newcombe_paired_method10_ci(100, 0, 0, 100)  # N=200, e*f*g*h > 0, t=10000 -> phi < 1
    assert ci_zero["delta_wrapped_minus_baseline"] == 0.0
    assert ci_zero["ci_lower"] < 0.0 < ci_zero["ci_upper"]
    assert abs(ci_zero["ci_lower"] + ci_zero["ci_upper"]) < 1e-6
    assert ci_zero["ci_includes_zero"]

    # 4. All discordant pairs favor one condition (b = 0, c > 0) including degenerate margins
    # Degenerate margin: wrapped 100% (b=0, d=0 -> h=0 -> e*f*g*h == 0 -> phi_hat == 0.0)
    ci_deg = newcombe_paired_method10_ci(186, 0, 14, 0)  # dev-like numbers at N=200
    assert ci_deg["degenerate_margin"] is True
    assert ci_deg["phi_hat"] == 0.0
    assert abs(ci_deg["delta_wrapped_minus_baseline"] - 0.07) < 1e-9
    assert ci_deg["ci_lower"] > 0.0  # should be strictly positive paired CI
    mc_deg = exact_mcnemar_test(0, 14)
    assert abs(mc_deg["p_value"] - 2.0 / (2 ** 14)) < 1e-10

    # All-same 100% degenerate margin (a=200, b=0, c=0, d=0)
    ci_all_pass = newcombe_paired_method10_ci(200, 0, 0, 0)
    assert ci_all_pass["degenerate_margin"] is True
    assert ci_all_pass["phi_hat"] == 0.0
    assert ci_all_pass["ci_lower"] < 0.0 < ci_all_pass["ci_upper"]

    # 5. Negative cross-product (a*d - b*c <= 0): phi_hat < 0 without +n/2 shrinkage
    ci_neg_corr = newcombe_paired_method10_ci(10, 40, 40, 10)  # ad-bc = 100-1600 = -1500
    assert ci_neg_corr["phi_hat"] < 0.0
    assert abs(ci_neg_corr["phi_hat"] - (-1500.0 / math.sqrt(50 * 50 * 50 * 50))) < 1e-6

    # Verify Newcombe paired CI is narrower than independent-proportions Wilson CI under strong positive corr
    ci_indep = newcombe_paired_method10_ci(180, 0, 10, 10)  # N=200, strong positive corr
    ci_no_corr_equiv_width = (
        math.sqrt((0.95 - wilson_interval(190, 200, NormalDist().inv_cdf(0.975))[0]) ** 2 + (wilson_interval(180, 200, NormalDist().inv_cdf(0.975))[1] - 0.90) ** 2)
    )
    assert (0.05 - ci_indep["ci_lower"]) < ci_no_corr_equiv_width

    return {
        "all_statistical_unit_tests_passed": True,
        "synthetic_checks_verified": [
            "paired_table_orientation_and_delta_sign",
            "symmetry_under_baseline_wrapped_swap",
            "zero_discordant_pairs_mcnemar_and_positive_width_ci",
            "all_discordant_one_sided_and_degenerate_margin_handling",
            "negative_cross_product_correlation_handling",
            "correlation_reduction_vs_independent_wilson",
        ],
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run_statistical_unit_tests(), indent=2))
