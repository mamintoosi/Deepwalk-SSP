# -*- coding: utf-8 -*-
"""
Statistical Analysis Module for DeepWalk-SSP
==============================================
Non-parametric statistical tests and effect size calculations.
"""

import numpy as np
from scipy import stats
from scipy.stats import wilcoxon, rankdata, norm


def wilcoxon_signed_rank_test(x, y, alternative='two-sided'):
    """
    Perform Wilcoxon signed-rank test.
    
    Args:
        x: Array of observations from method 1 (e.g., DeepWalk).
        y: Array of observations from method 2 (e.g., BoW).
        alternative: 'two-sided', 'greater', or 'less'.
        
    Returns:
        Dictionary with test statistic, p-value, and interpretation.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Remove pairs where both are NaN
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid], y[valid]
    
    if len(x) < 5:
        return {"statistic": np.nan, "p_value": np.nan, 
                "interpretation": "Insufficient data", "n": len(x)}
    
    try:
        stat, p_value = wilcoxon(x, y, alternative=alternative)
        return {
            "statistic": stat,
            "p_value": p_value,
            "n": len(x),
            "interpretation": interpret_p_value(p_value),
        }
    except ValueError as e:
        return {"statistic": np.nan, "p_value": np.nan,
                "interpretation": f"Test failed: {e}", "n": len(x)}


def interpret_p_value(p, alpha=0.05):
    """Interpret p-value significance."""
    if np.isnan(p):
        return "N/A"
    if p < 0.001:
        return "Highly significant (***)"
    elif p < 0.01:
        return "Very significant (**)"
    elif p < alpha:
        return "Significant (*)"
    else:
        return "Not significant (ns)"


def compute_effect_size_r(statistic, n):
    """
    Compute effect size r from Wilcoxon statistic.
    
    r = Z / sqrt(N)
    
    Args:
        statistic: Wilcoxon test statistic.
        n: Number of pairs.
        
    Returns:
        Effect size r (0.1 small, 0.3 medium, 0.5 large).
    """
    if n < 5:
        return np.nan
    
    # Compute Z-score from Wilcoxon statistic
    # For large samples, the Wilcoxon statistic is approximately normal
    mean_w = n * (n + 1) / 4
    std_w = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    
    if std_w == 0:
        return 0.0
    
    z = (statistic - mean_w) / std_w
    r = abs(z) / np.sqrt(n)
    
    return min(r, 1.0)  # Cap at 1.0


def compute_cliffs_delta(x, y):
    """
    Compute Cliff's delta effect size.
    
    Non-parametric effect size measure:
    - 0 to 0.147: negligible
    - 0.147 to 0.33: small
    - 0.33 to 0.474: medium
    - > 0.474: large
    
    Returns:
        Cliff's delta value and interpretation.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid], y[valid]
    
    if len(x) == 0 or len(y) == 0:
        return np.nan, "N/A"
    
    n_x, n_y = len(x), len(y)
    dominance = 0
    
    for xi in x:
        for yj in y:
            if xi > yj:
                dominance += 1
            elif xi < yj:
                dominance -= 1
    
    delta = dominance / (n_x * n_y)
    
    abs_d = abs(delta)
    if abs_d < 0.147:
        interpretation = "Negligible"
    elif abs_d < 0.33:
        interpretation = "Small"
    elif abs_d < 0.474:
        interpretation = "Medium"
    else:
        interpretation = "Large"
    
    return delta, interpretation


def compute_confidence_interval_mean(data, confidence=0.95):
    """
    Compute confidence interval for the mean using bootstrap.
    
    Args:
        data: Array of observations.
        confidence: Confidence level (default 0.95).
        
    Returns:
        Tuple of (mean, ci_lower, ci_upper).
    """
    data = np.asarray(data)
    data = data[~np.isnan(data)]
    
    if len(data) < 3:
        return np.mean(data), np.nan, np.nan
    
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)
    
    # Use t-distribution for small samples
    t_val = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    margin = t_val * se
    
    return mean, mean - margin, mean + margin


def compute_bootstrap_ci(data, n_bootstrap=10000, confidence=0.95, 
                          statistic=np.mean):
    """
    Compute bootstrap confidence interval.
    
    Args:
        data: Array of observations.
        n_bootstrap: Number of bootstrap resamples.
        confidence: Confidence level.
        statistic: Function to compute (default: mean).
        
    Returns:
        Tuple of (point_estimate, ci_lower, ci_upper).
    """
    data = np.asarray(data)
    data = data[~np.isnan(data)]
    
    if len(data) < 3:
        s = statistic(data)
        return s, np.nan, np.nan
    
    rng = np.random.RandomState(42)
    boot_stats = []
    
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=len(data), replace=True)
        boot_stats.append(statistic(sample))
    
    boot_stats = np.array(boot_stats)
    alpha = 1 - confidence
    ci_lower = np.percentile(boot_stats, 100 * alpha / 2)
    ci_upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    
    return statistic(data), ci_lower, ci_upper


def benjamini_hochberg_correction(p_values, alpha=0.05):
    """
    Apply Benjamini-Hochberg correction for multiple comparisons.
    
    Args:
        p_values: List of p-values.
        alpha: Significance level.
        
    Returns:
        Dictionary with adjusted p-values and which are significant.
    """
    p_vals = np.array(p_values)
    m = len(p_vals)
    
    # Sort p-values
    sorted_indices = np.argsort(p_vals)
    sorted_p = p_vals[sorted_indices]
    
    # Compute adjusted p-values
    adjusted_p = np.zeros(m)
    for i, (rank, p) in enumerate(zip(range(1, m + 1), sorted_p)):
        adjusted = p * m / rank
        adjusted_p[i] = min(adjusted, 1.0)
    
    # Enforce monotonicity
    for i in range(m - 2, -1, -1):
        adjusted_p[i] = min(adjusted_p[i], adjusted_p[i + 1])
    
    # Map back to original order
    result = np.zeros(m)
    result[sorted_indices] = adjusted_p
    
    significant = result < alpha
    
    return {
        "adjusted_p_values": result.tolist(),
        "significant": significant.tolist(),
        "alpha": alpha,
    }


def run_full_statistical_analysis(deepwalk_scores, baseline_scores, 
                                    method_names=None, metric_name="Silhouette Score"):
    """
    Comprehensive statistical comparison between DeepWalk and baseline.
    
    Args:
        deepwalk_scores: {method_name: array of scores across datasets}
        baseline_scores: {method_name: array of scores across datasets}
        method_names: List of method names.
        metric_name: Name of metric for reporting.
        
    Returns:
        Complete statistical analysis results.
    """
    if method_names is None:
        method_names = sorted(set(deepwalk_scores.keys()) & set(baseline_scores.keys()))
    
    results = {
        "metric": metric_name,
        "methods": [],
        "p_values": [],
        "effect_sizes": [],
        "effect_interpretations": [],
        "confidence_intervals": [],
        "cliffs_delta": [],
        "cliffs_interpretation": [],
    }
    
    all_p_values = []
    
    for method in method_names:
        dw = np.array(deepwalk_scores[method])
        bl = np.array(baseline_scores[method])
        
        # Wilcoxon signed-rank test
        test_result = wilcoxon_signed_rank_test(dw, bl, alternative='greater')
        
        # Effect size r
        r = compute_effect_size_r(test_result.get("statistic", 0), test_result.get("n", 0))
        
        # Cliff's delta
        delta, delta_interp = compute_cliffs_delta(dw, bl)
        
        # Confidence interval for mean difference
        diff = dw - bl
        _, ci_lower, ci_upper = compute_bootstrap_ci(diff)
        
        results["methods"].append(method)
        results["p_values"].append(test_result["p_value"])
        results["effect_sizes"].append(r)
        results["effect_interpretations"].append(
            "Large" if r >= 0.5 else "Medium" if r >= 0.3 else "Small" if r >= 0.1 else "Negligible"
        )
        results["confidence_intervals"].append({
            "mean_diff": np.nanmean(diff),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        })
        results["cliffs_delta"].append(delta)
        results["cliffs_interpretation"].append(delta_interp)
        
        all_p_values.append(test_result["p_value"])
    
    # Multiple comparisons correction
    if all_p_values:
        bh_results = benjamini_hochberg_correction(all_p_values)
        results["adjusted_p_values"] = bh_results["adjusted_p_values"]
        results["significant_after_correction"] = bh_results["significant"]
    
    return results
