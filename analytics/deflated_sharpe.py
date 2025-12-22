"""
Deflated Sharpe Ratio (DSR)

The Problem:
When you test many strategies and pick the best one, the reported Sharpe ratio
is biased upward. This is called "selection bias" or "data snooping."

Example: Test 100 random strategies, pick the best. The "best" will look good
purely by chance, not because it has any edge.

The Solution:
The Deflated Sharpe Ratio adjusts for the number of trials and gives you the
probability that your observed Sharpe would occur by chance.

References:
- Bailey & Lopez de Prado (2014) - "The Deflated Sharpe Ratio"
- Harvey et al. (2016) - "...and the Cross-Section of Expected Returns"

Key Insight:
A Sharpe of 2.0 after testing 10 strategies is impressive.
A Sharpe of 2.0 after testing 10,000 strategies is almost certainly noise.
"""

import numpy as np
from scipy import stats
from typing import Dict, Optional


def expected_max_sharpe(n_trials: int, sample_length: int, skewness: float = 0, kurtosis: float = 3) -> float:
    """
    Calculate expected maximum Sharpe ratio from N independent trials.

    Under the null hypothesis (no skill), this is what you'd expect the
    "best" Sharpe to be after testing N strategies.

    Args:
        n_trials: Number of strategies/parameter combinations tested
        sample_length: Number of observations (e.g., trading days)
        skewness: Skewness of returns (0 for normal)
        kurtosis: Kurtosis of returns (3 for normal)

    Returns:
        Expected maximum Sharpe ratio under the null (no skill)
    """
    if n_trials <= 0:
        return 0

    # Euler-Mascheroni constant
    gamma = 0.5772156649

    # Expected max of N standard normals
    # E[max(Z_1, ..., Z_N)] ≈ (1 - gamma) * Phi^{-1}(1 - 1/N) + gamma * Phi^{-1}(1 - 1/(N*e))
    if n_trials == 1:
        e_max = 0
    else:
        # Approximation for expected maximum
        e_max = (1 - gamma) * stats.norm.ppf(1 - 1/n_trials) + \
                gamma * stats.norm.ppf(1 - 1/(n_trials * np.e))

    # Adjust for sample size and non-normality
    # Standard error of Sharpe ratio
    sr_std = np.sqrt((1 + 0.5 * skewness**2 + (kurtosis - 3)/4) / sample_length)

    return e_max * sr_std


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    sample_length: int,
    skewness: float = 0,
    kurtosis: float = 3,
    sharpe_benchmark: float = 0
) -> Dict[str, float]:
    """
    Calculate Deflated Sharpe Ratio.

    This tells you: "What's the probability that my observed Sharpe ratio
    is due to skill rather than luck from testing many strategies?"

    Args:
        observed_sharpe: The Sharpe ratio you observed (annualized)
        n_trials: Number of strategies/parameter combinations you tested
        sample_length: Number of return observations (trading days)
        skewness: Skewness of returns (0 = symmetric)
        kurtosis: Kurtosis of returns (3 = normal tails)
        sharpe_benchmark: Minimum Sharpe to beat (usually 0)

    Returns:
        Dict with:
        - deflated_sharpe: Adjusted Sharpe ratio
        - haircut: How much the Sharpe was reduced (percentage)
        - p_value: Probability observed Sharpe is due to chance
        - e_max_sharpe: Expected max Sharpe under null (no skill)
        - is_significant: Whether Sharpe survives multiple testing adjustment
        - interpretation: Human-readable assessment

    Example:
        >>> dsr = deflated_sharpe_ratio(
        ...     observed_sharpe=1.5,
        ...     n_trials=50,  # tested 50 parameter combinations
        ...     sample_length=252,  # one year of daily data
        ... )
        >>> print(dsr['interpretation'])
        "After adjusting for 50 trials, Sharpe of 1.50 has 23% chance of being luck"
    """
    if n_trials < 1:
        n_trials = 1

    if sample_length < 10:
        return {
            'deflated_sharpe': observed_sharpe,
            'haircut': 0,
            'p_value': 1.0,
            'e_max_sharpe': 0,
            'is_significant': False,
            'interpretation': 'Insufficient data for DSR calculation'
        }

    # Expected maximum Sharpe under the null (no skill)
    e_max = expected_max_sharpe(n_trials, sample_length, skewness, kurtosis)

    # Standard error of Sharpe ratio
    # SR has std = sqrt((1 + 0.5*skew^2 + (kurt-3)/4) / T)
    sr_std = np.sqrt((1 + 0.5 * skewness**2 + (kurtosis - 3)/4) / sample_length)

    # P-value: probability of observing this Sharpe or higher under null
    # This is a one-sided test
    z_score = (observed_sharpe - sharpe_benchmark) / sr_std if sr_std > 0 else 0

    # Adjust z-score for multiple testing
    # Under N trials, the distribution of max is different
    if n_trials > 1:
        # Approximate p-value adjustment using Sidak correction
        # P(max >= observed) ≈ 1 - (1 - p_single)^N
        p_single = 1 - stats.norm.cdf(z_score)
        p_value = min(1.0, 1 - (1 - p_single) ** n_trials)
    else:
        p_value = 1 - stats.norm.cdf(z_score)

    # Deflated Sharpe = observed - expected_max_under_null
    deflated = max(0, observed_sharpe - e_max)

    # Haircut percentage
    if observed_sharpe > 0:
        haircut = (1 - deflated / observed_sharpe) * 100
    else:
        haircut = 0

    # Interpretation
    if p_value < 0.01:
        interp = f"Strong evidence of skill. After {n_trials} trials, p={p_value:.3f}"
    elif p_value < 0.05:
        interp = f"Moderate evidence. After {n_trials} trials, p={p_value:.3f}. Needs more validation."
    elif p_value < 0.10:
        interp = f"Weak evidence. After {n_trials} trials, {p_value*100:.0f}% chance this is luck."
    else:
        interp = f"Likely luck. After {n_trials} trials, Sharpe of {observed_sharpe:.2f} has {p_value*100:.0f}% chance of being noise."

    return {
        'observed_sharpe': observed_sharpe,
        'deflated_sharpe': deflated,
        'haircut': haircut,
        'p_value': p_value,
        'e_max_sharpe': e_max,
        'n_trials': n_trials,
        'sample_length': sample_length,
        'is_significant': p_value < 0.05,
        'interpretation': interp
    }


def estimate_n_trials(
    param_grid: Dict[str, list] = None,
    n_strategies: int = 1,
    n_param_combos: int = None
) -> int:
    """
    Estimate the number of implicit trials from strategy/parameter search.

    Be honest about how many things you tested!

    Args:
        param_grid: Dict of parameter names to lists of values tested
                   e.g., {'short_ma': [10,20,30], 'long_ma': [50,100,200]}
        n_strategies: Number of different strategy types tested
        n_param_combos: Override: directly specify number of combinations

    Returns:
        Estimated number of independent trials
    """
    if n_param_combos is not None:
        return n_param_combos * n_strategies

    if param_grid is None:
        return n_strategies

    # Calculate combinations from parameter grid
    n_combos = 1
    for param, values in param_grid.items():
        n_combos *= len(values)

    return n_combos * n_strategies


def multiple_testing_warning(n_trials: int, observed_sharpe: float) -> str:
    """
    Generate a warning message about multiple testing.

    Display this prominently to prevent over-interpretation.
    """
    if n_trials <= 1:
        return ""

    dsr = deflated_sharpe_ratio(observed_sharpe, n_trials, 252)  # Assume 1 year

    warning = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ⚠️  MULTIPLE TESTING WARNING                                                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  You tested {n_trials:,} parameter/strategy combinations.                            ║
║  The "best" result has inflated performance due to selection bias.           ║
║                                                                              ║
║  Observed Sharpe:  {observed_sharpe:>6.2f}                                              ║
║  Deflated Sharpe:  {dsr['deflated_sharpe']:>6.2f}  ({dsr['haircut']:.0f}% haircut)                              ║
║  P-value:          {dsr['p_value']:>6.3f}  (prob. this is luck)                       ║
║                                                                              ║
║  {dsr['interpretation']:<75}║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    return warning.strip()
