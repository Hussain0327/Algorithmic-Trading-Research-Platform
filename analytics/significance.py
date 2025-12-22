"""
Statistical Significance Testing for Backtesting Results

This module provides rigorous statistical tests to evaluate whether strategy
performance is statistically significant or likely due to random chance.

Key insight: A backtest showing 15% returns means nothing without knowing
the probability that result occurred by chance.

METHODOLOGY NOTES:
- Block bootstrap preserves autocorrelation structure (Politis & Romano 1994)
- Permutation tests assume exchangeability under null hypothesis
- All tests document their assumptions explicitly
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, Optional


# =============================================================================
# TEST ASSUMPTIONS DOCUMENTATION
# =============================================================================

BOOTSTRAP_ASSUMPTIONS = {
    'name': 'Block Bootstrap Sharpe CI',
    'assumes': [
        'Returns are stationary (distribution does not change over time)',
        'Finite variance (no infinite-variance fat tails)',
        'Block size captures relevant autocorrelation structure'
    ],
    'preserves': [
        'Autocorrelation within blocks',
        'Marginal distribution of returns'
    ],
    'does_not_account_for': [
        'Multiple testing (run many strategies, pick best)',
        'Regime changes (bull/bear market shifts)',
        'Look-ahead bias in strategy construction'
    ]
}

PERMUTATION_ASSUMPTIONS = {
    'name': 'Permutation Test vs Benchmark',
    'assumes': [
        'Strategy and benchmark returns are exchangeable under H0',
        'Returns within each series may be dependent',
        'Both series cover the same time period'
    ],
    'preserves': [
        'Sample sizes of each group',
        'Pooled distribution of all returns'
    ],
    'does_not_account_for': [
        'Time-series structure (shuffles across time)',
        'Different risk profiles (vol, skew, kurtosis)',
        'Transaction costs already embedded in strategy returns'
    ],
    'null_hypothesis': 'Strategy returns come from same distribution as benchmark'
}

MONTE_CARLO_ASSUMPTIONS = {
    'name': 'Monte Carlo vs Random Trading',
    'assumes': [
        'Random entry/exit points are uniformly distributed',
        'Number of trades similar to strategy',
        'Same capital and cost structure'
    ],
    'preserves': [
        'Price path (same underlying data)',
        'Approximate trade frequency'
    ],
    'does_not_account_for': [
        'Signal-based entry timing',
        'Correlation between strategy signals and price moves',
        'Survivorship bias in price data'
    ],
    'null_hypothesis': 'Strategy performs no better than random entry/exit'
}

def get_test_assumptions() -> Dict[str, Dict]:
    """Return assumptions for all significance tests."""
    return {
        'bootstrap': BOOTSTRAP_ASSUMPTIONS,
        'permutation': PERMUTATION_ASSUMPTIONS,
        'monte_carlo': MONTE_CARLO_ASSUMPTIONS
    }


# =============================================================================
# BLOCK BOOTSTRAP (preserves autocorrelation)
# =============================================================================

def estimate_block_size(returns: pd.Series) -> int:
    """
    Estimate optimal block size for block bootstrap.

    Uses the rule of thumb: block_size ~ n^(1/3) for weakly dependent data.
    Also considers autocorrelation decay.

    References:
    - Politis & Romano (1994) - Stationary Bootstrap
    - Lahiri (2003) - Resampling Methods for Dependent Data
    """
    n = len(returns)
    if n < 20:
        return max(2, n // 4)

    # Rule of thumb for weakly dependent data
    block_size = int(np.ceil(n ** (1/3)))

    # Adjust based on autocorrelation if significant
    try:
        acf_1 = returns.autocorr(lag=1)
        if pd.notna(acf_1) and abs(acf_1) > 0.1:
            # Higher autocorrelation -> larger blocks
            block_size = int(block_size * (1 + abs(acf_1)))
    except:
        pass

    return max(2, min(block_size, n // 4))


def block_bootstrap_sample(returns: np.ndarray, block_size: int) -> np.ndarray:
    """
    Generate one block bootstrap sample.

    Randomly selects blocks of consecutive observations and concatenates
    them to form a resampled series of the same length.
    """
    n = len(returns)
    n_blocks = int(np.ceil(n / block_size))

    # Randomly select starting indices for blocks
    max_start = n - block_size
    if max_start < 0:
        # Series too short for block bootstrap, fall back to i.i.d.
        return np.random.choice(returns, size=n, replace=True)

    starts = np.random.randint(0, max_start + 1, size=n_blocks)

    # Build resampled series from blocks
    resampled = []
    for start in starts:
        resampled.extend(returns[start:start + block_size])

    return np.array(resampled[:n])


def bootstrap_sharpe_confidence_interval(
    returns: pd.Series,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    risk_free_rate: float = 0.02,
    block_size: int = None,
    method: str = 'block'
) -> Dict[str, float]:
    """
    Calculate bootstrap confidence interval for Sharpe ratio.

    Uses BLOCK BOOTSTRAP by default to preserve autocorrelation structure.
    This is critical for financial returns which exhibit serial dependence.

    Args:
        returns: Daily returns series
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        risk_free_rate: Annual risk-free rate
        block_size: Block size for block bootstrap (auto-estimated if None)
        method: 'block' (default, preserves autocorrelation) or 'iid' (naive)

    Returns:
        Dict with point estimate, confidence bounds, and methodology info

    References:
        - Politis & Romano (1994) - The Stationary Bootstrap
        - Ledoit & Wolf (2008) - Robust Performance Hypothesis Testing

    Assumptions (see BOOTSTRAP_ASSUMPTIONS):
        - Returns are stationary
        - Finite variance
        - Block size captures autocorrelation structure
    """
    returns = returns.dropna()
    if len(returns) < 20:
        return {
            'sharpe': 0, 'ci_lower': 0, 'ci_upper': 0, 'std_error': 0,
            'ci_includes_zero': True, 'method': method, 'block_size': None,
            'assumptions': BOOTSTRAP_ASSUMPTIONS
        }

    daily_rf = risk_free_rate / 252
    n = len(returns)
    returns_arr = returns.values

    # Auto-estimate block size if not provided
    if method == 'block' and block_size is None:
        block_size = estimate_block_size(returns)

    def calc_sharpe(r):
        excess = r - daily_rf
        if len(r) == 0 or np.std(excess) == 0:
            return 0
        return np.sqrt(252) * np.mean(excess) / np.std(excess)

    point_estimate = calc_sharpe(returns_arr)

    # Bootstrap resampling
    bootstrap_sharpes = []

    for _ in range(n_bootstrap):
        if method == 'block':
            sample = block_bootstrap_sample(returns_arr, block_size)
        else:
            # Naive i.i.d. bootstrap (breaks autocorrelation - use with caution)
            sample = np.random.choice(returns_arr, size=n, replace=True)

        bootstrap_sharpes.append(calc_sharpe(sample))

    bootstrap_sharpes = np.array(bootstrap_sharpes)
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_sharpes, alpha / 2 * 100)
    ci_upper = np.percentile(bootstrap_sharpes, (1 - alpha / 2) * 100)

    return {
        'sharpe': point_estimate,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'std_error': bootstrap_sharpes.std(),
        'ci_includes_zero': ci_lower <= 0 <= ci_upper,
        'method': method,
        'block_size': block_size if method == 'block' else None,
        'n_bootstrap': n_bootstrap,
        'assumptions': BOOTSTRAP_ASSUMPTIONS
    }


def permutation_test_vs_baseline(
    strategy_returns: pd.Series,
    baseline_returns: pd.Series,
    n_permutations: int = 10000,
    metric: str = 'mean'
) -> Dict[str, float]:
    """
    Permutation test to determine if strategy outperformance is significant.

    Tests whether the observed difference in performance could have arisen
    by chance if strategy and baseline returns came from the same distribution.

    IMPORTANT CAVEAT: This test shuffles returns i.i.d., which breaks
    time-series dependence. The p-value should be interpreted as approximate.
    For highly autocorrelated returns, consider block permutation methods.

    Args:
        strategy_returns: Strategy daily returns
        baseline_returns: Baseline (e.g., buy-and-hold) daily returns
        n_permutations: Number of permutations
        metric: 'mean' for average return, 'sharpe' for risk-adjusted

    Returns:
        Dict with test statistic, p-value, and assumptions

    Null Hypothesis:
        Strategy and baseline returns are exchangeable (come from same distribution)

    Assumptions (see PERMUTATION_ASSUMPTIONS):
        - Exchangeability under H0
        - Same time period coverage
        - Shuffling i.i.d. is approximate for dependent data
    """
    strategy_returns = strategy_returns.dropna()
    baseline_returns = baseline_returns.dropna()

    # Align lengths
    min_len = min(len(strategy_returns), len(baseline_returns))
    strategy_returns = strategy_returns.iloc[:min_len]
    baseline_returns = baseline_returns.iloc[:min_len]

    if min_len < 10:
        return {
            'observed_diff': 0, 'p_value': 1.0, 'significant_at_05': False,
            'significant_at_01': False, 'assumptions': PERMUTATION_ASSUMPTIONS
        }

    def calc_metric(r):
        if metric == 'sharpe':
            if r.std() == 0:
                return 0
            return np.sqrt(252) * r.mean() / r.std()
        return r.mean() * 252  # Annualized mean return

    observed_diff = calc_metric(strategy_returns) - calc_metric(baseline_returns)

    # Pool all returns
    combined = np.concatenate([strategy_returns.values, baseline_returns.values])
    n_strategy = len(strategy_returns)

    # Permutation test
    count_extreme = 0
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        perm_strategy = pd.Series(combined[:n_strategy])
        perm_baseline = pd.Series(combined[n_strategy:])
        perm_diff = calc_metric(perm_strategy) - calc_metric(perm_baseline)

        if perm_diff >= observed_diff:
            count_extreme += 1

    p_value = (count_extreme + 1) / (n_permutations + 1)

    return {
        'observed_diff': observed_diff,
        'p_value': p_value,
        'significant_at_05': p_value < 0.05,
        'significant_at_01': p_value < 0.01,
        'metric_used': metric,
        'n_permutations': n_permutations,
        'assumptions': PERMUTATION_ASSUMPTIONS,
        'caveat': 'i.i.d. permutation breaks time-series dependence; interpret cautiously'
    }


def monte_carlo_under_null(
    prices: pd.Series,
    n_simulations: int = 1000,
    strategy_return: float = None,
    n_trades_observed: int = None
) -> Dict[str, float]:
    """
    Monte Carlo simulation to test if strategy beats random entry/exit.

    Generates random trading signals and computes distribution of returns
    under the null hypothesis that the strategy has no edge (random timing).

    This answers: "Could I have achieved similar returns by trading randomly?"

    Args:
        prices: Price series used in backtest
        n_simulations: Number of random strategy simulations
        strategy_return: Actual strategy return to compare against
        n_trades_observed: Number of trades in actual strategy (for matching)

    Returns:
        Dict with null distribution statistics, p-value, and assumptions

    Null Hypothesis:
        Strategy performs no better than random entry/exit timing

    Assumptions (see MONTE_CARLO_ASSUMPTIONS):
        - Random entry/exit points uniformly distributed
        - Trade count similar to actual strategy
        - Same price path (no alternative universes)
    """
    prices = prices.dropna()
    if len(prices) < 20:
        return {
            'p_value': 1.0, 'null_mean': 0, 'null_std': 0,
            'assumptions': MONTE_CARLO_ASSUMPTIONS
        }

    random_returns = []
    n = len(prices)

    for _ in range(n_simulations):
        # Random number of trades (match observed if provided)
        if n_trades_observed is not None:
            n_trades = max(1, n_trades_observed // 2)  # pairs of buy/sell
        else:
            n_trades = np.random.randint(2, max(3, n // 20))

        # Random entry/exit points
        n_points = min(n_trades * 2, n - 2)
        if n_points < 2:
            random_returns.append(0)
            continue

        trade_points = sorted(np.random.choice(range(1, n-1), size=n_points, replace=False))

        # Simulate random strategy
        capital = 10000
        position = 0

        for i, idx in enumerate(trade_points):
            price = prices.iloc[idx]
            if i % 2 == 0 and position == 0:  # Buy
                position = capital / price
                capital = 0
            elif i % 2 == 1 and position > 0:  # Sell
                capital = position * price
                position = 0

        # Close any open position at end
        if position > 0:
            capital = position * prices.iloc[-1]

        final_return = (capital - 10000) / 10000 * 100
        random_returns.append(final_return)

    random_returns = np.array(random_returns)
    null_mean = random_returns.mean()
    null_std = random_returns.std()

    result = {
        'null_mean': null_mean,
        'null_std': null_std,
        'null_median': np.median(random_returns),
        'null_5th_percentile': np.percentile(random_returns, 5),
        'null_95th_percentile': np.percentile(random_returns, 95),
        'n_simulations': n_simulations,
        'assumptions': MONTE_CARLO_ASSUMPTIONS
    }

    if strategy_return is not None:
        # One-tailed test: what fraction of random strategies beat ours?
        p_value = np.mean(random_returns >= strategy_return)
        result['strategy_return'] = strategy_return
        result['p_value'] = p_value
        result['percentile_rank'] = np.mean(random_returns <= strategy_return) * 100
        result['significant_at_05'] = p_value < 0.05

    return result


def analyze_return_distribution(returns: pd.Series) -> Dict[str, float]:
    """
    Test statistical properties of the return distribution.

    Important for understanding if standard assumptions hold and
    whether risk metrics are reliable.

    Args:
        returns: Daily returns series

    Returns:
        Dict with distribution statistics and normality tests
    """
    returns = returns.dropna()

    if len(returns) < 20:
        return {'error': 'Insufficient data for distribution analysis'}

    # Basic statistics
    mean = returns.mean()
    std = returns.std()
    skewness = stats.skew(returns)
    kurtosis = stats.kurtosis(returns)  # Excess kurtosis (normal = 0)

    # Normality tests
    if len(returns) >= 20:
        shapiro_stat, shapiro_p = stats.shapiro(returns[:min(5000, len(returns))])
    else:
        shapiro_stat, shapiro_p = 0, 1

    # Jarque-Bera test
    jb_stat, jb_p = stats.jarque_bera(returns)

    return {
        'mean_daily': mean,
        'std_daily': std,
        'skewness': skewness,
        'excess_kurtosis': kurtosis,
        'is_fat_tailed': kurtosis > 1,  # Higher than normal
        'is_negatively_skewed': skewness < -0.5,
        'shapiro_wilk_p': shapiro_p,
        'jarque_bera_p': jb_p,
        'is_normal_shapiro': shapiro_p > 0.05,
        'is_normal_jb': jb_p > 0.05,
        'var_95': np.percentile(returns, 5),  # 95% VaR
        'cvar_95': returns[returns <= np.percentile(returns, 5)].mean()  # Expected shortfall
    }


def calculate_benchmark_returns(prices: pd.Series) -> pd.Series:
    """
    Calculate buy-and-hold benchmark returns for comparison.

    Args:
        prices: Price series

    Returns:
        Daily returns series for buy-and-hold strategy
    """
    return prices.pct_change().dropna()


def strategy_significance_report(
    strategy_results: Dict,
    prices: pd.Series,
    n_bootstrap: int = 5000,
    n_permutations: int = 5000
) -> Dict:
    """
    Generate comprehensive statistical significance report for a strategy.

    This is the main function to call after running a backtest.

    Args:
        strategy_results: Results dict from BacktestEngine
        prices: Price series used in backtest
        n_bootstrap: Bootstrap samples for confidence intervals
        n_permutations: Permutations for hypothesis tests

    Returns:
        Complete significance report
    """
    equity_curve = strategy_results['equity_curve']
    strategy_returns = equity_curve['equity'].pct_change().dropna()
    benchmark_returns = calculate_benchmark_returns(prices)

    report = {
        'sharpe_confidence': bootstrap_sharpe_confidence_interval(
            strategy_returns, n_bootstrap=n_bootstrap
        ),
        'vs_benchmark': permutation_test_vs_baseline(
            strategy_returns,
            benchmark_returns,
            n_permutations=n_permutations
        ),
        'vs_random': monte_carlo_under_null(
            prices,
            n_simulations=n_permutations,
            strategy_return=strategy_results['return_pct']
        ),
        'return_distribution': analyze_return_distribution(strategy_returns)
    }

    # Summary interpretation
    sharpe_significant = not report['sharpe_confidence'].get('ci_includes_zero', True)
    beats_benchmark = report['vs_benchmark'].get('significant_at_05', False)
    beats_random = report['vs_random'].get('significant_at_05', False)

    report['summary'] = {
        'sharpe_statistically_significant': sharpe_significant,
        'beats_benchmark_significantly': beats_benchmark,
        'beats_random_trading': beats_random,
        'overall_evidence': _interpret_evidence(sharpe_significant, beats_benchmark, beats_random)
    }

    return report


def _interpret_evidence(sharpe_sig: bool, beats_bench: bool, beats_random: bool) -> str:
    """Interpret the statistical evidence."""
    score = sum([sharpe_sig, beats_bench, beats_random])

    if score == 3:
        return "Strong evidence of genuine edge"
    elif score == 2:
        return "Moderate evidence - warrants further investigation"
    elif score == 1:
        return "Weak evidence - likely noise or overfitting"
    else:
        return "No statistical evidence of edge over random"
