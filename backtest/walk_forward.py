"""
Walk-Forward Validation for Trading Strategies

A single train/test split is easy to overfit. Walk-forward validation
provides a more robust estimate of out-of-sample performance by testing
across multiple time periods.

This answers: "Does this strategy work consistently, or did I get lucky
with one particular split?"

Methods:
- Rolling: Fixed-size training window, slides forward
- Expanding: Training window grows over time (anchored start)

References:
- Pardo (2008) - The Evaluation and Optimization of Trading Strategies
- Bailey et al. (2014) - Pseudo-Mathematics and Financial Charlatanism
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Generator, Optional
from dataclasses import dataclass

from analytics.metrics import calculate_metrics


@dataclass
class FoldResult:
    """Results from a single walk-forward fold."""
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_metrics: Dict
    test_metrics: Dict
    n_train_days: int
    n_test_days: int


class WalkForwardValidator:
    """
    Walk-forward validation for trading strategies.

    Instead of a single train/test split, this divides data into multiple
    sequential folds and tests performance across all of them.

    This is critical for detecting:
    - Overfitting to a specific time period
    - Regime-dependent performance
    - Lucky splits that don't generalize
    """

    def __init__(
        self,
        n_splits: int = 5,
        train_ratio: float = 0.7,
        method: str = 'rolling',
        min_train_days: int = 60,
        min_test_days: int = 20
    ):
        """
        Initialize walk-forward validator.

        Args:
            n_splits: Number of train/test splits (folds)
            train_ratio: Proportion of each fold used for training
            method: 'rolling' (fixed window) or 'expanding' (growing window)
            min_train_days: Minimum training period length
            min_test_days: Minimum test period length
        """
        self.n_splits = n_splits
        self.train_ratio = train_ratio
        self.method = method
        self.min_train_days = min_train_days
        self.min_test_days = min_test_days

    def split(self, data: pd.DataFrame) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
        """
        Generate train/test splits for walk-forward validation.

        Yields:
            (train_data, test_data) tuples for each fold
        """
        n = len(data)
        fold_size = n // self.n_splits

        if fold_size < self.min_train_days + self.min_test_days:
            raise ValueError(
                f"Insufficient data for {self.n_splits} folds. "
                f"Need at least {(self.min_train_days + self.min_test_days) * self.n_splits} days, "
                f"have {n} days."
            )

        for i in range(self.n_splits):
            if self.method == 'expanding':
                # Expanding window: train on all data from start
                train_start = 0
                train_end = (i + 1) * fold_size
            else:
                # Rolling window: fixed-size training window
                train_start = i * fold_size
                train_end = train_start + int(fold_size * self.train_ratio)

            test_start = train_end
            test_end = min((i + 2) * fold_size, n) if i < self.n_splits - 1 else n

            # Ensure minimum periods
            if train_end - train_start < self.min_train_days:
                continue
            if test_end - test_start < self.min_test_days:
                continue

            train_data = data.iloc[train_start:train_end].copy()
            test_data = data.iloc[test_start:test_end].copy()

            yield train_data, test_data

    def validate(
        self,
        data: pd.DataFrame,
        strategy,
        engine,
        recalibrate: bool = True
    ) -> Dict:
        """
        Run walk-forward validation.

        Args:
            data: Full price data
            strategy: Strategy instance with generate_signals method
            engine: BacktestEngine instance
            recalibrate: If True, regenerate signals for each fold (proper walk-forward)
                        If False, use signals from full data (faster but less rigorous)

        Returns:
            Dict with:
            - fold_results: List of FoldResult for each fold
            - aggregate: Aggregated statistics across folds
            - consistency: Measures of performance stability
        """
        fold_results = []

        for fold_id, (train_data, test_data) in enumerate(self.split(data)):
            # Generate signals for this fold
            if recalibrate:
                train_signals = strategy.generate_signals(train_data)
                test_signals = strategy.generate_signals(test_data)
            else:
                # Use pre-computed signals (less rigorous)
                full_signals = strategy.generate_signals(data)
                train_signals = full_signals.iloc[:len(train_data)]
                test_signals = full_signals.iloc[len(train_data):len(train_data)+len(test_data)]

            # Run backtests
            train_result = engine._simulate(train_signals)
            test_result = engine._simulate(test_signals)

            train_metrics = calculate_metrics(train_result)
            test_metrics = calculate_metrics(test_result)

            fold_result = FoldResult(
                fold_id=fold_id,
                train_start=train_data.index[0] if hasattr(train_data.index[0], 'strftime') else fold_id,
                train_end=train_data.index[-1] if hasattr(train_data.index[-1], 'strftime') else fold_id,
                test_start=test_data.index[0] if hasattr(test_data.index[0], 'strftime') else fold_id,
                test_end=test_data.index[-1] if hasattr(test_data.index[-1], 'strftime') else fold_id,
                train_metrics=train_metrics,
                test_metrics=test_metrics,
                n_train_days=len(train_data),
                n_test_days=len(test_data)
            )
            fold_results.append(fold_result)

        if not fold_results:
            return {
                'fold_results': [],
                'aggregate': {},
                'consistency': {},
                'error': 'No valid folds generated'
            }

        # Aggregate metrics across folds
        aggregate = self._aggregate_results(fold_results)
        consistency = self._calculate_consistency(fold_results)

        return {
            'fold_results': fold_results,
            'aggregate': aggregate,
            'consistency': consistency,
            'n_folds': len(fold_results),
            'method': self.method
        }

    def _aggregate_results(self, fold_results: List[FoldResult]) -> Dict:
        """Aggregate metrics across all folds."""
        test_returns = [f.test_metrics.get('total_return', 0) for f in fold_results]
        test_sharpes = [f.test_metrics.get('sharpe', 0) for f in fold_results]
        test_drawdowns = [f.test_metrics.get('max_drawdown', 0) for f in fold_results]
        test_winrates = [f.test_metrics.get('win_rate', 0) for f in fold_results]

        return {
            'return': {
                'mean': np.mean(test_returns),
                'std': np.std(test_returns),
                'min': np.min(test_returns),
                'max': np.max(test_returns),
                'median': np.median(test_returns)
            },
            'sharpe': {
                'mean': np.mean(test_sharpes),
                'std': np.std(test_sharpes),
                'min': np.min(test_sharpes),
                'max': np.max(test_sharpes),
                'median': np.median(test_sharpes)
            },
            'max_drawdown': {
                'mean': np.mean(test_drawdowns),
                'std': np.std(test_drawdowns),
                'worst': np.min(test_drawdowns)  # Most negative
            },
            'win_rate': {
                'mean': np.mean(test_winrates),
                'std': np.std(test_winrates)
            }
        }

    def _calculate_consistency(self, fold_results: List[FoldResult]) -> Dict:
        """
        Calculate consistency metrics across folds.

        High consistency = strategy works reliably
        Low consistency = strategy is regime-dependent or overfit
        """
        test_returns = [f.test_metrics.get('total_return', 0) for f in fold_results]
        test_sharpes = [f.test_metrics.get('sharpe', 0) for f in fold_results]

        n_positive_returns = sum(1 for r in test_returns if r > 0)
        n_positive_sharpe = sum(1 for s in test_sharpes if s > 0)
        n_folds = len(fold_results)

        # Coefficient of variation (lower = more consistent)
        sharpe_mean = np.mean(test_sharpes)
        sharpe_cv = np.std(test_sharpes) / abs(sharpe_mean) if sharpe_mean != 0 else float('inf')

        return {
            'pct_positive_returns': n_positive_returns / n_folds * 100,
            'pct_positive_sharpe': n_positive_sharpe / n_folds * 100,
            'sharpe_coefficient_of_variation': sharpe_cv,
            'is_consistent': n_positive_sharpe >= n_folds * 0.6,  # 60%+ folds positive
            'interpretation': self._interpret_consistency(n_positive_sharpe, n_folds, sharpe_cv)
        }

    def _interpret_consistency(self, n_positive: int, n_folds: int, cv: float) -> str:
        """Generate human-readable consistency interpretation."""
        pct = n_positive / n_folds

        if pct >= 0.8 and cv < 1.0:
            return "Strong consistency - strategy works across most time periods"
        elif pct >= 0.6:
            return "Moderate consistency - works in majority of periods, some regime sensitivity"
        elif pct >= 0.4:
            return "Weak consistency - performance is regime-dependent"
        else:
            return "Inconsistent - likely overfit or random; avoid in production"


def walk_forward_summary(results: Dict) -> str:
    """Generate a text summary of walk-forward results."""
    if 'error' in results:
        return f"Walk-forward validation failed: {results['error']}"

    agg = results['aggregate']
    cons = results['consistency']
    n_folds = results['n_folds']

    lines = [
        f"Walk-Forward Validation ({n_folds} folds, {results['method']} method)",
        "=" * 50,
        "",
        "Out-of-Sample Performance Across Folds:",
        f"  Sharpe Ratio:  {agg['sharpe']['mean']:.2f} +/- {agg['sharpe']['std']:.2f}",
        f"                 (range: {agg['sharpe']['min']:.2f} to {agg['sharpe']['max']:.2f})",
        f"  Total Return:  {agg['return']['mean']:.2f}% +/- {agg['return']['std']:.2f}%",
        f"  Max Drawdown:  {agg['max_drawdown']['mean']:.2f}% (worst: {agg['max_drawdown']['worst']:.2f}%)",
        "",
        "Consistency Metrics:",
        f"  Positive Sharpe in {cons['pct_positive_sharpe']:.0f}% of folds",
        f"  Sharpe CV: {cons['sharpe_coefficient_of_variation']:.2f}",
        "",
        f"Assessment: {cons['interpretation']}"
    ]

    return "\n".join(lines)
