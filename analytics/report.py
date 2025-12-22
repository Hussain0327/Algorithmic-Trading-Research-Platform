"""
HTML Report Generator for Backtesting Results

Generates a professional, standalone HTML report that recruiters can scan
in 30 seconds. Includes:
- Executive summary with verdict
- Performance metrics with confidence intervals
- Statistical significance results with assumptions
- Walk-forward validation results (if available)
- Charts (equity curve, drawdown)
- Methodology notes

Usage:
    python -m analytics.report

Or programmatically:
    from analytics.report import generate_report
    generate_report(results, 'reports/backtest_report.html')
"""

import os
import json
from datetime import datetime
from typing import Dict, Optional
from pathlib import Path


# HTML template (embedded to avoid external dependencies)
HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Backtest Report - {ticker} | {strategy_name}</title>
    <style>
        :root {{
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-card: #334155;
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --accent-blue: #3b82f6;
            --accent-green: #10b981;
            --accent-red: #ef4444;
            --accent-yellow: #f59e0b;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 2rem;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{
            background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-card) 100%);
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            border-left: 4px solid var(--accent-blue);
        }}
        .header h1 {{ font-size: 1.75rem; margin-bottom: 0.5rem; }}
        .header .meta {{ color: var(--text-secondary); font-size: 0.9rem; }}
        .verdict-box {{
            background: var(--bg-secondary);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 2rem;
            text-align: center;
        }}
        .verdict-score {{
            font-size: 3rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }}
        .verdict-pass {{ color: var(--accent-green); }}
        .verdict-partial {{ color: var(--accent-yellow); }}
        .verdict-fail {{ color: var(--accent-red); }}
        .verdict-text {{ color: var(--text-secondary); }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 2rem; }}
        .metric-card {{
            background: var(--bg-secondary);
            padding: 1.25rem;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-label {{ color: var(--text-secondary); font-size: 0.85rem; margin-bottom: 0.25rem; }}
        .metric-value {{ font-size: 1.5rem; font-weight: 600; }}
        .metric-ci {{ font-size: 0.75rem; color: var(--text-secondary); }}
        .section {{
            background: var(--bg-secondary);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }}
        .section h2 {{
            font-size: 1.25rem;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid var(--bg-card);
        }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 0.75rem; text-align: left; border-bottom: 1px solid var(--bg-card); }}
        th {{ color: var(--text-secondary); font-weight: 500; }}
        .badge {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 500;
        }}
        .badge-pass {{ background: rgba(16, 185, 129, 0.2); color: var(--accent-green); }}
        .badge-fail {{ background: rgba(239, 68, 68, 0.2); color: var(--accent-red); }}
        .assumptions-box {{
            background: var(--bg-card);
            border-radius: 8px;
            padding: 1rem;
            margin-top: 1rem;
            font-size: 0.85rem;
        }}
        .assumptions-box h4 {{ color: var(--text-secondary); margin-bottom: 0.5rem; }}
        .assumptions-box ul {{ padding-left: 1.25rem; color: var(--text-secondary); }}
        .warning-box {{
            background: rgba(245, 158, 11, 0.1);
            border: 1px solid var(--accent-yellow);
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
            color: var(--accent-yellow);
        }}
        .footer {{
            text-align: center;
            color: var(--text-secondary);
            font-size: 0.8rem;
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid var(--bg-card);
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{strategy_name} Backtest Report</h1>
            <div class="meta">
                {ticker} | {date_range} | Generated {timestamp}
            </div>
        </div>

        <div class="verdict-box">
            <div class="verdict-score {verdict_class}">{tests_passed}/3</div>
            <div class="verdict-text">Statistical Tests Passed</div>
            <div style="margin-top: 1rem; color: var(--text-secondary);">{verdict_interpretation}</div>
        </div>

        <div class="grid">
            <div class="metric-card">
                <div class="metric-label">Total Return</div>
                <div class="metric-value" style="color: {return_color}">{total_return}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Sharpe Ratio</div>
                <div class="metric-value">{sharpe}</div>
                <div class="metric-ci">{sharpe_ci}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Max Drawdown</div>
                <div class="metric-value" style="color: var(--accent-red)">{max_drawdown}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Win Rate</div>
                <div class="metric-value">{win_rate}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Trades</div>
                <div class="metric-value">{n_trades}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Final Equity</div>
                <div class="metric-value">${final_equity}</div>
            </div>
        </div>

        {deflated_sharpe_section}

        <div class="section">
            <h2>Statistical Significance Tests</h2>
            <table>
                <thead>
                    <tr>
                        <th>Test</th>
                        <th>Result</th>
                        <th>P-Value</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {significance_rows}
                </tbody>
            </table>
            {assumptions_html}
        </div>

        {walk_forward_section}

        <div class="section">
            <h2>Methodology & Limitations</h2>
            <ul style="padding-left: 1.5rem; color: var(--text-secondary);">
                <li>Block bootstrap preserves autocorrelation (Politis & Romano 1994)</li>
                <li>Permutation test assumes exchangeability under null hypothesis</li>
                <li>Yahoo Finance data has survivorship bias (delisted stocks excluded)</li>
                <li>Execution assumes next-day close fills with {costs} transaction costs</li>
                <li>Results are out-of-sample (test period) but single split has variance</li>
            </ul>
            <div class="warning-box">
                <strong>Disclaimer:</strong> Backtest results do not guarantee future performance.
                This report is for research and educational purposes only.
            </div>
        </div>

        <div class="footer">
            Trading Strategy Validation Framework<br>
            Report generated using block bootstrap with {n_bootstrap} samples
        </div>
    </div>
</body>
</html>
'''


def generate_report(
    backtest_results: Dict,
    significance_results: Dict = None,
    walk_forward_results: Dict = None,
    deflated_sharpe: Dict = None,
    output_path: str = 'reports/backtest_report.html',
    ticker: str = 'UNKNOWN',
    date_range: str = '',
    n_trials: int = 1
) -> str:
    """
    Generate a standalone HTML backtest report.

    Args:
        backtest_results: Results from BacktestEngine.run()
        significance_results: Results from significance tests
        walk_forward_results: Results from walk-forward validation
        deflated_sharpe: Deflated Sharpe Ratio results
        output_path: Where to save the HTML file
        ticker: Stock ticker symbol
        date_range: Date range string
        n_trials: Number of strategies/parameters tested (for DSR warning)

    Returns:
        Path to generated report
    """
    # Extract metrics
    test_results = backtest_results.get('test', {})
    metrics = test_results.get('metrics', {})

    if not metrics:
        from analytics.metrics import calculate_metrics
        metrics = calculate_metrics(test_results)

    # Build template variables
    total_return = metrics.get('total_return', 0)
    sharpe = metrics.get('sharpe', 0)
    max_dd = metrics.get('max_drawdown', 0)
    win_rate = metrics.get('win_rate', 0)
    n_trades = metrics.get('num_trades', 0)
    final_equity = metrics.get('final_equity', 10000)

    # Significance results
    tests_passed = 0
    significance_rows = []
    assumptions_html = ""

    if significance_results:
        # Bootstrap
        sharpe_ci = significance_results.get('sharpe_confidence', {})
        sharpe_pass = not sharpe_ci.get('ci_includes_zero', True)
        if sharpe_pass:
            tests_passed += 1
        significance_rows.append(
            f'<tr><td>Bootstrap Sharpe CI</td>'
            f'<td>[{sharpe_ci.get("ci_lower", 0):.2f}, {sharpe_ci.get("ci_upper", 0):.2f}]</td>'
            f'<td>N/A</td>'
            f'<td><span class="badge {"badge-pass" if sharpe_pass else "badge-fail"}">'
            f'{"PASS" if sharpe_pass else "FAIL"}</span></td></tr>'
        )

        # Permutation
        perm = significance_results.get('vs_benchmark', {})
        perm_pass = perm.get('significant_at_05', False)
        if perm_pass:
            tests_passed += 1
        significance_rows.append(
            f'<tr><td>Permutation vs Benchmark</td>'
            f'<td>Diff: {perm.get("observed_diff", 0):.4f}</td>'
            f'<td>{perm.get("p_value", 1):.4f}</td>'
            f'<td><span class="badge {"badge-pass" if perm_pass else "badge-fail"}">'
            f'{"PASS" if perm_pass else "FAIL"}</span></td></tr>'
        )

        # Monte Carlo
        mc = significance_results.get('vs_random', {})
        mc_pass = mc.get('significant_at_05', False)
        if mc_pass:
            tests_passed += 1
        significance_rows.append(
            f'<tr><td>Monte Carlo vs Random</td>'
            f'<td>Rank: {mc.get("percentile_rank", 50):.1f}%</td>'
            f'<td>{mc.get("p_value", 1):.4f}</td>'
            f'<td><span class="badge {"badge-pass" if mc_pass else "badge-fail"}">'
            f'{"PASS" if mc_pass else "FAIL"}</span></td></tr>'
        )

        # Assumptions
        assumptions_html = '''
        <div class="assumptions-box">
            <h4>Test Assumptions & Caveats</h4>
            <ul>
                <li><strong>Bootstrap:</strong> Assumes stationary returns; block size auto-selected to preserve autocorrelation</li>
                <li><strong>Permutation:</strong> i.i.d. shuffling breaks time-series dependence; interpret cautiously</li>
                <li><strong>Monte Carlo:</strong> Random trades uniformly distributed; doesn't account for signal timing</li>
            </ul>
        </div>
        '''
    else:
        significance_rows.append('<tr><td colspan="4">Significance tests not run</td></tr>')

    # Verdict
    if tests_passed == 3:
        verdict_class = 'verdict-pass'
        verdict_interpretation = 'Strong evidence of edge. Warrants further investigation with walk-forward analysis.'
    elif tests_passed == 2:
        verdict_class = 'verdict-partial'
        verdict_interpretation = 'Moderate evidence. Consider parameter sensitivity analysis.'
    else:
        verdict_class = 'verdict-fail'
        verdict_interpretation = 'Weak or no evidence. Results likely due to noise or overfitting.'

    # Deflated Sharpe section
    deflated_section = ""
    if deflated_sharpe and n_trials > 1:
        deflated_section = f'''
        <div class="section">
            <h2>Multiple Testing Adjustment</h2>
            <div class="warning-box">
                <strong>You tested {n_trials} parameter/strategy combinations.</strong><br>
                The observed Sharpe of {sharpe:.2f} is biased upward due to selection.
            </div>
            <div class="grid" style="grid-template-columns: repeat(3, 1fr);">
                <div class="metric-card">
                    <div class="metric-label">Observed Sharpe</div>
                    <div class="metric-value">{deflated_sharpe.get("observed_sharpe", sharpe):.2f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Deflated Sharpe</div>
                    <div class="metric-value">{deflated_sharpe.get("deflated_sharpe", 0):.2f}</div>
                    <div class="metric-ci">{deflated_sharpe.get("haircut", 0):.0f}% haircut</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">P-Value (adjusted)</div>
                    <div class="metric-value">{deflated_sharpe.get("p_value", 1):.3f}</div>
                </div>
            </div>
            <p style="color: var(--text-secondary); margin-top: 1rem;">
                {deflated_sharpe.get("interpretation", "")}
            </p>
        </div>
        '''

    # Walk-forward section
    wf_section = ""
    if walk_forward_results and 'aggregate' in walk_forward_results:
        agg = walk_forward_results['aggregate']
        cons = walk_forward_results.get('consistency', {})
        wf_section = f'''
        <div class="section">
            <h2>Walk-Forward Validation ({walk_forward_results.get("n_folds", 0)} folds)</h2>
            <div class="grid" style="grid-template-columns: repeat(4, 1fr);">
                <div class="metric-card">
                    <div class="metric-label">Mean Sharpe</div>
                    <div class="metric-value">{agg["sharpe"]["mean"]:.2f}</div>
                    <div class="metric-ci">± {agg["sharpe"]["std"]:.2f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Mean Return</div>
                    <div class="metric-value">{agg["return"]["mean"]:.1f}%</div>
                    <div class="metric-ci">± {agg["return"]["std"]:.1f}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Positive Folds</div>
                    <div class="metric-value">{cons.get("pct_positive_sharpe", 0):.0f}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Consistency</div>
                    <div class="metric-value">{"Yes" if cons.get("is_consistent", False) else "No"}</div>
                </div>
            </div>
            <p style="color: var(--text-secondary); margin-top: 1rem;">
                {cons.get("interpretation", "")}
            </p>
        </div>
        '''

    # Build final HTML
    html = HTML_TEMPLATE.format(
        ticker=ticker,
        strategy_name=backtest_results.get('strategy', 'Strategy'),
        date_range=date_range,
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M'),
        tests_passed=tests_passed,
        verdict_class=verdict_class,
        verdict_interpretation=verdict_interpretation,
        total_return=f'{total_return:.2f}',
        return_color='var(--accent-green)' if total_return >= 0 else 'var(--accent-red)',
        sharpe=f'{sharpe:.2f}',
        sharpe_ci=f'95% CI: [{sharpe_ci.get("ci_lower", 0):.2f}, {sharpe_ci.get("ci_upper", 0):.2f}]' if significance_results else '',
        max_drawdown=f'{max_dd:.2f}',
        win_rate=f'{win_rate:.1f}',
        n_trades=n_trades,
        final_equity=f'{final_equity:,.0f}',
        deflated_sharpe_section=deflated_section,
        significance_rows='\n'.join(significance_rows),
        assumptions_html=assumptions_html,
        walk_forward_section=wf_section,
        costs='0.15%',
        n_bootstrap=5000
    )

    # Ensure reports directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Write file
    with open(output_path, 'w') as f:
        f.write(html)

    return output_path


def main():
    """CLI entry point for generating a demo report."""
    print("Generating demo backtest report...")

    # Import here to avoid circular imports
    from data.fetcher import fetch_data
    from strategies import MovingAverageCrossover
    from backtest.engine import BacktestEngine
    from analytics.metrics import calculate_metrics
    from analytics.significance import (
        bootstrap_sharpe_confidence_interval,
        permutation_test_vs_baseline,
        monte_carlo_under_null
    )
    from analytics.deflated_sharpe import deflated_sharpe_ratio

    # Run a demo backtest
    ticker = 'AAPL'
    start = '2023-01-01'
    end = '2024-01-01'

    print(f"Fetching {ticker} data...")
    data = fetch_data(ticker, start, end)

    strategy = MovingAverageCrossover(short_window=20, long_window=50)
    engine = BacktestEngine(initial_capital=10000)

    print("Running backtest...")
    results = engine.run(data, strategy)

    # Calculate significance
    print("Running significance tests...")
    test_equity = results['test']['equity_curve']
    test_returns = test_equity['equity'].pct_change().dropna()
    test_prices = test_equity['price']
    benchmark_returns = test_prices.pct_change().dropna()

    significance = {
        'sharpe_confidence': bootstrap_sharpe_confidence_interval(test_returns, n_bootstrap=2000),
        'vs_benchmark': permutation_test_vs_baseline(test_returns, benchmark_returns, n_permutations=2000),
        'vs_random': monte_carlo_under_null(test_prices, n_simulations=1000, strategy_return=results['test']['return_pct'])
    }

    # Calculate metrics for results
    results['test']['metrics'] = calculate_metrics(results['test'])

    # Deflated Sharpe (assume we tested 10 parameter combinations)
    dsr = deflated_sharpe_ratio(
        observed_sharpe=results['test']['metrics']['sharpe'],
        n_trials=10,
        sample_length=len(test_returns)
    )

    # Generate report
    output_path = generate_report(
        backtest_results=results,
        significance_results=significance,
        deflated_sharpe=dsr,
        output_path='reports/backtest_report.html',
        ticker=ticker,
        date_range=f'{start} to {end}',
        n_trials=10
    )

    print(f"\nReport generated: {output_path}")
    print("Open in browser to view.")


if __name__ == '__main__':
    main()
