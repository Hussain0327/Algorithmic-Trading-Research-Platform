import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

from data.fetcher import fetch_data
from strategies import (
    MovingAverageCrossover, RSIStrategy, MomentumStrategy,
    PairsTradingStrategy, SpreadMeanReversionStrategy
)
from backtest.engine import BacktestEngine
from analytics.metrics import calculate_metrics
from analytics.significance import (
    bootstrap_sharpe_confidence_interval,
    permutation_test_vs_baseline,
    monte_carlo_under_null,
    test_return_distribution
)

# Page config
st.set_page_config(
    page_title='QuantLab | Trading Research Platform',
    page_icon='📊',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Custom CSS for professional dark theme
st.markdown("""
<style>
    /* Main theme */
    .stApp {
        background: linear-gradient(180deg, #0e1117 0%, #1a1f2e 100%);
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #1e3a5f 0%, #2d5a87 100%);
        padding: 1.5rem 2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        border-left: 4px solid #00d4aa;
    }

    .main-header h1 {
        color: #ffffff;
        font-size: 2rem;
        margin: 0;
        font-weight: 700;
    }

    .main-header p {
        color: #a0aec0;
        margin: 0.5rem 0 0 0;
        font-size: 0.95rem;
    }

    /* Disclaimer box - FIXED: dark text on light background */
    .disclaimer-box {
        background-color: #fef3c7;
        border: 1px solid #f59e0b;
        border-left: 4px solid #f59e0b;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-bottom: 1.5rem;
        color: #92400e;
        font-size: 0.9rem;
    }

    .disclaimer-box strong {
        color: #78350f;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border-radius: 12px;
        padding: 1.25rem;
        border: 1px solid #475569;
        text-align: center;
    }

    .metric-card .label {
        color: #94a3b8;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }

    .metric-card .value {
        color: #ffffff;
        font-size: 1.75rem;
        font-weight: 700;
    }

    .metric-card .value.positive { color: #10b981; }
    .metric-card .value.negative { color: #ef4444; }

    /* Stats boxes */
    .stat-box {
        background: #1e293b;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #334155;
    }

    .stat-box h4 {
        color: #e2e8f0;
        margin: 0 0 0.75rem 0;
        font-size: 1rem;
        border-bottom: 1px solid #475569;
        padding-bottom: 0.5rem;
    }

    /* Significance badges */
    .sig-pass {
        background: linear-gradient(135deg, #065f46 0%, #047857 100%);
        color: #ffffff;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        font-size: 0.85rem;
    }

    .sig-fail {
        background: linear-gradient(135deg, #991b1b 0%, #dc2626 100%);
        color: #ffffff;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        font-size: 0.85rem;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1e293b;
        border-radius: 10px;
        padding: 0.5rem;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background-color: #3b82f6 !important;
        color: white !important;
    }

    /* Sidebar styling */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    }

    /* Button styling */
    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
        border: none;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border-radius: 8px;
    }

    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(90deg, #2563eb 0%, #1d4ed8 100%);
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Section headers */
    .section-header {
        color: #e2e8f0;
        font-size: 1.25rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #3b82f6;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>📊 QuantLab</h1>
    <p>Algorithmic Trading Research & Backtesting Platform</p>
</div>
""", unsafe_allow_html=True)

# Disclaimer - FIXED styling
st.markdown("""
<div class="disclaimer-box">
    <strong>Research Framework Disclaimer:</strong> This platform provides baseline strategy implementations
    for educational and research purposes. Backtest results do not guarantee future performance.
    Statistical significance tests help identify potential edges, but all results require
    out-of-sample validation before any real-world application.
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")

    # Market Data Section
    st.markdown("**📈 Market Data**")
    ticker = st.text_input('Ticker Symbol', value='AAPL', help="Enter any valid stock ticker")

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input('Start', value=datetime.now() - timedelta(days=730))
    with col2:
        end_date = st.date_input('End', value=datetime.now())

    st.markdown("---")

    # Strategy Section
    st.markdown("**🎯 Strategy Selection**")
    strategy_name = st.selectbox(
        'Strategy',
        ['MA Crossover', 'RSI', 'Momentum', 'Pairs Trading', 'Bollinger Bands'],
        help="Select a trading strategy to backtest"
    )

    # Dynamic strategy parameters
    st.markdown("**Parameters**")
    if strategy_name == 'MA Crossover':
        st.caption("Trend-following: trades moving average crossovers")
        short_window = st.slider('Short MA', 5, 50, 20)
        long_window = st.slider('Long MA', 20, 200, 50)
        strategy = MovingAverageCrossover(short_window, long_window)
    elif strategy_name == 'RSI':
        st.caption("Mean reversion: buys oversold, sells overbought")
        period = st.slider('RSI Period', 5, 30, 14)
        oversold = st.slider('Oversold', 10, 40, 30)
        overbought = st.slider('Overbought', 60, 90, 70)
        strategy = RSIStrategy(period, oversold, overbought)
    elif strategy_name == 'Momentum':
        st.caption("Trend-following: trades recent price direction")
        lookback = st.slider('Lookback', 5, 60, 20)
        strategy = MomentumStrategy(lookback)
    elif strategy_name == 'Pairs Trading':
        st.caption("Statistical arbitrage: z-score mean reversion")
        lookback = st.slider('Lookback', 10, 50, 20)
        entry_z = st.slider('Entry Z', 1.0, 3.0, 2.0)
        exit_z = st.slider('Exit Z', 0.0, 1.5, 0.5)
        strategy = PairsTradingStrategy(lookback, entry_z, exit_z)
    else:  # Bollinger Bands
        st.caption("Mean reversion: Bollinger Band breakouts")
        lookback = st.slider('Lookback', 10, 50, 20)
        num_std = st.slider('Std Devs', 1.0, 3.0, 2.0)
        strategy = SpreadMeanReversionStrategy(lookback, num_std)

    st.markdown("---")

    # Backtest Settings
    st.markdown("**💰 Backtest Settings**")
    initial_capital = st.number_input('Initial Capital ($)', value=10000, step=1000)
    train_split = st.slider('Train/Test Split', 0.5, 0.9, 0.7)

    st.markdown("---")

    # Statistical Testing
    st.markdown("**📊 Statistical Testing**")
    run_significance = st.checkbox('Run Significance Tests', value=True)
    if run_significance:
        n_bootstrap = st.slider('Samples', 1000, 10000, 5000)
    else:
        n_bootstrap = 5000

    st.markdown("---")

    # Run Button
    run_backtest = st.button('🚀 Run Backtest', type='primary', use_container_width=True)

# Main Content Area
if run_backtest:
    # Fetch data
    with st.spinner('📡 Fetching market data...'):
        data = fetch_data(ticker, str(start_date), str(end_date))

    if data is None or len(data) == 0:
        st.error('❌ No data found for this ticker/date range. Please check the ticker symbol.')
    else:
        # Run backtest
        with st.spinner('⚡ Running backtest simulation...'):
            engine = BacktestEngine(initial_capital=initial_capital)
            results = engine.run(data, strategy, train_pct=train_split)

            train_metrics = calculate_metrics(results['train'])
            test_metrics = calculate_metrics(results['test'])

        # Key Metrics Summary Bar
        st.markdown('<div class="section-header">📈 Performance Summary</div>', unsafe_allow_html=True)

        # Top metrics row
        m1, m2, m3, m4, m5, m6 = st.columns(6)

        test_return = test_metrics.get('total_return', 0)
        test_sharpe = test_metrics.get('sharpe', 0)
        test_drawdown = test_metrics.get('max_drawdown', 0)
        test_winrate = test_metrics.get('win_rate', 0)
        test_trades = test_metrics.get('num_trades', 0)
        final_equity = test_metrics.get('final_equity', initial_capital)

        with m1:
            delta_color = "normal" if test_return >= 0 else "inverse"
            st.metric("Test Return", f"{test_return:.2f}%", delta=f"{test_return:.2f}%", delta_color=delta_color)
        with m2:
            st.metric("Sharpe Ratio", f"{test_sharpe:.2f}")
        with m3:
            st.metric("Max Drawdown", f"{test_drawdown:.2f}%")
        with m4:
            st.metric("Win Rate", f"{test_winrate:.1f}%")
        with m5:
            st.metric("Trades", f"{test_trades}")
        with m6:
            profit = final_equity - initial_capital
            st.metric("Final Equity", f"${final_equity:,.0f}", delta=f"${profit:,.0f}")

        st.markdown("---")

        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Charts", "📈 Analysis", "📋 Trade Log", "🔬 Statistics"])

        with tab1:
            # Chart type selection
            chart_col1, chart_col2 = st.columns([3, 1])
            with chart_col2:
                chart_type = st.radio("Chart Type", ["Line", "Candlestick"], horizontal=True)

            # Main chart
            test_equity = results['test']['equity_curve']
            trades = results['test']['trades']

            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                row_heights=[0.5, 0.3, 0.2],
                subplot_titles=('Portfolio Equity vs Benchmark', f'{ticker} Price & Signals', 'Drawdown'),
                vertical_spacing=0.08
            )

            # Row 1: Equity curve
            fig.add_trace(go.Scatter(
                x=test_equity['date'],
                y=test_equity['equity'],
                name='Strategy',
                line=dict(color='#3b82f6', width=2),
                fill='tozeroy',
                fillcolor='rgba(59, 130, 246, 0.1)'
            ), row=1, col=1)

            # Buy and hold comparison
            initial_shares = initial_capital / test_equity['price'].iloc[0]
            buy_hold_equity = initial_shares * test_equity['price']
            fig.add_trace(go.Scatter(
                x=test_equity['date'],
                y=buy_hold_equity,
                name='Buy & Hold',
                line=dict(color='#6b7280', width=2, dash='dash')
            ), row=1, col=1)

            # Row 2: Price chart
            if chart_type == "Candlestick" and 'open' in data.columns:
                # Filter data to test period
                test_start = test_equity['date'].iloc[0]
                test_data = data[data.index >= test_start].copy()

                fig.add_trace(go.Candlestick(
                    x=test_data.index,
                    open=test_data['open'],
                    high=test_data['high'],
                    low=test_data['low'],
                    close=test_data['close'],
                    name='OHLC',
                    increasing_line_color='#10b981',
                    decreasing_line_color='#ef4444'
                ), row=2, col=1)
            else:
                fig.add_trace(go.Scatter(
                    x=test_equity['date'],
                    y=test_equity['price'],
                    name='Price',
                    line=dict(color='#e2e8f0', width=1.5)
                ), row=2, col=1)

            # Trade markers
            buys = [t for t in trades if t['type'] == 'buy']
            sells = [t for t in trades if t['type'] == 'sell']

            if buys:
                fig.add_trace(go.Scatter(
                    x=[t['date'] for t in buys],
                    y=[t['price'] for t in buys],
                    mode='markers',
                    marker=dict(symbol='triangle-up', size=14, color='#10b981', line=dict(width=1, color='white')),
                    name='Buy'
                ), row=2, col=1)

            if sells:
                fig.add_trace(go.Scatter(
                    x=[t['date'] for t in sells],
                    y=[t['price'] for t in sells],
                    mode='markers',
                    marker=dict(symbol='triangle-down', size=14, color='#ef4444', line=dict(width=1, color='white')),
                    name='Sell'
                ), row=2, col=1)

            # Row 3: Drawdown chart
            equity = test_equity['equity']
            rolling_max = equity.expanding().max()
            drawdown = (equity - rolling_max) / rolling_max * 100

            fig.add_trace(go.Scatter(
                x=test_equity['date'],
                y=drawdown,
                name='Drawdown',
                fill='tozeroy',
                fillcolor='rgba(239, 68, 68, 0.3)',
                line=dict(color='#ef4444', width=1)
            ), row=3, col=1)

            # Layout
            fig.update_layout(
                height=800,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                plot_bgcolor='#1e293b',
                paper_bgcolor='#0e1117',
                font=dict(color='#e2e8f0'),
                xaxis_rangeslider_visible=False
            )

            fig.update_xaxes(gridcolor='#334155', showgrid=True)
            fig.update_yaxes(gridcolor='#334155', showgrid=True)
            fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
            fig.update_yaxes(title_text="Price ($)", row=2, col=1)
            fig.update_yaxes(title_text="DD %", row=3, col=1)

            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            # Detailed analysis
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Training Period Performance")
                st.markdown(f"""
                | Metric | Value |
                |--------|-------|
                | Total Return | **{train_metrics.get('total_return', 0):.2f}%** |
                | Sharpe Ratio | {train_metrics.get('sharpe', 0):.2f} |
                | Max Drawdown | {train_metrics.get('max_drawdown', 0):.2f}% |
                | Win Rate | {train_metrics.get('win_rate', 0):.1f}% |
                | Number of Trades | {train_metrics.get('num_trades', 0)} |
                | Final Equity | ${train_metrics.get('final_equity', 0):,.2f} |
                """)

            with col2:
                st.markdown("#### Test Period Performance (Out-of-Sample)")
                st.markdown(f"""
                | Metric | Value |
                |--------|-------|
                | Total Return | **{test_metrics.get('total_return', 0):.2f}%** |
                | Sharpe Ratio | {test_metrics.get('sharpe', 0):.2f} |
                | Max Drawdown | {test_metrics.get('max_drawdown', 0):.2f}% |
                | Win Rate | {test_metrics.get('win_rate', 0):.1f}% |
                | Number of Trades | {test_metrics.get('num_trades', 0)} |
                | Final Equity | ${test_metrics.get('final_equity', 0):,.2f} |
                """)

            # Performance comparison chart
            st.markdown("#### Train vs Test Comparison")
            comparison_fig = go.Figure()

            metrics_names = ['Return %', 'Sharpe', 'Win Rate %']
            train_vals = [train_metrics.get('total_return', 0), train_metrics.get('sharpe', 0), train_metrics.get('win_rate', 0)]
            test_vals = [test_metrics.get('total_return', 0), test_metrics.get('sharpe', 0), test_metrics.get('win_rate', 0)]

            comparison_fig.add_trace(go.Bar(name='Train', x=metrics_names, y=train_vals, marker_color='#3b82f6'))
            comparison_fig.add_trace(go.Bar(name='Test', x=metrics_names, y=test_vals, marker_color='#10b981'))

            comparison_fig.update_layout(
                barmode='group',
                height=300,
                plot_bgcolor='#1e293b',
                paper_bgcolor='#0e1117',
                font=dict(color='#e2e8f0')
            )
            st.plotly_chart(comparison_fig, use_container_width=True)

        with tab3:
            # Trade log
            if trades:
                trades_df = pd.DataFrame(trades)
                trades_df['date'] = pd.to_datetime(trades_df['date'])
                trades_df['value'] = trades_df['price'] * trades_df['shares']

                # Summary stats
                st.markdown(f"**Total Trades:** {len(trades)} | **Buys:** {len(buys)} | **Sells:** {len(sells)}")

                # Format the dataframe
                display_df = trades_df.copy()
                display_df['price'] = display_df['price'].apply(lambda x: f"${x:.2f}")
                display_df['value'] = display_df['value'].apply(lambda x: f"${x:,.2f}")
                display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')

                st.dataframe(
                    display_df,
                    use_container_width=True,
                    column_config={
                        "type": st.column_config.TextColumn("Type", width="small"),
                        "date": st.column_config.TextColumn("Date", width="medium"),
                        "price": st.column_config.TextColumn("Price", width="small"),
                        "shares": st.column_config.NumberColumn("Shares", width="small"),
                        "value": st.column_config.TextColumn("Value", width="medium"),
                    }
                )
            else:
                st.info('No trades executed in test period')

        with tab4:
            # Statistical significance testing
            if run_significance:
                with st.spinner('🔬 Running statistical significance tests...'):
                    test_equity = results['test']['equity_curve']
                    test_returns = test_equity['equity'].pct_change().dropna()
                    test_prices = test_equity['price']
                    benchmark_returns = test_prices.pct_change().dropna()

                    sharpe_ci = bootstrap_sharpe_confidence_interval(test_returns, n_bootstrap=n_bootstrap)
                    perm_test = permutation_test_vs_baseline(test_returns, benchmark_returns, n_permutations=n_bootstrap)
                    mc_test = monte_carlo_under_null(test_prices, n_simulations=min(n_bootstrap, 2000), strategy_return=results['test']['return_pct'])
                    dist_test = test_return_distribution(test_returns)

                # Results
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### 1. Sharpe Ratio Bootstrap CI")
                    sharpe_sig = not sharpe_ci.get('ci_includes_zero', True)

                    st.markdown(f"""
                    - **Point Estimate:** {sharpe_ci['sharpe']:.3f}
                    - **95% CI:** [{sharpe_ci['ci_lower']:.3f}, {sharpe_ci['ci_upper']:.3f}]
                    - **Std Error:** {sharpe_ci['std_error']:.3f}
                    """)

                    if sharpe_sig:
                        st.success("✅ PASS - CI excludes zero")
                    else:
                        st.error("❌ FAIL - CI includes zero")

                    st.markdown("---")

                    st.markdown("#### 2. Permutation Test vs Buy-and-Hold")
                    bench_sig = perm_test.get('significant_at_05', False)

                    st.markdown(f"""
                    - **Observed Diff:** {perm_test['observed_diff']:.4f}
                    - **p-value:** {perm_test['p_value']:.4f}
                    """)

                    if bench_sig:
                        st.success("✅ PASS - Beats benchmark (p < 0.05)")
                    else:
                        st.error("❌ FAIL - No significant difference")

                with col2:
                    st.markdown("#### 3. Monte Carlo vs Random Trading")
                    random_sig = mc_test.get('significant_at_05', False)

                    st.markdown(f"""
                    - **Strategy Return:** {mc_test.get('strategy_return', 0):.2f}%
                    - **Random Mean:** {mc_test['null_mean']:.2f}%
                    - **Percentile Rank:** {mc_test.get('percentile_rank', 50):.1f}%
                    - **p-value:** {mc_test.get('p_value', 1):.4f}
                    """)

                    if random_sig:
                        st.success("✅ PASS - Beats random trading")
                    else:
                        st.error("❌ FAIL - Not better than random")

                    st.markdown("---")

                    st.markdown("#### 4. Return Distribution")
                    st.markdown(f"""
                    - **Daily Mean:** {dist_test.get('mean_daily', 0)*100:.4f}%
                    - **Daily Std:** {dist_test.get('std_daily', 0)*100:.4f}%
                    - **Skewness:** {dist_test.get('skewness', 0):.3f}
                    - **Kurtosis:** {dist_test.get('excess_kurtosis', 0):.3f}
                    - **95% VaR:** {dist_test.get('var_95', 0)*100:.2f}%
                    """)

                    if dist_test.get('is_fat_tailed', False):
                        st.warning("⚠️ Fat tails detected")
                    if dist_test.get('is_negatively_skewed', False):
                        st.warning("⚠️ Negative skew detected")

                # Overall verdict
                st.markdown("---")
                st.markdown("### 🎯 Overall Statistical Verdict")

                tests_passed = sum([sharpe_sig, bench_sig, random_sig])

                verdict_col1, verdict_col2 = st.columns([1, 3])

                with verdict_col1:
                    # Visual score
                    score_color = "#10b981" if tests_passed == 3 else "#f59e0b" if tests_passed == 2 else "#ef4444"
                    st.markdown(f"""
                    <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #1e293b 0%, #334155 100%); border-radius: 12px; border: 2px solid {score_color};">
                        <div style="font-size: 3rem; font-weight: 700; color: {score_color};">{tests_passed}/3</div>
                        <div style="color: #94a3b8; font-size: 0.9rem;">Tests Passed</div>
                    </div>
                    """, unsafe_allow_html=True)

                with verdict_col2:
                    if tests_passed == 3:
                        st.success("""
                        **Strong Evidence of Edge** - Strategy passes all significance tests.
                        This warrants further investigation with walk-forward analysis and paper trading.
                        """)
                    elif tests_passed == 2:
                        st.warning("""
                        **Moderate Evidence** - Strategy passes 2/3 tests. Results are promising
                        but not conclusive. Consider parameter sensitivity analysis.
                        """)
                    elif tests_passed == 1:
                        st.error("""
                        **Weak Evidence** - Strategy passes only 1/3 tests. Performance is likely
                        due to noise or overfitting.
                        """)
                    else:
                        st.error("""
                        **No Statistical Evidence** - Strategy fails all tests. Returns are
                        indistinguishable from random chance.
                        """)
            else:
                st.info("Enable 'Run Significance Tests' in the sidebar to see statistical analysis.")

        # Strategy info expander
        with st.expander("📝 Strategy Parameters & Configuration"):
            st.json(results['params'])

else:
    # Landing state - show instructions
    st.markdown("""
    <div style="text-align: center; padding: 3rem; background: linear-gradient(135deg, #1e293b 0%, #334155 100%); border-radius: 12px; margin-top: 2rem;">
        <h2 style="color: #e2e8f0; margin-bottom: 1rem;">Welcome to QuantLab</h2>
        <p style="color: #94a3b8; font-size: 1.1rem; max-width: 600px; margin: 0 auto;">
            Configure your backtest parameters in the sidebar, then click <strong>Run Backtest</strong> to analyze your trading strategy.
        </p>
        <div style="margin-top: 2rem; display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
            <div style="text-align: center;">
                <div style="font-size: 2rem;">📈</div>
                <div style="color: #94a3b8; font-size: 0.9rem;">5 Strategies</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 2rem;">🔬</div>
                <div style="color: #94a3b8; font-size: 0.9rem;">Statistical Testing</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 2rem;">📊</div>
                <div style="color: #94a3b8; font-size: 0.9rem;">Interactive Charts</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 2rem;">✅</div>
                <div style="color: #94a3b8; font-size: 0.9rem;">Train/Test Validation</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
