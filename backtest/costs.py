"""
Transaction Cost Models

Models the real-world costs of trading that erode backtested returns.
Many "profitable" strategies become unprofitable once realistic costs are applied.

Cost Components:
1. Commission: Broker fee per trade (increasingly zero for retail)
2. Slippage: Difference between expected and actual execution price
3. Spread: Bid-ask spread (especially important for less liquid assets)
4. Market Impact: Price movement caused by your own order (for large orders)

References:
- Kissell (2013) - The Science of Algorithmic Trading and Portfolio Management
- Almgren & Chriss (2000) - Optimal Execution of Portfolio Transactions
"""

from typing import Literal


def calculate_costs(
    trade_value: float,
    commission_pct: float = 0.001,
    slippage_pct: float = 0.0005,
    model: Literal['fixed', 'spread', 'impact'] = 'fixed',
    price: float = None,
    volume: float = None,
    avg_daily_volume: float = None
) -> float:
    """
    Calculate transaction costs for a trade.

    Args:
        trade_value: Dollar value of the trade
        commission_pct: Commission as percentage of trade value (default 0.1%)
        slippage_pct: Slippage as percentage of trade value (default 0.05%)
        model: Cost model to use
            - 'fixed': Simple percentage-based (default)
            - 'spread': Bid-ask spread estimate based on price
            - 'impact': Square-root market impact model
        price: Asset price (required for 'spread' model)
        volume: Number of shares traded (required for 'impact' model)
        avg_daily_volume: Average daily volume (required for 'impact' model)

    Returns:
        Total transaction cost in dollars

    Cost Models:
        Fixed: cost = value * (commission + slippage)
            Simple and conservative. Good for liquid large-caps.

        Spread: cost = value * spread_estimate(price)
            Estimates bid-ask spread based on price level.
            Lower-priced stocks typically have wider spreads.

        Impact: cost = value * sigma * sqrt(volume / ADV)
            Square-root market impact model for large orders.
            Cost increases with order size relative to liquidity.
    """
    trade_value = abs(trade_value)

    if model == 'fixed':
        commission = trade_value * commission_pct
        slippage = trade_value * slippage_pct
        return commission + slippage

    elif model == 'spread':
        # Estimate spread based on price level
        # Lower-priced stocks typically have wider spreads
        spread = estimate_spread(price) if price else 0.001
        return trade_value * spread + trade_value * commission_pct

    elif model == 'impact':
        # Square-root market impact model
        # Cost = sigma * sqrt(shares / ADV)
        if volume and avg_daily_volume and avg_daily_volume > 0:
            participation = volume / avg_daily_volume
            # Assume 2% daily volatility, 0.1 impact coefficient
            impact_pct = 0.02 * 0.1 * (participation ** 0.5)
            return trade_value * impact_pct + trade_value * commission_pct
        else:
            return trade_value * (commission_pct + slippage_pct)

    else:
        return trade_value * (commission_pct + slippage_pct)


def estimate_spread(price: float) -> float:
    """
    Estimate bid-ask spread as percentage of price.

    Empirical observation: spreads are roughly inversely proportional to price
    for liquid stocks. This is a rough heuristic.

    Args:
        price: Stock price

    Returns:
        Estimated spread as decimal (e.g., 0.001 = 0.1%)

    References:
        - Stoll (2000) - Friction (Journal of Finance)
    """
    if price is None or price <= 0:
        return 0.002  # Default 0.2%

    if price > 100:
        return 0.0005  # 0.05% for high-priced liquid stocks
    elif price > 50:
        return 0.001   # 0.1%
    elif price > 20:
        return 0.0015  # 0.15%
    elif price > 10:
        return 0.002   # 0.2%
    elif price > 5:
        return 0.003   # 0.3%
    else:
        return 0.005   # 0.5% for penny stocks


def total_cost_summary(
    n_trades: int,
    avg_trade_value: float,
    model: str = 'fixed',
    commission_pct: float = 0.001,
    slippage_pct: float = 0.0005
) -> dict:
    """
    Summarize total costs over multiple trades.

    Useful for understanding how costs compound.
    """
    cost_per_trade = calculate_costs(avg_trade_value, commission_pct, slippage_pct, model)
    total_cost = cost_per_trade * n_trades
    cost_pct = total_cost / (avg_trade_value * n_trades) * 100

    return {
        'cost_per_trade': cost_per_trade,
        'total_cost': total_cost,
        'cost_pct_of_capital': cost_pct,
        'model_used': model
    }
