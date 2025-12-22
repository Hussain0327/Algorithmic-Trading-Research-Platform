"""
Market Data Fetcher

Fetches historical OHLCV data from Yahoo Finance.

IMPORTANT LIMITATIONS:
1. Survivorship Bias: Yahoo Finance only includes currently listed stocks.
   Delisted companies (bankruptcies, acquisitions) are excluded, which
   inflates historical returns of any strategy that "would have" held them.

2. Adjusted Prices: We use adjusted close by default, which handles splits
   and dividends. However, adjustments are backward-looking and may change.

3. Data Quality: Free data sources have known issues with corporate actions,
   missing days, and incorrect values. Always validate with multiple sources
   for production use.

For research purposes only.
"""

import yfinance as yf
import pandas as pd
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Data quality warnings
DATA_WARNINGS = {
    'survivorship_bias': (
        "Yahoo Finance data has survivorship bias: delisted stocks are excluded. "
        "Historical backtests will overstate returns compared to real-time trading."
    ),
    'adjusted_prices': (
        "Using adjusted prices for splits/dividends. "
        "Adjustments are backward-looking and may be revised."
    ),
    'data_quality': (
        "Free data sources may have errors. "
        "Validate critical results with premium data providers."
    )
}


def fetch_data(ticker, start, end, adjusted=True, warn=False):
    """
    Fetch historical OHLCV data from Yahoo Finance.

    Args:
        ticker: Stock symbol (e.g., 'AAPL', 'MSFT')
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        adjusted: If True, use adjusted close (handles splits/dividends)
        warn: If True, print data quality warnings

    Returns:
        DataFrame with columns: date, open, high, low, close, volume

    Data Limitations:
        - Survivorship bias (delisted stocks excluded)
        - Adjusted prices may be revised
        - Free data may have errors

    Example:
        >>> data = fetch_data('AAPL', '2023-01-01', '2024-01-01')
        >>> print(data.head())
    """
    if warn:
        print("=" * 60)
        print("DATA QUALITY WARNINGS")
        print("=" * 60)
        for key, msg in DATA_WARNINGS.items():
            print(f"  - {msg}")
        print("=" * 60)

    df = yf.download(ticker, start=start, end=end, progress=False)
    df = df.reset_index()

    # Flatten multi-index columns if needed
    if isinstance(df.columns[0], tuple):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]

    # Use adjusted close if requested
    if adjusted and 'adj close' in df.columns:
        df['close'] = df['adj close']
        df = df.drop(columns=['adj close'])

    return df


def fetch_multiple(tickers, start, end):
    data = {}
    for t in tickers:
        data[t] = fetch_data(t, start, end)
    return data


# TODO: add caching so we dont hit the api every time
