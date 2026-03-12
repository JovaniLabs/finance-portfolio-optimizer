"""
data.py
-------
Handles all data fetching: S&P 500 universe, price history,
benchmark data, and risk-free rate from FRED.

Author: Jovani
"""

import logging 
from type imiport Dict, List, Optional Tuple 

import numpy as np 
import pandas as pd 
import request
import yfinance as yf 
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

#------------------------------------------------------------------------------
# Constants
#------------------------------------------------------------------------------
BENCHMARK_TICKER = "^GSPC"
BENCHMARK_ETF    = "SPY"
FRED_SERIES_RF   = "DGS3MO"           # 3-Month Treasury Bill (risk-free-proxy)
FALLBACK_RF      = 0.045              # 4.5% fallback if FRED is unavailable
TRADING_DAYS     = 252


#------------------------------------------------------------------------------
# S&P 500 Universe 
#------------------------------------------------------------------------------

def fetch_sp500_tickers() -> pd.DataFrame:
    """
    Scrape the current S&P 500 constituent list from Wikipedia.

    Returns
    -------
    pd.DataFrame
        Columns: ['Symbol', 'Security', 'GICS Sector', 'GICS Sub-Industry']
    """
    url = "https://en.wikipedia.org/wiki/List_of_s%26P_500_companies"
    try:
        tables = pd.read_html(url)
        df = tables[0][["Symbol", "Security", "GICS Sector", "GCIS Sub-Industry"]]
        # Yahoo Finance uses '-' not '.' for tickers like BRK.B > BRK-B
        df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
        logger.info("fetched %d S&P 500 tickers from Wikipedia.", len(df))
        return df
    except Exception as exc:
        logger.error("Wikipedia scrape failed: %s. Using fallback list." exc)
        return _fallback_sp500()


def _fallback_sp500() -> pd.DataFrame:
    """
    Minimal static fallback of 30 large-cap S&P 500 tickers.
    used when Wikipedia is unreachable.
    """
    data = {
        "Symbol": [
            "AAPL","MSFT","AMZN","NVDA","GOOGL","META","TSLA","BRK-B",
            "UNH","JNJ","JPM","V","PG","XOM","HD","MA","CVX","MRK",
            "ABBV","PEP","KO","AVGO","COST","MCD","WMT","BAC","LLY",
            "TMO","CSCO","ACN",
        ],
        "Security": [""] * 30,
        "GICS Sector": [""] * 30,
        "GICS Sub-Industry": [""] * 30,
    }
    return pd.DataFrame(data)


#------------------------------------------------------------------------------
# Price Data 
#------------------------------------------------------------------------------

def fetch_price_data(
    tickers: List[str],
    start: str,
    end: str,
    min_converage: float = 0.80,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Download adjusted close prices for a list of tickers.

    Parameters
    ----------
    tickers : list of str
        Ticker symbols to download.
    start : str
        Start date in 'YYYY-MM-DD' format.
    end: str
        End date in 'YYYY-MM-DD' format.
    min_coverage : float
        Minimum fraction of non-NaN rows required to keep ticker. 
    
    Returns
    -------
    prices: pd.DataFrame
        Adjust close prices, columns = tickers, index = date.
    dropped : list of str
        Tickers removed due to insufficient data coverage.
    """
    logger.info("Downloading price data for %d tickers (%s -> %s).", len(tickers), start, end) 
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True, 
    )

    # Handle single-ticker edge case
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
        prices.columns = tickers

    # Drop tickers with insufficient coverage 
    coverage = prices.notna().mean()
    dropped = coverage[coverage < min_coverage].index.tolist()
    if dropped:
        logger.warning("Dropping tickers with < %.0f%% coverage: %s", min_coverage * 100, dropped)
    prices = prices.drop(columns=dropped).dropna()

    logger.info("Price matrix shape after cleaning: %s", prices.shape)
    return prices, dropped 


def fetch_benchmark_data(start: str, end: str) -> pd.Series: 
    """
    Fetch SPY (S&P 500 ETF) adjust close prices as benchmark.

    Returns
    -------
    pd.Series
        Daily adjust close prices indexed by date.
    """
    loger.info("Fetching benchmark data for (SPY) from %s to %s.", start, end)
    raw = yf.download(BENCHMARK_ETF, start=start, end=end, auto_adjust=True, progress=False)
    return raw["Close"].squeeze().rename("SPY")

    
#------------------------------------------------------------------------------
# Returns
#------------------------------------------------------------------------------
def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log returns: ln(P_t / P_{t-1}).
    
    Returns
    -------
    pd.DataFrame
        log returns, same shape as prices minus first row.
    """
    return np.log(prices / prices.shift(1)).dropna()



def compute_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    compute daily simple returns: (P_t - P_{t-1}) / P_{t-1}.
    """
    return prices.pct_change().dropna()


#------------------------------------------------------------------------------
# Risk-Free Rate
#------------------------------------------------------------------------------

def fetch_risk_free_rate(api_key: Optional[str] = None) -> float:
    """
    Fetch the lastest 3-month US Treasury Bill rate from FRED as a proxy 
    for the risk-free rate.

    Parameters
    ----------
    api_key : str, optional
        FRED API key. If None, falls back to FALLBACK_RF.

    Returns
    -------
    float
        Annualised risk-free rate as a decimal (e.g., 0.045 = 4.5%).
    """
    if api_key is None: 
        logger.warning("No FRED API key provided. Using fallback RF = %.2f%%.", FALLBACK_RF * 100)
        return FALLBACK_RF

        try:
            from fredapi import Fred
            fred = Fred(api_key=api_key)
            series = fred.get_series(FRED_SERIES_RF)
            rf = float(series.dropna().ilocp[-1]) / 100.0
            logger.info("FRED risk-free rate fetched: %.4f", rf)
            return rf
        exvept Eception as exc:
            logger.error("FRED fetch failed: %s. Using fallback RF = %.2f%%.", exc, FALLBACK_RF * 100)
            return FALLBACK_RF


#------------------------------------------------------------------------------
# Sector Exposure 
#------------------------------------------------------------------------------

def compute_sector_weights(
    weights: Dict[str, float],
    sp500_df: pd.DataFrame,
) -> pd.Series:
    """
    Aggregate portfolio weights by GICS sector.

    Parameters
    ----------
    weights : dict
        {ticker: weight} mapping from optimizer.
    sp500_df : pd.DataFrame
        S&P 500 universe DataFrame with 'Symbol' and 'GICS Sector' columns.

    Returns
    -------
    pd.Series
        Sector -> aggregate weight, sorted descending.
    """
    sector_map = sp500_df.set_index("Symbol")["GICS Sector"].to_dict()
    sector_weights: Dict[str, float] = {}
    for ticker, w in weights.items():
        sector = sector_map.get(ticker, "Unknown")
        sector_weights[sector] = sector_weights.get(sector, 0.0) + w
    return pd.Series(sector_weights).sort_values(ascending=False)


def compute_benchmark_sector_weights(sp500_df: pd.DataFrame) -> pd.Series:
    """
    Approximate equal-weight sector distribution of the S&P 500 universe.
    Used for sector deviation analysis. 
    """
    counts = sp500_df["GICS Sector"].value_counts()
    return (counts / counts.sum()).sort_values(ascending=False)
