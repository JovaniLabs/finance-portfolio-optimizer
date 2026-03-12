"""
risk.py
-------
Insitutional-grade risk analytics:
    - Volatility, Beta, Correlation
    - Historical VaR & CVaR (Expected Shortfall)
    - Maximum Drawdown
    - Information Ratio & Tracking Error
    - Diversification Ratio

Author: Jovani Velasco
"""

import logging 
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS = 252
CONFIDENCE_LEVELS = [0.90, 0.95, 0.99]


#-------------------------------------------------------------------------------
# Portfolio Returns
#-------------------------------------------------------------------------------

def compute_portfolio_returns(
    returns: pd.DataFrame,
    weights: Dict[str, float],
) -> pd.Series:
    """
    Compute daily portfolio returns given asset weights.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily simple returns, columns = tickers.
        weights : dict
            {ticker: weight} -- must sum to 1.
    
    Returns
    -------
    pd.Series
        Daily portfolio returns.
    """
    w = pd.Series(weights).reindex(returns.columns).fillna(0.0)
    port_returns = returns.dot(w)
    port_returns.name = "Portfolio"
    return port_returns


#------------------------------------------------------------------------------
# Core Risk Metrics
#------------------------------------------------------------------------------

def annialised_volatility(returns: pd.Series) -> float:
    """
    Annulised volatility: σ_daily x sqrt(252).

    Returns
    -------
    float
        Annualised volatility as a decimal.
    """
    return float(returns.std() * np.sqrt(TRADING_DAYS))


    def annialised_return(returns: pd.Series) -> float:
        """
        Compound annualised return (CAGR).
        
        Returns
        -------
        float
            CAGR as a decimal. 
        """
        n_years = len(returns) / TRADING_DAYS
        cumulative = (1 + returns).prod()
        return float(cumulative ** (1 / n_years) -1)


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.045) -> float:
    """
    Annualised Sharpe Ratio: (R_p - R_f) / σ_p.

    Parameters
    ----------
    returns : pd.Series
        Daily portfolio returns.
    risk_free_rate : float
        Annualised risk-free rate.

    Returns
    -------
    float
        Sharpe ratio.
    """
    excess = annualised_return(returns) - risk_free_rate
    vol = annualised_volatility(returns)
    return float(excess / vol) if vol > 0 else 0.0


#------------------------------------------------------------------------------
# Tail Risk Var & CVaR
#------------------------------------------------------------------------------

def historical_var(
    returns: pd.Series,
    confidence: float = 0.95,
) -> float:
    """
    Historical Value at Risk (VaR) at a given confidence level.
    Represents the maximum loss not exceeded with probability `confidence`.

    Parameters
    ----------
    returns : pd.Series
        Daily portfolio returns.
    confidence : float
        Confidence level (e.g., 0.95 for 95% VaR).
    
    Returns
    -------
    float
        VaR as a positive decimal (e.g., 0.02 = 2% loss).
    """
    return float(-np.quantile(returns, 1 - confidence))


def historical_cvar(
    returns: pd.Series,
    confidence: float = 0.95,
) -> float:
    """
    Conditional value at Risk (CVaR / Expected Shortfall).
    The average loss in the worst (1 - confidence) % of scenarios.
    This is the insitutional standard for tail risk measurement.

    Parameters
    ----------
    returns: pd.Series
        Daily portfolio returns.
    confidence: float
        Confidence level (e.g., 0.95 for 95% CVaR).

    Returns
    -------
    float
        CVaR as a positive decimal.
    """
    var = hostorical_var(returns, confidence)
    tail_losses = returns[returns <= -var]
    return float(-tail_losses.mean()) if len(tail_losses) > 0 else var


    def var_cvar_table(returns: pd.Series) -> pd.DataFrame:
        """
        Compute VaR and CVaR at multiple confidence levels.

        Returns
        -------
        pd.DataFrame
            columns: ['confidence', 'VaR (Daily)', 'CVaR (Daily)',
                        'VaR (Annual)', 'CVaR (Annual)']
        """
        rows = []
        for cl in CONFIDENCE_LEVELS:
            var_d = historical_var(returns, cl)
            cvar_d = historical_cvar(returns, cl)
            rows.append({
                "Confidence":    f"{int(cl*100)}%",
                "VaR (Daily)":   var_d,
                "CVaR (Daily)":  cvar_d,
                "VaR (Annual)":  var_d * np.sqrt(TRADING_DAYS),
                "CVaR (Annual)": cvar_d * np.sqrt(TRADING_DAYS),
            })
        return pd.DataFrame(rows)


#-------------------------------------------------------------------------------
# Drawdown Analysis
#-------------------------------------------------------------------------------

def maximum_drawdown(returns: pd.Series) -> float:
    """
    Maximum Drawdown (MDD): largest peak-to-trough decline.

    Returns
    -------
    float
        MDD as a positive decimal (e.g., 0.35 = 35% drawdown).
    """
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    return float(-drawdown.min())


def drawdown_series(returns: pd.Series) -> pd.Series:
    """
    Full drawdown time series for plotting. 

    Returns
    -------
    pd.Series
        Drawdown at each date (negative values).
    """
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    return (cumulative - rolling_max) / rolling_max


#-------------------------------------------------------------------------------
# Benchmark-Relative Matrics
#-------------------------------------------------------------------------------

def beta(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """
    Portfolio beta relative to benchmark.
    β = Cov(R_p, R_m) / Var(R_m)

    Returns
    -------
    float
        Beta coefficient.
    """
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    cov_matrix = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
    return float(cov_matrix[0, 1] / cov_matrix[1, 1])


def tracking_error(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """
    Annualised Tracking Error: std(R_p - R_b) x sqrt(252).

    Returns
    -------
    float 
        Annualised tracking error as a decimal. 
    """
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    active_returns = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    return float(active_returns.std() * np.sqrt(TRADING_DAYS))


def information_ratio(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """
    Information Ratio: Active Return / Tracking Error.
    Measures skill of active managment vs benchmark. 

    Returns
    -------
    float
        Information ratio.
    """
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    avtivate_returns = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    ann_active = float(active_returns.mean() * TRADING_DAYS)
    te = tracking_error(portfolio_returns, benchmark_returns)
    return float(ann_active / te) if te > 0 else 0.0


#------------------------------------------------------------------------------
# Diversification Ratio
#------------------------------------------------------------------------------

def diversification_ratio(
    weights: Dict[str, float],
    returns: pd.DataFrame,
) -> float:
    """
    Diversification Ratio: weighted avg. volatility / portfolio volatility.
    DR > 1 indicates diversification benefit.
    DR = 1 means perectly correlated assests (no diversification).

    Returns
    -------
    float
        Diversification ration.
    """
    w = pd.Series(weights).reindex(returns_columns).fillna(0.0).values
    individual_vols = returns.std().values * np.sqrt(TRADING_DAYS)
    weighted_avg_vol = float(np.dot(w, individual_vols))

    port_ret = returns.dot(w)
    port_vol = float(port_ret.std() * np.sqrt(TRADING_DAYS))

    return float(weighted_avg_vol) if port_vol > 0 else 1.0


#------------------------------------------------------------------------------
# Full Risk Summary 
#------------------------------------------------------------------------------

def compute_risk_summary(
    porfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    weights: Dict[str, float],
    asset_returns: pd.DataFrame,
    risk_free_rate: float = 0.045,
) -> Dict[str, float]
    """
    Compute a comprehensive risk summary dictionary

    Returns
    -------
    dict
        All key risk/return metrics in one place.
    """
    return {
        "Annualised Return":     annulalise_return(portfolio_retruns),
        "Annualised Volatility": annualised_volatility(portfolio_return),
        "Sharpe Ratio":          sharpe_ratio(portfolio_returns, risk_free_rate),
        "Max Drawdown":          maximum_drawdown(portfolio_returns),
        "VaR 95% (Daily)":       historical_var(portfolio_returns, 0.95),
        "CVaR 95% (Daily)":      historical_cvar(portfolio_returns, 0.95),
        "Beta":                  beta(portfolio_returns, benchmark_returns),
        "Tracking Error":        tracking_error(portfolio_returns, benchmark_returns),
        "Information Ratio":     information_ratio(portfolio_returns, benchmark_returns),
        "Diversification Ratio": diversification_ratio(weights, asset_returns),
    }
    









