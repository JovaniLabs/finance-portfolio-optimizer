"""
optimize.py
-----------
Institutioanl Mean-Variance Optimization engine using:
    -Ledoit-Wolf Shrinkage Covariance (robust to estimation error)
    -CAPM / Mean Historical Expected Returns
    -Utility Function: maximize w'μ - (λ/2) w'Σw
    -Constraints: long-only, wights caps, min weights

Author: Jovani
"""

import logging
from typing import Dict, List, Optional, Tuple 

import numpy as np 
import pandas as pd 
from pypfopt_import (
    EfficientFrontier,
    expected_returns, 
    objective_functions,
    risk_models,
)
from pypfopt.exceptions import OptimizationError

logger = logging.getLogger(__name__)

#------------------------------------------------------------------------------
# Constants
#------------------------------------------------------------------------------
LAMBDA_MIN = 0.5 # Risk-aversion: very aggressive (slider = 1)
LAMBDA_MAX = 20.0 # Risk-aversion: very conservative (slider = 10)


#------------------------------------------------------------------------------
# Covariance & Expected Returns 
#------------------------------------------------------------------------------

def build_expected_returns(
    prices: pd.DataFrame,
    method: str = "capm",
    risk_free_rate: float = 0.045,
) -> pd.Series:
    """
    Estimate expected returns for each assest.

    Parameters
    ----------
    prices: pd.DataFrame
        Adjusted close prices.
    method : str
        'capm' -> CAPM-implied returns (recommended, less noisy)
        'mean' -> Mean historical returns
        'ema' -> Exponentially weighted mean (more recent data weighted higher)
    risk_free_rate : float 
        Annualised risk-free rate. 

    Returns
    -------
    pd.Series
        Expected annualised returns per asset.
    """
    if method == "capm":
        mu = expected_returns.capm_return(
            prices, 
            risk_free_rate=risk_free_rate,
            fequency=252,
        )
    elif method == "ema":
        mu = expected_returns.ema_historical_return(prices, frequency=252)
    else:
        mu = expected_returns.mean_historical_return(prices, frequency=252)

    logger.info("Expected returns computed via '%s' method.", method)
    return mu


def build_covariance_matrix(
    prices: pd.DataFrame,
    method: str = "ledoit_wolf",
) -> pd.DataFrame:
    """
    Estimate the covariance matrix using shrinkage methods.

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted close prices.
    method : str
        'ledoit_wolf' -> Ledoit-Wolf shrinkage (institutional standard)
        'oracle'      -> Oracle Approximating Shrinkage (OAS)
        'sample'      -> Sample covariance (not recommended for small samples)

    Returns
    -------
    pd.DataFrame
        Annualised covariance matrix. 
    """
    if method == "ledoit_wolf":
        S = risk_models.CovarianceShrinkage(prices, frequency=252).ledoit_wolf()
    elif method == "oracle":
        S = risk_models.CovarianceShrinkage(prices, frequency=252).oracle_approximating()
    else:
        S = risk_models.sample_cov(prices, frequency=252)

    logger.info("Covariance matrix built via '%s' method. Shape: %s", method, S.shape)
    return S 


#------------------------------------------------------------------------------
# Risk-Aversion Mapping
#------------------------------------------------------------------------------

def slider_to_lambda(slider_value: int) -> float:
    """
    Map a user-facing risk tolerance slider (1-10) to a risk-aversion
    coefficient (λ) for the utility function: max w'μ - (λ/2) w'Σw.

    Slider = 1 -> Aggressive -> Low λ (chase returns)
    Slider = 10 -> Conservative -> High λ (minimize risk)

    Parameters
    ----------
    slider_value : int 
        Integer from 1 (aggressive) to 10 (conservative).

    Returns
    -------
    float
        Risk-aversion coefficient λ.
    """
    # Linear interpolation in log-space for better spread
    t = (slider_value - 1) / 9.0
    lam = np.exp(np.log(LAMBDA_MIN) + t * (np.log(LAMBDA_MAX) - np.log(LAMBDA_MIN)))
    logger.debug("Slider %d -> λ = %.4f", slider_value, lam)
    return float(lam)


#------------------------------------------------------------------------------
# Core Optimizer
#------------------------------------------------------------------------------

def optimize_portfolio(
    prices: pd.DataFrame,
    risk_tolerance: int = 5,
    risk_free_rate: float = 0.045,
    max_weight: float = 0.10,
    min_weight: float = 0.01,
    expected_return_method: str = "capm",
    cov_method: str = "ledoit_wolf",
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    







