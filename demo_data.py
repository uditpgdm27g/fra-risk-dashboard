"""SYNTHETIC price generator, used ONLY to test the pipeline when real data is unavailable.

These are NOT real prices. The dashboard shows a loud banner whenever this is used.
"""
import numpy as np
import pandas as pd

from config import BENCHMARK, PORTFOLIO_A, PORTFOLIO_B


def synthetic_prices(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2007-01-01", "2026-10-09")
    n = len(idx)
    # market volatility regimes
    mkt_vol = np.full(n, 0.009)
    for s, e, v in [("2008-01-01", "2009-03-09", 0.026), ("2020-02-20", "2020-05-30", 0.027)]:
        mkt_vol[(idx >= s) & (idx <= e)] = v
    mkt_vol[(idx >= "2017-01-01") & (idx <= "2017-12-31")] = 0.0045
    mkt_drift = np.full(n, 0.0004)
    mkt_drift[(idx >= "2008-01-01") & (idx <= "2009-03-09")] = -0.0035
    mkt_drift[(idx >= "2020-02-20") & (idx <= "2020-03-23")] = -0.012
    mkt = mkt_drift + mkt_vol * rng.standard_normal(n)

    data = {BENCHMARK: 1000 * np.cumprod(1 + mkt)}
    for tickers, beta_rng, idio in [(PORTFOLIO_A, (1.1, 1.6), 0.016), (PORTFOLIO_B, (0.35, 0.8), 0.008)]:
        for t in tickers:
            b = rng.uniform(*beta_rng)
            r = 0.0002 + b * mkt + idio * rng.standard_normal(n)
            data[t] = 100 * np.cumprod(1 + r)
    return pd.DataFrame(data, index=idx)
