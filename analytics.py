"""Risk analytics: VaR / Expected Shortfall, risk-label metrics and the Markowitz frontier.

Pure functions (no Streamlit) so everything can be unit-tested and reused in a notebook.
Conventions: simple daily returns, fixed weights rebalanced daily, 252 trading days/year.
VaR and ES are returned as positive loss fractions of portfolio value.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats

TD = 252


# ----------------------------------------------------------------------------- data helpers
def returns_from_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns; short gaps (<=3 days) are forward-filled first."""
    return prices.sort_index().ffill(limit=3).pct_change(fill_method=None).iloc[1:]


def window(df: pd.DataFrame | pd.Series, start, end):
    return df.loc[pd.Timestamp(start): pd.Timestamp(end)]


def usable_tickers(rets: pd.DataFrame, windows, min_cov: float = 0.97) -> list[str]:
    """Tickers with (almost) complete data in every requested window."""
    ok = list(rets.columns)
    for s, e in windows:
        w = window(rets, s, e)
        if len(w) == 0:
            return []
        cov = w.notna().mean()
        ok = [t for t in ok if cov[t] >= min_cov]
    return ok


def normalise(weights: pd.Series) -> pd.Series:
    w = weights.clip(lower=0).astype(float)
    if w.sum() <= 0:
        w = pd.Series(1.0, index=w.index)
    return w / w.sum()


# ----------------------------------------------------------------------------- basic metrics
def max_drawdown(r: pd.Series) -> float:
    wealth = (1 + r).cumprod()
    return float((wealth / wealth.cummax() - 1).min())


def beta(r: pd.Series, m: pd.Series) -> float:
    d = pd.concat([r, m], axis=1).dropna()
    if len(d) < 10:
        return float("nan")
    c = np.cov(d.iloc[:, 0], d.iloc[:, 1])
    return float(c[0, 1] / c[1, 1])


def avg_pairwise_corr(rets: pd.DataFrame) -> float:
    n = rets.shape[1]
    if n < 2:
        return float("nan")
    c = rets.corr().values
    return float((c.sum() - n) / (n * (n - 1)))


def ann_vol(r: pd.Series) -> float:
    return float(r.std(ddof=1) * np.sqrt(TD))


def ann_ret(r: pd.Series) -> float:
    return float(r.mean() * TD)


def stock_risk_table(rets: pd.DataFrame, bench: pd.Series) -> pd.DataFrame:
    """Per-stock label metrics over whatever window `rets` covers."""
    rows = {}
    for t in rets.columns:
        r = rets[t].dropna()
        rows[t] = {
            "Ann. volatility": ann_vol(r),
            "Beta vs Nifty": beta(r, bench),
            "Max drawdown": max_drawdown(r),
            "Ann. return": ann_ret(r),
        }
    return pd.DataFrame(rows).T


# ----------------------------------------------------------------------------- VaR / ES
def hist_var_es(r: pd.Series, cl: float) -> tuple[float, float, int]:
    """Historical VaR and ES. Also returns the number of observations in the tail."""
    q = np.quantile(r, 1 - cl)
    tail = r[r <= q]
    return float(-q), float(-tail.mean()), int(len(tail))


def normal_var_es(r: pd.Series, cl: float) -> tuple[float, float]:
    mu, sd = r.mean(), r.std(ddof=1)
    z = stats.norm.ppf(1 - cl)
    return float(-(mu + z * sd)), float(-(mu - sd * stats.norm.pdf(z) / (1 - cl)))


def cornish_fisher_var(r: pd.Series, cl: float) -> float:
    """Modified VaR that adjusts the normal quantile for skewness and excess kurtosis."""
    mu, sd = r.mean(), r.std(ddof=1)
    s, k = stats.skew(r), stats.kurtosis(r)
    z = stats.norm.ppf(1 - cl)
    zcf = z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24 - (2 * z**3 - 5 * z) * s**2 / 36
    return float(-(mu + zcf * sd))


def breach_rate(r: pd.Series, var_level: float) -> float:
    """Share of days whose loss exceeded a previously estimated 1-day VaR."""
    return float((r < -var_level).mean())


def period_summary(port: pd.Series, bench: pd.Series, assets: pd.DataFrame, cl: float, rf: float) -> dict:
    """Everything Test 1 needs for one portfolio in one window (1-day horizon, fractions)."""
    h_var, h_es, n_tail = hist_var_es(port, cl)
    n_var, n_es = normal_var_es(port, cl)
    vol = ann_vol(port)
    return {
        "Observations": len(port),
        "Tail obs (hist.)": n_tail,
        "Period return": float((1 + port).prod() - 1),
        "Ann. volatility": vol,
        "Beta vs Nifty": beta(port, bench),
        "Max drawdown": max_drawdown(port),
        "Worst day": float(port.min()),
        "Sharpe": (ann_ret(port) - rf) / vol if vol > 0 else float("nan"),
        "Avg pairwise corr.": avg_pairwise_corr(assets),
        "VaR hist": h_var,
        "VaR normal": n_var,
        "VaR Cornish-Fisher": cornish_fisher_var(port, cl),
        "ES hist": h_es,
        "ES normal": n_es,
        "Skew": float(stats.skew(port)),
        "Excess kurtosis": float(stats.kurtosis(port)),
    }


# ----------------------------------------------------------------------------- Markowitz
def era_stats(rets: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Annualised mean vector and covariance matrix for an era."""
    return rets.mean() * TD, rets.cov() * TD


def solve_min_var(mu: np.ndarray, cov: np.ndarray, target: float | None = None) -> np.ndarray | None:
    """Long-only min-variance weights, optionally with expected return >= target."""
    n = len(mu)
    k = 100.0  # scaling for numerical conditioning
    fun = lambda w: k * float(w @ cov @ w)
    jac = lambda w: 2 * k * (cov @ w)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(n)}]
    if target is not None:
        cons.append({"type": "ineq", "fun": lambda w: float(w @ mu) - target, "jac": lambda w: mu})
    starts = [np.ones(n) / n]
    top = np.zeros(n)
    top[np.argmax(mu)] = 1.0
    starts.append(top)
    order = np.argsort(-mu)[: max(2, n // 3)]
    tilt = np.zeros(n)
    tilt[order] = 1.0 / len(order)
    starts.append(tilt)
    best = None
    for x0 in starts:
        res = optimize.minimize(fun, x0, jac=jac, bounds=[(0, 1)] * n, constraints=cons,
                                method="SLSQP", options={"maxiter": 600, "ftol": 1e-13})
        if not res.success:
            continue
        w = np.clip(res.x, 0, 1)
        w = w / w.sum()
        if target is not None and w @ mu < target - 1e-6:
            continue
        if best is None or fun(w) < best[0]:
            best = (fun(w), w)
    if best is None:
        return None
    w = best[1].copy()
    w[w < 1e-5] = 0.0
    return w / w.sum()


def frontier_curve(mu: pd.Series, cov: pd.DataFrame, n_pts: int = 35) -> pd.DataFrame:
    """Efficient frontier (long-only) from the min-variance point to the max-return asset."""
    m, c = mu.values, cov.values
    w0 = solve_min_var(m, c)
    r_lo = float(w0 @ m) if w0 is not None else float(m.min())
    r_hi = float(m.max())
    pts = []
    for t in np.linspace(r_lo, r_hi, n_pts):
        w = solve_min_var(m, c, t if t > r_lo + 1e-9 else None)
        if w is not None:
            pts.append((float(np.sqrt(w @ c @ w)), float(w @ m)))
    return pd.DataFrame(pts, columns=["vol", "ret"])


def optimiser_at_target(mu: pd.Series, cov: pd.DataFrame, target: float) -> dict:
    """What would the optimiser do for `target` return in this era?

    status: 'optimal'    -> min-variance portfolio hitting the target
            'gmv'        -> target is below the min-variance portfolio's return, so the optimiser
                            simply holds the global minimum-variance portfolio
            'infeasible' -> target exceeds the best single-stock return in this era
    """
    m, c = mu.values, cov.values
    max_ret = float(m.max())
    out = {"status": "infeasible", "weights": None, "ret": None, "vol": None, "max_ret": max_ret}
    gmv = solve_min_var(m, c)
    if gmv is None:
        return out
    gmv_ret = float(gmv @ m)
    out["gmv_ret"] = gmv_ret
    if target > max_ret + 1e-9:
        return out
    if target <= gmv_ret + 1e-9:
        w, status = gmv, "gmv"
    else:
        w = solve_min_var(m, c, target)
        status = "optimal"
        if w is None:
            return out
    out.update(status=status, weights=pd.Series(w, index=mu.index),
               ret=float(w @ m), vol=float(np.sqrt(w @ c @ w)))
    return out


def turnover(w_a: pd.Series, w_b: pd.Series) -> float:
    """Half the L1 distance: share of the portfolio that would have to be re-allocated."""
    return float(0.5 * (w_a - w_b).abs().sum())
