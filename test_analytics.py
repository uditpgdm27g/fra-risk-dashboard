"""Sanity checks: run with `python test_analytics.py`. Uses synthetic data only."""
import numpy as np
import pandas as pd
from scipy import stats

import analytics as an
from config import BENCHMARK, PORTFOLIO_A, PORTFOLIO_B
from demo_data import synthetic_prices

rng = np.random.default_rng(1)

# 1. VaR / ES on a large normal sample should match closed forms
r = pd.Series(rng.normal(0.0005, 0.01, 400_000))
for cl in (0.95, 0.99):
    hv, he, _ = an.hist_var_es(r, cl)
    nv, ne = an.normal_var_es(r, cl)
    z = stats.norm.ppf(1 - cl)
    assert abs(nv - -(0.0005 + z * 0.01)) < 2e-4, (nv,)
    assert abs(hv - nv) < 3e-4 and abs(he - ne) < 5e-4, (hv, nv, he, ne)
    assert he > hv  # ES always exceeds VaR
    assert abs(an.cornish_fisher_var(r, cl) - nv) < 5e-4  # ~normal => CF ~ normal
print("VaR/ES ok")

# 2. Max drawdown on a known path
assert abs(an.max_drawdown(pd.Series([0.1, -0.5, 0.2])) - (-0.5)) < 1e-12
# 3. Beta of 2*market is 2
m = pd.Series(rng.normal(0, 0.01, 2000))
assert abs(an.beta(2 * m, m) - 2) < 1e-9
print("drawdown / beta ok")

# 4. Optimiser: two uncorrelated assets, closed-form check
mu = pd.Series([0.10, 0.20], index=["x", "y"])
cov = pd.DataFrame(np.diag([0.04, 0.16]), index=mu.index, columns=mu.index)
gmv = an.solve_min_var(mu.values, cov.values)
assert np.allclose(gmv, [0.8, 0.2], atol=1e-4), gmv  # inverse-variance weights
res = an.optimiser_at_target(mu, cov, 0.15)
assert res["status"] == "optimal" and np.allclose(res["weights"], [0.5, 0.5], atol=1e-3)
assert an.optimiser_at_target(mu, cov, 0.12)["status"] == "gmv"  # 0.12 = gmv return exactly -> gmv
assert an.optimiser_at_target(mu, cov, 0.25)["status"] == "infeasible"
print("optimiser ok")

# 5. Full pipeline on synthetic data
px = synthetic_prices()
rets = an.returns_from_prices(px)
bench = rets[BENCHMARK]
wins = [("2008-01-01", "2009-03-09"), ("2017-01-01", "2017-12-31")]
ok = an.usable_tickers(rets.drop(columns=BENCHMARK), wins)
assert len(ok) == len(PORTFOLIO_A) + len(PORTFOLIO_B)
for name, tick in (("A", PORTFOLIO_A), ("B", PORTFOLIO_B)):
    cols = list(tick)
    w = pd.Series(1 / len(cols), index=cols)
    for s, e in wins:
        rr = an.window(rets, s, e)
        s_ = an.period_summary(an.window(rets[cols], s, e).fillna(0) @ w, an.window(bench, s, e),
                               an.window(rets[cols], s, e), 0.95, 0.065)
        assert s_["ES hist"] >= s_["VaR hist"] > 0
    mu, cov = an.era_stats(an.window(rets[cols], "2017-01-01", "2017-12-31"))
    fr = an.frontier_curve(mu, cov)
    assert fr["vol"].is_monotonic_increasing or (np.diff(fr["vol"]) > -1e-6).all()
    assert (np.diff(fr["ret"]) >= -1e-9).all()
    tgt = float(w @ mu)
    out = an.optimiser_at_target(mu, cov, tgt)
    assert out["status"] in ("optimal", "gmv")
    # optimiser must not be riskier than equal weight at >= same return
    assert out["vol"] <= np.sqrt(w.values @ cov.values @ w.values) + 1e-9
    assert abs(out["weights"].sum() - 1) < 1e-9 and (out["weights"] >= 0).all()
print("pipeline ok")
print("ALL TESTS PASSED")
