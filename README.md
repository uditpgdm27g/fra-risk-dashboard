# Did the risk label survive the crash?

FRA Group Project 1: two ₹15 lakh portfolios of Indian equities (A = high risk, B = low risk), stress-tested with
VaR and Expected Shortfall through a crisis and a calm period, then rebuilt with the Markowitz efficient frontier
using each era's data.

## Run it (3 commands)

```bash
pip install -r requirements.txt
python fetch_data.py          # downloads real prices from Yahoo -> data/prices.csv
streamlit run app.py
```

`python test_analytics.py` runs the maths checks (VaR/ES vs closed forms, optimiser vs an analytic 2-asset solution).

## Make it a shareable link (free)

1. Create a GitHub repo and push this folder **including `data/prices.csv`** (so the hosted app never depends on Yahoo).
2. Go to https://share.streamlit.io, sign in with GitHub, choose the repo, main file `app.py`, Deploy.
3. Put the resulting `https://<name>.streamlit.app` link on your resume and in the slides.

## Before you present: check these

- Open the **Portfolios** tab and read the *Data coverage* table. If `fetch_data.py` warns that a ticker has no data
  (corporate actions such as demergers can change Yahoo tickers), swap it in `config.py` and re-run.
- Each portfolio needs at least 10 stocks after dropping; there are 12 each so a couple can fail safely.
- Change the rupee amount in the sidebar or `INVESTMENT` in `config.py`.
- Replace the default stock lists and the periods in `config.py` with your own if your group chose differently, and be ready to justify them.

## What each file does

| File | Purpose |
|---|---|
| `config.py` | Stock lists, sectors, periods, ₹ amount, risk-free rate |
| `analytics.py` | VaR/ES (historical, normal, Cornish-Fisher), beta, drawdown, frontier optimiser |
| `fetch_data.py` | Pulls adjusted prices into `data/prices.csv` |
| `app.py` | The dashboard |
| `demo_data.py` | Synthetic prices used only to test the code (never presented as real) |

## 15-minute talk track

1. **Setup (2 min)** the question, ₹15L each, 12 stocks per portfolio, no overlap.
2. **Why A is high risk (2 min)** the label table: volatility, beta, drawdown, sectors.
3. **Test 1 (4 min)** crisis vs calm: ES in ₹, the A/B ratio, B vs Nifty, correlation, calm-VaR breach rate.
4. **Test 2 (4 min)** target return, frontier per era, weight heatmap, unreachable targets.
5. **Dashboard demo + takeaways (3 min)** move the confidence slider and period selectors live.

Be ready for: why these periods, why historical VaR over normal, why equal weights, what survivorship bias does to the results.
