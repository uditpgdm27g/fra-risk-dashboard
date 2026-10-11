"""Download adjusted close prices from Yahoo Finance and save them to data/prices.csv.

Run once on your own machine (needs internet), then commit data/prices.csv so the hosted
dashboard does not depend on Yahoo being reachable from the cloud:

    python fetch_data.py
"""
import os
import sys

import pandas as pd
import yfinance as yf

from config import BENCHMARK, DATA_START, PORTFOLIO_A, PORTFOLIO_B


def main(out: str = "data/prices.csv") -> None:
    tickers = list(PORTFOLIO_A) + list(PORTFOLIO_B) + [BENCHMARK]
    px = yf.download(tickers, start=DATA_START, auto_adjust=True, progress=False)["Close"]
    px = px.dropna(how="all")
    missing = [t for t in tickers if t not in px.columns or px[t].dropna().empty]
    if missing:
        print("WARNING: no data for", missing, "- replace these tickers in config.py")
    first = px.apply(lambda s: s.first_valid_index())
    print("First available date per ticker:\n", first.sort_values().to_string())
    os.makedirs(os.path.dirname(out), exist_ok=True)
    px.to_csv(out)
    print(f"Saved {px.shape[0]} rows x {px.shape[1]} tickers to {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
