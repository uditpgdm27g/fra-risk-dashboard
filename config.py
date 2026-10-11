"""Project configuration: stock universe, historical periods and defaults.

Every ticker below was already listed on NSE before January 2008, so it can be
used in the 2008 crisis window. (Some tickers can still be dropped at run time
if Yahoo has no clean history for a window; the app reports that.)
"""

INVESTMENT = 1_500_000  # Rs 15 lakh per portfolio. Change here or in the dashboard sidebar.
BENCHMARK = "^NSEI"  # Nifty 50
RISK_FREE = 0.065  # assumed annual risk-free rate (India 10y G-sec, approx.). Edit freely.
TODAY_YEARS = 3  # "today" window = last N years of data
DATA_START = "2007-01-01"

# ticker: (company, sector)
PORTFOLIO_A = {  # High risk: cyclicals, metals, PSU banks, infra / real estate, high-beta
    "TATASTEEL.NS": ("Tata Steel", "Metals"),
    "JSWSTEEL.NS": ("JSW Steel", "Metals"),
    "HINDALCO.NS": ("Hindalco", "Metals"),
    "VEDL.NS": ("Vedanta", "Metals & Mining"),
    "SAIL.NS": ("SAIL", "Metals (PSU)"),
    "JINDALSTEL.NS": ("Jindal Steel", "Metals"),
    "DLF.NS": ("DLF", "Real Estate"),
    "ADANIENT.NS": ("Adani Enterprises", "Conglomerate"),
    "PNB.NS": ("Punjab National Bank", "PSU Bank"),
    "BANKBARODA.NS": ("Bank of Baroda", "PSU Bank"),
    "BHEL.NS": ("BHEL", "Capital Goods (PSU)"),
    "ASHOKLEY.NS": ("Ashok Leyland", "Autos (CV)"),
}

PORTFOLIO_B = {  # Low risk: FMCG, pharma, IT, utilities, large defensives
    "HINDUNILVR.NS": ("Hindustan Unilever", "FMCG"),
    "NESTLEIND.NS": ("Nestle India", "FMCG"),
    "ITC.NS": ("ITC", "FMCG"),
    "COLPAL.NS": ("Colgate-Palmolive India", "FMCG"),
    "DABUR.NS": ("Dabur", "FMCG"),
    "BRITANNIA.NS": ("Britannia", "FMCG"),
    "ASIANPAINT.NS": ("Asian Paints", "Consumer"),
    "CIPLA.NS": ("Cipla", "Pharma"),
    "SUNPHARMA.NS": ("Sun Pharma", "Pharma"),
    "TCS.NS": ("TCS", "IT"),
    "INFY.NS": ("Infosys", "IT"),
    "POWERGRID.NS": ("Power Grid", "Utilities"),
}

# Historical windows (inclusive). The user can also add a custom one in the app.
PERIODS = {
    "GFC crash (Jan 2008 - Mar 2009)": ("2008-01-01", "2009-03-09"),
    "COVID shock (Jan - Jun 2020)": ("2020-01-01", "2020-06-30"),
    "Calm bull run (2017)": ("2017-01-01", "2017-12-31"),
}
DEFAULT_CRISIS = "GFC crash (Jan 2008 - Mar 2009)"
DEFAULT_CALM = "Calm bull run (2017)"
