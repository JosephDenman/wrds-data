"""
Shared test fixtures — synthetic CRSP, Compustat, and CCM data.

All tests use synthetic data to avoid WRDS credentials.
The synthetic data is designed to exercise edge cases in each correction.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_crsp_daily() -> pd.DataFrame:
    """
    Synthetic CRSP daily stock data for 5 securities over 300 trading days.

    Includes edge cases:
        - PERMNO 10001: normal stock (SHRCD=11, EXCHCD=1)
        - PERMNO 10002: ADR (SHRCD=31, should be filtered)
        - PERMNO 10003: stock that delists (will have delisting event)
        - PERMNO 10004: penny stock (PRC < $5)
        - PERMNO 10005: short history (only 50 days)
        - Negative prices (bid-ask midpoints) sprinkled in
    """
    np.random.seed(42)
    rows = []
    base_date = date(2020, 1, 2)

    for permno, n_days, shrcd, exchcd, base_price in [
        (10001, 300, 11, 1, 50.0),   # Normal NYSE stock
        (10002, 300, 31, 1, 100.0),  # ADR — should be filtered by ShareCodeFilter
        (10003, 250, 11, 3, 30.0),   # NASDAQ stock that will delist
        (10004, 300, 11, 2, 3.0),    # AMEX penny stock
        (10005, 50, 11, 1, 75.0),    # Short history — filtered by MinHistoryFilter
    ]:
        price = base_price
        for day_offset in range(n_days):
            trading_date = base_date + timedelta(days=day_offset)
            # Skip weekends
            if trading_date.weekday() >= 5:
                continue

            ret = np.random.normal(0.0005, 0.02)
            price = price * (1 + ret)

            # Sprinkle some negative prices (bid-ask midpoint convention)
            prc = -price if np.random.random() < 0.05 else price

            # OPENPRC: available from 1992-06-15; simulate with slight offset from close
            openprc = price * (1 + np.random.normal(0, 0.005))

            rows.append({
                "permno": permno,
                "date": trading_date,
                "prc": prc,
                "openprc": openprc,
                "askhi": price * (1 + abs(np.random.normal(0, 0.01))),
                "bidlo": price * (1 - abs(np.random.normal(0, 0.01))),
                "vol": max(1, int(np.random.lognormal(10, 1))),
                "ret": ret,
                "retx": ret - 0.0001,  # Approx return ex-dividends
                "shrout": 50000,  # 50M shares outstanding (in thousands)
                "shrcd": shrcd,
                "exchcd": exchcd,
                "cfacpr": 1.0,
                "cfacshr": 1.0,
            })

    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_delisting() -> pd.DataFrame:
    """
    Synthetic CRSP delisting events.

    Includes:
        - PERMNO 10003: performance-related delisting (code 400, missing DLRET)
        - PERMNO 99999: exchange-dropped delisting (code 500, has DLRET)
    """
    return pd.DataFrame([
        {
            "permno": 10003,
            "dlstdt": date(2020, 9, 1),
            "dlret": np.nan,  # Missing — should be imputed as -0.30
            "dlstcd": 400,
            "nwperm": np.nan,
            "nwcomp": np.nan,
        },
        {
            "permno": 99999,
            "dlstdt": date(2020, 6, 15),
            "dlret": -0.45,  # Actual return provided
            "dlstcd": 520,
            "nwperm": np.nan,
            "nwcomp": np.nan,
        },
    ])


@pytest.fixture
def synthetic_compustat() -> pd.DataFrame:
    """
    Synthetic Compustat annual fundamentals.

    Includes:
        - GVKEY 001: normal company (all fields present)
        - GVKEY 002: financial company (SIC 6020, should be excluded)
        - GVKEY 003: missing SEQ (tests book equity fallback hierarchy)
        - GVKEY 004: non-USD company (should be filtered)
    """
    return pd.DataFrame([
        # GVKEY 001: normal company, 2019
        {
            "gvkey": "001", "datadate": date(2019, 12, 31), "fyear": 2019,
            "at": 1000.0, "lt": 400.0, "seq": 500.0, "ceq": 450.0,
            "pstk": 50.0, "pstkrv": 55.0, "pstkl": 52.0,
            "txditc": 20.0, "txdb": 15.0, "itcb": 5.0,
            "sale": 810.0, "revt": 800.0, "cogs": 500.0, "xsga": 100.0, "xint": 30.0,
            "capx": 80.0, "sic": "3571", "curcd": "USD",
            "datafmt": "STD", "popsrc": "D", "consol": "C", "indfmt": "INDL",
            "rdq": date(2020, 2, 15),
        },
        # GVKEY 001: same company, 2020
        {
            "gvkey": "001", "datadate": date(2020, 12, 31), "fyear": 2020,
            "at": 1100.0, "lt": 420.0, "seq": 550.0, "ceq": 500.0,
            "pstk": 50.0, "pstkrv": 55.0, "pstkl": 52.0,
            "txditc": 22.0, "txdb": 17.0, "itcb": 5.0,
            "sale": 910.0, "revt": 900.0, "cogs": 550.0, "xsga": 110.0, "xint": 28.0,
            "capx": 90.0, "sic": "3571", "curcd": "USD",
            "datafmt": "STD", "popsrc": "D", "consol": "C", "indfmt": "INDL",
            "rdq": date(2021, 2, 20),
        },
        # GVKEY 002: financial company
        {
            "gvkey": "002", "datadate": date(2019, 12, 31), "fyear": 2019,
            "at": 5000.0, "lt": 4500.0, "seq": 400.0, "ceq": 350.0,
            "pstk": 50.0, "pstkrv": 55.0, "pstkl": 52.0,
            "txditc": 10.0, "txdb": 8.0, "itcb": 2.0,
            "sale": 210.0, "revt": 200.0, "cogs": 50.0, "xsga": 40.0, "xint": 100.0,
            "capx": 20.0, "sic": "6020", "curcd": "USD",
            "datafmt": "STD", "popsrc": "D", "consol": "C", "indfmt": "INDL",
            "rdq": date(2020, 3, 1),
        },
        # GVKEY 003: missing SEQ and TXDITC (tests fallback hierarchies)
        {
            "gvkey": "003", "datadate": date(2019, 12, 31), "fyear": 2019,
            "at": 800.0, "lt": 300.0, "seq": np.nan, "ceq": 420.0,
            "pstk": 30.0, "pstkrv": np.nan, "pstkl": 35.0,
            "txditc": np.nan, "txdb": 10.0, "itcb": 5.0,
            "sale": 610.0, "revt": 600.0, "cogs": 400.0, "xsga": 80.0, "xint": 20.0,
            "capx": 60.0, "sic": "2800", "curcd": "USD",
            "datafmt": "STD", "popsrc": "D", "consol": "C", "indfmt": "INDL",
            "rdq": date(2020, 2, 28),
        },
        # GVKEY 004: non-USD
        {
            "gvkey": "004", "datadate": date(2019, 12, 31), "fyear": 2019,
            "at": 500.0, "lt": 200.0, "seq": 250.0, "ceq": 230.0,
            "pstk": 20.0, "pstkrv": 25.0, "pstkl": 22.0,
            "txditc": 5.0, "txdb": 3.0, "itcb": 2.0,
            "sale": 310.0, "revt": 300.0, "cogs": 200.0, "xsga": 30.0, "xint": 10.0,
            "capx": 40.0, "sic": "2000", "curcd": "GBP",
            "datafmt": "STD", "popsrc": "D", "consol": "C", "indfmt": "INDL",
            "rdq": date(2020, 3, 15),
        },
    ])


@pytest.fixture
def synthetic_ccm() -> pd.DataFrame:
    """
    Synthetic CCM linking table.

    Maps:
        - GVKEY 001 → PERMNO 10001 (valid link, LC, primary)
        - GVKEY 002 → PERMNO 10002 (valid link, LC, primary)
        - GVKEY 003 → PERMNO 10003 (valid link, LU, primary candidate)
        - GVKEY 999 → PERMNO 10001 (duplicate link, LD, should be filtered)
    """
    return pd.DataFrame([
        {
            "gvkey": "001", "lpermno": 10001, "linktype": "LC",
            "linkprim": "P", "linkdt": date(1990, 1, 1),
            "linkenddt": pd.NaT,  # Still active
        },
        {
            "gvkey": "002", "lpermno": 10002, "linktype": "LC",
            "linkprim": "P", "linkdt": date(1990, 1, 1),
            "linkenddt": pd.NaT,
        },
        {
            "gvkey": "003", "lpermno": 10003, "linktype": "LU",
            "linkprim": "C", "linkdt": date(2000, 1, 1),
            "linkenddt": date(2021, 12, 31),
        },
        {
            "gvkey": "999", "lpermno": 10001, "linktype": "LD",
            "linkprim": "N", "linkdt": date(1990, 1, 1),
            "linkenddt": date(2000, 12, 31),
        },
    ])


@pytest.fixture
def synthetic_names() -> pd.DataFrame:
    """Synthetic CRSP stock name history."""
    return pd.DataFrame([
        {
            "permno": 10001, "ticker": "AAPL", "comnam": "APPLE INC",
            "shrcd": 11, "exchcd": 1, "siccd": 3571,
            "namedt": date(1990, 1, 1), "nameendt": pd.NaT,
            "permco": 20001,
        },
        {
            "permno": 10002, "ticker": "BABA", "comnam": "ALIBABA GROUP",
            "shrcd": 31, "exchcd": 1, "siccd": 5961,
            "namedt": date(2014, 9, 19), "nameendt": pd.NaT,
            "permco": 20002,
        },
        {
            "permno": 10003, "ticker": "DLIST", "comnam": "DELISTED CORP",
            "shrcd": 11, "exchcd": 3, "siccd": 2800,
            "namedt": date(2000, 1, 1), "nameendt": date(2020, 9, 1),
            "permco": 20003,
        },
        {
            "permno": 10004, "ticker": "PENY", "comnam": "PENNY STOCK CO",
            "shrcd": 11, "exchcd": 2, "siccd": 3990,
            "namedt": date(2010, 1, 1), "nameendt": pd.NaT,
            "permco": 20004,
        },
        {
            "permno": 10005, "ticker": "SHRT", "comnam": "SHORT HISTORY INC",
            "shrcd": 11, "exchcd": 1, "siccd": 7370,
            "namedt": date(2019, 6, 1), "nameendt": pd.NaT,
            "permco": 20005,
        },
    ])
