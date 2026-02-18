"""
CRSP dataset definitions (v1 legacy and v2 CIZ format).

Supports both CRSP v1 (legacy SIZ format, data through Dec 2024) and
CRSP v2 (CIZ format, data from July 2022 onward, sole format after Jan 2025).

V1 column names follow legacy CRSP conventions:
    - PRC: closing price (negative = bid-ask midpoint)
    - OPENPRC: opening price (available from 1992-06-15 onward)
    - ASKHI: highest ask or trade price (proxy for daily high)
    - BIDLO: lowest bid or trade price (proxy for daily low)
    - VOL: trading volume (in shares, not dollars)
    - RET: holding period return (includes dividends)
    - RETX: holding period return excluding dividends
    - SHROUT: shares outstanding (in thousands)
    - CFACPR: cumulative price adjustment factor (for splits)
    - CFACSHR: cumulative share adjustment factor

V2 column names use the CIZ naming convention (dly* prefix for daily):
    - DLYPRC → prc, DLYOPEN → openprc, DLYHIGH → askhi, DLYLOW → bidlo
    - DLYVOL → vol, DLYRET → ret, DLYRETX → retx, SHROUT → shrout

All definitions map WRDS column names to the same canonical names,
so downstream code is version-agnostic.

Key v2 differences:
    - Delisting returns are INCLUDED in DLYRET/MTHRET (no separate table needed)
    - Share/exchange filtering via stksecurityinfohist JOIN (not SHRCD/EXCHCD)
    - Exchange codes are alphanumeric: 'N' (NYSE), 'A' (AMEX), 'Q' (NASDAQ)

References:
    - CRSP Data Descriptions: https://www.crsp.org/products/documentation
    - CRSP Cross-Reference Guide (SIZ to CIZ): https://www.crsp.org/wp-content/uploads/guides/CRSP_Cross_Reference_Guide_1.0_to_2.0.pdf
    - Shumway (1997): "The Delisting Bias in CRSP Data"
    - tidyfinance Python package CRSP v2 implementation
"""

from __future__ import annotations

from wrds_data.datasets.base import DatasetDefinition

CRSP_DAILY = DatasetDefinition(
    name="crsp_daily",
    wrds_table="crsp.dsf",
    date_column="date",
    entity_column="permno",
    columns={
        "permno": "permno",
        "date": "date",
        "prc": "prc",
        "openprc": "openprc",
        "askhi": "askhi",
        "bidlo": "bidlo",
        "vol": "vol",
        "ret": "ret",
        "retx": "retx",
        "shrout": "shrout",
        "cfacpr": "cfacpr",
        "cfacshr": "cfacshr",
    },
    description="CRSP Daily Stock File — daily prices, returns, volume, shares outstanding",
    default_chunk_years=1,
)

CRSP_MONTHLY = DatasetDefinition(
    name="crsp_monthly",
    wrds_table="crsp.msf",
    date_column="date",
    entity_column="permno",
    columns={
        "permno": "permno",
        "date": "date",
        "prc": "prc",
        "askhi": "askhi",
        "bidlo": "bidlo",
        "vol": "vol",
        "ret": "ret",
        "retx": "retx",
        "shrout": "shrout",
        "cfacpr": "cfacpr",
        "cfacshr": "cfacshr",
    },
    description="CRSP Monthly Stock File — monthly prices, returns, volume",
    default_chunk_years=5,
)

CRSP_DELISTING = DatasetDefinition(
    name="crsp_delisting",
    wrds_table="crsp.dsedelist",
    date_column="dlstdt",
    entity_column="permno",
    columns={
        "permno": "permno",
        "dlstdt": "dlstdt",
        "dlret": "dlret",
        "dlstcd": "dlstcd",
        "nwperm": "nwperm",
        "nwcomp": "nwcomp",
    },
    description=(
        "CRSP Delisting Events — delisting dates, returns, and reason codes. "
        "Critical for survivorship bias correction (Shumway 1997)."
    ),
    default_chunk_years=10,
)

CRSP_NAMES = DatasetDefinition(
    name="crsp_names",
    wrds_table="crsp.dsenames",
    date_column="namedt",
    entity_column="permno",
    columns={
        "permno": "permno",
        "ticker": "ticker",
        "comnam": "comnam",
        "shrcd": "shrcd",
        "exchcd": "exchcd",
        "siccd": "siccd",
        "namedt": "namedt",
        "nameendt": "nameendt",
        "permco": "permco",
    },
    description=(
        "CRSP Stock Name History — ticker-PERMNO mapping, share/exchange codes, "
        "SIC codes, company names with effective date ranges."
    ),
    default_chunk_years=20,
    is_reference_table=True,
)


# ===========================================================================
# V2 (CIZ Format) Definitions — data from July 2022, sole format after Jan 2025
# ===========================================================================

# SQL fragment for the stksecurityinfohist JOIN used by all v2 queries.
# Filters to: common equity, US-incorporated, active trading, major exchanges.
# This replaces the v1 ShareCodeFilter + ExchangeCodeFilter corrections AND
# includes delisting returns directly in the return series.
_SSIH_JOIN_DAILY = """
    INNER JOIN crsp.stksecurityinfohist AS ssih
        ON dsf.permno = ssih.permno
        AND ssih.secinfostartdt <= dsf.dlycaldt
        AND dsf.dlycaldt <= ssih.secinfoenddt
"""

_SSIH_FILTERS = """
    AND ssih.sharetype = 'NS'
    AND ssih.securitytype = 'EQTY'
    AND ssih.securitysubtype = 'COM'
    AND ssih.usincflg = 'Y'
    AND ssih.issuertype IN ('ACOR', 'CORP')
    AND ssih.primaryexch IN ('N', 'A', 'Q')
    AND ssih.conditionaltype IN ('RW', 'NW')
    AND ssih.tradingstatusflg = 'A'
"""

CRSP_DAILY_V2 = DatasetDefinition(
    name="crsp_daily",
    wrds_table="crsp.dsf_v2",
    date_column="date",
    entity_column="permno",
    columns={
        # Identifiers
        "permno": "permno",
        # Dates
        "date": "dlycaldt",
        # Prices
        "prc": "dlyprc",
        "openprc": "dlyopen",
        "askhi": "dlyhigh",
        "bidlo": "dlylow",
        # Volume & shares
        "vol": "dlyvol",
        "shrout": "shrout",
        # Returns (delisting returns INCLUDED in v2)
        "ret": "dlyret",
        "retx": "dlyretx",
        # Security info (from stksecurityinfohist JOIN)
        "exchcd": "primaryexch",
        "siccd": "siccd",
    },
    description=(
        "CRSP Daily Stock File v2 (CIZ format) — daily OHLCV, returns, "
        "shares outstanding. Includes delisting returns in DLYRET. "
        "Joined with stksecurityinfohist for common equity on major exchanges."
    ),
    default_chunk_years=1,
    sql_template=f"""
        SELECT dsf.permno, dsf.dlycaldt, dsf.dlyprc, dsf.dlyopen,
               dsf.dlyhigh, dsf.dlylow, dsf.dlyvol, dsf.shrout,
               dsf.dlyret, dsf.dlyretx,
               ssih.primaryexch, ssih.siccd
        FROM crsp.dsf_v2 AS dsf
        {_SSIH_JOIN_DAILY}
        WHERE dsf.dlycaldt >= :start_date AND dsf.dlycaldt <= :end_date
        {_SSIH_FILTERS}
    """,
)

_SSIH_JOIN_MONTHLY = """
    INNER JOIN crsp.stksecurityinfohist AS ssih
        ON msf.permno = ssih.permno
        AND ssih.secinfostartdt <= msf.mthcaldt
        AND msf.mthcaldt <= ssih.secinfoenddt
"""

CRSP_MONTHLY_V2 = DatasetDefinition(
    name="crsp_monthly",
    wrds_table="crsp.msf_v2",
    date_column="date",
    entity_column="permno",
    columns={
        # Identifiers
        "permno": "permno",
        # Dates
        "date": "mthcaldt",
        # Prices
        "prc": "mthprc",
        # Volume & shares
        "shrout": "shrout",
        # Returns (delisting returns INCLUDED in v2)
        "ret": "mthret",
        # Security info (from stksecurityinfohist JOIN)
        "exchcd": "primaryexch",
        "siccd": "siccd",
    },
    description=(
        "CRSP Monthly Stock File v2 (CIZ format) — monthly prices, returns, "
        "shares outstanding. Includes delisting returns in MTHRET. "
        "Joined with stksecurityinfohist for common equity on major exchanges."
    ),
    default_chunk_years=5,
    sql_template=f"""
        SELECT msf.permno, msf.mthcaldt, msf.mthret, msf.shrout,
               msf.mthprc,
               ssih.primaryexch, ssih.siccd
        FROM crsp.msf_v2 AS msf
        {_SSIH_JOIN_MONTHLY}
        WHERE msf.mthcaldt >= :start_date AND msf.mthcaldt <= :end_date
        {_SSIH_FILTERS}
    """,
)


# ===========================================================================
# Version-selection helpers
# ===========================================================================

def get_crsp_daily(version: str = "v2") -> DatasetDefinition:
    """Return CRSP daily dataset definition for the specified version."""
    if version == "v2":
        return CRSP_DAILY_V2
    elif version == "v1":
        return CRSP_DAILY
    else:
        raise ValueError(f"Unknown CRSP version: {version!r}. Use 'v1' or 'v2'.")


def get_crsp_monthly(version: str = "v2") -> DatasetDefinition:
    """Return CRSP monthly dataset definition for the specified version."""
    if version == "v2":
        return CRSP_MONTHLY_V2
    elif version == "v1":
        return CRSP_MONTHLY
    else:
        raise ValueError(f"Unknown CRSP version: {version!r}. Use 'v1' or 'v2'.")
