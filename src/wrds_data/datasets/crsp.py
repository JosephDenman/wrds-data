"""
CRSP dataset definitions.

Column names follow CRSP conventions:
    - PRC: closing price (negative = bid-ask midpoint)
    - ASKHI: highest ask or trade price (proxy for daily high)
    - BIDLO: lowest bid or trade price (proxy for daily low)
    - VOL: trading volume (in shares, not dollars)
    - RET: holding period return (includes dividends)
    - RETX: holding period return excluding dividends
    - SHROUT: shares outstanding (in thousands)
    - CFACPR: cumulative price adjustment factor (for splits)
    - CFACSHR: cumulative share adjustment factor

References:
    - CRSP Data Descriptions: https://www.crsp.org/products/documentation
    - Shumway (1997): "The Delisting Bias in CRSP Data"
    - Shumway & Warther (1999): "The Delisting Bias in CRSP's Nasdaq Data"
"""

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
)
