"""
Compustat dataset definitions.

Column names follow Compustat conventions:
    - AT: total assets
    - LT: total liabilities
    - SEQ: stockholders' equity
    - CEQ: common equity
    - PSTK: preferred stock (par value)
    - PSTKRV: preferred stock (redemption value)
    - PSTKL: preferred stock (liquidating value)
    - TXDITC: deferred taxes and investment tax credit
    - REVT: revenue (total)
    - COGS: cost of goods sold
    - XSGA: selling, general, and administrative expense
    - XINT: interest and related expense
    - CAPX: capital expenditures
    - SIC: standard industrial classification code

Data format filters:
    - datafmt='STD': standardized format
    - popsrc='D': domestic population
    - consol='C': consolidated statements
    - indfmt='INDL': industrial format

References:
    - Fama & French (1993): book equity calculation
    - Davis, Fama, French (2000): book equity hierarchy
    - Novy-Marx (2013): operating profitability
"""

from wrds_data.datasets.base import DatasetDefinition

COMPUSTAT_ANNUAL = DatasetDefinition(
    name="compustat_annual",
    wrds_table="comp.funda",
    date_column="datadate",
    entity_column="gvkey",
    columns={
        "gvkey": "gvkey",
        "datadate": "datadate",
        "fyear": "fyear",
        # Assets & Liabilities
        "at": "at",
        "lt": "lt",
        # Equity components (Fama-French book equity hierarchy)
        "seq": "seq",
        "ceq": "ceq",
        "pstk": "pstk",
        "pstkrv": "pstkrv",
        "pstkl": "pstkl",
        "txditc": "txditc",
        # Income statement
        "revt": "revt",
        "cogs": "cogs",
        "xsga": "xsga",
        "xint": "xint",
        # Investment
        "capx": "capx",
        # Classification
        "sic": "sic",
        "curcd": "curcd",
        # Data format identifiers (for filtering)
        "datafmt": "datafmt",
        "popsrc": "popsrc",
        "consol": "consol",
        "indfmt": "indfmt",
        # Report date (for point-in-time alignment)
        "rdq": "rdq",
    },
    description="Compustat Annual Fundamentals — balance sheet, income statement, cash flow",
    default_chunk_years=10,
)

COMPUSTAT_QUARTERLY = DatasetDefinition(
    name="compustat_quarterly",
    wrds_table="comp.fundq",
    date_column="datadate",
    entity_column="gvkey",
    columns={
        "gvkey": "gvkey",
        "datadate": "datadate",
        "fyearq": "fyearq",
        "fqtr": "fqtr",
        # Assets & Liabilities
        "atq": "atq",
        "ltq": "ltq",
        # Equity
        "seqq": "seqq",
        "ceqq": "ceqq",
        # Income
        "revtq": "revtq",
        "cogsq": "cogsq",
        # Classification
        "sic": "sic",
        "curcdq": "curcdq",
        # Data format
        "datafmt": "datafmt",
        "popsrc": "popsrc",
        "consol": "consol",
        "indfmt": "indfmt",
        # Report date
        "rdq": "rdq",
    },
    description="Compustat Quarterly Fundamentals",
    default_chunk_years=10,
)
