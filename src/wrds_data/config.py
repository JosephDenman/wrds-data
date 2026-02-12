"""
Configuration dataclasses for wrds-data.

All configuration is Python dataclasses with sensible defaults.
Every correction is individually toggleable. All defaults represent
the conservative, academically standard settings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

@dataclass
class WRDSConnectionConfig:
    """Credentials for live WRDS PostgreSQL access."""

    username: str = ""
    password: str = ""
    host: str = "wrds-pgdata.wharton.upenn.edu"
    port: int = 9737
    dbname: str = "wrds"

    def __post_init__(self) -> None:
        if not self.username:
            self.username = os.getenv("WRDS_USERNAME", "")
        if not self.password:
            self.password = os.getenv("WRDS_PASSWORD", "")

    @property
    def connection_string(self) -> str:
        return (
            f"postgresql+psycopg2://{self.username}:{self.password}"
            f"@{self.host}:{self.port}/{self.dbname}"
        )


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

@dataclass
class StorageConfig:
    """Where local data lives and which backend to use."""

    cache_dir: Path = field(default_factory=lambda: Path(
        os.getenv("WRDS_DATA_DIR", "data/wrds")
    ))
    backend: str = "parquet"  # "wrds" | "parquet" | "auto"
    fallback_to_wrds: bool = True  # If parquet missing, query WRDS live
    chunk_size_mb: int = 500  # Target Parquet file size for downloads


# ---------------------------------------------------------------------------
# CRSP Corrections
# ---------------------------------------------------------------------------

@dataclass
class CRSPCorrectionConfig:
    """
    Controls for CRSP data corrections.

    All defaults are the conservative, academically standard settings
    per Fama-French methodology and Shumway (1997).
    """

    # Share code filter: keep only ordinary common shares
    share_code_filter: bool = True
    share_codes: tuple[int, ...] = (10, 11)

    # Exchange code filter: major US exchanges
    exchange_code_filter: bool = True
    exchange_codes: tuple[int, ...] = (1, 2, 3)  # NYSE, AMEX, NASDAQ

    # Price sign correction: CRSP uses negative prices for bid-ask midpoints
    price_sign_correction: bool = True

    # Delisting return adjustment (Shumway 1997)
    # Critical for survivorship bias correction
    delisting_adjustment: bool = True
    delisting_return_otc: float = -0.30  # DLSTCD 400-499 (performance-related)
    delisting_return_exchange: float = -0.55  # DLSTCD 500+ (dropped by exchange)

    # Penny stock filter
    penny_stock_filter: bool = True
    penny_stock_threshold: float = 5.0  # Exclude stocks below this price

    # Minimum trading history
    min_history_filter: bool = True
    min_trading_days: int = 252  # ~1 year of trading days

    # Volume validation
    volume_validation: bool = True


# ---------------------------------------------------------------------------
# Compustat Corrections
# ---------------------------------------------------------------------------

@dataclass
class CompustatCorrectionConfig:
    """
    Controls for Compustat data corrections.

    Defaults follow Fama-French methodology for book equity calculation
    and standard academic filters.
    """

    # Standard data format filters
    standard_filter: bool = True

    # Currency filter: USD-denominated only
    currency_filter: bool = True

    # Industry exclusion
    industry_exclusion: bool = True
    excluded_sic_ranges: list[tuple[int, int]] = field(
        default_factory=lambda: [(6000, 6999)]  # Financials
    )

    # Book equity calculation (Fama-French hierarchy)
    book_equity_calculation: bool = True

    # Point-in-time alignment: prevents look-ahead bias
    point_in_time_alignment: bool = True
    pit_lag_days: int = 1  # Days after report date (rdq) data becomes public
    pit_fallback_days: int = 180  # If rdq missing, use datadate + this many days

    # Duplicate removal: keep latest restatement
    duplicate_removal: bool = True


# ---------------------------------------------------------------------------
# CCM Linking Corrections
# ---------------------------------------------------------------------------

@dataclass
class CCMCorrectionConfig:
    """Controls for CRSP-Compustat Merged linking table corrections."""

    # Link type filter: LC (confirmed), LU (unconfirmed but usable)
    link_type_filter: bool = True
    valid_link_types: tuple[str, ...] = ("LC", "LU")

    # Link date enforcement: only use links valid for the observation date
    link_date_enforcement: bool = True

    # Primary link preference: prefer P (primary) and C (primary candidate)
    primary_link_preference: bool = True
    preferred_link_prim: tuple[str, ...] = ("P", "C")


# ---------------------------------------------------------------------------
# Derived Quantities
# ---------------------------------------------------------------------------

@dataclass
class DerivedConfig:
    """Controls for computed/derived financial quantities."""

    # Market capitalization: |PRC| * SHROUT
    market_cap: bool = True

    # Book-to-market ratio: BE / ME
    book_to_market: bool = True

    # Operating profitability (Novy-Marx 2013): (REVT - COGS) / AT
    operating_profitability: bool = True

    # Investment rate: CAPX / AT
    investment_rate: bool = True


# ---------------------------------------------------------------------------
# Aggregate Correction Config
# ---------------------------------------------------------------------------

@dataclass
class CorrectionConfig:
    """All correction configurations grouped by data source."""

    crsp: CRSPCorrectionConfig = field(default_factory=CRSPCorrectionConfig)
    compustat: CompustatCorrectionConfig = field(default_factory=CompustatCorrectionConfig)
    ccm: CCMCorrectionConfig = field(default_factory=CCMCorrectionConfig)
    derived: DerivedConfig = field(default_factory=DerivedConfig)


# ---------------------------------------------------------------------------
# Universe Sampling
# ---------------------------------------------------------------------------

@dataclass
class UniverseSamplingConfig:
    """
    Configuration for universe sampling — selecting a representative subset
    of stocks for training when the full universe is too large.

    The sampler uses stratified sampling (sector × liquidity tier) to maintain
    proportional representation across sectors and market cap ranges.

    The max_symbols constraint exists because CRSP has ~8,000+ active
    securities on any given day, but training on all of them is infeasible
    (memory, compute, API limits). In tft-finance, max_symbols_per_batch
    cannot exceed 2500.
    """

    # Target number of symbols to sample
    n_symbols: int = 2000

    # Hard upper bound (tft-finance constraint: max_symbols_per_batch <= 2500)
    max_symbols: int = 2500

    # Random seed for reproducible sampling
    random_seed: int = 42

    # --- Pre-filtering (before stratification) ---

    # Minimum average price over lookback period
    min_price: float = 5.0

    # Minimum average daily dollar volume over lookback period
    min_dollar_volume: float = 500_000

    # Minimum average daily share volume
    min_share_volume: float = 50_000

    # Minimum number of trading days in lookback period
    min_trading_days: int = 20

    # --- Lookback period for computing statistics ---

    # Number of calendar days to look back for computing sampling statistics
    lookback_days: int = 63  # ~3 months of trading days

    # --- Stratification ---

    # Number of liquidity tiers (small, mid, large by dollar volume)
    n_liquidity_tiers: int = 3

    # Sector balance: ensure at least this many symbols per sector
    min_symbols_per_sector: int = 5

    # Maximum symbols per sector (None = no cap)
    max_symbols_per_sector: int | None = None

    # --- Rebalancing ---

    # How often to rebuild the sampled universe
    rebalance_frequency: str = "quarterly"  # "quarterly" | "monthly" | "annually"


# ---------------------------------------------------------------------------
# Top-Level Config
# ---------------------------------------------------------------------------

@dataclass
class WRDSDataConfig:
    """
    Top-level configuration for wrds-data library.

    Usage::

        config = WRDSDataConfig(
            connection=WRDSConnectionConfig(username="myuser"),
            storage=StorageConfig(cache_dir=Path("/data/wrds"), backend="auto"),
        )
        provider = WRDSDataProvider(config)
    """

    connection: WRDSConnectionConfig = field(default_factory=WRDSConnectionConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    corrections: CorrectionConfig = field(default_factory=CorrectionConfig)
