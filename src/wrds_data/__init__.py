"""
wrds-data: WRDS financial data library with academic-quality corrections.

Core classes:
    WRDSDataProvider — main API for fetching corrected CRSP/Compustat data
    WRDSDataConfig — configuration (connection, storage, corrections)
    WRDSDataSource — tft-finance adapter (IDataSource drop-in)

Quick start::

    from wrds_data import WRDSDataProvider, WRDSDataConfig
    from datetime import date

    config = WRDSDataConfig()
    provider = WRDSDataProvider(config)

    # Fetch corrected daily prices
    prices = provider.daily_prices(ticker="AAPL", start=date(2010, 1, 1))

    # Full CRSP+Compustat merge
    merged = provider.merged(start=date(2000, 1, 1), end=date(2023, 12, 31))

    # Use as tft-finance data source
    from wrds_data import WRDSDataSource
    adapter = WRDSDataSource(provider)
    # adapter.get_data("AAPL", start_date, end_date)
"""

from wrds_data.adapter import WRDSDataSource
from wrds_data.config import (
    CCMCorrectionConfig,
    CompustatCorrectionConfig,
    CorrectionConfig,
    CRSPCorrectionConfig,
    DerivedConfig,
    StorageConfig,
    UniverseSamplingConfig,
    WRDSConnectionConfig,
    WRDSDataConfig,
)
from wrds_data.provider import WRDSDataProvider
from wrds_data.sampling import WRDSUniverseSampler
from wrds_data.sectors import WRDSSectorDataSource

__version__ = "0.1.0"

__all__ = [
    # Main API
    "WRDSDataProvider",
    "WRDSDataSource",
    # Universe sampling
    "WRDSUniverseSampler",
    "UniverseSamplingConfig",
    # Sector/industry classification
    "WRDSSectorDataSource",
    # Configuration
    "WRDSDataConfig",
    "WRDSConnectionConfig",
    "StorageConfig",
    "CorrectionConfig",
    "CRSPCorrectionConfig",
    "CompustatCorrectionConfig",
    "CCMCorrectionConfig",
    "DerivedConfig",
]
