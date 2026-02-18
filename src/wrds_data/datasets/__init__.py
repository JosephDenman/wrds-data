"""Dataset definitions for WRDS tables."""

from wrds_data.datasets.base import DatasetDefinition
from wrds_data.datasets.ccm import CCM_LINK
from wrds_data.datasets.compustat import COMPUSTAT_ANNUAL, COMPUSTAT_QUARTERLY
from wrds_data.datasets.crsp import (
    CRSP_DAILY,
    CRSP_DAILY_V2,
    CRSP_DELISTING,
    CRSP_MONTHLY,
    CRSP_MONTHLY_V2,
    CRSP_NAMES,
    get_crsp_daily,
    get_crsp_monthly,
)
from wrds_data.datasets.registry import DatasetRegistry

__all__ = [
    "DatasetDefinition",
    "DatasetRegistry",
    # V1 (legacy) definitions
    "CRSP_DAILY",
    "CRSP_MONTHLY",
    "CRSP_DELISTING",
    "CRSP_NAMES",
    # V2 (CIZ) definitions
    "CRSP_DAILY_V2",
    "CRSP_MONTHLY_V2",
    # Version-selection helpers
    "get_crsp_daily",
    "get_crsp_monthly",
    # Compustat & CCM (version-independent)
    "COMPUSTAT_ANNUAL",
    "COMPUSTAT_QUARTERLY",
    "CCM_LINK",
]
