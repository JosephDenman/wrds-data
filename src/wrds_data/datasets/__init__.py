"""Dataset definitions for WRDS tables."""

from wrds_data.datasets.base import DatasetDefinition
from wrds_data.datasets.ccm import CCM_LINK
from wrds_data.datasets.compustat import COMPUSTAT_ANNUAL, COMPUSTAT_QUARTERLY
from wrds_data.datasets.crsp import (
    CRSP_DAILY,
    CRSP_DELISTING,
    CRSP_MONTHLY,
    CRSP_NAMES,
)
from wrds_data.datasets.registry import DatasetRegistry

__all__ = [
    "DatasetDefinition",
    "DatasetRegistry",
    "CRSP_DAILY",
    "CRSP_MONTHLY",
    "CRSP_DELISTING",
    "CRSP_NAMES",
    "COMPUSTAT_ANNUAL",
    "COMPUSTAT_QUARTERLY",
    "CCM_LINK",
]
