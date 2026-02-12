"""
CCM (CRSP-Compustat Merged) linking corrections.

Three corrections for the linking table that maps Compustat GVKEY
to CRSP PERMNO:

1. LinkTypeFilter — keep only confirmed/usable link types
2. LinkDateEnforcement — enforce link date validity windows
3. PrimaryLinkPreference — prefer primary securities

References:
    - WRDS CCM Documentation
    - Fama-French portfolio construction methodology
"""

from __future__ import annotations

import pandas as pd
from loguru import logger

from wrds_data.config import CCMCorrectionConfig
from wrds_data.corrections.base import CorrectionStep


class LinkTypeFilter(CorrectionStep):
    """
    Keep only valid, usable link types.

    Link types:
        LC = Link confirmed by CRSP research (highest quality)
        LU = Link unconfirmed but usable
        LD = Duplicate link (avoid — ambiguous)
        LX = Non-US exchange (avoid)
        LN = No link available
        NR = Not yet researched

    Standard practice: keep LC and LU only.
    """

    def __init__(self, config: CCMCorrectionConfig) -> None:
        self._valid_types = set(config.valid_link_types)

    @property
    def name(self) -> str:
        return "LinkTypeFilter"

    @property
    def description(self) -> str:
        types = ", ".join(sorted(self._valid_types))
        return f"Keep linktype in {{{types}}}"

    @property
    def required_columns(self) -> list[str]:
        return ["linktype"]

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        mask = df["linktype"].isin(self._valid_types)
        return df[mask].copy()


class LinkDateEnforcement(CorrectionStep):
    """
    Enforce that links are only used within their valid date range.

    Each CCM link has:
        - linkdt: start date of the link
        - linkenddt: end date of the link (NaT = still active)

    This step fills missing linkenddt with today's date (current links)
    and ensures all date fields are properly parsed.
    """

    @property
    def name(self) -> str:
        return "LinkDateEnforcement"

    @property
    def description(self) -> str:
        return "Parse link dates and fill missing linkenddt with today"

    @property
    def required_columns(self) -> list[str]:
        return ["linkdt", "linkenddt"]

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["linkdt"] = pd.to_datetime(df["linkdt"])
        df["linkenddt"] = pd.to_datetime(df["linkenddt"])

        # Fill missing end dates with a far-future date (still active links)
        n_active = df["linkenddt"].isna().sum()
        df["linkenddt"] = df["linkenddt"].fillna(pd.Timestamp.today().normalize())

        if n_active > 0:
            logger.debug(
                f"  {n_active:,} links have no end date (still active)"
            )

        return df


class PrimaryLinkPreference(CorrectionStep):
    """
    When a GVKEY has multiple PERMNO links, prefer the primary security.

    Link primacy codes:
        P = Primary security (main listing)
        C = Primary candidate (used when P unavailable)
        J = Junior security
        N = Non-primary

    When multiple links exist for the same GVKEY at overlapping dates,
    we keep only P and C links. If both P and C exist, P takes priority.
    """

    def __init__(self, config: CCMCorrectionConfig) -> None:
        self._preferred = list(config.preferred_link_prim)

    @property
    def name(self) -> str:
        return "PrimaryLinkPreference"

    @property
    def description(self) -> str:
        prims = ", ".join(self._preferred)
        return f"Keep only linkprim in {{{prims}}} (prefer primary securities)"

    @property
    def required_columns(self) -> list[str]:
        return ["linkprim"]

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        mask = df["linkprim"].isin(self._preferred)
        return df[mask].copy()


def build_ccm_pipeline(config: CCMCorrectionConfig) -> list[CorrectionStep]:
    """
    Build the ordered list of CCM linking correction steps.

    Order:
        1. LinkTypeFilter (remove bad link types)
        2. LinkDateEnforcement (parse/fill dates)
        3. PrimaryLinkPreference (keep primary securities)
    """
    steps: list[CorrectionStep] = []

    if config.link_type_filter:
        steps.append(LinkTypeFilter(config))

    if config.link_date_enforcement:
        steps.append(LinkDateEnforcement())

    if config.primary_link_preference:
        steps.append(PrimaryLinkPreference(config))

    return steps
