"""
Compustat data corrections.

Six corrections following standard academic methodology:

1. StandardFilter — keep only standardized, domestic, consolidated, industrial data
2. CurrencyFilter — USD-denominated observations only
3. IndustryExclusionFilter — exclude financials (and optionally utilities)
4. BookEquityCalculation — Fama-French book equity hierarchy
5. PointInTimeAlignment — prevent look-ahead bias using report dates
6. DuplicateRemoval — handle restatements by keeping latest

References:
    - Fama, E. & French, K. (1993). "Common Risk Factors in the Returns
      on Stocks and Bonds."
    - Davis, J., Fama, E. & French, K. (2000). "Characteristics, Covariances,
      and Average Returns: 1929 to 1997."
    - Novy-Marx, R. (2013). "The Other Side of Value: The Gross Profitability
      Premium."
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from wrds_data.config import CompustatCorrectionConfig
from wrds_data.corrections.base import CorrectionStep


class StandardFilter(CorrectionStep):
    """
    Keep only standardized domestic consolidated industrial data.

    Compustat includes multiple data formats. The standard academic
    filters are:
        - datafmt = 'STD' (standardized, not restated)
        - popsrc = 'D' (domestic)
        - consol = 'C' (consolidated statements)
        - indfmt = 'INDL' (industrial format)
    """

    @property
    def name(self) -> str:
        return "StandardFilter"

    @property
    def description(self) -> str:
        return "Keep datafmt=STD, popsrc=D, consol=C, indfmt=INDL"

    @property
    def required_columns(self) -> list[str]:
        return ["datafmt", "popsrc", "consol", "indfmt"]

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        mask = (
            (df["datafmt"] == "STD")
            & (df["popsrc"] == "D")
            & (df["consol"] == "C")
            & (df["indfmt"] == "INDL")
        )
        return df[mask].copy()


class CurrencyFilter(CorrectionStep):
    """Keep only USD-denominated observations."""

    @property
    def name(self) -> str:
        return "CurrencyFilter"

    @property
    def description(self) -> str:
        return "Keep only USD-denominated observations (curcd='USD')"

    @property
    def required_columns(self) -> list[str]:
        return ["curcd"]

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        mask = df["curcd"] == "USD"
        return df[mask].copy()


class IndustryExclusionFilter(CorrectionStep):
    """
    Exclude financial firms (and optionally utilities).

    Financial firms (SIC 6000-6999) have fundamentally different balance
    sheet structures that make book equity comparisons misleading.
    Utilities (SIC 4900-4999) are sometimes excluded for similar reasons.

    This is standard in Fama-French portfolio construction.
    """

    def __init__(self, config: CompustatCorrectionConfig) -> None:
        self._excluded_ranges = config.excluded_sic_ranges

    @property
    def name(self) -> str:
        return "IndustryExclusionFilter"

    @property
    def description(self) -> str:
        ranges = ", ".join(f"{lo}-{hi}" for lo, hi in self._excluded_ranges)
        return f"Exclude SIC ranges: {ranges}"

    @property
    def required_columns(self) -> list[str]:
        return ["sic"]

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Convert SIC to numeric, handling NaN
        sic = pd.to_numeric(df["sic"], errors="coerce")
        mask = pd.Series(True, index=df.index)
        for lo, hi in self._excluded_ranges:
            mask = mask & ~sic.between(lo, hi)
        return df[mask].copy()


class BookEquityCalculation(CorrectionStep):
    """
    Calculate book equity using the Fama-French hierarchy.

    Book equity = Stockholders' Equity + Deferred Taxes - Preferred Stock

    Stockholders' Equity hierarchy:
        1. SEQ (stockholders' equity)
        2. CEQ + PSTK (common equity + preferred stock par value)
        3. AT - LT (total assets - total liabilities)

    Preferred Stock hierarchy:
        1. PSTKRV (redemption value)
        2. PSTKL (liquidating value)
        3. PSTK (par value)

    This is the standard book equity calculation from:
        Davis, Fama, French (2000), Table 1.

    The result is stored in a new column 'be' (book equity).
    """

    @property
    def name(self) -> str:
        return "BookEquityCalculation"

    @property
    def description(self) -> str:
        return (
            "Fama-French book equity: SEQ (or CEQ+PSTK, or AT-LT) "
            "+ TXDITC (or TXDB+ITCB) - preferred stock (PSTKRV or PSTKL or PSTK)"
        )

    @property
    def required_columns(self) -> list[str]:
        # We need at least some of these; the hierarchy handles missingness
        return ["gvkey", "datadate"]

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # --- Stockholders' Equity (SE) hierarchy ---
        se = pd.Series(np.nan, index=df.index)

        # Level 1: SEQ
        if "seq" in df.columns:
            se = df["seq"].copy()

        # Level 2: CEQ + PSTK (fill where SEQ is missing, but CEQ exists)
        # Only use this fallback when CEQ is non-NaN; PSTK can default to 0
        if "ceq" in df.columns:
            pstk_fill = df["pstk"].fillna(0) if "pstk" in df.columns else 0
            fallback = df["ceq"] + pstk_fill  # NaN if CEQ is NaN
            se = se.fillna(fallback)

        # Level 3: AT - LT (fill where both above are missing)
        if "at" in df.columns and "lt" in df.columns:
            fallback = df["at"] - df["lt"]
            se = se.fillna(fallback)

        # --- Preferred Stock (PS) hierarchy ---
        ps = pd.Series(np.nan, index=df.index)

        # Level 1: PSTKRV (redemption value, most conservative)
        if "pstkrv" in df.columns:
            ps = df["pstkrv"].copy()

        # Level 2: PSTKL (liquidating value)
        if "pstkl" in df.columns:
            ps = ps.fillna(df["pstkl"])

        # Level 3: PSTK (par value)
        if "pstk" in df.columns:
            ps = ps.fillna(df["pstk"])

        # Fill remaining NaN as 0 (no preferred stock)
        ps = ps.fillna(0)

        # --- Deferred Taxes (tidyfinance convention) ---
        # Primary: TXDITC (deferred taxes and investment tax credit)
        # Fallback: TXDB + ITCB (balance sheet deferred taxes + investment tax credit)
        # Final: 0 if all are missing
        txditc = pd.Series(np.nan, index=df.index)
        if "txditc" in df.columns:
            txditc = df["txditc"].copy()

        # Build fallback from txdb + itcb
        txdb = df["txdb"].fillna(0) if "txdb" in df.columns else pd.Series(0.0, index=df.index)
        itcb = df["itcb"].fillna(0) if "itcb" in df.columns else pd.Series(0.0, index=df.index)
        txdb_itcb = txdb + itcb

        # combine_first: use txditc where available, fall back to txdb + itcb
        txditc = txditc.combine_first(txdb_itcb).fillna(0)

        # --- Book Equity ---
        df["be"] = se + txditc - ps

        n_computed = df["be"].notna().sum()
        n_total = len(df)
        logger.debug(
            f"  Book equity computed for {n_computed:,}/{n_total:,} observations "
            f"({n_computed / max(n_total, 1) * 100:.1f}%)"
        )

        return df


class PointInTimeAlignment(CorrectionStep):
    """
    Add a 'public_date' column representing when fundamentals become public.

    This prevents look-ahead bias in backtests. Financial reports are
    not available on the fiscal period end date (datadate); they become
    public on or after the report date (rdq).

    Logic:
        - If rdq is available: public_date = rdq + pit_lag_days
        - If rdq is missing: public_date = datadate + pit_fallback_days

    The public_date column is used downstream when merging fundamentals
    with daily price data via merge_asof.
    """

    def __init__(self, config: CompustatCorrectionConfig) -> None:
        self._lag_days = config.pit_lag_days
        self._fallback_days = config.pit_fallback_days

    @property
    def name(self) -> str:
        return "PointInTimeAlignment"

    @property
    def description(self) -> str:
        return (
            f"Set public_date = rdq + {self._lag_days}d "
            f"(fallback: datadate + {self._fallback_days}d)"
        )

    @property
    def required_columns(self) -> list[str]:
        return ["datadate"]

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["datadate"] = pd.to_datetime(df["datadate"])

        if "rdq" in df.columns:
            df["rdq"] = pd.to_datetime(df["rdq"])
            # Use rdq + lag where available, fallback to datadate + fallback_days
            df["public_date"] = df["rdq"] + pd.Timedelta(days=self._lag_days)
            missing_rdq = df["rdq"].isna()
            df.loc[missing_rdq, "public_date"] = (
                df.loc[missing_rdq, "datadate"]
                + pd.Timedelta(days=self._fallback_days)
            )
            n_rdq = (~missing_rdq).sum()
            n_fallback = missing_rdq.sum()
            logger.debug(
                f"  Point-in-time: {n_rdq:,} using rdq, "
                f"{n_fallback:,} using datadate fallback"
            )
        else:
            logger.warning(
                "  Column 'rdq' not available — using datadate + "
                f"{self._fallback_days}d for all observations"
            )
            df["public_date"] = df["datadate"] + pd.Timedelta(
                days=self._fallback_days
            )

        return df


class DuplicateRemoval(CorrectionStep):
    """
    Remove duplicate observations, keeping the latest restatement.

    Compustat may have multiple observations for the same firm-year
    due to restatements. Standard practice: per (gvkey, fyear), keep
    the row with the latest datadate.
    """

    @property
    def name(self) -> str:
        return "DuplicateRemoval"

    @property
    def description(self) -> str:
        return "Per (gvkey, fyear), keep row with latest datadate"

    @property
    def required_columns(self) -> list[str]:
        return ["gvkey", "datadate"]

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["datadate"] = pd.to_datetime(df["datadate"])

        # Determine grouping column
        fyear_col = "fyear" if "fyear" in df.columns else None
        if fyear_col is None:
            # For quarterly data, try fyearq
            fyear_col = "fyearq" if "fyearq" in df.columns else None

        if fyear_col is not None:
            n_before = len(df)
            df = df.sort_values(["gvkey", fyear_col, "datadate"])
            df = df.drop_duplicates(subset=["gvkey", fyear_col], keep="last")
            n_removed = n_before - len(df)
            if n_removed > 0:
                logger.debug(f"  Removed {n_removed:,} duplicate observations")
        else:
            logger.warning(
                "  Neither 'fyear' nor 'fyearq' found — "
                "deduplicating by (gvkey, datadate)"
            )
            df = df.drop_duplicates(subset=["gvkey", "datadate"], keep="last")

        return df


def build_compustat_pipeline(
    config: CompustatCorrectionConfig,
) -> list[CorrectionStep]:
    """
    Build the ordered list of Compustat correction steps from config.

    The order matters:
        1. StandardFilter first (removes non-standard data)
        2. CurrencyFilter (removes non-USD)
        3. DuplicateRemoval (before calculations that depend on uniqueness)
        4. IndustryExclusionFilter (before book equity, which is N/A for financials)
        5. BookEquityCalculation (needs clean, unique data)
        6. PointInTimeAlignment (last, adds the public_date column)
    """
    steps: list[CorrectionStep] = []

    if config.standard_filter:
        steps.append(StandardFilter())

    if config.currency_filter:
        steps.append(CurrencyFilter())

    if config.duplicate_removal:
        steps.append(DuplicateRemoval())

    if config.industry_exclusion:
        steps.append(IndustryExclusionFilter(config))

    if config.book_equity_calculation:
        steps.append(BookEquityCalculation())

    if config.point_in_time_alignment:
        steps.append(PointInTimeAlignment(config))

    return steps
