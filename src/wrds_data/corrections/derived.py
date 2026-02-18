"""
Derived financial quantities.

These are computed from the corrected CRSP and Compustat data:

1. MarketCapCalculation — |PRC| × SHROUT
2. BookToMarketCalculation — BE / ME
3. OperatingProfitabilityCalculation — (SALE - COGS - XSGA - XINT) / BE  (FF5)
4. InvestmentRateCalculation — AT / AT_lag - 1  (asset growth, FF5)

Unlike corrections that filter rows, these ADD columns to the DataFrame.
They are applied after the CRSP-Compustat merge.

References:
    - Fama, E. & French, K. (1993). Book-to-market ratio construction.
    - Fama, E. & French, K. (2015). "A Five-Factor Asset Pricing Model."
    - tidyfinance Python package conventions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from wrds_data.corrections.base import CorrectionStep


class MarketCapCalculation(CorrectionStep):
    """
    Calculate market capitalization.

    market_cap = |PRC| × SHROUT

    CRSP SHROUT is in thousands, PRC is in dollars.
    Result is in thousands of dollars.
    """

    @property
    def name(self) -> str:
        return "MarketCapCalculation"

    @property
    def description(self) -> str:
        return "Compute market_cap = |prc| * shrout (in $thousands)"

    @property
    def required_columns(self) -> list[str]:
        return ["prc", "shrout"]

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["market_cap"] = df["prc"].abs() * df["shrout"]

        n_valid = df["market_cap"].notna().sum()
        logger.debug(
            f"  Market cap computed for {n_valid:,}/{len(df):,} observations"
        )
        return df


class BookToMarketCalculation(CorrectionStep):
    """
    Calculate book-to-market ratio.

    bm = be / market_cap

    Requires:
        - 'be' column from BookEquityCalculation
        - 'market_cap' column from MarketCapCalculation

    Negative or zero book equity produces NaN (not a meaningful ratio).
    """

    @property
    def name(self) -> str:
        return "BookToMarketCalculation"

    @property
    def description(self) -> str:
        return "Compute bm = book_equity / market_cap"

    @property
    def required_columns(self) -> list[str]:
        return ["be", "market_cap"]

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Only compute where both BE > 0 and market_cap > 0
        valid = (df["be"] > 0) & (df["market_cap"] > 0)
        df["bm"] = np.where(valid, df["be"] / df["market_cap"], np.nan)

        n_valid = valid.sum()
        n_neg_be = (df["be"] <= 0).sum()
        logger.debug(
            f"  Book-to-market computed for {n_valid:,} observations "
            f"({n_neg_be:,} with non-positive book equity set to NaN)"
        )
        return df


class OperatingProfitabilityCalculation(CorrectionStep):
    """
    Calculate operating profitability (Fama-French 2015 / tidyfinance convention).

    op = (sale - cogs - xsga - xint) / be

    Revenue minus cost of goods sold, SGA expenses, and interest expense,
    scaled by book equity. This follows the Fama-French five-factor model
    definition used in tidyfinance.

    Requires 'be' column from BookEquityCalculation.
    Falls back to 'revt' if 'sale' is not available.
    Missing cost components (cogs, xsga, xint) are filled with 0.

    References:
        - Fama, E. & French, K. (2015). "A Five-Factor Asset Pricing Model."
        - tidyfinance Python package: download_data_compustat() implementation
    """

    @property
    def name(self) -> str:
        return "OperatingProfitabilityCalculation"

    @property
    def description(self) -> str:
        return "Compute op = (sale - cogs - xsga - xint) / be (FF5 / tidyfinance)"

    @property
    def required_columns(self) -> list[str]:
        return ["be"]

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Revenue: prefer 'sale', fall back to 'revt'
        if "sale" in df.columns:
            revenue = df["sale"].copy()
            if "revt" in df.columns:
                revenue = revenue.fillna(df["revt"])
        else:
            revenue = df.get("revt", pd.Series(np.nan, index=df.index))

        cogs = df.get("cogs", pd.Series(0.0, index=df.index)).fillna(0)
        xsga = df.get("xsga", pd.Series(0.0, index=df.index)).fillna(0)
        xint = df.get("xint", pd.Series(0.0, index=df.index)).fillna(0)

        valid = (df["be"] > 0) & revenue.notna()
        df["op"] = np.where(
            valid, (revenue - cogs - xsga - xint) / df["be"], np.nan
        )

        n_valid = valid.sum()
        logger.debug(
            f"  Operating profitability computed for {n_valid:,}/{len(df):,} "
            f"observations"
        )
        return df


class InvestmentRateCalculation(CorrectionStep):
    """
    Calculate investment rate as asset growth (Fama-French 2015 / tidyfinance).

    inv = at / at_lag - 1

    Total asset growth rate, where at_lag is the previous year's total assets
    for the same firm (gvkey). This follows the Fama-French five-factor model
    definition (CMA factor) as implemented in tidyfinance.

    Requires data sorted by (gvkey, datadate/fyear) so that lag is meaningful.
    First observation per firm will have NaN investment rate.

    References:
        - Fama, E. & French, K. (2015). "A Five-Factor Asset Pricing Model."
        - tidyfinance Python package: download_data_compustat() implementation
    """

    @property
    def name(self) -> str:
        return "InvestmentRateCalculation"

    @property
    def description(self) -> str:
        return "Compute inv = at / at_lag - 1 (asset growth, FF5 / tidyfinance)"

    @property
    def required_columns(self) -> list[str]:
        return ["gvkey", "at"]

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Sort by firm and time to ensure correct lagging
        sort_col = "fyear" if "fyear" in df.columns else "datadate"
        df = df.sort_values(["gvkey", sort_col])

        # Compute lagged total assets per firm
        df["at_lag"] = df.groupby("gvkey")["at"].shift(1)

        # Investment = asset growth rate
        valid = (df["at_lag"] > 0) & df["at"].notna()
        df["inv"] = np.where(valid, df["at"] / df["at_lag"] - 1, np.nan)

        # Clean up temporary column
        df = df.drop(columns=["at_lag"])

        n_valid = valid.sum()
        logger.debug(
            f"  Investment rate (asset growth) computed for {n_valid:,}/{len(df):,} "
            f"observations"
        )
        return df


def build_derived_pipeline(
    config: "DerivedConfig",
) -> list[CorrectionStep]:
    """
    Build derived quantity calculation steps from config.

    Order matters:
        1. MarketCap first (needed by BookToMarket)
        2. BookToMarket second (depends on MarketCap and BookEquity)
        3. OperatingProfitability (independent)
        4. InvestmentRate (independent)
    """
    from wrds_data.config import DerivedConfig

    steps: list[CorrectionStep] = []

    if config.market_cap:
        steps.append(MarketCapCalculation())

    if config.book_to_market:
        steps.append(BookToMarketCalculation())

    if config.operating_profitability:
        steps.append(OperatingProfitabilityCalculation())

    if config.investment_rate:
        steps.append(InvestmentRateCalculation())

    return steps
