"""
Derived financial quantities.

These are computed from the corrected CRSP and Compustat data:

1. MarketCapCalculation — |PRC| × SHROUT
2. BookToMarketCalculation — BE / ME
3. OperatingProfitabilityCalculation — (REVT - COGS) / AT  (Novy-Marx 2013)
4. InvestmentRateCalculation — CAPX / AT

Unlike corrections that filter rows, these ADD columns to the DataFrame.
They are applied after the CRSP-Compustat merge.

References:
    - Fama, E. & French, K. (1993). Book-to-market ratio construction.
    - Novy-Marx, R. (2013). "The Other Side of Value."
    - Fama, E. & French, K. (2015). Five-factor model: profitability & investment.
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
    Calculate operating profitability (Novy-Marx 2013).

    op = (revt - cogs) / at

    Gross profitability scaled by total assets. This is a strong
    predictor of stock returns, particularly when combined with value.
    """

    @property
    def name(self) -> str:
        return "OperatingProfitabilityCalculation"

    @property
    def description(self) -> str:
        return "Compute op = (revt - cogs) / at (Novy-Marx 2013)"

    @property
    def required_columns(self) -> list[str]:
        return ["at"]

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        revt = df.get("revt", pd.Series(np.nan, index=df.index))
        cogs = df.get("cogs", pd.Series(0.0, index=df.index)).fillna(0)

        valid = (df["at"] > 0) & revt.notna()
        df["op"] = np.where(valid, (revt - cogs) / df["at"], np.nan)

        n_valid = valid.sum()
        logger.debug(
            f"  Operating profitability computed for {n_valid:,}/{len(df):,} "
            f"observations"
        )
        return df


class InvestmentRateCalculation(CorrectionStep):
    """
    Calculate investment rate.

    inv = capx / at

    Capital expenditures scaled by total assets. Used in Fama-French
    five-factor model (CMA factor).
    """

    @property
    def name(self) -> str:
        return "InvestmentRateCalculation"

    @property
    def description(self) -> str:
        return "Compute inv = capx / at"

    @property
    def required_columns(self) -> list[str]:
        return ["at"]

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        capx = df.get("capx", pd.Series(np.nan, index=df.index))

        valid = (df["at"] > 0) & capx.notna()
        df["inv"] = np.where(valid, capx / df["at"], np.nan)

        n_valid = valid.sum()
        logger.debug(
            f"  Investment rate computed for {n_valid:,}/{len(df):,} observations"
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
