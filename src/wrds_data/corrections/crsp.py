"""
CRSP data corrections.

Seven corrections following standard academic methodology:

1. ShareCodeFilter — ordinary common shares only (SHRCD 10, 11)
2. ExchangeCodeFilter — major US exchanges (NYSE, AMEX, NASDAQ)
3. PriceSignCorrection — handle CRSP negative price convention
4. DelistingAdjustment — survivorship bias correction (Shumway 1997)
5. PennyStockFilter — exclude low-priced stocks
6. MinimumHistoryFilter — exclude securities with too few observations
7. VolumeValidation — remove rows with invalid volume/shares data

These corrections are applied to CRSP daily/monthly stock data AFTER
it has been fetched and before it is used for analysis or merged with
other datasets.

References:
    - Shumway, T. (1997). "The Delisting Bias in CRSP Data."
      Journal of Finance, 52(1), 327-340.
    - Shumway, T. & Warther, V. (1999). "The Delisting Bias in CRSP's
      Nasdaq Data and Its Implications for the Size Effect."
      Journal of Finance, 54(6), 2361-2379.
    - Fama, E. & French, K. (1993). "Common Risk Factors in the Returns
      on Stocks and Bonds." Journal of Financial Economics, 33(1), 3-56.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from wrds_data.config import CRSPCorrectionConfig
from wrds_data.corrections.base import CorrectionStep


class ShareCodeFilter(CorrectionStep):
    """
    Keep only ordinary common shares.

    CRSP share codes (SHRCD):
        10 = common shares, not further defined
        11 = common shares, certificates (standard domestic common stock)
        12 = shares of beneficial interest (REITs, etc.)
        ...

    Standard practice: keep SHRCD in {10, 11} to focus on ordinary
    common equity, excluding ADRs, REITs, closed-end funds, etc.
    """

    def __init__(self, config: CRSPCorrectionConfig) -> None:
        self._share_codes = set(config.share_codes)

    @property
    def name(self) -> str:
        return "ShareCodeFilter"

    @property
    def description(self) -> str:
        codes = ", ".join(str(c) for c in sorted(self._share_codes))
        return f"Keep only ordinary common shares (SHRCD in {{{codes}}})"

    @property
    def required_columns(self) -> list[str]:
        return ["shrcd"]

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        mask = df["shrcd"].isin(self._share_codes)
        return df[mask].copy()


class ExchangeCodeFilter(CorrectionStep):
    """
    Keep only stocks traded on major US exchanges.

    CRSP exchange codes (EXCHCD):
        1 = NYSE
        2 = AMEX (now NYSE American)
        3 = NASDAQ
        4 = Arca
        ...

    Standard practice: keep EXCHCD in {1, 2, 3}.
    """

    def __init__(self, config: CRSPCorrectionConfig) -> None:
        self._exchange_codes = set(config.exchange_codes)

    @property
    def name(self) -> str:
        return "ExchangeCodeFilter"

    @property
    def description(self) -> str:
        codes = ", ".join(str(c) for c in sorted(self._exchange_codes))
        return f"Keep only major US exchanges (EXCHCD in {{{codes}}})"

    @property
    def required_columns(self) -> list[str]:
        return ["exchcd"]

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        mask = df["exchcd"].isin(self._exchange_codes)
        return df[mask].copy()


class PriceSignCorrection(CorrectionStep):
    """
    Correct CRSP's negative price convention.

    In CRSP, a negative price indicates the value is a bid-ask midpoint
    (used when no closing trade occurred). The magnitude is still the
    correct price level.

    This step takes the absolute value of PRC.
    """

    @property
    def name(self) -> str:
        return "PriceSignCorrection"

    @property
    def description(self) -> str:
        return "Convert negative prices (bid-ask midpoints) to absolute values"

    @property
    def required_columns(self) -> list[str]:
        return ["prc"]

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        n_negative = (df["prc"] < 0).sum()
        if n_negative > 0:
            logger.debug(
                f"  Correcting {n_negative:,} negative prices "
                f"({n_negative / len(df) * 100:.1f}% of rows)"
            )
        df["prc"] = df["prc"].abs()

        # Also correct ASKHI and BIDLO if present
        if "askhi" in df.columns:
            df["askhi"] = df["askhi"].abs()
        if "bidlo" in df.columns:
            df["bidlo"] = df["bidlo"].abs()

        return df


class DelistingAdjustment(CorrectionStep):
    """
    Adjust returns for delisting events to correct survivorship bias.

    When a stock delists, CRSP's daily return series ends. Without
    correction, the final-period return understates the true loss
    shareholders experience.

    Method (Shumway 1997):
        1. Merge delisting data (dsedelist) onto daily stock data.
        2. For each delisting event, compute the "missing" return.
        3. If the delisting return (DLRET) is available, use it.
        4. If DLRET is missing and the delisting was performance-related
           (DLSTCD 400-499), impute -30%.
        5. If DLRET is missing and the stock was dropped by the exchange
           (DLSTCD 500+), impute -55%.
        6. Compound the delisting return with the last available return.

    This correction is THE most important correction for avoiding
    survivorship bias in CRSP data. Omitting it biases returns upward.

    This step requires that the delisting DataFrame is provided via
    ``set_delisting_data()`` before ``apply()`` is called.
    """

    def __init__(self, config: CRSPCorrectionConfig) -> None:
        self._return_otc = config.delisting_return_otc
        self._return_exchange = config.delisting_return_exchange
        self._delisting_df: pd.DataFrame | None = None

    def set_delisting_data(self, delisting_df: pd.DataFrame) -> None:
        """
        Provide the CRSP delisting events DataFrame.

        Expected columns: permno, dlstdt, dlret, dlstcd.
        """
        self._delisting_df = delisting_df.copy()

    @property
    def name(self) -> str:
        return "DelistingAdjustment"

    @property
    def description(self) -> str:
        return (
            "Shumway (1997) delisting return adjustment: impute "
            f"{self._return_otc:.0%} for performance-related delistings (400-499), "
            f"{self._return_exchange:.0%} for exchange-dropped (500+)"
        )

    @property
    def required_columns(self) -> list[str]:
        return ["permno", "date", "ret"]

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._delisting_df is None or len(self._delisting_df) == 0:
            logger.warning(
                "No delisting data provided — skipping delisting adjustment. "
                "This will introduce survivorship bias."
            )
            return df

        df = df.copy()
        delist = self._delisting_df.copy()

        # Ensure date types match
        df["date"] = pd.to_datetime(df["date"])
        delist["dlstdt"] = pd.to_datetime(delist["dlstdt"])

        # Impute missing delisting returns based on delisting code
        delist["dlret_imputed"] = delist["dlret"].copy()

        # Performance-related delistings (codes 400-499)
        mask_perf = (
            delist["dlstcd"].between(400, 499) & delist["dlret"].isna()
        )
        delist.loc[mask_perf, "dlret_imputed"] = self._return_otc
        n_imputed_perf = mask_perf.sum()

        # Exchange-dropped delistings (codes 500+)
        mask_exch = (
            (delist["dlstcd"] >= 500) & delist["dlret"].isna()
        )
        delist.loc[mask_exch, "dlret_imputed"] = self._return_exchange
        n_imputed_exch = mask_exch.sum()

        if n_imputed_perf > 0 or n_imputed_exch > 0:
            logger.debug(
                f"  Imputed {n_imputed_perf} performance-related and "
                f"{n_imputed_exch} exchange-dropped delisting returns"
            )

        # Merge delisting returns onto stock data
        # Match on permno and date == dlstdt (the delisting date)
        delist_merge = delist[["permno", "dlstdt", "dlret_imputed"]].dropna(
            subset=["dlret_imputed"]
        )
        delist_merge = delist_merge.rename(columns={"dlstdt": "date"})

        df = df.merge(
            delist_merge,
            on=["permno", "date"],
            how="left",
        )

        # Compound the delisting return with the existing return
        # If the stock has a return on the delisting date, compound:
        #   adjusted_ret = (1 + ret) * (1 + dlret) - 1
        # If no return exists, use dlret directly
        has_dlret = df["dlret_imputed"].notna()
        has_ret = df["ret"].notna()

        # Case 1: both returns exist — compound them
        both = has_dlret & has_ret
        df.loc[both, "ret"] = (
            (1 + df.loc[both, "ret"]) * (1 + df.loc[both, "dlret_imputed"]) - 1
        )

        # Case 2: only delisting return — use it directly
        only_dlret = has_dlret & ~has_ret
        df.loc[only_dlret, "ret"] = df.loc[only_dlret, "dlret_imputed"]

        n_adjusted = has_dlret.sum()
        if n_adjusted > 0:
            logger.debug(f"  Adjusted {n_adjusted:,} returns for delisting events")

        df = df.drop(columns=["dlret_imputed"])
        return df


class PennyStockFilter(CorrectionStep):
    """
    Remove observations where the stock price is below a threshold.

    Penny stocks have different microstructure characteristics (wider
    bid-ask spreads, lower liquidity) that can distort analysis.
    Standard threshold: $5.

    Note: Applied AFTER PriceSignCorrection (prices should be positive).
    """

    def __init__(self, config: CRSPCorrectionConfig) -> None:
        self._threshold = config.penny_stock_threshold

    @property
    def name(self) -> str:
        return "PennyStockFilter"

    @property
    def description(self) -> str:
        return f"Remove observations where |PRC| < ${self._threshold:.2f}"

    @property
    def required_columns(self) -> list[str]:
        return ["prc"]

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        mask = df["prc"].abs() >= self._threshold
        return df[mask].copy()


class MinimumHistoryFilter(CorrectionStep):
    """
    Remove securities with fewer than N trading days.

    Securities with very short histories lack sufficient data for
    meaningful statistical analysis. Standard: 252 trading days (~1 year).
    """

    def __init__(self, config: CRSPCorrectionConfig) -> None:
        self._min_days = config.min_trading_days

    @property
    def name(self) -> str:
        return "MinimumHistoryFilter"

    @property
    def description(self) -> str:
        return f"Remove PERMNOs with fewer than {self._min_days} trading days"

    @property
    def required_columns(self) -> list[str]:
        return ["permno"]

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        counts = df.groupby("permno").size()
        valid_permnos = counts[counts >= self._min_days].index
        n_dropped = len(counts) - len(valid_permnos)
        if n_dropped > 0:
            logger.debug(
                f"  Dropping {n_dropped:,} PERMNOs with < {self._min_days} "
                f"trading days (keeping {len(valid_permnos):,})"
            )
        return df[df["permno"].isin(valid_permnos)].copy()


class VolumeValidation(CorrectionStep):
    """
    Remove rows with invalid volume or shares outstanding data.

    Rows with zero or negative volume or shares outstanding are data
    quality issues that would corrupt analyses depending on these fields.
    """

    @property
    def name(self) -> str:
        return "VolumeValidation"

    @property
    def description(self) -> str:
        return "Remove rows where volume <= 0 or shares outstanding <= 0"

    @property
    def required_columns(self) -> list[str]:
        return ["vol"]

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        mask = df["vol"] > 0
        if "shrout" in df.columns:
            mask = mask & (df["shrout"] > 0)
        return df[mask].copy()


def build_crsp_pipeline(config: CRSPCorrectionConfig) -> list[CorrectionStep]:
    """
    Build the ordered list of CRSP correction steps from config.

    Steps are returned in the canonical order. The caller wraps them
    in a CorrectionPipeline.

    Note: DelistingAdjustment is included but requires
    ``set_delisting_data()`` to be called before the pipeline runs.
    """
    steps: list[CorrectionStep] = []

    if config.share_code_filter:
        steps.append(ShareCodeFilter(config))

    if config.exchange_code_filter:
        steps.append(ExchangeCodeFilter(config))

    if config.price_sign_correction:
        steps.append(PriceSignCorrection())

    if config.delisting_adjustment:
        steps.append(DelistingAdjustment(config))

    if config.penny_stock_filter:
        steps.append(PennyStockFilter(config))

    if config.volume_validation:
        steps.append(VolumeValidation())

    # MinimumHistoryFilter goes last — after other filters have removed rows,
    # we check if enough remain per PERMNO.
    if config.min_history_filter:
        steps.append(MinimumHistoryFilter(config))

    return steps
