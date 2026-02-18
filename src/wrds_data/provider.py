"""
WRDSDataProvider — the main user-facing API.

This is the single class users interact with to fetch CRSP and Compustat
data with proper academic corrections applied.

Usage::

    from wrds_data import WRDSDataProvider, WRDSDataConfig

    config = WRDSDataConfig()
    provider = WRDSDataProvider(config)

    # Fetch corrected daily prices for a single ticker
    prices = provider.daily_prices(ticker="AAPL", start=date(2010, 1, 1))

    # Fetch fundamentals
    fundies = provider.fundamentals(ticker="AAPL")

    # Full CRSP+Compustat merge with all corrections
    merged = provider.merged(start=date(2000, 1, 1), end=date(2023, 12, 31))

    # Download all datasets to local Parquet
    provider.download()
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
from loguru import logger

from wrds_data.backend.base import DataBackend
from wrds_data.backend.parquet_backend import ParquetBackend
from wrds_data.backend.wrds_backend import WRDSBackend
from wrds_data.config import UniverseSamplingConfig, WRDSDataConfig
from wrds_data.corrections.base import CorrectionPipeline
from wrds_data.corrections.compustat import build_compustat_pipeline
from wrds_data.corrections.crsp import DelistingAdjustment, build_crsp_pipeline
from wrds_data.corrections.derived import build_derived_pipeline
from wrds_data.corrections.linking import build_ccm_pipeline
from wrds_data.datasets.base import DatasetDefinition
from wrds_data.datasets.ccm import CCM_LINK
from wrds_data.datasets.compustat import COMPUSTAT_ANNUAL, COMPUSTAT_QUARTERLY
from wrds_data.datasets.crsp import CRSP_DAILY, CRSP_DELISTING, CRSP_NAMES
from wrds_data.datasets.registry import DatasetRegistry
from wrds_data.download.downloader import BulkDownloader
from wrds_data.exceptions import ConfigurationError, DataNotAvailableError
from wrds_data.universe import UniverseResolver


class WRDSDataProvider:
    """
    Main API for accessing corrected WRDS financial data.

    Handles:
        - Backend selection (WRDS live, local Parquet, or auto)
        - Data correction pipelines (CRSP, Compustat, CCM)
        - Ticker ↔ PERMNO resolution
        - CRSP-Compustat merging via CCM linking table
        - Bulk data downloads
    """

    def __init__(self, config: WRDSDataConfig) -> None:
        self._config = config
        self._backend = self._create_backend()
        self._universe = UniverseResolver(self._backend)

        # Build correction pipelines
        crsp_steps = build_crsp_pipeline(config.corrections.crsp)
        self._crsp_pipeline = CorrectionPipeline(crsp_steps)

        compustat_steps = build_compustat_pipeline(config.corrections.compustat)
        self._compustat_pipeline = CorrectionPipeline(compustat_steps)

        ccm_steps = build_ccm_pipeline(config.corrections.ccm)
        self._ccm_pipeline = CorrectionPipeline(ccm_steps)

        derived_steps = build_derived_pipeline(config.corrections.derived)
        self._derived_pipeline = CorrectionPipeline(derived_steps)

        logger.info(
            f"WRDSDataProvider initialized "
            f"(backend={config.storage.backend}, "
            f"crsp_corrections={len(crsp_steps)}, "
            f"compustat_corrections={len(compustat_steps)})"
        )

    def _create_backend(self) -> DataBackend:
        """Create the appropriate backend from config."""
        backend_type = self._config.storage.backend

        if backend_type == "wrds":
            return WRDSBackend(self._config.connection)

        elif backend_type == "parquet":
            return ParquetBackend(self._config.storage)

        elif backend_type == "auto":
            # Try parquet first, fall back to WRDS
            parquet = ParquetBackend(self._config.storage)
            if parquet.is_available(CRSP_DAILY):
                logger.info("Auto-detected local Parquet data, using ParquetBackend")
                return parquet
            else:
                logger.info("No local data found, using WRDSBackend")
                return WRDSBackend(self._config.connection)

        else:
            raise ConfigurationError(
                f"Unknown backend type: '{backend_type}'. "
                f"Must be 'wrds', 'parquet', or 'auto'."
            )

    # ------------------------------------------------------------------
    # Core Data Access
    # ------------------------------------------------------------------

    def daily_prices(
        self,
        permno: int | None = None,
        ticker: str | None = None,
        start: date | None = None,
        end: date | None = None,
        apply_corrections: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch CRSP daily stock data with corrections applied.

        Provide either ``permno`` or ``ticker`` (not both).
        If neither is provided, returns data for all securities.

        Args:
            permno: CRSP permanent security number.
            ticker: Stock ticker symbol (resolved to PERMNO via dsenames).
            start: Start date (inclusive). None = all available.
            end: End date (inclusive). None = all available.
            apply_corrections: If True, run the CRSP correction pipeline.

        Returns:
            DataFrame with canonical column names (permno, date, prc, ret, etc.)
        """
        # Resolve ticker to PERMNO if needed
        resolved_permno = self._resolve_identifier(permno, ticker)

        # Build filters
        filters: dict[str, Any] = {}
        if resolved_permno is not None:
            filters["permno"] = resolved_permno

        date_range = self._make_date_range(start, end)

        # Fetch raw data
        df = self._backend.query(CRSP_DAILY, date_range=date_range, filters=filters)

        if len(df) == 0:
            logger.warning("No CRSP daily data returned for the given parameters")
            return df

        # Merge name info for share/exchange code filtering
        if apply_corrections and self._needs_name_info():
            df = self._merge_name_info(df)

        # Apply corrections
        if apply_corrections:
            # Provide delisting data to the DelistingAdjustment step
            self._inject_delisting_data(date_range)
            df = self._crsp_pipeline.run(df)

        return df

    def fundamentals(
        self,
        permno: int | None = None,
        ticker: str | None = None,
        gvkey: str | None = None,
        start: date | None = None,
        end: date | None = None,
        frequency: str = "annual",
        apply_corrections: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch Compustat fundamentals with corrections applied.

        Provide ``gvkey`` (Compustat identifier), or ``permno``/``ticker``
        (resolved to GVKEY via CCM linking table).

        Args:
            permno: CRSP permanent security number (resolved via CCM).
            ticker: Stock ticker (resolved to PERMNO, then to GVKEY via CCM).
            gvkey: Compustat company identifier (direct lookup).
            start: Start date (inclusive).
            end: End date (inclusive).
            frequency: "annual" or "quarterly".
            apply_corrections: If True, run the Compustat correction pipeline.

        Returns:
            DataFrame with Compustat fundamentals (including 'be' if BookEquity
            calculation is enabled, and 'public_date' if PIT alignment is enabled).
        """
        dataset = COMPUSTAT_ANNUAL if frequency == "annual" else COMPUSTAT_QUARTERLY

        # Resolve to GVKEY if needed
        filters: dict[str, Any] = {}
        if gvkey is not None:
            filters["gvkey"] = gvkey
        elif permno is not None or ticker is not None:
            resolved_permno = self._resolve_identifier(permno, ticker)
            if resolved_permno is not None:
                gvkeys = self._permno_to_gvkeys(resolved_permno)
                if gvkeys:
                    filters["gvkey"] = gvkeys

        date_range = self._make_date_range(start, end)

        df = self._backend.query(dataset, date_range=date_range, filters=filters)

        if len(df) == 0:
            logger.warning("No Compustat data returned for the given parameters")
            return df

        if apply_corrections:
            df = self._compustat_pipeline.run(df)

        return df

    def merged(
        self,
        start: date | None = None,
        end: date | None = None,
        apply_corrections: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch CRSP + Compustat merged data via CCM linking.

        This is the full pipeline:
            1. Fetch and correct CRSP daily data
            2. Fetch and correct Compustat annual data
            3. Fetch and correct CCM linking table
            4. Merge CRSP ↔ Compustat via CCM links
            5. Compute derived quantities (market cap, B/M, etc.)

        Args:
            start: Start date for CRSP data.
            end: End date for CRSP data.
            apply_corrections: If True, apply all correction pipelines.

        Returns:
            Merged DataFrame with CRSP prices, Compustat fundamentals,
            and derived quantities.
        """
        logger.info("Building merged CRSP-Compustat dataset...")

        # 1. Fetch CRSP daily data
        crsp = self.daily_prices(start=start, end=end, apply_corrections=apply_corrections)
        if len(crsp) == 0:
            raise DataNotAvailableError("No CRSP data available for merge")

        # 2. Fetch Compustat annual data
        compustat = self.fundamentals(
            start=start, end=end, apply_corrections=apply_corrections
        )

        # 3. Fetch and correct CCM linking table
        ccm = self._backend.query(CCM_LINK)
        if apply_corrections:
            ccm = self._ccm_pipeline.run(ccm)

        # 4. Merge via CCM
        merged = self._merge_crsp_compustat(crsp, compustat, ccm)

        # 5. Compute derived quantities
        if apply_corrections and len(self._derived_pipeline) > 0:
            merged = self._derived_pipeline.run(merged)

        logger.info(
            f"Merged dataset: {len(merged):,} rows, "
            f"{merged['permno'].nunique():,} securities"
        )
        return merged

    # ------------------------------------------------------------------
    # Universe
    # ------------------------------------------------------------------

    def universe(
        self,
        as_of: date | None = None,
        share_codes: tuple[int, ...] | None = (10, 11),
        exchange_codes: tuple[int, ...] | None = (1, 2, 3),
    ) -> pd.DataFrame:
        """
        Return all valid securities as of a date.

        Args:
            as_of: Reference date. None = all securities ever listed.
            share_codes: SHRCD filter. Default: common shares.
            exchange_codes: EXCHCD filter. Default: NYSE, AMEX, NASDAQ.

        Returns:
            DataFrame with: permno, ticker, comnam, shrcd, exchcd, siccd
        """
        return self._universe.universe(as_of, share_codes, exchange_codes)

    def resolve_ticker(self, ticker: str, as_of: date | None = None) -> int:
        """Ticker → PERMNO."""
        return self._universe.resolve_ticker(ticker, as_of)

    def resolve_permno(self, permno: int, as_of: date | None = None) -> str:
        """PERMNO → ticker."""
        return self._universe.resolve_permno(permno, as_of)

    # ------------------------------------------------------------------
    # Universe Sampling
    # ------------------------------------------------------------------

    def sample_universe(
        self,
        as_of: date | None = None,
        n_symbols: int | None = None,
        sampling_config: UniverseSamplingConfig | None = None,
        random_seed: int | None = None,
    ) -> list[str]:
        """
        Sample a representative subset of stock tickers.

        Uses stratified sampling (sector × liquidity tier) to maintain
        proportional representation. The result is suitable for use as
        the symbol list in tft-finance's DataConfig.

        The default max_symbols (2500) matches tft-finance's
        max_symbols_per_batch constraint.

        Args:
            as_of: Reference date for sampling. None = today.
            n_symbols: Target number of symbols. Capped at config.max_symbols.
            sampling_config: Override the default sampling configuration.
            random_seed: Override the random seed for reproducibility.

        Returns:
            List of ticker symbols (sorted alphabetically).
        """
        from wrds_data.sampling import WRDSUniverseSampler

        cfg = sampling_config or UniverseSamplingConfig()
        sampler = WRDSUniverseSampler(self, cfg)
        return sampler.sample(as_of=as_of, n_symbols=n_symbols, random_seed=random_seed)

    def sample_universe_historical(
        self,
        start: date,
        end: date,
        n_symbols: int | None = None,
        rebalance_frequency: str | None = None,
        sampling_config: UniverseSamplingConfig | None = None,
        random_seed: int | None = None,
    ) -> list[str]:
        """
        Sample a survivorship-bias-free universe across a historical period.

        Takes multiple point-in-time snapshots (quarterly, monthly, or annually)
        of the CRSP universe throughout the training period and unions them.
        Stocks that were delisted partway through the period are included from
        the snapshots taken while they were still active.

        Args:
            start: Training period start date.
            end: Training period end date.
            n_symbols: Target symbols per snapshot.
            rebalance_frequency: "quarterly", "monthly", or "annually".
            sampling_config: Override default sampling configuration.
            random_seed: Override random seed.

        Returns:
            List of ticker symbols (sorted, deduplicated).
        """
        from wrds_data.sampling import WRDSUniverseSampler

        cfg = sampling_config or UniverseSamplingConfig()
        sampler = WRDSUniverseSampler(self, cfg)
        return sampler.sample_historical(
            start=start,
            end=end,
            n_symbols=n_symbols,
            rebalance_frequency=rebalance_frequency,
            random_seed=random_seed,
        )

    def sector_industry(
        self,
        symbols: list[str],
        as_of: date | None = None,
    ) -> pd.DataFrame:
        """
        Get sector and industry classification for symbols.

        Uses SIC codes from the CRSP names table, mapped to:
        - sector: GICS-like broad sector name (comparable to yfinance)
        - industry: Fama-French 12-industry classification
        - ff49: Fama-French 49-industry classification

        This replaces yfinance-based sector lookups with WRDS-native data.

        Args:
            symbols: List of ticker symbols.
            as_of: Date for the lookup. None = most recent mapping.

        Returns:
            DataFrame with columns:
            [symbol, sector, industry, sic, ff12, ff49]
        """
        from wrds_data.sectors import WRDSSectorDataSource

        source = WRDSSectorDataSource(self)
        return source.fetch_sectors(symbols, as_of=as_of)

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def download(
        self,
        datasets: list[str] | None = None,
        start_year: int = 1960,
        force: bool = False,
    ) -> dict[str, int]:
        """
        Download datasets from WRDS to local Parquet storage.

        Requires a live WRDS connection (even if the primary backend is Parquet).

        Args:
            datasets: Dataset names to download. None = all registered.
            start_year: Earliest year to download.
            force: Re-download even if files exist.

        Returns:
            Dict mapping dataset name → total rows downloaded.
        """
        # Use a WRDS backend for downloading, regardless of primary backend
        wrds = WRDSBackend(self._config.connection)
        try:
            downloader = BulkDownloader(
                wrds, self._config.storage, start_year=start_year
            )
            return downloader.download(datasets, force=force)
        finally:
            wrds.close()

    def download_status(self) -> dict[str, dict[str, int]]:
        """Check download status for all registered datasets."""
        downloader = BulkDownloader(
            self._backend
            if isinstance(self._backend, WRDSBackend)
            else WRDSBackend(self._config.connection),
            self._config.storage,
        )
        return downloader.status()

    # ------------------------------------------------------------------
    # Raw / Extensibility
    # ------------------------------------------------------------------

    def fetch_raw(
        self,
        dataset: str | DatasetDefinition,
        start: date | None = None,
        end: date | None = None,
        filters: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """
        Fetch raw (uncorrected) data from any registered dataset.

        Args:
            dataset: Dataset name or DatasetDefinition object.
            start: Start date.
            end: End date.
            filters: Column-value filters.

        Returns:
            Raw DataFrame with canonical column names.
        """
        if isinstance(dataset, str):
            registry = DatasetRegistry.instance()
            dataset = registry.get(dataset)

        date_range = self._make_date_range(start, end)
        return self._backend.query(dataset, date_range=date_range, filters=filters)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_identifier(
        self, permno: int | None, ticker: str | None
    ) -> int | None:
        """Resolve permno/ticker to a single PERMNO."""
        if permno is not None and ticker is not None:
            raise ValueError("Provide either 'permno' or 'ticker', not both")
        if ticker is not None:
            return self._universe.resolve_ticker(ticker)
        return permno

    @staticmethod
    def _make_date_range(
        start: date | None, end: date | None
    ) -> tuple[date, date] | None:
        """Build a date range tuple, or None if no dates specified."""
        if start is not None and end is not None:
            return (start, end)
        if start is not None:
            return (start, date.today())
        if end is not None:
            return (date(1960, 1, 1), end)
        return None

    def _needs_name_info(self) -> bool:
        """Check if any CRSP correction needs shrcd/exchcd from dsenames."""
        for step in self._crsp_pipeline.steps:
            if hasattr(step, "required_columns"):
                if "shrcd" in step.required_columns or "exchcd" in step.required_columns:
                    return True
        return False

    def _merge_name_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """Merge shrcd/exchcd from CRSP names table onto daily data."""
        names = self._backend.query(
            CRSP_NAMES,
            columns=["permno", "shrcd", "exchcd", "namedt", "nameendt"],
        )
        names["namedt"] = pd.to_datetime(names["namedt"])
        names["nameendt"] = pd.to_datetime(names["nameendt"]).fillna(
            pd.Timestamp("2099-12-31")
        )

        df["date"] = pd.to_datetime(df["date"])

        # merge_asof: for each (permno, date), find the name record
        # where namedt <= date. Then filter by nameendt >= date.
        df = df.sort_values(["permno", "date"])
        names = names.sort_values(["permno", "namedt"])

        merged = pd.merge_asof(
            df,
            names[["permno", "namedt", "nameendt", "shrcd", "exchcd"]],
            left_on="date",
            right_on="namedt",
            by="permno",
            direction="backward",
        )

        # Enforce nameendt boundary
        valid_name = merged["nameendt"] >= merged["date"]
        merged.loc[~valid_name, "shrcd"] = pd.NA
        merged.loc[~valid_name, "exchcd"] = pd.NA

        # Drop helper columns
        merged = merged.drop(columns=["namedt", "nameendt"], errors="ignore")

        return merged

    def _inject_delisting_data(
        self, date_range: tuple[date, date] | None
    ) -> None:
        """Load delisting data and inject it into the DelistingAdjustment step."""
        for step in self._crsp_pipeline.steps:
            if isinstance(step, DelistingAdjustment):
                delist_df = self._backend.query(
                    CRSP_DELISTING, date_range=date_range
                )
                step.set_delisting_data(delist_df)
                logger.debug(f"Loaded {len(delist_df):,} delisting events")
                break

    def _permno_to_gvkeys(self, permno: int) -> list[str]:
        """Look up all GVKEYs linked to a PERMNO via CCM."""
        ccm = self._backend.query(CCM_LINK, filters={"lpermno": permno})
        if len(ccm) == 0:
            return []
        return ccm["gvkey"].unique().tolist()

    def _merge_crsp_compustat(
        self,
        crsp: pd.DataFrame,
        compustat: pd.DataFrame,
        ccm: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merge CRSP and Compustat via CCM linking table.

        Uses merge_asof with direction='backward' for point-in-time
        alignment: each CRSP observation gets the most recent Compustat
        fundamentals available as of that date.
        """
        if len(compustat) == 0 or len(ccm) == 0:
            logger.warning("No Compustat or CCM data — returning CRSP data only")
            return crsp

        # Step 1: Attach GVKEY to CRSP data via CCM
        crsp["date"] = pd.to_datetime(crsp["date"])
        ccm["linkdt"] = pd.to_datetime(ccm["linkdt"])
        ccm["linkenddt"] = pd.to_datetime(ccm["linkenddt"])

        # Merge CCM links onto CRSP by permno
        ccm_slim = ccm[["gvkey", "lpermno", "linkdt", "linkenddt"]].rename(
            columns={"lpermno": "permno"}
        )

        crsp_ccm = crsp.merge(ccm_slim, on="permno", how="inner")

        # Enforce link date windows
        valid_link = (
            (crsp_ccm["date"] >= crsp_ccm["linkdt"])
            & (crsp_ccm["date"] <= crsp_ccm["linkenddt"])
        )
        crsp_ccm = crsp_ccm[valid_link].copy()
        crsp_ccm = crsp_ccm.drop(columns=["linkdt", "linkenddt"])

        if len(crsp_ccm) == 0:
            logger.warning("No valid CCM links — returning CRSP data only")
            return crsp

        # Step 2: Determine the merge date column for Compustat
        # Use public_date if available (from PointInTimeAlignment), else datadate
        if "public_date" in compustat.columns:
            comp_date_col = "public_date"
        else:
            comp_date_col = "datadate"
            compustat["datadate"] = pd.to_datetime(compustat["datadate"])

        compustat[comp_date_col] = pd.to_datetime(compustat[comp_date_col])

        # Step 3: merge_asof — for each CRSP date, find the most recent
        # Compustat observation that was publicly available
        crsp_ccm = crsp_ccm.sort_values(["gvkey", "date"])
        compustat = compustat.sort_values(["gvkey", comp_date_col])

        # Select Compustat columns to merge (avoid duplicates)
        crsp_cols = set(crsp_ccm.columns)
        comp_merge_cols = ["gvkey", comp_date_col] + [
            c for c in compustat.columns
            if c not in crsp_cols and c not in ["gvkey", comp_date_col]
        ]
        comp_for_merge = compustat[comp_merge_cols].copy()

        merged = pd.merge_asof(
            crsp_ccm,
            comp_for_merge,
            left_on="date",
            right_on=comp_date_col,
            by="gvkey",
            direction="backward",
        )

        n_with_fundies = merged[
            merged.columns.intersection(["be", "at", "revt"])
        ].notna().any(axis=1).sum()

        logger.info(
            f"CRSP-Compustat merge: {len(merged):,} rows, "
            f"{n_with_fundies:,} with fundamentals attached"
        )

        return merged

    def close(self) -> None:
        """Release backend resources."""
        self._backend.close()

    def __enter__(self) -> WRDSDataProvider:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
