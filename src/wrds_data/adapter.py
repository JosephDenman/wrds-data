"""
tft-finance adapter.

Implements tft-finance's IDataSource interface so that WRDSDataProvider
can be used as a drop-in replacement for AlpacaDataSource.

Usage in tft-finance::

    from wrds_data import WRDSDataProvider, WRDSDataConfig, WRDSDataSource

    provider = WRDSDataProvider(WRDSDataConfig())
    adapter = WRDSDataSource(provider)

    # Drop-in replacement for AlpacaDataSource:
    data_manager = DataManager(data_source=adapter, config=system_config)

Column mapping from CRSP to tft-finance expected format:

    tft-finance     CRSP field      Notes
    -----------     ----------      -----
    close           abs(PRC)        Price sign correction applied
    high            ASKHI           Highest ask/trade price
    low             BIDLO           Lowest bid/trade price
    open            OPENPRC         Opening price
    volume          VOL             Trading volume in shares

All columns are required. Missing columns raise SchemaValidationError.
Missing individual values are left as NaN (no proxy substitution).
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Dict, List

import pandas as pd
from loguru import logger

from wrds_data.provider import WRDSDataProvider


# We don't import IDataSource directly to avoid making tft-finance a dependency.
# Instead, WRDSDataSource implements the same interface (duck typing).
# If tft-finance is installed, it will also satisfy isinstance checks via ABC.


class WRDSDataSource:
    """
    Adapter that makes WRDSDataProvider compatible with tft-finance's
    DataManager.

    Implements:
        - get_data(symbol, start_date, end_date) → DataFrame
        - get_data_batch(symbols, start_date, end_date) → dict[str, DataFrame]
        - get_sector_industry(symbols) → DataFrame

    The output DataFrame is indexed by date and has columns:
    [open, high, low, close, volume]

    All columns come from real CRSP fields. Missing columns raise
    SchemaValidationError; missing values remain NaN.
    """

    def __init__(self, provider: WRDSDataProvider) -> None:
        self._provider = provider
        self._ticker_permno_cache: dict[str, int | None] = {}

    def get_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a single symbol.

        Args:
            symbol: Ticker symbol (e.g. "AAPL").
            start_date: Start date.
            end_date: End date.

        Returns:
            DataFrame indexed by date with columns:
            [open, high, low, close, volume]
        """
        permno = self._resolve_ticker_cached(symbol)
        if permno is None:
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"]
            )

        start = start_date.date() if isinstance(start_date, datetime) else start_date
        end = end_date.date() if isinstance(end_date, datetime) else end_date

        try:
            raw = self._provider.daily_prices(
                permno=permno,
                start=start,
                end=end,
                apply_corrections=True,
            )
        except Exception as e:
            logger.warning(f"Failed to fetch data for {symbol} (PERMNO={permno}): {e}")
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"]
            )

        if len(raw) == 0:
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"]
            )

        return self._to_ohlcv(raw)

    def get_data_batch(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
    ) -> Dict[str, pd.DataFrame]:
        """
        Batch fetch OHLCV data for multiple symbols.

        More efficient than calling get_data() in a loop because it
        fetches all CRSP data at once and then splits by symbol.

        Args:
            symbols: List of ticker symbols.
            start_date: Start date.
            end_date: End date.

        Returns:
            Dict mapping symbol → OHLCV DataFrame.
        """
        start = start_date.date() if isinstance(start_date, datetime) else start_date
        end = end_date.date() if isinstance(end_date, datetime) else end_date

        # Resolve all tickers to PERMNOs
        permno_map = self._provider._universe.ticker_to_permno_map(symbols)
        if not permno_map:
            logger.warning("No tickers could be resolved to PERMNOs")
            return {}

        permnos = list(permno_map.values())
        ticker_by_permno = {v: k for k, v in permno_map.items()}

        # Fetch all data at once (much more efficient than per-symbol)
        try:
            raw = self._provider.daily_prices(
                start=start,
                end=end,
                apply_corrections=True,
            )
        except Exception as e:
            logger.error(f"Batch fetch failed: {e}")
            return {}

        if len(raw) == 0:
            return {}

        # Filter to requested PERMNOs only
        raw = raw[raw["permno"].isin(permnos)]

        # Split by PERMNO and convert each to OHLCV
        result: Dict[str, pd.DataFrame] = {}
        for permno, group_df in raw.groupby("permno"):
            ticker = ticker_by_permno.get(int(permno))
            if ticker is not None:
                ohlcv = self._to_ohlcv(group_df)
                if len(ohlcv) > 0:
                    result[ticker] = ohlcv

        logger.info(
            f"Batch fetch: {len(result)}/{len(symbols)} symbols returned data"
        )
        return result

    def get_sector_industry(
        self, symbols: List[str]
    ) -> pd.DataFrame:
        """
        Get sector/industry classification for symbols using WRDS SIC codes.

        Uses the centralized sector/industry mapping from wrds_data.sectors,
        which provides:
        - sector: GICS-like broad sector name (comparable to yfinance)
        - industry: Fama-French 12-industry classification
        - sic: raw SIC code
        - ff12, ff49: Fama-French 12 and 49-industry codes

        This replaces the yfinance-based SectorDataSource in tft-finance.
        The output format is compatible with tft-finance's DataManager
        (columns: symbol, sector, industry).

        Args:
            symbols: List of ticker symbols.

        Returns:
            DataFrame with columns: symbol, sector, industry
            (plus sic, ff12, ff49 for additional detail)
        """
        return self._provider.sector_industry(symbols)

    def get_sector_mapping(self, symbols: List[str]) -> Dict[str, str]:
        """
        Get symbol → sector dict (same interface as SectorDataSource.to_mapping).

        This can be passed directly to tft-finance's UniverseSampler.

        Args:
            symbols: List of ticker symbols.

        Returns:
            Dict mapping symbol → sector name.
        """
        from wrds_data.sectors import WRDSSectorDataSource

        df = self.get_sector_industry(symbols)
        return WRDSSectorDataSource.to_mapping(df, "sector")

    def get_industry_mapping(self, symbols: List[str]) -> Dict[str, str]:
        """
        Get symbol → industry dict.

        Args:
            symbols: List of ticker symbols.

        Returns:
            Dict mapping symbol → Fama-French 12-industry name.
        """
        from wrds_data.sectors import WRDSSectorDataSource

        df = self.get_sector_industry(symbols)
        return WRDSSectorDataSource.to_mapping(df, "industry")

    def sample_universe(
        self,
        as_of: "date | None" = None,
        n_symbols: int | None = None,
    ) -> List[str]:
        """
        Sample a representative universe of stock tickers for training.

        This is the WRDS equivalent of tft-finance's alpaca_sample mode.
        Uses stratified sampling (sector × liquidity tier) to ensure
        proportional representation.

        The default cap of 2500 symbols matches tft-finance's
        max_symbols_per_batch constraint.

        Args:
            as_of: Reference date. None = today.
            n_symbols: Target number of symbols.

        Returns:
            List of ticker symbols (sorted alphabetically).
        """
        return self._provider.sample_universe(as_of=as_of, n_symbols=n_symbols)

    def sample_universe_historical(
        self,
        start_date: "date",
        end_date: "date",
        n_symbols: int | None = None,
        rebalance_frequency: str | None = None,
    ) -> List[str]:
        """
        Sample a survivorship-bias-free universe across a historical period.

        Takes point-in-time snapshots at regular intervals (quarterly by
        default) throughout the training period and unions the results.
        This ensures stocks that were delisted during the training window
        are still included for the periods when they were active.

        Args:
            start_date: Training period start date.
            end_date: Training period end date.
            n_symbols: Target symbols per snapshot.
            rebalance_frequency: "quarterly", "monthly", or "annually".

        Returns:
            List of ticker symbols (sorted, deduplicated).
        """
        start = start_date.date() if isinstance(start_date, datetime) else start_date
        end = end_date.date() if isinstance(end_date, datetime) else end_date

        return self._provider.sample_universe_historical(
            start=start,
            end=end,
            n_symbols=n_symbols,
            rebalance_frequency=rebalance_frequency,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_ticker_cached(self, ticker: str) -> int | None:
        """Resolve ticker to PERMNO with caching."""
        if ticker not in self._ticker_permno_cache:
            try:
                self._ticker_permno_cache[ticker] = self._provider.resolve_ticker(ticker)
            except Exception:
                logger.debug(f"Could not resolve ticker '{ticker}' to PERMNO")
                self._ticker_permno_cache[ticker] = None
        return self._ticker_permno_cache[ticker]

    _REQUIRED_COLUMNS = ("prc", "askhi", "bidlo", "openprc", "vol")

    def _to_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert CRSP daily data to OHLCV format expected by tft-finance.

        Maps:
            close  ← prc (already sign-corrected)
            high   ← askhi
            low    ← bidlo
            open   ← openprc
            volume ← vol

        All columns are required. Missing values are left as NaN.

        Raises:
            SchemaValidationError: If any required column is missing.
        """
        missing = [c for c in self._REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            from wrds_data.exceptions import SchemaValidationError
            raise SchemaValidationError(missing, context="adapter._to_ohlcv")

        result = pd.DataFrame(index=pd.to_datetime(df["date"]))
        result.index.name = None

        result["open"] = df["openprc"].values
        result["high"] = df["askhi"].values
        result["low"] = df["bidlo"].values
        result["close"] = df["prc"].values
        result["volume"] = df["vol"].values

        # Sort by date
        result = result.sort_index()

        return result

    # Legacy SIC mapping methods (kept for backward compatibility)
    # New code should use wrds_data.sectors.sic_to_sector / sic_to_ff12

    @staticmethod
    def _sic_to_sector(sic: int) -> str:
        """Map SIC code to sector. Delegates to wrds_data.sectors."""
        from wrds_data.sectors import sic_to_sector
        return sic_to_sector(sic)

    @staticmethod
    def _sic_to_industry(sic: int) -> str:
        """Map SIC code to FF12 industry. Delegates to wrds_data.sectors."""
        from wrds_data.sectors import sic_to_ff12
        return sic_to_ff12(sic)
