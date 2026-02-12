"""
Universe resolution: ticker ↔ PERMNO mapping.

WRDS uses PERMNO as the permanent security identifier. Tickers change
over time (mergers, name changes, re-listings), so the mapping from
ticker → PERMNO depends on the as-of date.

The dsenames table provides the authoritative mapping with effective
date ranges (namedt, nameendt).
"""

from __future__ import annotations

from datetime import date

import pandas as pd
from loguru import logger

from wrds_data.backend.base import DataBackend
from wrds_data.datasets.crsp import CRSP_NAMES
from wrds_data.exceptions import TickerResolutionError


class UniverseResolver:
    """
    Resolves ticker ↔ PERMNO mappings and provides universe listings.

    Caches the names table in memory after first load for fast lookups.
    """

    def __init__(self, backend: DataBackend) -> None:
        self._backend = backend
        self._names_df: pd.DataFrame | None = None

    def _ensure_names_loaded(self) -> pd.DataFrame:
        """Load the CRSP names table if not cached."""
        if self._names_df is None:
            logger.debug("Loading CRSP names table for universe resolution...")
            self._names_df = self._backend.query(CRSP_NAMES)
            self._names_df["namedt"] = pd.to_datetime(self._names_df["namedt"])
            self._names_df["nameendt"] = pd.to_datetime(self._names_df["nameendt"])
            # Fill missing end dates with far future
            self._names_df["nameendt"] = self._names_df["nameendt"].fillna(
                pd.Timestamp("2099-12-31")
            )
            logger.debug(
                f"Loaded {len(self._names_df):,} name history records "
                f"for {self._names_df['permno'].nunique():,} PERMNOs"
            )
        return self._names_df

    def resolve_ticker(self, ticker: str, as_of: date | None = None) -> int:
        """
        Map a ticker symbol to its PERMNO.

        Args:
            ticker: Stock ticker symbol (e.g. "AAPL").
            as_of: Date for the lookup. None = most recent mapping.

        Returns:
            PERMNO integer.

        Raises:
            TickerResolutionError: If the ticker cannot be resolved.
        """
        names = self._ensure_names_loaded()
        ticker_upper = ticker.upper().strip()

        mask = names["ticker"] == ticker_upper
        if as_of is not None:
            as_of_ts = pd.Timestamp(as_of)
            mask = mask & (names["namedt"] <= as_of_ts) & (names["nameendt"] >= as_of_ts)

        matches = names[mask]

        if len(matches) == 0:
            raise TickerResolutionError(
                ticker, str(as_of) if as_of else ""
            )

        # If multiple matches, take the one with the latest namedt
        # (most recent assignment of this ticker)
        best = matches.sort_values("namedt", ascending=False).iloc[0]
        return int(best["permno"])

    def resolve_permno(self, permno: int, as_of: date | None = None) -> str:
        """
        Map a PERMNO to its ticker symbol.

        Args:
            permno: CRSP permanent security number.
            as_of: Date for the lookup. None = most recent mapping.

        Returns:
            Ticker string.

        Raises:
            TickerResolutionError: If the PERMNO cannot be resolved.
        """
        names = self._ensure_names_loaded()

        mask = names["permno"] == permno
        if as_of is not None:
            as_of_ts = pd.Timestamp(as_of)
            mask = mask & (names["namedt"] <= as_of_ts) & (names["nameendt"] >= as_of_ts)

        matches = names[mask]

        if len(matches) == 0:
            raise TickerResolutionError(
                str(permno), str(as_of) if as_of else ""
            )

        best = matches.sort_values("namedt", ascending=False).iloc[0]
        return str(best["ticker"])

    def universe(
        self,
        as_of: date | None = None,
        share_codes: tuple[int, ...] | None = (10, 11),
        exchange_codes: tuple[int, ...] | None = (1, 2, 3),
    ) -> pd.DataFrame:
        """
        Return all valid securities as of a given date.

        Args:
            as_of: Date for the universe. None = all securities ever listed.
            share_codes: Filter to these share codes. None = no filter.
            exchange_codes: Filter to these exchange codes. None = no filter.

        Returns:
            DataFrame with columns: permno, ticker, comnam, shrcd, exchcd, siccd
        """
        names = self._ensure_names_loaded()
        df = names.copy()

        if as_of is not None:
            as_of_ts = pd.Timestamp(as_of)
            df = df[(df["namedt"] <= as_of_ts) & (df["nameendt"] >= as_of_ts)]

        if share_codes is not None:
            df = df[df["shrcd"].isin(share_codes)]

        if exchange_codes is not None:
            df = df[df["exchcd"].isin(exchange_codes)]

        # Deduplicate: per PERMNO, keep the latest name record
        df = df.sort_values("namedt", ascending=False)
        df = df.drop_duplicates(subset=["permno"], keep="first")

        result = df[["permno", "ticker", "comnam", "shrcd", "exchcd", "siccd"]].copy()
        result = result.sort_values("ticker").reset_index(drop=True)

        logger.debug(f"Universe: {len(result):,} securities" +
                     (f" as of {as_of}" if as_of else ""))
        return result

    def ticker_to_permno_map(
        self,
        tickers: list[str],
        as_of: date | None = None,
    ) -> dict[str, int]:
        """
        Batch-resolve a list of tickers to PERMNOs.

        Returns a dict mapping ticker → permno. Tickers that cannot be
        resolved are omitted (with a warning).
        """
        result: dict[str, int] = {}
        unresolved: list[str] = []

        for ticker in tickers:
            try:
                result[ticker] = self.resolve_ticker(ticker, as_of)
            except TickerResolutionError:
                unresolved.append(ticker)

        if unresolved:
            logger.warning(
                f"Could not resolve {len(unresolved)} tickers: "
                f"{unresolved[:10]}{'...' if len(unresolved) > 10 else ''}"
            )

        return result

    def permno_to_ticker_map(
        self,
        permnos: list[int],
        as_of: date | None = None,
    ) -> dict[int, str]:
        """
        Batch-resolve a list of PERMNOs to tickers.

        Returns a dict mapping permno → ticker. PERMNOs that cannot be
        resolved are omitted (with a warning).
        """
        result: dict[int, str] = {}
        unresolved: list[int] = []

        for permno in permnos:
            try:
                result[permno] = self.resolve_permno(permno, as_of)
            except TickerResolutionError:
                unresolved.append(permno)

        if unresolved:
            logger.warning(
                f"Could not resolve {len(unresolved)} PERMNOs: "
                f"{unresolved[:10]}{'...' if len(unresolved) > 10 else ''}"
            )

        return result
