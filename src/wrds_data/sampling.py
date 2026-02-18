"""
Universe sampling — stratified sampling for representative stock subsets.

When the full CRSP universe has 8,000+ securities, training a model on all
of them is infeasible. This module provides stratified sampling that:

1. Pre-filters by tradability (price, volume, history)
2. Stratifies by sector × liquidity tier
3. Proportionally samples from each stratum
4. Within each stratum, uses sqrt(liquidity)-weighted sampling to favor
   liquid stocks while maintaining diversity

The max_symbols constraint (default 2500) exists because tft-finance's
cross-sectional batching has a practical upper bound on batch width.

Usage::

    from wrds_data import WRDSDataProvider, WRDSDataConfig
    from wrds_data.sampling import WRDSUniverseSampler
    from wrds_data.config import UniverseSamplingConfig

    provider = WRDSDataProvider(WRDSDataConfig())
    sampler = WRDSUniverseSampler(provider, UniverseSamplingConfig(n_symbols=2000))

    # Sample a universe as of a specific date
    symbols = sampler.sample(as_of=date(2023, 1, 1))
    # Returns: list of ~2000 ticker symbols

    # Get full summary with statistics
    summary = sampler.sample_with_stats(as_of=date(2023, 1, 1))
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd
from loguru import logger

from wrds_data.config import UniverseSamplingConfig
from wrds_data.sectors import sic_to_sector


class WRDSUniverseSampler:
    """
    Stratified universe sampler using CRSP data.

    Sampling strategy:
    1. Start with all common stocks on major exchanges (SHRCD 10,11; EXCHCD 1,2,3)
    2. Compute trailing statistics (price, volume, dollar volume) over lookback window
    3. Pre-filter by minimum price, volume, and trading history
    4. Assign each stock to a stratum (sector × liquidity tier)
    5. Proportionally sample from each stratum
    6. Within stratum, use sqrt(dollar volume) weighting for diversity

    This is the WRDS-native equivalent of tft-finance's UniverseSampler,
    but operates directly on CRSP data rather than requiring pre-computed
    symbol_stats from Alpaca.
    """

    def __init__(
        self,
        provider: "WRDSDataProvider",
        config: UniverseSamplingConfig | None = None,
    ) -> None:
        from wrds_data.provider import WRDSDataProvider
        self._provider = provider
        self._config = config or UniverseSamplingConfig()

    # ------------------------------------------------------------------
    # Point-in-time historical sampling (survivorship-bias-free)
    # ------------------------------------------------------------------

    def sample_historical(
        self,
        start: date,
        end: date,
        n_symbols: int | None = None,
        rebalance_frequency: str | None = None,
        random_seed: int | None = None,
    ) -> List[str]:
        """
        Sample a survivorship-bias-free universe across a historical period.

        Instead of sampling at a single date (which excludes stocks that were
        delisted before that date), this method samples the universe at multiple
        points throughout the training period and unions the results.

        A stock that was active in 2012 but delisted in 2015 will appear in
        the snapshots taken at/before 2015, and thus be included in the union.
        The downstream quarterly universe filter then handles per-quarter
        tradability masking.

        Args:
            start: Training period start date.
            end: Training period end date.
            n_symbols: Target symbols per snapshot. None = use config default.
                       The final union will typically be larger than this since
                       different snapshots contribute different stocks.
            rebalance_frequency: How often to snapshot. One of "quarterly",
                                 "annually", "monthly". None = use config default.
            random_seed: Override config seed for reproducibility.

        Returns:
            List of ticker symbols (sorted, deduplicated union across all
            snapshots).
        """
        freq = rebalance_frequency or self._config.rebalance_frequency
        seed = random_seed if random_seed is not None else self._config.random_seed
        n = min(n_symbols or self._config.n_symbols, self._config.max_symbols)

        # Generate rebalance dates (end of each period)
        freq_map = {"quarterly": "QE", "monthly": "ME", "annually": "YE"}
        pd_freq = freq_map.get(freq)
        if pd_freq is None:
            raise ValueError(
                f"Unknown rebalance_frequency: '{freq}'. "
                f"Valid options: {list(freq_map.keys())}"
            )

        rebalance_dates = pd.date_range(start, end, freq=pd_freq)

        # Ensure the end date itself is included (it may fall mid-period)
        end_ts = pd.Timestamp(end)
        if len(rebalance_dates) == 0 or rebalance_dates[-1] < end_ts:
            rebalance_dates = rebalance_dates.union(
                pd.DatetimeIndex([end_ts])
            ).sort_values()

        # Also include the start date to capture stocks that may delist early
        start_ts = pd.Timestamp(start)
        if rebalance_dates[0] > start_ts:
            rebalance_dates = pd.DatetimeIndex([start_ts]).union(
                rebalance_dates
            ).sort_values()

        logger.info(
            f"Point-in-time historical sampling: {len(rebalance_dates)} snapshots "
            f"({freq}) from {start} to {end}"
        )

        all_symbols: Set[str] = set()
        snapshot_counts: List[int] = []

        for i, rebalance_ts in enumerate(rebalance_dates):
            snapshot_date = rebalance_ts.date()

            # Use consistent seed per snapshot for reproducibility,
            # but vary it so each snapshot isn't identical
            snapshot_seed = seed + i

            try:
                snapshot_symbols = self._sample_impl(
                    as_of=snapshot_date,
                    n_symbols=n,
                    random_seed=snapshot_seed,
                )
            except Exception as e:
                logger.warning(
                    f"  Snapshot {i+1}/{len(rebalance_dates)} ({snapshot_date}) "
                    f"failed: {e}"
                )
                snapshot_counts.append(0)
                continue

            new_symbols = set(snapshot_symbols) - all_symbols
            all_symbols.update(snapshot_symbols)
            snapshot_counts.append(len(snapshot_symbols))

            logger.info(
                f"  [{i+1}/{len(rebalance_dates)}] as_of={snapshot_date}: "
                f"{len(snapshot_symbols)} sampled, "
                f"{len(new_symbols)} new, "
                f"{len(all_symbols)} cumulative"
            )

        logger.info(
            f"Historical sampling complete: {len(all_symbols)} unique symbols "
            f"from {len(rebalance_dates)} snapshots "
            f"(avg {np.mean(snapshot_counts):.0f} per snapshot)"
        )

        return sorted(all_symbols)

    def sample(
        self,
        as_of: date | None = None,
        n_symbols: int | None = None,
        random_seed: int | None = None,
    ) -> List[str]:
        """
        Sample a representative universe of stock tickers.

        Args:
            as_of: Reference date. Lookback statistics computed from
                   (as_of - lookback_days) to as_of. None = today.
            n_symbols: Override config.n_symbols. Capped at config.max_symbols.
            random_seed: Override config.random_seed.

        Returns:
            List of ticker symbols (sorted alphabetically).
        """
        n = min(n_symbols or self._config.n_symbols, self._config.max_symbols)
        seed = random_seed if random_seed is not None else self._config.random_seed

        result = self._sample_impl(as_of=as_of, n_symbols=n, random_seed=seed)

        logger.info(f"Sampled universe: {len(result)} symbols (target: {n})")
        return sorted(result)

    def sample_with_stats(
        self,
        as_of: date | None = None,
        n_symbols: int | None = None,
        random_seed: int | None = None,
    ) -> pd.DataFrame:
        """
        Sample universe and return full statistics.

        Returns a DataFrame with one row per sampled symbol, including:
        - symbol, permno, sector, liquidity_tier
        - avg_price, avg_volume, adv_dollars
        - n_trading_days

        This is useful for debugging sampling distributions.
        """
        n = min(n_symbols or self._config.n_symbols, self._config.max_symbols)
        seed = random_seed if random_seed is not None else self._config.random_seed

        stats = self._compute_universe_stats(as_of)
        if len(stats) == 0:
            return pd.DataFrame()

        filtered = self._apply_prefilters(stats)
        if len(filtered) == 0:
            return pd.DataFrame()

        stratified = self._assign_strata(filtered)
        sampled_tickers = self._stratified_sample(stratified, n, seed)

        return stratified[stratified["ticker"].isin(sampled_tickers)].copy()

    def get_sampling_summary(
        self,
        as_of: date | None = None,
    ) -> Dict[str, Any]:
        """
        Get a summary of the universe without sampling.

        Returns dict with:
        - total_securities: all common stocks on major exchanges
        - after_prefilter: securities passing price/volume filters
        - sector_distribution: {sector: count}
        - liquidity_distribution: {tier: count}
        """
        stats = self._compute_universe_stats(as_of)
        filtered = self._apply_prefilters(stats)
        stratified = self._assign_strata(filtered)

        return {
            "total_securities": len(stats),
            "after_prefilter": len(filtered),
            "sector_distribution": stratified["sector"].value_counts().to_dict(),
            "liquidity_distribution": stratified["liquidity_tier"].value_counts().to_dict(),
        }

    # ------------------------------------------------------------------
    # Private implementation
    # ------------------------------------------------------------------

    def _sample_impl(
        self,
        as_of: date | None,
        n_symbols: int,
        random_seed: int,
    ) -> List[str]:
        """Core sampling implementation."""
        # Step 1: Compute statistics for all eligible securities
        stats = self._compute_universe_stats(as_of)
        logger.info(f"Universe candidates: {len(stats)} securities")

        if len(stats) == 0:
            logger.warning("No eligible securities found")
            return []

        # Step 2: Apply pre-filters
        filtered = self._apply_prefilters(stats)
        logger.info(f"After pre-filters: {len(filtered)} securities")

        if len(filtered) <= n_symbols:
            logger.info(
                f"Requested {n_symbols} but only {len(filtered)} pass filters. "
                f"Returning all."
            )
            return filtered["ticker"].tolist()

        # Step 3: Assign strata (sector × liquidity tier)
        stratified = self._assign_strata(filtered)

        # Step 4: Stratified sampling
        sampled = self._stratified_sample(stratified, n_symbols, random_seed)

        return sampled

    def _compute_universe_stats(self, as_of: date | None) -> pd.DataFrame:
        """
        Compute trailing statistics for all eligible securities.

        Fetches CRSP daily data for the lookback window and computes:
        - avg_price, avg_volume, adv_dollars, n_trading_days, sector
        """
        if as_of is None:
            as_of = date.today()

        lookback_start = as_of - timedelta(days=self._config.lookback_days)

        # Get universe of common stocks on major exchanges
        universe = self._provider.universe(
            as_of=as_of,
            share_codes=(10, 11),
            exchange_codes=(1, 2, 3),
        )

        if len(universe) == 0:
            return pd.DataFrame()

        # Fetch raw daily prices for the lookback period (no corrections —
        # we just need price/volume for statistics)
        raw = self._provider.daily_prices(
            start=lookback_start,
            end=as_of,
            apply_corrections=False,
        )

        if len(raw) == 0:
            return pd.DataFrame()

        # Filter to universe PERMNOs
        universe_permnos = set(universe["permno"].values)
        raw = raw[raw["permno"].isin(universe_permnos)].copy()

        # Compute per-security statistics
        raw["abs_prc"] = raw["prc"].abs()
        raw["dollar_vol"] = raw["abs_prc"] * raw["vol"]

        stats = raw.groupby("permno").agg(
            avg_price=("abs_prc", "mean"),
            avg_volume=("vol", "mean"),
            adv_dollars=("dollar_vol", "mean"),
            n_trading_days=("date", "count"),
        ).reset_index()

        # Merge in ticker and SIC code from universe
        stats = stats.merge(
            universe[["permno", "ticker", "siccd"]],
            on="permno",
            how="left",
        )

        # Add sector classification (GICS-like, not FF12 — better coverage)
        stats["sector"] = stats["siccd"].apply(
            lambda sic: sic_to_sector(int(sic)) if pd.notna(sic) else "Unknown"
        )

        return stats

    def _apply_prefilters(self, stats: pd.DataFrame) -> pd.DataFrame:
        """Apply minimum price, volume, and history filters."""
        cfg = self._config
        mask = pd.Series(True, index=stats.index)

        # Price filter
        price_mask = stats["avg_price"] >= cfg.min_price
        n_excluded_price = (~price_mask & mask).sum()
        mask &= price_mask

        # Dollar volume filter
        dv_mask = stats["adv_dollars"] >= cfg.min_dollar_volume
        n_excluded_dv = (~dv_mask & mask).sum()
        mask &= dv_mask

        # Share volume filter
        sv_mask = stats["avg_volume"] >= cfg.min_share_volume
        n_excluded_sv = (~sv_mask & mask).sum()
        mask &= sv_mask

        # Minimum trading days
        td_mask = stats["n_trading_days"] >= cfg.min_trading_days
        n_excluded_td = (~td_mask & mask).sum()
        mask &= td_mask

        total_excluded = len(stats) - mask.sum()
        if total_excluded > 0:
            logger.debug(f"Pre-filter exclusions:")
            if n_excluded_price > 0:
                logger.debug(f"  Price < ${cfg.min_price}: {n_excluded_price}")
            if n_excluded_dv > 0:
                logger.debug(f"  Dollar volume < ${cfg.min_dollar_volume:,.0f}: {n_excluded_dv}")
            if n_excluded_sv > 0:
                logger.debug(f"  Share volume < {cfg.min_share_volume:,.0f}: {n_excluded_sv}")
            if n_excluded_td > 0:
                logger.debug(f"  Trading days < {cfg.min_trading_days}: {n_excluded_td}")

        return stats[mask].copy()

    def _assign_strata(self, stats: pd.DataFrame) -> pd.DataFrame:
        """Assign each stock to a sector × liquidity tier stratum."""
        stats = stats.copy()

        # Liquidity tiers based on dollar volume percentiles
        n_tiers = self._config.n_liquidity_tiers
        tier_labels = {1: "small", 2: "mid", 3: "large"}
        if n_tiers == 2:
            tier_labels = {1: "small", 2: "large"}

        if len(stats) >= n_tiers:
            try:
                labels = [tier_labels[i + 1] for i in range(n_tiers)]
                stats["liquidity_tier"] = pd.qcut(
                    stats["adv_dollars"],
                    q=n_tiers,
                    labels=labels,
                    duplicates="drop",
                )
            except ValueError:
                stats["liquidity_tier"] = "mid"
        else:
            stats["liquidity_tier"] = "mid"

        # Combined stratum
        stats["stratum"] = (
            stats["sector"].astype(str) + "_" +
            stats["liquidity_tier"].astype(str)
        )

        return stats

    def _stratified_sample(
        self,
        stats: pd.DataFrame,
        n_symbols: int,
        random_seed: int,
    ) -> List[str]:
        """
        Proportionally sample from each stratum.

        Within each stratum, uses probability ∝ sqrt(adv_dollars) to
        favor liquid stocks while maintaining diversity.
        """
        rng = np.random.RandomState(random_seed)

        # Calculate target samples per stratum (proportional to population)
        stratum_counts = stats["stratum"].value_counts()
        stratum_proportions = stratum_counts / len(stats)

        target_per_stratum = (stratum_proportions * n_symbols).round().astype(int)

        # Ensure at least 1 per non-empty stratum
        target_per_stratum = target_per_stratum.clip(lower=1)

        # Enforce min_symbols_per_sector
        if self._config.min_symbols_per_sector > 0:
            sector_targets = {}
            for stratum in target_per_stratum.index:
                sector = stratum.rsplit("_", 1)[0]
                sector_targets[sector] = sector_targets.get(sector, 0) + target_per_stratum[stratum]

            for sector, total in sector_targets.items():
                if total < self._config.min_symbols_per_sector:
                    # Proportionally increase this sector's strata
                    sector_strata = [s for s in target_per_stratum.index if s.rsplit("_", 1)[0] == sector]
                    deficit = self._config.min_symbols_per_sector - total
                    for stratum in sector_strata:
                        pop = stratum_counts.get(stratum, 0)
                        if pop > target_per_stratum[stratum]:
                            add = min(deficit, pop - target_per_stratum[stratum])
                            target_per_stratum[stratum] += add
                            deficit -= add
                        if deficit <= 0:
                            break

        # Adjust total to hit exact target
        total_target = target_per_stratum.sum()

        # If over, reduce from largest strata
        while total_target > n_symbols:
            largest = target_per_stratum.idxmax()
            if target_per_stratum[largest] > 1:
                target_per_stratum[largest] -= 1
                total_target -= 1
            else:
                break

        # If under, add to largest strata (that have room)
        while total_target < n_symbols:
            for stratum in target_per_stratum.sort_values(ascending=False).index:
                pop = stratum_counts.get(stratum, 0)
                if target_per_stratum[stratum] < pop:
                    target_per_stratum[stratum] += 1
                    total_target += 1
                    break
            else:
                break  # No stratum has room

        # Enforce max_symbols_per_sector if configured
        if self._config.max_symbols_per_sector is not None:
            sector_totals: Dict[str, int] = {}
            for stratum, target in target_per_stratum.items():
                sector = stratum.rsplit("_", 1)[0]
                sector_totals[sector] = sector_totals.get(sector, 0) + target

            for sector, total in sector_totals.items():
                if total > self._config.max_symbols_per_sector:
                    excess = total - self._config.max_symbols_per_sector
                    sector_strata = [
                        s for s in target_per_stratum.index
                        if s.rsplit("_", 1)[0] == sector
                    ]
                    # Reduce proportionally
                    for stratum in sorted(sector_strata,
                                          key=lambda s: target_per_stratum[s],
                                          reverse=True):
                        reduce = min(excess, target_per_stratum[stratum] - 1)
                        if reduce > 0:
                            target_per_stratum[stratum] -= reduce
                            excess -= reduce
                        if excess <= 0:
                            break

        # Sample from each stratum
        sampled: List[str] = []
        for stratum, target in target_per_stratum.items():
            stratum_stocks = stats[stats["stratum"] == stratum]
            n_available = len(stratum_stocks)

            if n_available == 0:
                continue

            n_to_sample = min(target, n_available)

            if n_to_sample >= n_available:
                chosen = stratum_stocks["ticker"].tolist()
            else:
                # Weighted sampling: prob ∝ sqrt(dollar volume)
                weights = np.sqrt(stratum_stocks["adv_dollars"].values.astype(float))
                weights = weights / weights.sum()

                indices = rng.choice(
                    len(stratum_stocks),
                    size=n_to_sample,
                    replace=False,
                    p=weights,
                )
                chosen = stratum_stocks["ticker"].iloc[indices].tolist()

            sampled.extend(chosen)

        # Log sector distribution
        sampled_set = set(sampled)
        sampled_df = stats[stats["ticker"].isin(sampled_set)]
        sector_counts_sampled = sampled_df["sector"].value_counts()
        logger.info(f"Sampled sector distribution ({len(sampled)} total):")
        for sector, count in sector_counts_sampled.items():
            pct = count / len(sampled) * 100
            logger.info(f"  {sector}: {count} ({pct:.1f}%)")

        return sampled
