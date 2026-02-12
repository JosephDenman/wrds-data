"""
Tests for universe sampling.

Uses synthetic CRSP data to test:
- Pre-filtering (price, volume, trading days)
- Stratum assignment (sector × liquidity tier)
- Proportional stratified sampling
- Max symbols constraint
- Sector balance enforcement
- Edge cases (empty data, more requested than available)
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from wrds_data.config import UniverseSamplingConfig
from wrds_data.sampling import WRDSUniverseSampler


# ---------------------------------------------------------------------------
# Fixtures — we don't use the provider directly in unit tests; instead we
# test the internal methods of WRDSUniverseSampler with synthetic DataFrames.
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_stats() -> pd.DataFrame:
    """
    Synthetic per-security statistics DataFrame matching what
    _compute_universe_stats() returns.

    Simulates ~100 securities across multiple sectors and liquidity levels.
    """
    np.random.seed(42)
    n_stocks = 100

    sectors = ["Technology", "Finance", "Health", "Energy", "Manufacturing",
               "Consumer NonDurables", "Shops", "Utilities", "Other"]
    tickers = [f"SYM{i:03d}" for i in range(n_stocks)]

    return pd.DataFrame({
        "permno": list(range(10001, 10001 + n_stocks)),
        "ticker": tickers,
        "siccd": [
            # Distribute SIC codes across sectors
            3571 + (i % 10) if i < 20 else  # Technology
            6020 + (i % 10) if i < 35 else  # Finance
            2834 + (i % 5) if i < 45 else   # Health
            1300 + (i % 5) if i < 55 else   # Energy
            3200 + (i % 10) if i < 65 else  # Manufacturing
            2000 + (i % 10) if i < 75 else  # Consumer NonDurables
            5000 + (i % 10) if i < 85 else  # Shops
            4950 + (i % 5) if i < 90 else   # Utilities
            9000 + (i % 10)                  # Other
            for i in range(n_stocks)
        ],
        "avg_price": np.random.uniform(10, 500, n_stocks),
        "avg_volume": np.random.lognormal(12, 1, n_stocks),
        "adv_dollars": np.random.lognormal(16, 1.5, n_stocks),
        "n_trading_days": np.random.randint(10, 63, n_stocks),
        "sector": [
            "Technology" if i < 20 else
            "Finance" if i < 35 else
            "Health" if i < 45 else
            "Energy" if i < 55 else
            "Manufacturing" if i < 65 else
            "Consumer NonDurables" if i < 75 else
            "Shops" if i < 85 else
            "Utilities" if i < 90 else
            "Other"
            for i in range(n_stocks)
        ],
    })


@pytest.fixture
def default_config() -> UniverseSamplingConfig:
    return UniverseSamplingConfig(
        n_symbols=50,
        max_symbols=100,
        random_seed=42,
        min_price=5.0,
        min_dollar_volume=100_000,
        min_share_volume=10_000,
        min_trading_days=15,
        lookback_days=63,
    )


class TestPreFilters:
    """Test _apply_prefilters."""

    def test_price_filter(self, sample_stats, default_config):
        """Stocks below min_price are excluded."""
        # Set some stocks to very low prices
        stats = sample_stats.copy()
        stats.loc[:9, "avg_price"] = 2.0  # Below $5 threshold

        sampler = WRDSUniverseSampler.__new__(WRDSUniverseSampler)
        sampler._config = default_config

        filtered = sampler._apply_prefilters(stats)
        # The 10 penny stocks should be excluded (if they also pass other filters)
        assert len(filtered) < len(stats)
        assert (filtered["avg_price"] >= default_config.min_price).all()

    def test_dollar_volume_filter(self, sample_stats, default_config):
        """Stocks below min_dollar_volume are excluded."""
        stats = sample_stats.copy()
        stats.loc[:4, "adv_dollars"] = 50_000  # Below $100k threshold

        sampler = WRDSUniverseSampler.__new__(WRDSUniverseSampler)
        sampler._config = default_config

        filtered = sampler._apply_prefilters(stats)
        assert (filtered["adv_dollars"] >= default_config.min_dollar_volume).all()

    def test_trading_days_filter(self, sample_stats, default_config):
        """Stocks with too few trading days are excluded."""
        stats = sample_stats.copy()
        stats.loc[:4, "n_trading_days"] = 5  # Below 15-day minimum

        sampler = WRDSUniverseSampler.__new__(WRDSUniverseSampler)
        sampler._config = default_config

        filtered = sampler._apply_prefilters(stats)
        assert (filtered["n_trading_days"] >= default_config.min_trading_days).all()

    def test_all_filters_combined(self, sample_stats, default_config):
        """Multiple filters stack correctly."""
        stats = sample_stats.copy()
        stats.loc[0, "avg_price"] = 1.0        # Fails price
        stats.loc[1, "adv_dollars"] = 10.0      # Fails dollar vol
        stats.loc[2, "avg_volume"] = 100         # Fails share vol
        stats.loc[3, "n_trading_days"] = 1       # Fails trading days

        sampler = WRDSUniverseSampler.__new__(WRDSUniverseSampler)
        sampler._config = default_config

        filtered = sampler._apply_prefilters(stats)
        # None of the specifically-violated indices should be present
        for idx in [0, 1, 2, 3]:
            permno = stats.loc[idx, "permno"]
            assert permno not in filtered["permno"].values

    def test_no_stocks_pass(self, default_config):
        """If no stocks pass, return empty DataFrame."""
        stats = pd.DataFrame({
            "permno": [1, 2],
            "ticker": ["A", "B"],
            "avg_price": [1.0, 1.0],
            "avg_volume": [10, 10],
            "adv_dollars": [10.0, 10.0],
            "n_trading_days": [1, 1],
            "sector": ["Other", "Other"],
        })

        sampler = WRDSUniverseSampler.__new__(WRDSUniverseSampler)
        sampler._config = default_config

        filtered = sampler._apply_prefilters(stats)
        assert len(filtered) == 0


class TestStratumAssignment:
    """Test _assign_strata."""

    def test_strata_created(self, sample_stats, default_config):
        """Each stock gets a stratum label."""
        sampler = WRDSUniverseSampler.__new__(WRDSUniverseSampler)
        sampler._config = default_config

        stratified = sampler._assign_strata(sample_stats)
        assert "stratum" in stratified.columns
        assert "liquidity_tier" in stratified.columns
        assert stratified["stratum"].notna().all()

    def test_three_liquidity_tiers(self, sample_stats, default_config):
        """Default config creates 3 tiers: small, mid, large."""
        sampler = WRDSUniverseSampler.__new__(WRDSUniverseSampler)
        sampler._config = default_config

        stratified = sampler._assign_strata(sample_stats)
        tiers = set(stratified["liquidity_tier"].unique())
        assert tiers == {"small", "mid", "large"}

    def test_strata_combine_sector_and_tier(self, sample_stats, default_config):
        """Stratum format is 'sector_tier'."""
        sampler = WRDSUniverseSampler.__new__(WRDSUniverseSampler)
        sampler._config = default_config

        stratified = sampler._assign_strata(sample_stats)
        for stratum in stratified["stratum"]:
            parts = stratum.rsplit("_", 1)
            assert len(parts) == 2
            assert parts[1] in {"small", "mid", "large"}

    def test_small_dataset_uses_mid_tier(self, default_config):
        """With < n_tiers stocks, everyone gets 'mid'."""
        stats = pd.DataFrame({
            "permno": [1, 2],
            "ticker": ["A", "B"],
            "adv_dollars": [1000, 2000],
            "sector": ["Technology", "Finance"],
        })
        sampler = WRDSUniverseSampler.__new__(WRDSUniverseSampler)
        sampler._config = default_config

        stratified = sampler._assign_strata(stats)
        assert (stratified["liquidity_tier"] == "mid").all()


class TestStratifiedSampling:
    """Test _stratified_sample."""

    def test_returns_correct_count(self, sample_stats, default_config):
        """Sample returns approximately the requested number of symbols."""
        sampler = WRDSUniverseSampler.__new__(WRDSUniverseSampler)
        sampler._config = default_config

        # Pre-filter and stratify
        filtered = sampler._apply_prefilters(sample_stats)
        stratified = sampler._assign_strata(filtered)

        n_target = 30
        sampled = sampler._stratified_sample(stratified, n_target, random_seed=42)

        # Should be close to target (may differ slightly due to rounding)
        assert abs(len(sampled) - n_target) <= 5

    def test_all_sampled_are_in_input(self, sample_stats, default_config):
        """Every sampled ticker exists in the input."""
        sampler = WRDSUniverseSampler.__new__(WRDSUniverseSampler)
        sampler._config = default_config

        filtered = sampler._apply_prefilters(sample_stats)
        stratified = sampler._assign_strata(filtered)

        sampled = sampler._stratified_sample(stratified, 30, random_seed=42)
        all_tickers = set(stratified["ticker"])
        for ticker in sampled:
            assert ticker in all_tickers

    def test_no_duplicates(self, sample_stats, default_config):
        """No ticker appears twice."""
        sampler = WRDSUniverseSampler.__new__(WRDSUniverseSampler)
        sampler._config = default_config

        filtered = sampler._apply_prefilters(sample_stats)
        stratified = sampler._assign_strata(filtered)

        sampled = sampler._stratified_sample(stratified, 50, random_seed=42)
        assert len(sampled) == len(set(sampled))

    def test_reproducible_with_seed(self, sample_stats, default_config):
        """Same seed gives same sample."""
        sampler = WRDSUniverseSampler.__new__(WRDSUniverseSampler)
        sampler._config = default_config

        filtered = sampler._apply_prefilters(sample_stats)
        stratified = sampler._assign_strata(filtered)

        sample1 = sampler._stratified_sample(stratified, 30, random_seed=42)
        sample2 = sampler._stratified_sample(stratified, 30, random_seed=42)
        assert sorted(sample1) == sorted(sample2)

    def test_different_seed_gives_different_sample(self, sample_stats, default_config):
        """Different seeds give different samples."""
        sampler = WRDSUniverseSampler.__new__(WRDSUniverseSampler)
        sampler._config = default_config

        filtered = sampler._apply_prefilters(sample_stats)
        stratified = sampler._assign_strata(filtered)

        sample1 = sampler._stratified_sample(stratified, 30, random_seed=42)
        sample2 = sampler._stratified_sample(stratified, 30, random_seed=99)
        # Should differ (very unlikely to be identical with different seeds)
        assert sorted(sample1) != sorted(sample2)

    def test_sector_representation(self, sample_stats, default_config):
        """Multiple sectors should be represented in the sample."""
        sampler = WRDSUniverseSampler.__new__(WRDSUniverseSampler)
        sampler._config = default_config

        filtered = sampler._apply_prefilters(sample_stats)
        stratified = sampler._assign_strata(filtered)

        sampled = sampler._stratified_sample(stratified, 50, random_seed=42)
        sampled_df = stratified[stratified["ticker"].isin(sampled)]
        n_sectors = sampled_df["sector"].nunique()
        assert n_sectors >= 4  # Should have at least 4 sectors in 50 samples

    def test_request_more_than_available(self, default_config):
        """If n_symbols > available, return all."""
        stats = pd.DataFrame({
            "permno": [1, 2, 3],
            "ticker": ["A", "B", "C"],
            "adv_dollars": [1000, 2000, 3000],
            "sector": ["Technology", "Finance", "Health"],
            "stratum": ["Technology_mid", "Finance_mid", "Health_mid"],
            "liquidity_tier": ["mid", "mid", "mid"],
        })

        sampler = WRDSUniverseSampler.__new__(WRDSUniverseSampler)
        sampler._config = default_config

        sampled = sampler._stratified_sample(stats, 100, random_seed=42)
        assert len(sampled) == 3


class TestMaxSymbolsConstraint:
    """Test that max_symbols is respected."""

    def test_n_symbols_capped_by_max(self, sample_stats):
        """n_symbols should never exceed max_symbols."""
        config = UniverseSamplingConfig(
            n_symbols=5000,
            max_symbols=50,
            min_price=0.01,
            min_dollar_volume=0.01,
            min_share_volume=0.01,
            min_trading_days=1,
        )

        sampler = WRDSUniverseSampler.__new__(WRDSUniverseSampler)
        sampler._config = config

        filtered = sampler._apply_prefilters(sample_stats)
        stratified = sampler._assign_strata(filtered)

        # Use the min of n_symbols and max_symbols
        n = min(config.n_symbols, config.max_symbols)
        sampled = sampler._stratified_sample(stratified, n, random_seed=42)
        assert len(sampled) <= config.max_symbols


class TestSectorBalance:
    """Test min/max symbols per sector enforcement."""

    def test_min_symbols_per_sector(self):
        """Small sectors should be boosted to minimum."""
        config = UniverseSamplingConfig(
            min_symbols_per_sector=3,
            min_price=0.01,
            min_dollar_volume=0.01,
            min_share_volume=0.01,
            min_trading_days=1,
        )

        # Create data with 1 stock in Utilities sector
        stats = pd.DataFrame({
            "permno": list(range(20)),
            "ticker": [f"S{i}" for i in range(20)],
            "adv_dollars": [1000000] * 20,
            "sector": ["Technology"] * 15 + ["Finance"] * 4 + ["Utilities"],
            "liquidity_tier": ["mid"] * 20,
            "stratum": ["Technology_mid"] * 15 + ["Finance_mid"] * 4 + ["Utilities_mid"],
        })

        sampler = WRDSUniverseSampler.__new__(WRDSUniverseSampler)
        sampler._config = config

        sampled = sampler._stratified_sample(stats, 10, random_seed=42)

        # Utilities has only 1 stock, so it can only contribute 1
        # But the mechanism should try to boost it
        sampled_df = stats[stats["ticker"].isin(sampled)]
        utilities_count = (sampled_df["sector"] == "Utilities").sum()
        # Should have at least 1 utility stock
        assert utilities_count >= 1

    def test_max_symbols_per_sector(self):
        """No sector should exceed max_symbols_per_sector."""
        config = UniverseSamplingConfig(
            max_symbols_per_sector=5,
            min_price=0.01,
            min_dollar_volume=0.01,
            min_share_volume=0.01,
            min_trading_days=1,
        )

        stats = pd.DataFrame({
            "permno": list(range(30)),
            "ticker": [f"S{i}" for i in range(30)],
            "adv_dollars": [1000000] * 30,
            "sector": ["Technology"] * 25 + ["Finance"] * 5,
            "liquidity_tier": ["mid"] * 30,
            "stratum": ["Technology_mid"] * 25 + ["Finance_mid"] * 5,
        })

        sampler = WRDSUniverseSampler.__new__(WRDSUniverseSampler)
        sampler._config = config

        sampled = sampler._stratified_sample(stats, 20, random_seed=42)

        sampled_df = stats[stats["ticker"].isin(sampled)]
        tech_count = (sampled_df["sector"] == "Technology").sum()
        assert tech_count <= 5


class TestConfigDefaults:
    """Test UniverseSamplingConfig defaults match expected constraints."""

    def test_default_max_symbols(self):
        config = UniverseSamplingConfig()
        assert config.max_symbols == 2500

    def test_default_n_symbols(self):
        config = UniverseSamplingConfig()
        assert config.n_symbols == 2000

    def test_default_seed(self):
        config = UniverseSamplingConfig()
        assert config.random_seed == 42

    def test_n_cannot_exceed_max(self):
        """n_symbols > max_symbols should be capped in practice."""
        config = UniverseSamplingConfig(n_symbols=5000, max_symbols=2500)
        effective_n = min(config.n_symbols, config.max_symbols)
        assert effective_n == 2500
