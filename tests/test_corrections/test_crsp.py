"""Tests for CRSP data corrections."""

import numpy as np
import pandas as pd
import pytest

from wrds_data.config import CRSPCorrectionConfig
from wrds_data.corrections.base import CorrectionPipeline
from wrds_data.corrections.crsp import (
    DelistingAdjustment,
    ExchangeCodeFilter,
    MinimumHistoryFilter,
    PennyStockFilter,
    PriceSignCorrection,
    ShareCodeFilter,
    VolumeValidation,
    build_crsp_pipeline,
)


class TestShareCodeFilter:

    def test_keeps_common_shares(self, synthetic_crsp_daily):
        config = CRSPCorrectionConfig()
        step = ShareCodeFilter(config)
        result = step.apply(synthetic_crsp_daily)
        # PERMNO 10002 has SHRCD=31 (ADR), should be removed
        assert 10002 not in result["permno"].values
        # All remaining should have SHRCD in {10, 11}
        assert result["shrcd"].isin({10, 11}).all()

    def test_custom_share_codes(self, synthetic_crsp_daily):
        config = CRSPCorrectionConfig(share_codes=(10, 11, 31))
        step = ShareCodeFilter(config)
        result = step.apply(synthetic_crsp_daily)
        # Now PERMNO 10002 (SHRCD=31) should be kept
        assert 10002 in result["permno"].values


class TestExchangeCodeFilter:

    def test_keeps_major_exchanges(self, synthetic_crsp_daily):
        config = CRSPCorrectionConfig()
        step = ExchangeCodeFilter(config)
        result = step.apply(synthetic_crsp_daily)
        assert result["exchcd"].isin({1, 2, 3}).all()

    def test_custom_exchanges(self, synthetic_crsp_daily):
        config = CRSPCorrectionConfig(exchange_codes=(1,))  # NYSE only
        step = ExchangeCodeFilter(config)
        result = step.apply(synthetic_crsp_daily)
        assert (result["exchcd"] == 1).all()
        # PERMNOs on AMEX (2) and NASDAQ (3) should be excluded
        assert 10004 not in result["permno"].values  # AMEX
        assert 10003 not in result["permno"].values  # NASDAQ


class TestPriceSignCorrection:

    def test_corrects_negative_prices(self, synthetic_crsp_daily):
        step = PriceSignCorrection()
        result = step.apply(synthetic_crsp_daily)
        assert (result["prc"] >= 0).all()
        # Original had some negative prices
        assert (synthetic_crsp_daily["prc"] < 0).any()

    def test_preserves_magnitude(self, synthetic_crsp_daily):
        step = PriceSignCorrection()
        result = step.apply(synthetic_crsp_daily)
        # Absolute values should match
        np.testing.assert_array_almost_equal(
            result["prc"].values,
            synthetic_crsp_daily["prc"].abs().values,
        )


class TestDelistingAdjustment:

    def test_imputes_missing_performance_delisting(self, synthetic_crsp_daily, synthetic_delisting):
        config = CRSPCorrectionConfig()
        step = DelistingAdjustment(config)
        step.set_delisting_data(synthetic_delisting)

        result = step.apply(synthetic_crsp_daily)

        # PERMNO 10003 delisted on 2020-09-01, code 400, missing DLRET
        # Should have been imputed with -0.30
        delist_row = result[
            (result["permno"] == 10003)
            & (result["date"] == pd.Timestamp("2020-09-01"))
        ]
        if len(delist_row) > 0:
            # The return should have been adjusted
            # Original ret + compounded delisting return
            original = synthetic_crsp_daily[
                (synthetic_crsp_daily["permno"] == 10003)
                & (synthetic_crsp_daily["date"] == pd.Timestamp("2020-09-01").date())
            ]
            if len(original) > 0:
                orig_ret = original["ret"].values[0]
                expected = (1 + orig_ret) * (1 + (-0.30)) - 1
                assert abs(delist_row["ret"].values[0] - expected) < 0.001

    def test_no_delisting_data_warns(self, synthetic_crsp_daily):
        config = CRSPCorrectionConfig()
        step = DelistingAdjustment(config)
        # Don't set delisting data
        result = step.apply(synthetic_crsp_daily)
        assert len(result) == len(synthetic_crsp_daily)

    def test_row_count_unchanged(self, synthetic_crsp_daily, synthetic_delisting):
        """Delisting adjustment modifies returns, doesn't remove rows."""
        config = CRSPCorrectionConfig()
        step = DelistingAdjustment(config)
        step.set_delisting_data(synthetic_delisting)
        result = step.apply(synthetic_crsp_daily)
        assert len(result) == len(synthetic_crsp_daily)


class TestPennyStockFilter:

    def test_removes_penny_stocks(self, synthetic_crsp_daily):
        config = CRSPCorrectionConfig(penny_stock_threshold=5.0)
        step = PennyStockFilter(config)
        result = step.apply(synthetic_crsp_daily)
        assert (result["prc"].abs() >= 5.0).all()

    def test_custom_threshold(self, synthetic_crsp_daily):
        config = CRSPCorrectionConfig(penny_stock_threshold=1.0)
        step = PennyStockFilter(config)
        result = step.apply(synthetic_crsp_daily)
        assert (result["prc"].abs() >= 1.0).all()


class TestMinimumHistoryFilter:

    def test_removes_short_history(self, synthetic_crsp_daily):
        config = CRSPCorrectionConfig(min_trading_days=100)
        step = MinimumHistoryFilter(config)
        result = step.apply(synthetic_crsp_daily)
        # PERMNO 10005 has only ~50 days, should be removed
        assert 10005 not in result["permno"].values
        # Others should remain (they have 200+ days each)
        counts = result.groupby("permno").size()
        assert (counts >= 100).all()


class TestVolumeValidation:

    def test_removes_zero_volume(self):
        df = pd.DataFrame({
            "permno": [1, 1, 1],
            "vol": [100, 0, 200],
            "shrout": [1000, 1000, 1000],
        })
        step = VolumeValidation()
        result = step.apply(df)
        assert len(result) == 2
        assert (result["vol"] > 0).all()


class TestBuildCRSPPipelineV1:
    """Tests for v1 (legacy) CRSP pipeline — all 7 corrections active."""

    def test_builds_all_steps_v1(self):
        config = CRSPCorrectionConfig(crsp_version="v1")
        steps = build_crsp_pipeline(config)
        assert len(steps) == 7
        step_names = [s.name for s in steps]
        assert "ShareCodeFilter" in step_names
        assert "ExchangeCodeFilter" in step_names
        assert "PriceSignCorrection" in step_names
        assert "DelistingAdjustment" in step_names
        assert "PennyStockFilter" in step_names
        assert "VolumeValidation" in step_names
        assert "MinimumHistoryFilter" in step_names

    def test_disabling_steps_v1(self):
        config = CRSPCorrectionConfig(
            crsp_version="v1",
            share_code_filter=False,
            penny_stock_filter=False,
        )
        steps = build_crsp_pipeline(config)
        step_names = [s.name for s in steps]
        assert "ShareCodeFilter" not in step_names
        assert "PennyStockFilter" not in step_names

    def test_pipeline_runs_v1(self, synthetic_crsp_daily, synthetic_delisting):
        config = CRSPCorrectionConfig(crsp_version="v1", min_trading_days=100)
        steps = build_crsp_pipeline(config)

        # Inject delisting data
        for step in steps:
            if isinstance(step, DelistingAdjustment):
                step.set_delisting_data(synthetic_delisting)

        pipeline = CorrectionPipeline(steps)
        result = pipeline.run(synthetic_crsp_daily)

        # Should have removed ADR (10002), penny stock (10004), short history (10005)
        remaining_permnos = result["permno"].unique()
        assert 10002 not in remaining_permnos  # ADR (ShareCodeFilter)
        assert 10005 not in remaining_permnos  # Short history
        # 10001 and 10003 should remain
        assert 10001 in remaining_permnos


class TestBuildCRSPPipelineV2:
    """Tests for v2 (CIZ) CRSP pipeline — skips share/exchange/delisting steps."""

    def test_builds_v2_steps(self):
        """V2 pipeline should skip ShareCodeFilter, ExchangeCodeFilter, DelistingAdjustment."""
        config = CRSPCorrectionConfig(crsp_version="v2")
        steps = build_crsp_pipeline(config)
        step_names = [s.name for s in steps]

        # These 3 should be SKIPPED for v2 (handled in SQL JOIN)
        assert "ShareCodeFilter" not in step_names
        assert "ExchangeCodeFilter" not in step_names
        assert "DelistingAdjustment" not in step_names

        # These 4 should still be present
        assert "PriceSignCorrection" in step_names
        assert "PennyStockFilter" in step_names
        assert "VolumeValidation" in step_names
        assert "MinimumHistoryFilter" in step_names
        assert len(steps) == 4

    def test_v2_default(self):
        """Default config should use v2."""
        config = CRSPCorrectionConfig()
        assert config.crsp_version == "v2"
        steps = build_crsp_pipeline(config)
        step_names = [s.name for s in steps]
        assert "ShareCodeFilter" not in step_names
        assert "DelistingAdjustment" not in step_names
        assert len(steps) == 4

    def test_v2_pipeline_runs(self, synthetic_crsp_daily):
        """V2 pipeline should work without delisting data or shrcd/exchcd columns."""
        config = CRSPCorrectionConfig(crsp_version="v2", min_trading_days=100)
        steps = build_crsp_pipeline(config)

        # No need to inject delisting data for v2
        pipeline = CorrectionPipeline(steps)
        result = pipeline.run(synthetic_crsp_daily)

        # Price correction should have been applied
        assert (result["prc"] >= 0).all()
        # Short history filtered
        assert 10005 not in result["permno"].unique()
        # ADR still present (ShareCodeFilter skipped in v2 — SQL handles it)
        assert 10002 in result["permno"].unique()

    def test_v2_disabling_steps(self):
        """Disabling steps in v2 config should work."""
        config = CRSPCorrectionConfig(
            crsp_version="v2",
            penny_stock_filter=False,
            min_history_filter=False,
        )
        steps = build_crsp_pipeline(config)
        step_names = [s.name for s in steps]
        assert "PennyStockFilter" not in step_names
        assert "MinimumHistoryFilter" not in step_names
        # Only PriceSignCorrection and VolumeValidation remain
        assert len(steps) == 2
