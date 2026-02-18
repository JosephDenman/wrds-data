"""Tests for derived quantity calculations."""

import numpy as np
import pandas as pd
import pytest

from wrds_data.config import DerivedConfig
from wrds_data.corrections.derived import (
    BookToMarketCalculation,
    InvestmentRateCalculation,
    MarketCapCalculation,
    OperatingProfitabilityCalculation,
    build_derived_pipeline,
)


class TestMarketCapCalculation:

    def test_computes_market_cap(self):
        df = pd.DataFrame({
            "prc": [50.0, -100.0, 25.0],  # -100 = bid-ask midpoint
            "shrout": [1000, 2000, 500],
        })
        step = MarketCapCalculation()
        result = step.apply(df)
        assert "market_cap" in result.columns
        # |50| * 1000 = 50000, |100| * 2000 = 200000, |25| * 500 = 12500
        expected = [50000, 200000, 12500]
        np.testing.assert_array_almost_equal(result["market_cap"].values, expected)


class TestBookToMarketCalculation:

    def test_computes_bm(self):
        df = pd.DataFrame({
            "be": [100.0, 200.0, -50.0, np.nan],
            "market_cap": [500.0, 400.0, 300.0, 200.0],
        })
        step = BookToMarketCalculation()
        result = step.apply(df)
        assert "bm" in result.columns
        assert abs(result["bm"].values[0] - 0.2) < 0.001  # 100/500
        assert abs(result["bm"].values[1] - 0.5) < 0.001  # 200/400
        assert np.isnan(result["bm"].values[2])  # Negative BE → NaN
        assert np.isnan(result["bm"].values[3])  # NaN BE → NaN


class TestOperatingProfitabilityCalculation:

    def test_computes_op_with_sale(self):
        """OP = (sale - cogs - xsga - xint) / be (tidyfinance/FF5 formula)."""
        df = pd.DataFrame({
            "sale": [800.0, 600.0],
            "cogs": [500.0, 400.0],
            "xsga": [100.0, 80.0],
            "xint": [30.0, 20.0],
            "be": [465.0, 430.0],
        })
        step = OperatingProfitabilityCalculation()
        result = step.apply(df)
        assert "op" in result.columns
        # (800 - 500 - 100 - 30) / 465 = 170 / 465 ≈ 0.3656
        assert abs(result["op"].values[0] - (170.0 / 465.0)) < 0.001
        # (600 - 400 - 80 - 20) / 430 = 100 / 430 ≈ 0.2326
        assert abs(result["op"].values[1] - (100.0 / 430.0)) < 0.001

    def test_falls_back_to_revt(self):
        """When sale is missing, fall back to revt."""
        df = pd.DataFrame({
            "revt": [800.0],
            "cogs": [500.0],
            "xsga": [100.0],
            "xint": [30.0],
            "be": [465.0],
        })
        step = OperatingProfitabilityCalculation()
        result = step.apply(df)
        # (800 - 500 - 100 - 30) / 465 = 170 / 465
        assert abs(result["op"].values[0] - (170.0 / 465.0)) < 0.001

    def test_handles_missing_cost_components(self):
        """Missing cogs/xsga/xint should fill to 0."""
        df = pd.DataFrame({
            "sale": [800.0],
            "cogs": [np.nan],
            "xsga": [np.nan],
            "xint": [np.nan],
            "be": [465.0],
        })
        step = OperatingProfitabilityCalculation()
        result = step.apply(df)
        # (800 - 0 - 0 - 0) / 465 = 800 / 465
        assert abs(result["op"].values[0] - (800.0 / 465.0)) < 0.001

    def test_negative_be_produces_nan(self):
        """Negative book equity should produce NaN."""
        df = pd.DataFrame({
            "sale": [800.0],
            "cogs": [500.0],
            "xsga": [100.0],
            "xint": [30.0],
            "be": [-100.0],
        })
        step = OperatingProfitabilityCalculation()
        result = step.apply(df)
        assert np.isnan(result["op"].values[0])


class TestInvestmentRateCalculation:

    def test_computes_inv_as_asset_growth(self):
        """INV = at / at_lag - 1 (asset growth, tidyfinance/FF5 formula)."""
        df = pd.DataFrame({
            "gvkey": ["001", "001", "001"],
            "fyear": [2018, 2019, 2020],
            "at": [900.0, 1000.0, 1100.0],
        })
        step = InvestmentRateCalculation()
        result = step.apply(df)
        assert "inv" in result.columns
        # First year: NaN (no lag available)
        assert np.isnan(result["inv"].values[0])
        # 2019: 1000/900 - 1 ≈ 0.1111
        assert abs(result["inv"].values[1] - (1000.0 / 900.0 - 1)) < 0.001
        # 2020: 1100/1000 - 1 = 0.1
        assert abs(result["inv"].values[2] - 0.1) < 0.001

    def test_separate_firms_no_cross_lag(self):
        """Lagging should be within-firm only, not cross-firm."""
        df = pd.DataFrame({
            "gvkey": ["001", "001", "002", "002"],
            "fyear": [2019, 2020, 2019, 2020],
            "at": [1000.0, 1100.0, 500.0, 600.0],
        })
        step = InvestmentRateCalculation()
        result = step.apply(df)
        # Firm 001: first year NaN, second year 1100/1000 - 1 = 0.1
        firm1 = result[result["gvkey"] == "001"].sort_values("fyear")
        assert np.isnan(firm1["inv"].values[0])
        assert abs(firm1["inv"].values[1] - 0.1) < 0.001
        # Firm 002: first year NaN, second year 600/500 - 1 = 0.2
        firm2 = result[result["gvkey"] == "002"].sort_values("fyear")
        assert np.isnan(firm2["inv"].values[0])
        assert abs(firm2["inv"].values[1] - 0.2) < 0.001


class TestBuildDerivedPipeline:

    def test_builds_all(self):
        config = DerivedConfig()
        steps = build_derived_pipeline(config)
        assert len(steps) == 4

    def test_pipeline_runs(self):
        config = DerivedConfig()
        steps = build_derived_pipeline(config)

        from wrds_data.corrections.base import CorrectionPipeline
        pipeline = CorrectionPipeline(steps)

        df = pd.DataFrame({
            "gvkey": ["001", "001"],
            "fyear": [2019, 2020],
            "prc": [50.0, 100.0],
            "shrout": [1000, 2000],
            "be": [100.0, 300.0],
            "sale": [800.0, 600.0],
            "cogs": [500.0, 400.0],
            "xsga": [100.0, 80.0],
            "xint": [30.0, 20.0],
            "at": [1000.0, 1100.0],
        })

        result = pipeline.run(df)
        assert "market_cap" in result.columns
        assert "bm" in result.columns
        assert "op" in result.columns
        assert "inv" in result.columns
