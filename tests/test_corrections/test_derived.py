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

    def test_computes_op(self):
        df = pd.DataFrame({
            "revt": [800.0, 600.0],
            "cogs": [500.0, 400.0],
            "at": [1000.0, 800.0],
        })
        step = OperatingProfitabilityCalculation()
        result = step.apply(df)
        assert "op" in result.columns
        assert abs(result["op"].values[0] - 0.3) < 0.001   # (800-500)/1000
        assert abs(result["op"].values[1] - 0.25) < 0.001  # (600-400)/800

    def test_handles_missing_cogs(self):
        df = pd.DataFrame({
            "revt": [800.0],
            "cogs": [np.nan],
            "at": [1000.0],
        })
        step = OperatingProfitabilityCalculation()
        result = step.apply(df)
        # COGS fills to 0 → op = 800/1000 = 0.8
        assert abs(result["op"].values[0] - 0.8) < 0.001


class TestInvestmentRateCalculation:

    def test_computes_inv(self):
        df = pd.DataFrame({
            "capx": [80.0, 60.0],
            "at": [1000.0, 800.0],
        })
        step = InvestmentRateCalculation()
        result = step.apply(df)
        assert "inv" in result.columns
        assert abs(result["inv"].values[0] - 0.08) < 0.001  # 80/1000
        assert abs(result["inv"].values[1] - 0.075) < 0.001  # 60/800


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
            "prc": [50.0, 100.0],
            "shrout": [1000, 2000],
            "be": [100.0, 300.0],
            "revt": [800.0, 600.0],
            "cogs": [500.0, 400.0],
            "at": [1000.0, 800.0],
            "capx": [80.0, 60.0],
        })

        result = pipeline.run(df)
        assert "market_cap" in result.columns
        assert "bm" in result.columns
        assert "op" in result.columns
        assert "inv" in result.columns
