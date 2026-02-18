"""Tests for Compustat data corrections."""

import numpy as np
import pandas as pd
import pytest

from wrds_data.config import CompustatCorrectionConfig
from wrds_data.corrections.base import CorrectionPipeline
from wrds_data.corrections.compustat import (
    BookEquityCalculation,
    CurrencyFilter,
    DuplicateRemoval,
    IndustryExclusionFilter,
    PointInTimeAlignment,
    StandardFilter,
    build_compustat_pipeline,
)


class TestStandardFilter:

    def test_keeps_standard_only(self, synthetic_compustat):
        step = StandardFilter()
        result = step.apply(synthetic_compustat)
        assert (result["datafmt"] == "STD").all()
        assert (result["popsrc"] == "D").all()
        assert (result["consol"] == "C").all()
        assert (result["indfmt"] == "INDL").all()

    def test_filters_non_standard(self):
        df = pd.DataFrame([
            {"datafmt": "STD", "popsrc": "D", "consol": "C", "indfmt": "INDL"},
            {"datafmt": "SUMM_STD", "popsrc": "D", "consol": "C", "indfmt": "INDL"},
        ])
        step = StandardFilter()
        result = step.apply(df)
        assert len(result) == 1


class TestCurrencyFilter:

    def test_keeps_usd_only(self, synthetic_compustat):
        step = CurrencyFilter()
        result = step.apply(synthetic_compustat)
        assert (result["curcd"] == "USD").all()
        # GVKEY 004 (GBP) should be removed
        assert "004" not in result["gvkey"].values


class TestIndustryExclusionFilter:

    def test_excludes_financials(self, synthetic_compustat):
        config = CompustatCorrectionConfig()
        step = IndustryExclusionFilter(config)
        result = step.apply(synthetic_compustat)
        # GVKEY 002 (SIC 6020 = finance) should be removed
        assert "002" not in result["gvkey"].values
        # Others should remain
        assert "001" in result["gvkey"].values
        assert "003" in result["gvkey"].values

    def test_custom_exclusion_ranges(self, synthetic_compustat):
        config = CompustatCorrectionConfig(
            excluded_sic_ranges=[(6000, 6999), (4900, 4999)]
        )
        step = IndustryExclusionFilter(config)
        result = step.apply(synthetic_compustat)
        assert "002" not in result["gvkey"].values  # Financial


class TestBookEquityCalculation:

    def test_computes_book_equity_with_seq(self, synthetic_compustat):
        step = BookEquityCalculation()
        result = step.apply(synthetic_compustat)
        assert "be" in result.columns

        # GVKEY 001, fyear 2019: SEQ=500, TXDITC=20, PSTKRV=55
        # BE = 500 + 20 - 55 = 465
        row = result[(result["gvkey"] == "001") & (result["fyear"] == 2019)]
        assert len(row) == 1
        assert abs(row["be"].values[0] - 465.0) < 0.01

    def test_fallback_to_ceq_pstk_and_txdb_itcb(self, synthetic_compustat):
        step = BookEquityCalculation()
        result = step.apply(synthetic_compustat)

        # GVKEY 003: SEQ=NaN, CEQ=420, PSTK=30 → SE fallback = 420 + 30 = 450
        # PSTKRV=NaN, PSTKL=35 → PS = 35
        # TXDITC=NaN → fallback to TXDB(10) + ITCB(5) = 15
        # BE = 450 + 15 - 35 = 430
        row = result[(result["gvkey"] == "003") & (result["fyear"] == 2019)]
        assert len(row) == 1
        assert abs(row["be"].values[0] - 430.0) < 0.01

    def test_no_crash_on_all_nan(self):
        df = pd.DataFrame([{
            "gvkey": "999", "datadate": "2020-12-31",
            "at": np.nan, "lt": np.nan, "seq": np.nan,
            "ceq": np.nan, "pstk": np.nan,
        }])
        step = BookEquityCalculation()
        result = step.apply(df)
        assert "be" in result.columns
        assert result["be"].isna().all()


class TestPointInTimeAlignment:

    def test_uses_rdq_when_available(self, synthetic_compustat):
        config = CompustatCorrectionConfig(pit_lag_days=1)
        step = PointInTimeAlignment(config)
        result = step.apply(synthetic_compustat)
        assert "public_date" in result.columns

        # GVKEY 001, 2019: rdq = 2020-02-15, lag=1 → public_date = 2020-02-16
        row = result[(result["gvkey"] == "001") & (result["fyear"] == 2019)]
        assert row["public_date"].values[0] == pd.Timestamp("2020-02-16")

    def test_fallback_when_rdq_missing(self):
        df = pd.DataFrame([{
            "gvkey": "001", "datadate": "2020-12-31", "rdq": pd.NaT,
        }])
        config = CompustatCorrectionConfig(pit_fallback_days=180)
        step = PointInTimeAlignment(config)
        result = step.apply(df)
        # 2020-12-31 + 180 days ≈ 2021-06-29
        expected = pd.Timestamp("2020-12-31") + pd.Timedelta(days=180)
        assert result["public_date"].values[0] == expected


class TestDuplicateRemoval:

    def test_keeps_latest_datadate(self):
        df = pd.DataFrame([
            {"gvkey": "001", "datadate": "2020-06-30", "fyear": 2020, "at": 100},
            {"gvkey": "001", "datadate": "2020-12-31", "fyear": 2020, "at": 110},
        ])
        step = DuplicateRemoval()
        result = step.apply(df)
        assert len(result) == 1
        assert result["at"].values[0] == 110


class TestBuildCompustatPipeline:

    def test_builds_all_steps(self):
        config = CompustatCorrectionConfig()
        steps = build_compustat_pipeline(config)
        step_names = [s.name for s in steps]
        assert "StandardFilter" in step_names
        assert "BookEquityCalculation" in step_names
        assert "PointInTimeAlignment" in step_names

    def test_pipeline_runs(self, synthetic_compustat):
        config = CompustatCorrectionConfig()
        steps = build_compustat_pipeline(config)
        pipeline = CorrectionPipeline(steps)
        result = pipeline.run(synthetic_compustat)

        # Should have removed financial (002) and non-USD (004)
        assert "002" not in result["gvkey"].values
        assert "004" not in result["gvkey"].values
        # Should have book equity and public_date
        assert "be" in result.columns
        assert "public_date" in result.columns
        # Remaining: GVKEY 001 (2 years) + GVKEY 003 (1 year) = 3 rows
        assert len(result) == 3
