"""Tests for CCM linking corrections."""

import pandas as pd
import pytest

from wrds_data.config import CCMCorrectionConfig
from wrds_data.corrections.base import CorrectionPipeline
from wrds_data.corrections.linking import (
    LinkDateEnforcement,
    LinkTypeFilter,
    PrimaryLinkPreference,
    build_ccm_pipeline,
)


class TestLinkTypeFilter:

    def test_keeps_valid_types(self, synthetic_ccm):
        config = CCMCorrectionConfig()
        step = LinkTypeFilter(config)
        result = step.apply(synthetic_ccm)
        # Should keep LC and LU, remove LD
        assert "LD" not in result["linktype"].values
        assert "LC" in result["linktype"].values
        assert "LU" in result["linktype"].values
        assert len(result) == 3  # 3 valid links (001, 002, 003)


class TestLinkDateEnforcement:

    def test_fills_missing_enddt(self, synthetic_ccm):
        step = LinkDateEnforcement()
        result = step.apply(synthetic_ccm)
        # No NaT in linkenddt
        assert result["linkenddt"].notna().all()

    def test_parses_dates(self, synthetic_ccm):
        step = LinkDateEnforcement()
        result = step.apply(synthetic_ccm)
        assert pd.api.types.is_datetime64_any_dtype(result["linkdt"])
        assert pd.api.types.is_datetime64_any_dtype(result["linkenddt"])


class TestPrimaryLinkPreference:

    def test_keeps_primary(self, synthetic_ccm):
        config = CCMCorrectionConfig()
        step = PrimaryLinkPreference(config)
        result = step.apply(synthetic_ccm)
        # Should keep P and C, remove N
        assert result["linkprim"].isin({"P", "C"}).all()
        # GVKEY 999 (linkprim=N) should be removed
        assert "999" not in result["gvkey"].values


class TestBuildCCMPipeline:

    def test_pipeline_runs(self, synthetic_ccm):
        config = CCMCorrectionConfig()
        steps = build_ccm_pipeline(config)
        pipeline = CorrectionPipeline(steps)
        result = pipeline.run(synthetic_ccm)

        # After all filters: GVKEY 001 (LC, P), 002 (LC, P), 003 (LU, C)
        # GVKEY 999 removed (LD, N)
        assert len(result) == 3
        assert set(result["gvkey"].values) == {"001", "002", "003"}
