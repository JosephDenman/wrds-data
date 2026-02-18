"""Tests for dataset definitions and registry, including CRSP v1/v2."""

import pytest

from wrds_data.datasets import (
    CRSP_DAILY,
    CRSP_DAILY_V2,
    CRSP_DELISTING,
    CRSP_MONTHLY,
    CRSP_MONTHLY_V2,
    CRSP_NAMES,
    COMPUSTAT_ANNUAL,
    COMPUSTAT_QUARTERLY,
    CCM_LINK,
    DatasetDefinition,
    DatasetRegistry,
    get_crsp_daily,
    get_crsp_monthly,
)
from wrds_data.exceptions import DatasetNotFoundError


class TestDatasetDefinitions:
    """Test that all dataset definitions are valid and have required fields."""

    def test_crsp_daily_v1(self):
        assert CRSP_DAILY.name == "crsp_daily"
        assert CRSP_DAILY.wrds_table == "crsp.dsf"
        assert CRSP_DAILY.date_column == "date"
        assert CRSP_DAILY.entity_column == "permno"
        assert "prc" in CRSP_DAILY.columns
        assert "ret" in CRSP_DAILY.columns
        assert "openprc" in CRSP_DAILY.columns
        assert CRSP_DAILY.sql_template == ""

    def test_crsp_daily_v2(self):
        assert CRSP_DAILY_V2.name == "crsp_daily"
        assert CRSP_DAILY_V2.wrds_table == "crsp.dsf_v2"
        assert CRSP_DAILY_V2.date_column == "date"
        assert CRSP_DAILY_V2.entity_column == "permno"
        # V2 maps canonical names to CIZ column names
        assert CRSP_DAILY_V2.columns["date"] == "dlycaldt"
        assert CRSP_DAILY_V2.columns["prc"] == "dlyprc"
        assert CRSP_DAILY_V2.columns["openprc"] == "dlyopen"
        assert CRSP_DAILY_V2.columns["askhi"] == "dlyhigh"
        assert CRSP_DAILY_V2.columns["bidlo"] == "dlylow"
        assert CRSP_DAILY_V2.columns["vol"] == "dlyvol"
        assert CRSP_DAILY_V2.columns["ret"] == "dlyret"
        assert CRSP_DAILY_V2.columns["retx"] == "dlyretx"
        assert CRSP_DAILY_V2.columns["shrout"] == "shrout"
        assert CRSP_DAILY_V2.columns["exchcd"] == "primaryexch"
        assert CRSP_DAILY_V2.columns["siccd"] == "siccd"
        # V2 has a SQL template with stksecurityinfohist JOIN
        assert "stksecurityinfohist" in CRSP_DAILY_V2.sql_template
        assert "dsf_v2" in CRSP_DAILY_V2.sql_template

    def test_crsp_monthly_v2(self):
        assert CRSP_MONTHLY_V2.name == "crsp_monthly"
        assert CRSP_MONTHLY_V2.wrds_table == "crsp.msf_v2"
        assert CRSP_MONTHLY_V2.columns["date"] == "mthcaldt"
        assert CRSP_MONTHLY_V2.columns["prc"] == "mthprc"
        assert CRSP_MONTHLY_V2.columns["ret"] == "mthret"
        assert "stksecurityinfohist" in CRSP_MONTHLY_V2.sql_template

    def test_v2_canonical_names_match_v1(self):
        """V1 and V2 should share the same canonical column names for common fields."""
        common_canonical = {"permno", "date", "prc", "ret", "shrout", "vol"}
        for name in common_canonical:
            assert name in CRSP_DAILY.columns, f"{name} missing from v1"
            assert name in CRSP_DAILY_V2.columns, f"{name} missing from v2"

    def test_v2_wrds_to_canonical_mapping(self):
        """wrds_to_canonical should map CIZ column names back to canonical."""
        mapping = CRSP_DAILY_V2.wrds_to_canonical
        assert mapping["dlycaldt"] == "date"
        assert mapping["dlyprc"] == "prc"
        assert mapping["dlyret"] == "ret"
        assert mapping["dlyretx"] == "retx"
        assert mapping["dlyvol"] == "vol"


class TestVersionHelpers:
    """Test get_crsp_daily() and get_crsp_monthly() version selectors."""

    def test_get_crsp_daily_v2(self):
        ds = get_crsp_daily("v2")
        assert ds is CRSP_DAILY_V2
        assert ds.wrds_table == "crsp.dsf_v2"

    def test_get_crsp_daily_v1(self):
        ds = get_crsp_daily("v1")
        assert ds is CRSP_DAILY
        assert ds.wrds_table == "crsp.dsf"

    def test_get_crsp_daily_default_is_v2(self):
        ds = get_crsp_daily()
        assert ds is CRSP_DAILY_V2

    def test_get_crsp_monthly_v2(self):
        ds = get_crsp_monthly("v2")
        assert ds is CRSP_MONTHLY_V2
        assert ds.wrds_table == "crsp.msf_v2"

    def test_get_crsp_monthly_v1(self):
        ds = get_crsp_monthly("v1")
        assert ds is CRSP_MONTHLY
        assert ds.wrds_table == "crsp.msf"

    def test_invalid_version_raises(self):
        with pytest.raises(ValueError, match="Unknown CRSP version"):
            get_crsp_daily("v3")
        with pytest.raises(ValueError, match="Unknown CRSP version"):
            get_crsp_monthly("v3")


class TestDatasetRegistry:
    """Test the singleton dataset registry with v1/v2 support."""

    def setup_method(self):
        """Reset the singleton between tests."""
        DatasetRegistry._instance = None
        DatasetRegistry._initialized = False

    def test_instance_is_singleton(self):
        r1 = DatasetRegistry.instance()
        r2 = DatasetRegistry.instance()
        assert r1 is r2

    def test_default_registers_v2(self):
        """Default registry should register v2 CRSP datasets."""
        registry = DatasetRegistry.instance()
        crsp_daily = registry.get("crsp_daily")
        assert crsp_daily.wrds_table == "crsp.dsf_v2"
        crsp_monthly = registry.get("crsp_monthly")
        assert crsp_monthly.wrds_table == "crsp.msf_v2"

    def test_default_registers_compustat_and_ccm(self):
        registry = DatasetRegistry.instance()
        assert "compustat_annual" in registry
        assert "compustat_quarterly" in registry
        assert "ccm_link" in registry

    def test_set_crsp_version_v1(self):
        """Switching to v1 should register legacy tables."""
        registry = DatasetRegistry.instance()
        registry.set_crsp_version("v1")
        crsp_daily = registry.get("crsp_daily")
        assert crsp_daily.wrds_table == "crsp.dsf"
        crsp_monthly = registry.get("crsp_monthly")
        assert crsp_monthly.wrds_table == "crsp.msf"
        # V1 also has delisting and names datasets
        assert "crsp_delisting" in registry
        assert "crsp_names" in registry

    def test_set_crsp_version_v2_removes_v1_only(self):
        """Switching back to v2 should remove v1-only datasets.

        crsp_delisting is removed (delisting returns included in v2 DLYRET).
        crsp_names is kept (reference table needed for ticker resolution).
        """
        registry = DatasetRegistry.instance()
        registry.set_crsp_version("v1")
        assert "crsp_delisting" in registry
        registry.set_crsp_version("v2")
        assert "crsp_delisting" not in registry
        assert "crsp_names" in registry  # Reference table, always needed
        # V2 datasets should be back
        crsp_daily = registry.get("crsp_daily")
        assert crsp_daily.wrds_table == "crsp.dsf_v2"

    def test_invalid_version_raises(self):
        registry = DatasetRegistry.instance()
        with pytest.raises(ValueError, match="Unknown CRSP version"):
            registry.set_crsp_version("v3")

    def test_unknown_dataset_raises(self):
        registry = DatasetRegistry.instance()
        with pytest.raises(DatasetNotFoundError):
            registry.get("nonexistent_dataset")

    def test_list_datasets(self):
        registry = DatasetRegistry.instance()
        names = registry.list()
        assert isinstance(names, list)
        assert "crsp_daily" in names
        assert "crsp_monthly" in names
        assert "compustat_annual" in names
        assert names == sorted(names)  # Should be sorted

    def test_register_custom_dataset(self):
        registry = DatasetRegistry.instance()
        custom = DatasetDefinition(
            name="my_custom",
            wrds_table="my_schema.my_table",
            date_column="date",
            entity_column="id",
            columns={"date": "date", "id": "id", "value": "value"},
        )
        registry.register(custom)
        assert "my_custom" in registry
        assert registry.get("my_custom") is custom

    def test_len(self):
        registry = DatasetRegistry.instance()
        assert len(registry) >= 5  # crsp_daily, crsp_monthly, comp_ann, comp_qtr, ccm
