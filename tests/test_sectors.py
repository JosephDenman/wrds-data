"""
Tests for sector and industry classification from SIC codes.

Tests the three mapping functions:
- sic_to_sector: GICS-like broad sector
- sic_to_ff12: Fama-French 12-industry classification
- sic_to_ff49: Fama-French 49-industry classification
- WRDSSectorDataSource.to_mapping: dict conversion

References for SIC → FF12 mapping:
    https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
"""

import pandas as pd
import pytest

from wrds_data.sectors import (
    WRDSSectorDataSource,
    sic_to_ff12,
    sic_to_ff49,
    sic_to_sector,
)


class TestSicToFF12:
    """Test Fama-French 12-industry classification."""

    def test_consumer_nondurables(self):
        """Agriculture, food, tobacco, textiles, apparel, leather, toys."""
        assert sic_to_ff12(100) == "Consumer NonDurables"  # Agriculture
        assert sic_to_ff12(2000) == "Consumer NonDurables"  # Food
        assert sic_to_ff12(2399) == "Consumer NonDurables"  # Food
        assert sic_to_ff12(2700) == "Consumer NonDurables"  # Printing/publishing
        assert sic_to_ff12(3100) == "Consumer NonDurables"  # Leather
        assert sic_to_ff12(3940) == "Consumer NonDurables"  # Toys

    def test_consumer_durables(self):
        """Cars, TVs, furniture, household appliances."""
        assert sic_to_ff12(2500) == "Consumer Durables"  # Furniture
        assert sic_to_ff12(3630) == "Consumer Durables"  # Household appliances
        assert sic_to_ff12(3711) == "Consumer Durables"  # Motor vehicles
        assert sic_to_ff12(3900) == "Consumer Durables"  # Misc manufacturing

    def test_manufacturing(self):
        """Machinery, chemicals, paper, etc."""
        assert sic_to_ff12(2520) == "Manufacturing"  # Building materials
        assert sic_to_ff12(2600) == "Manufacturing"  # Paper
        assert sic_to_ff12(3200) == "Manufacturing"  # Stone/clay/glass
        assert sic_to_ff12(3500) == "Manufacturing"  # Industrial machinery

    def test_energy(self):
        """Oil, gas, coal extraction products."""
        assert sic_to_ff12(1200) == "Energy"  # Coal mining
        assert sic_to_ff12(1300) == "Energy"  # Oil and gas extraction
        assert sic_to_ff12(2900) == "Energy"  # Petroleum refining
        assert sic_to_ff12(4900) == "Energy"  # Electric services

    def test_technology(self):
        """Computers, software, electronic equipment."""
        assert sic_to_ff12(3570) == "Technology"  # Computer hardware
        assert sic_to_ff12(3674) == "Technology"  # Semiconductors
        assert sic_to_ff12(3810) == "Technology"  # Search/navigation equipment
        assert sic_to_ff12(7372) == "Technology"  # Prepackaged software

    def test_telecommunications(self):
        assert sic_to_ff12(4800) == "Telecommunications"
        assert sic_to_ff12(4899) == "Telecommunications"

    def test_shops(self):
        """Wholesale and retail."""
        assert sic_to_ff12(5000) == "Shops"
        assert sic_to_ff12(5311) == "Shops"  # Department stores
        assert sic_to_ff12(5999) == "Shops"

    def test_health(self):
        """Healthcare, drugs, medical equipment."""
        assert sic_to_ff12(2830) == "Health"  # Drugs
        assert sic_to_ff12(3693) == "Health"  # X-ray apparatus
        assert sic_to_ff12(3840) == "Health"  # Medical instruments
        assert sic_to_ff12(8000) == "Health"  # Health services

    def test_utilities(self):
        assert sic_to_ff12(4950) == "Utilities"
        assert sic_to_ff12(4960) == "Utilities"
        assert sic_to_ff12(4970) == "Utilities"

    def test_finance(self):
        assert sic_to_ff12(6000) == "Finance"  # Banking
        assert sic_to_ff12(6500) == "Finance"  # Real estate
        assert sic_to_ff12(6999) == "Finance"

    def test_other(self):
        """Everything else falls to Other."""
        assert sic_to_ff12(0) == "Other"
        assert sic_to_ff12(-1) == "Other"
        assert sic_to_ff12(9999) == "Other"

    def test_known_sic_codes(self):
        """Test specific well-known SIC codes."""
        # Apple (SIC 3571 - Electronic Computers)
        assert sic_to_ff12(3571) == "Technology"
        # JPMorgan (SIC 6020 - State Commercial Banks)
        assert sic_to_ff12(6020) == "Finance"
        # Pfizer (SIC 2834 - Pharmaceutical Preparations)
        assert sic_to_ff12(2834) == "Health"
        # ExxonMobil (SIC 2911 - Petroleum Refining)
        assert sic_to_ff12(2911) == "Energy"


class TestSicToSector:
    """Test GICS-like broad sector mapping."""

    def test_technology(self):
        assert sic_to_sector(3571) == "Technology"
        assert sic_to_sector(7372) == "Technology"

    def test_healthcare(self):
        assert sic_to_sector(2834) == "Healthcare"
        assert sic_to_sector(8000) == "Healthcare"

    def test_financial_services(self):
        assert sic_to_sector(6020) == "Financial Services"

    def test_energy(self):
        assert sic_to_sector(1300) == "Energy"
        assert sic_to_sector(2911) == "Energy"

    def test_utilities(self):
        assert sic_to_sector(4911) == "Utilities"

    def test_consumer_cyclical(self):
        assert sic_to_sector(5411) == "Consumer Cyclical"  # Grocery stores (retail)

    def test_consumer_defensive(self):
        assert sic_to_sector(2000) == "Consumer Defensive"  # Food

    def test_unknown_sic(self):
        assert sic_to_sector(0) == "Unknown"
        assert sic_to_sector(-1) == "Unknown"

    def test_all_sectors_populated(self):
        """Verify that we produce multiple distinct sectors across the SIC range."""
        sics = [100, 1300, 2000, 2834, 3571, 4811, 4911, 5411, 6020, 7372, 8000]
        sectors = {sic_to_sector(s) for s in sics}
        # Should have at least 5 distinct sectors from this diverse set
        assert len(sectors) >= 5


class TestSicToFF49:
    """Test Fama-French 49-industry classification."""

    def test_agriculture(self):
        assert sic_to_ff49(100) == "Agric"
        assert sic_to_ff49(200) == "Agric"

    def test_oil(self):
        assert sic_to_ff49(1300) == "Oil"
        assert sic_to_ff49(2911) == "Oil"

    def test_software(self):
        assert sic_to_ff49(7372) == "Softw"

    def test_utilities(self):
        assert sic_to_ff49(4911) == "Util"

    def test_banks(self):
        assert sic_to_ff49(6020) == "Banks"

    def test_other(self):
        assert sic_to_ff49(0) == "Other"
        assert sic_to_ff49(-1) == "Other"


class TestSectorDataSourceToMapping:
    """Test the to_mapping static method."""

    def test_sector_mapping(self):
        df = pd.DataFrame({
            "symbol": ["AAPL", "JPM", "XOM"],
            "sector": ["Technology", "Financial Services", "Energy"],
            "industry": ["Technology", "Finance", "Energy"],
        })
        mapping = WRDSSectorDataSource.to_mapping(df, "sector")
        assert mapping == {
            "AAPL": "Technology",
            "JPM": "Financial Services",
            "XOM": "Energy",
        }

    def test_industry_mapping(self):
        df = pd.DataFrame({
            "symbol": ["AAPL", "JPM"],
            "sector": ["Technology", "Financial Services"],
            "industry": ["Technology", "Finance"],
        })
        mapping = WRDSSectorDataSource.to_mapping(df, "industry")
        assert mapping == {"AAPL": "Technology", "JPM": "Finance"}

    def test_invalid_column_raises(self):
        df = pd.DataFrame({
            "symbol": ["AAPL"],
            "sector": ["Technology"],
        })
        with pytest.raises(ValueError, match="Column 'bogus' not found"):
            WRDSSectorDataSource.to_mapping(df, "bogus")

    def test_empty_dataframe(self):
        df = pd.DataFrame({"symbol": [], "sector": []})
        mapping = WRDSSectorDataSource.to_mapping(df, "sector")
        assert mapping == {}
