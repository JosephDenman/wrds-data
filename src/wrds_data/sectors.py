"""
Sector and industry classification from WRDS data.

Maps SIC codes (from CRSP dsenames and Compustat) to:
1. Broad sector names (~11 sectors, comparable to GICS sectors from yfinance)
2. Fama-French 12-industry classification (standard academic grouping)
3. Fama-French 49-industry classification (finer-grained academic grouping)

This module replaces yfinance-based sector lookups with WRDS-native
SIC code mappings, eliminating the external API dependency and providing
historically accurate classifications (SIC codes don't change retroactively
the way yfinance's "current" sector assignments do).

References:
    - Kenneth French's data library:
      https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
    - Fama & French (1997): "Industry costs of equity"
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd
from loguru import logger

from wrds_data.provider import WRDSDataProvider


def sic_to_ff12(sic: int) -> str:
    """
    Map a SIC code to the Fama-French 12-industry classification.

    This is the standard academic industry grouping used in Fama-French
    factor models. It provides a moderate level of granularity suitable
    for stratified sampling and sector-aware modeling.

    Args:
        sic: 4-digit SIC code.

    Returns:
        Industry name string.
    """
    if sic <= 0:
        return "Other"

    # 1. Consumer NonDurables
    if (100 <= sic <= 999 or 2000 <= sic <= 2399 or 2700 <= sic <= 2749
            or 2770 <= sic <= 2799 or 3100 <= sic <= 3199
            or 3940 <= sic <= 3989):
        return "Consumer NonDurables"

    # 2. Consumer Durables
    if (2500 <= sic <= 2519 or 2590 <= sic <= 2599 or 3630 <= sic <= 3659
            or 3710 <= sic <= 3711 or sic == 3714 or sic == 3716
            or 3750 <= sic <= 3751 or sic == 3792
            or 3900 <= sic <= 3939 or 3990 <= sic <= 3999):
        return "Consumer Durables"

    # 3. Manufacturing
    if (2520 <= sic <= 2589 or 2600 <= sic <= 2699 or 2750 <= sic <= 2769
            or 2800 <= sic <= 2829 or 2840 <= sic <= 2899
            or 3000 <= sic <= 3099 or 3200 <= sic <= 3569
            or 3580 <= sic <= 3629 or 3700 <= sic <= 3709
            or 3712 <= sic <= 3713 or sic == 3715
            or 3717 <= sic <= 3749 or 3752 <= sic <= 3791
            or 3793 <= sic <= 3799 or 3830 <= sic <= 3839
            or 3860 <= sic <= 3899):
        return "Manufacturing"

    # 4. Energy
    if 1200 <= sic <= 1399 or 2900 <= sic <= 2999 or 4900 <= sic <= 4949:
        return "Energy"

    # 5. Technology (HiTec)
    if (3570 <= sic <= 3579 or 3660 <= sic <= 3692 or 3694 <= sic <= 3699
            or 3810 <= sic <= 3829 or 7370 <= sic <= 7379):
        return "Technology"

    # 6. Telecommunications
    if 4800 <= sic <= 4899:
        return "Telecommunications"

    # 7. Shops (Wholesale & Retail)
    if 5000 <= sic <= 5999:
        return "Shops"

    # 8. Health (Healthcare, Medical Equipment, Drugs)
    if (2830 <= sic <= 2839 or sic == 3693 or 3840 <= sic <= 3859
            or 8000 <= sic <= 8099):
        return "Health"

    # 9. Utilities
    if 4950 <= sic <= 4959 or 4960 <= sic <= 4969 or 4970 <= sic <= 4979:
        return "Utilities"

    # 10. Finance (Money)
    if 6000 <= sic <= 6999:
        return "Finance"

    # 11. Other
    return "Other"


def sic_to_sector(sic: int) -> str:
    """
    Map a SIC code to a broad sector name comparable to GICS sectors.

    This provides a COMPLETE mapping — every SIC code 100-9999 resolves
    to a named sector. No code falls through to "Other".

    Coverage is derived from the SIC code rather than a proprietary
    classification. The mapping covers 12 sectors:
    Technology, Healthcare, Real Estate, Financial Services, Energy,
    Utilities, Communication Services, Consumer Cyclical,
    Consumer Defensive, Industrials, Basic Materials, and
    Conglomerates (for SIC 9999 nonclassifiable/SPACs).

    Args:
        sic: 4-digit SIC code.

    Returns:
        Sector name string. "Unknown" only for sic <= 0.
    """
    if sic <= 0:
        return "Unknown"

    # Technology
    if (3570 <= sic <= 3579 or 3660 <= sic <= 3692 or 3694 <= sic <= 3699
            or 3810 <= sic <= 3839 or 7370 <= sic <= 7379):
        return "Technology"

    # Healthcare (pharma 2830-2839, medical devices 3693/3840-3859, health services 8000-8099)
    if (2830 <= sic <= 2839 or sic == 3693 or 3840 <= sic <= 3859
            or 8000 <= sic <= 8099):
        return "Healthcare"

    # Real Estate — must come BEFORE Financial Services since 6500-6553 ⊂ 6000-6999
    if 6500 <= sic <= 6553:
        return "Real Estate"

    # Financial Services (banks, insurance, brokers, holding/investment offices)
    if 6000 <= sic <= 6999:
        return "Financial Services"

    # Energy
    if 1200 <= sic <= 1399 or 2900 <= sic <= 2999 or 4920 <= sic <= 4924:
        return "Energy"

    # Utilities
    if 4900 <= sic <= 4999:
        return "Utilities"

    # Communication Services (telecom, media, entertainment, motion pictures)
    if 4800 <= sic <= 4899 or 7800 <= sic <= 7999:
        return "Communication Services"

    # Consumer Discretionary (Cyclical)
    if (2500 <= sic <= 2599 or 3600 <= sic <= 3659 or 3710 <= sic <= 3716
            or 3750 <= sic <= 3751 or 3900 <= sic <= 3999
            or 5000 <= sic <= 5999 or 7000 <= sic <= 7369
            or 7380 <= sic <= 7799):
        return "Consumer Cyclical"

    # Consumer Staples (Defensive)
    if (100 <= sic <= 999 or 2000 <= sic <= 2399 or 2700 <= sic <= 2799
            or 2840 <= sic <= 2899):
        return "Consumer Defensive"

    # Industrials (construction, manufacturing, transport, engineering,
    #              environmental services, professional services, government)
    if (1500 <= sic <= 1999 or 3100 <= sic <= 3199 or 3400 <= sic <= 3569
            or 3580 <= sic <= 3599 or 3700 <= sic <= 3709
            or 3717 <= sic <= 3749 or 3752 <= sic <= 3799
            or 3800 <= sic <= 3809 or 3860 <= sic <= 3899
            or 4000 <= sic <= 4799
            or 8100 <= sic <= 8999
            or 9000 <= sic <= 9998):
        return "Industrials"

    # Basic Materials
    if (1000 <= sic <= 1199 or 1400 <= sic <= 1499 or 2400 <= sic <= 2499
            or 2600 <= sic <= 2699 or 2800 <= sic <= 2829
            or 3000 <= sic <= 3099 or 3200 <= sic <= 3399):
        return "Basic Materials"

    # Conglomerates — SIC 9999 "Nonclassifiable Establishments"
    # Mostly SPACs, blank check companies, and shell companies.
    # ~78% listed 2019+ during the SPAC boom. These are financial vehicles
    # but are separated from Financial Services to avoid overweighting.
    if sic == 9999:
        return "Conglomerates"

    return "Unknown"


def sic_to_ff49(sic: int) -> str:
    """
    Map a SIC code to the Fama-French 49-industry classification.

    This provides the finest-grained standard academic industry grouping.
    Useful for detailed sector analysis and as features for ML models.

    Args:
        sic: 4-digit SIC code.

    Returns:
        Industry name string.
    """
    if sic <= 0:
        return "Other"

    # Agriculture
    if 100 <= sic <= 299:
        return "Agric"
    # Food Products
    if 2000 <= sic <= 2046 or 2050 <= sic <= 2063 or 2070 <= sic <= 2079 or 2090 <= sic <= 2099 or 2040 <= sic <= 2046:
        return "Food"
    # Candy & Soda
    if 2064 <= sic <= 2068 or 2086 <= sic <= 2087 or 2095 <= sic <= 2095 or 2098 <= sic <= 2099:
        return "Soda"
    # Beer & Liquor
    if 2080 <= sic <= 2085:
        return "Beer"
    # Tobacco Products
    if 2100 <= sic <= 2199:
        return "Smoke"
    # Recreation
    if 3650 <= sic <= 3652 or 3732 <= sic <= 3732 or 3930 <= sic <= 3931 or 3940 <= sic <= 3949:
        return "Toys"
    # Entertainment
    if 7800 <= sic <= 7833 or 7840 <= sic <= 7841 or 7900 <= sic <= 7999:
        return "Fun"
    # Printing and Publishing
    if 2700 <= sic <= 2749 or 2770 <= sic <= 2799:
        return "Books"
    # Consumer Goods
    if 2047 <= sic <= 2047 or 2391 <= sic <= 2392 or 2510 <= sic <= 2519 or 2590 <= sic <= 2599 or 2840 <= sic <= 2844 or 3160 <= sic <= 3199 or 3991 <= sic <= 3991:
        return "Hshld"
    # Apparel
    if 2300 <= sic <= 2390 or 3020 <= sic <= 3021 or 3100 <= sic <= 3111 or 3130 <= sic <= 3159 or 3965 <= sic <= 3965:
        return "Clths"
    # Medical Equipment
    if 3693 <= sic <= 3693 or 3840 <= sic <= 3851:
        return "MedEq"
    # Pharmaceutical Products
    if 2830 <= sic <= 2836:
        return "Drugs"
    # Chemicals
    if 2800 <= sic <= 2829 or 2860 <= sic <= 2899:
        return "Chems"
    # Rubber and Plastics
    if 3000 <= sic <= 3099:
        return "Rubbr"
    # Textiles
    if 2200 <= sic <= 2284 or 2290 <= sic <= 2399:
        return "Txtls"
    # Construction Materials
    if 800 <= sic <= 899 or 2400 <= sic <= 2439 or 2450 <= sic <= 2459 or 2490 <= sic <= 2499 or 2660 <= sic <= 2661 or 2950 <= sic <= 2952 or 3200 <= sic <= 3299 or 3420 <= sic <= 3442 or 3446 <= sic <= 3452 or 3490 <= sic <= 3499 or 3996 <= sic <= 3996:
        return "BldMt"
    # Construction
    if 1500 <= sic <= 1511 or 1520 <= sic <= 1549 or 1600 <= sic <= 1799:
        return "Cnstr"
    # Steel Works
    if 3300 <= sic <= 3399 or 3460 <= sic <= 3479:
        return "Steel"
    # Fabricated Products
    if 3400 <= sic <= 3419 or 3443 <= sic <= 3444 or 3460 <= sic <= 3479:
        return "FabPr"
    # Machinery
    if 3510 <= sic <= 3536 or 3538 <= sic <= 3599:
        return "Mach"
    # Electrical Equipment
    if 3600 <= sic <= 3621 or 3623 <= sic <= 3629 or 3640 <= sic <= 3646 or 3648 <= sic <= 3649 or 3660 <= sic <= 3669 or 3680 <= sic <= 3699:
        return "ElcEq"
    # Automobiles and Trucks
    if 3710 <= sic <= 3711 or 3714 <= sic <= 3714 or 3716 <= sic <= 3716 or 3750 <= sic <= 3751 or 3792 <= sic <= 3792:
        return "Autos"
    # Aircraft
    if 3720 <= sic <= 3729:
        return "Aero"
    # Shipbuilding, Railroad Equipment
    if 3730 <= sic <= 3731 or 3740 <= sic <= 3743:
        return "Ships"
    # Defense
    if 3760 <= sic <= 3769 or 3795 <= sic <= 3795 or 3480 <= sic <= 3489:
        return "Guns"
    # Precious Metals
    if 1040 <= sic <= 1049:
        return "Gold"
    # Mining
    if 1000 <= sic <= 1039 or 1050 <= sic <= 1099 or 1200 <= sic <= 1299 or 1400 <= sic <= 1499:
        return "Mines"
    # Coal
    if 1200 <= sic <= 1299:
        return "Coal"
    # Petroleum and Natural Gas
    if 1300 <= sic <= 1389 or 2900 <= sic <= 2999:
        return "Oil"
    # Utilities
    if 4900 <= sic <= 4999:
        return "Util"
    # Telecommunication
    if 4800 <= sic <= 4899:
        return "Telcm"
    # Personal Services
    if 7020 <= sic <= 7021 or 7030 <= sic <= 7033 or 7200 <= sic <= 7299 or 7395 <= sic <= 7395 or 7500 <= sic <= 7500 or 7520 <= sic <= 7549 or 7600 <= sic <= 7699 or 8100 <= sic <= 8199 or 8200 <= sic <= 8299 or 8300 <= sic <= 8399 or 8400 <= sic <= 8499 or 8600 <= sic <= 8699 or 8800 <= sic <= 8899:
        return "PerSv"
    # Computer Software (must precede Business Services — 7370-7379 ⊂ 7300-7399)
    if 7370 <= sic <= 7379:
        return "Softw"
    # Computers
    if 3570 <= sic <= 3579 or 3680 <= sic <= 3689 or 3695 <= sic <= 3695:
        return "Comps"
    # Business Services (excluding 7370-7379 which is Computer Software)
    if 2750 <= sic <= 2759 or 7300 <= sic <= 7369 or 7380 <= sic <= 7384 or 7389 <= sic <= 7394 or 7396 <= sic <= 7397 or 7399 <= sic <= 7399 or 7510 <= sic <= 7519 or 8700 <= sic <= 8748 or 8900 <= sic <= 8999:
        return "BusSv"
    # Electronic Equipment
    if 3622 <= sic <= 3622 or 3661 <= sic <= 3669 or 3670 <= sic <= 3679 or 3810 <= sic <= 3810 or 3812 <= sic <= 3812:
        return "Chips"
    # Measuring and Control Equipment
    if 3811 <= sic <= 3811 or 3820 <= sic <= 3829:
        return "LabEq"
    # Paper Business Supplies
    if 2440 <= sic <= 2449 or 2520 <= sic <= 2549 or 2600 <= sic <= 2659 or 2670 <= sic <= 2699:
        return "Paper"
    # Shipping Containers
    if 2440 <= sic <= 2449 or 2640 <= sic <= 2659 or 3085 <= sic <= 3089 or 3411 <= sic <= 3412 or 3412 <= sic <= 3412:
        return "Boxes"
    # Transportation
    if 4000 <= sic <= 4099 or 4100 <= sic <= 4199 or 4200 <= sic <= 4299 or 4400 <= sic <= 4499 or 4500 <= sic <= 4599 or 4600 <= sic <= 4699 or 4700 <= sic <= 4799:
        return "Trans"
    # Wholesale
    if 5000 <= sic <= 5199:
        return "Whlsl"
    # Retail
    if 5200 <= sic <= 5999:
        return "Rtail"
    # Restaurants, Hotels, Motels
    if 5800 <= sic <= 5829 or 5890 <= sic <= 5899 or 7000 <= sic <= 7019 or 7040 <= sic <= 7049 or 7213 <= sic <= 7213:
        return "Meals"
    # Banking
    if 6000 <= sic <= 6199:
        return "Banks"
    # Insurance
    if 6300 <= sic <= 6399 or 6400 <= sic <= 6411:
        return "Insur"
    # Real Estate
    if 6500 <= sic <= 6553:
        return "RlEst"
    # Trading
    if 6200 <= sic <= 6299 or 6700 <= sic <= 6799:
        return "Fin"

    return "Other"


class WRDSSectorDataSource:
    """
    Provides sector and industry classification for stock symbols using
    SIC codes from WRDS (CRSP dsenames table).

    This is the WRDS-native replacement for tft-finance's SectorDataSource
    which uses yfinance. Advantages:
    1. No external API calls (SIC codes come from CRSP)
    2. Historically accurate (SIC codes are point-in-time)
    3. No rate limiting
    4. Consistent with academic literature (Fama-French classifications)

    The output format matches SectorDataSource.fetch_sectors() so it can be
    used as a drop-in replacement in tft-finance's DataManager.

    Usage::

        from wrds_data import WRDSDataProvider, WRDSDataConfig
        from wrds_data.sectors import WRDSSectorDataSource

        provider = WRDSDataProvider(WRDSDataConfig())
        sector_source = WRDSSectorDataSource(provider)

        # Get sectors for a list of tickers
        sector_df = sector_source.fetch_sectors(["AAPL", "MSFT", "JPM"])
        # Returns: DataFrame with columns [symbol, sector, industry, sic, ff12, ff49]

        # Get as dict mappings (same interface as SectorDataSource)
        sector_map = WRDSSectorDataSource.to_mapping(sector_df, "sector")
        industry_map = WRDSSectorDataSource.to_mapping(sector_df, "industry")
    """

    def __init__(self, provider: WRDSDataProvider) -> None:
        self._provider = provider

    def fetch_sectors(
        self,
        symbols: List[str],
        as_of: "date | None" = None,
    ) -> pd.DataFrame:
        """
        Fetch sector and industry classification for symbols.

        Uses SIC codes from the CRSP names table, mapped to:
        - sector: GICS-like broad sector name (comparable to yfinance)
        - industry: Fama-French 12-industry classification
        - ff49: Fama-French 49-industry classification
        - sic: raw SIC code

        Args:
            symbols: List of ticker symbols.
            as_of: Date for the lookup. None = most recent mapping.

        Returns:
            DataFrame with columns:
            [symbol, sector, industry, sic, ff12, ff49]
        """
        if not symbols:
            return pd.DataFrame(
                columns=["symbol", "sector", "industry", "sic", "ff12", "ff49"]
            )

        # Get the full universe with SIC codes
        universe = self._provider.universe(as_of=as_of)

        # Resolve tickers to PERMNOs
        permno_map = self._provider._universe.ticker_to_permno_map(symbols, as_of)
        ticker_by_permno = {v: k for k, v in permno_map.items()}

        rows = []
        for _, row in universe.iterrows():
            permno = int(row["permno"])
            ticker = ticker_by_permno.get(permno)
            if ticker is None:
                continue

            sic_raw = row.get("siccd", 0)
            sic = int(sic_raw) if pd.notna(sic_raw) else 0

            rows.append({
                "symbol": ticker,
                "sector": sic_to_sector(sic),
                "industry": sic_to_ff12(sic),
                "sic": sic,
                "ff12": sic_to_ff12(sic),
                "ff49": sic_to_ff49(sic),
            })

        # Handle unresolved symbols
        resolved_symbols = {r["symbol"] for r in rows}
        for sym in symbols:
            if sym not in resolved_symbols:
                rows.append({
                    "symbol": sym,
                    "sector": "Unknown",
                    "industry": "Other",
                    "sic": 0,
                    "ff12": "Other",
                    "ff49": "Other",
                })

        df = pd.DataFrame(rows)

        # Log distribution
        sector_counts = df["sector"].value_counts()
        logger.info(
            f"Sector classification for {len(symbols)} symbols "
            f"({len(sector_counts)} sectors, {len(df[df['sector'] != 'Unknown'])} resolved)"
        )
        for sector, count in sector_counts.head(10).items():
            logger.debug(f"  {sector}: {count}")

        return df

    @staticmethod
    def to_mapping(df: pd.DataFrame, column: str = "sector") -> Dict[str, str]:
        """
        Convert DataFrame to dict mapping for fast lookup.

        Same interface as tft-finance's SectorDataSource.to_mapping().

        Args:
            df: DataFrame with columns ['symbol', column]
            column: Which column to use ('sector', 'industry', 'ff12', 'ff49')

        Returns:
            Dict mapping symbol → classification value.
        """
        if column not in df.columns:
            raise ValueError(
                f"Column '{column}' not found. Available: {list(df.columns)}"
            )
        return dict(zip(df["symbol"], df[column]))
