"""
CRSP-Compustat Merged (CCM) linking table definition.

The CCM linking table maps Compustat GVKEY to CRSP PERMNO with effective
date ranges. This is essential for merging accounting data (Compustat)
with market data (CRSP).

Link types:
    - LC: Link confirmed by CRSP research
    - LU: Link unconfirmed but usable
    - LD: Duplicate link (avoid)
    - LX: Link to a security that trades on a non-US exchange (avoid)
    - LN: No link available
    - NR: Not researched

Link primacy:
    - P: Primary security for the company
    - C: Primary candidate (used when P is not available)
    - J, N: Junior or non-primary securities

References:
    - WRDS CCM documentation
    - Fama-French data construction methodology
"""

from wrds_data.datasets.base import DatasetDefinition

CCM_LINK = DatasetDefinition(
    name="ccm_link",
    wrds_table="crsp.ccmxpf_lnkhist",
    date_column="linkdt",
    entity_column="gvkey",
    columns={
        "gvkey": "gvkey",
        "lpermno": "lpermno",
        "linktype": "linktype",
        "linkprim": "linkprim",
        "linkdt": "linkdt",
        "linkenddt": "linkenddt",
    },
    description="CRSP-Compustat Merged Link History — GVKEY to PERMNO mapping with date ranges",
    default_chunk_years=50,  # Small table, download all at once
    is_reference_table=True,
)
