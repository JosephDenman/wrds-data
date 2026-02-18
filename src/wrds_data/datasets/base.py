"""Base dataset definition."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DatasetDefinition:
    """
    Metadata describing a single WRDS table.

    This is a pure data object — it does not fetch data itself.
    The backend layer uses these definitions to construct queries.

    Attributes:
        name: Human-readable dataset name (e.g. "crsp_daily").
        wrds_table: Fully-qualified WRDS table name (e.g. "crsp.dsf").
        date_column: Primary date column name in the WRDS table.
        entity_column: Primary entity identifier column (e.g. "permno", "gvkey").
        columns: Mapping of canonical column names to WRDS column names.
            Canonical names are used throughout the library; WRDS names are
            used only when constructing SQL queries.
        description: Human-readable description of the dataset.
        default_chunk_years: Number of years per chunk when downloading.
            Smaller values for large tables (daily data), larger for small tables.
        sql_template: Optional raw SQL query template. When set, the backend
            uses this instead of auto-generating SQL from columns/wrds_table.
            Must use :start_date and :end_date as named parameters for date
            range filtering. Column results are still renamed via wrds_to_canonical.
            Used for v2 CRSP queries that require JOINs with stksecurityinfohist.
        is_reference_table: If True, download the entire table without date
            filtering. Used for dimension/lookup tables (e.g. crsp_names, ccm_link)
            where the date_column represents a validity range, not observation dates.
    """

    name: str
    wrds_table: str
    date_column: str
    entity_column: str
    columns: dict[str, str] = field(default_factory=dict)
    description: str = ""
    default_chunk_years: int = 1
    sql_template: str = ""
    is_reference_table: bool = False

    @property
    def wrds_columns(self) -> list[str]:
        """List of WRDS column names to fetch."""
        return list(self.columns.values())

    @property
    def canonical_columns(self) -> list[str]:
        """List of canonical column names used internally."""
        return list(self.columns.keys())

    @property
    def wrds_to_canonical(self) -> dict[str, str]:
        """Reverse mapping: WRDS column name → canonical name."""
        return {v: k for k, v in self.columns.items()}

    def __repr__(self) -> str:
        return f"DatasetDefinition(name='{self.name}', table='{self.wrds_table}')"
