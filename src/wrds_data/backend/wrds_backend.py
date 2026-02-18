"""
Live WRDS PostgreSQL backend.

Connects directly to the WRDS database and executes SQL queries
constructed from DatasetDefinition metadata.
"""

from __future__ import annotations

import time
from datetime import date
from typing import Any

import pandas as pd
from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from wrds_data.backend.base import DataBackend
from wrds_data.config import WRDSConnectionConfig
from wrds_data.datasets.base import DatasetDefinition
from wrds_data.exceptions import ConnectionError, DataNotAvailableError


class WRDSBackend(DataBackend):
    """
    Backend that queries the live WRDS PostgreSQL database.

    Uses SQLAlchemy for connection management with explicit connection
    strings (not the wrds package's built-in pgpass mechanism) so that
    credentials can be managed programmatically.
    """

    MAX_RETRIES = 3
    RETRY_DELAY_SECONDS = 5

    def __init__(self, config: WRDSConnectionConfig) -> None:
        self._config = config
        self._engine: Engine | None = None

    @property
    def engine(self) -> Engine:
        """Lazy-initialize the SQLAlchemy engine."""
        if self._engine is None:
            if not self._config.username or not self._config.password:
                raise ConnectionError(
                    "WRDS credentials not configured. "
                    "Set WRDS_USERNAME and WRDS_PASSWORD environment variables, "
                    "or pass them via WRDSConnectionConfig."
                )
            try:
                self._engine = create_engine(
                    self._config.connection_string,
                    pool_pre_ping=True,
                    pool_size=2,
                    max_overflow=3,
                )
                # Test connection
                with self._engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                logger.info("Connected to WRDS database")
            except Exception as e:
                raise ConnectionError(f"Failed to connect to WRDS: {e}") from e
        return self._engine

    def query(
        self,
        dataset: DatasetDefinition,
        columns: list[str] | None = None,
        filters: dict[str, Any] | None = None,
        date_range: tuple[date, date] | None = None,
    ) -> pd.DataFrame:
        """
        Execute a SQL query against the WRDS database.

        Constructs a SELECT query from the dataset definition, applies
        filters and date range, then renames columns from WRDS names
        to canonical names.

        If the dataset has a sql_template, uses that directly instead of
        constructing SQL from column definitions. The template must use
        :start_date and :end_date as named parameters.
        """
        params: dict[str, Any] = {}

        if dataset.sql_template:
            # --- Template-based query (used for v2 CRSP with JOINs) ---
            sql = dataset.sql_template

            if date_range is not None:
                params["start_date"] = date_range[0]
                params["end_date"] = date_range[1]
        else:
            # --- Auto-generated query from column definitions ---
            if columns is not None:
                wrds_cols = [dataset.columns[c] for c in columns if c in dataset.columns]
            else:
                wrds_cols = dataset.wrds_columns

            select_clause = ", ".join(wrds_cols)
            sql = f"SELECT {select_clause} FROM {dataset.wrds_table}"

            where_clauses: list[str] = []

            # Date range filter
            if date_range is not None:
                start_date, end_date = date_range
                wrds_date_col = dataset.columns[dataset.date_column]
                where_clauses.append(f"{wrds_date_col} >= :start_date")
                where_clauses.append(f"{wrds_date_col} <= :end_date")
                params["start_date"] = start_date
                params["end_date"] = end_date

            # Column-value filters
            if filters:
                for canonical_name, value in filters.items():
                    if canonical_name not in dataset.columns:
                        logger.warning(
                            f"Filter column '{canonical_name}' not in dataset "
                            f"'{dataset.name}', skipping"
                        )
                        continue
                    wrds_col = dataset.columns[canonical_name]
                    param_name = f"filter_{canonical_name}"
                    if isinstance(value, (list, tuple, set)):
                        placeholders = ", ".join(
                            f":{param_name}_{i}" for i in range(len(value))
                        )
                        where_clauses.append(f"{wrds_col} IN ({placeholders})")
                        for i, v in enumerate(value):
                            params[f"{param_name}_{i}"] = v
                    else:
                        where_clauses.append(f"{wrds_col} = :{param_name}")
                        params[param_name] = value

            if where_clauses:
                sql += " WHERE " + " AND ".join(where_clauses)

        # Execute with retry logic
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                logger.debug(f"Executing query on {dataset.wrds_table} (attempt {attempt})")
                df = pd.read_sql(text(sql), self.engine, params=params)
                break
            except Exception as e:
                if attempt == self.MAX_RETRIES:
                    raise DataNotAvailableError(
                        f"Failed to query {dataset.wrds_table} after {self.MAX_RETRIES} "
                        f"attempts: {e}"
                    ) from e
                logger.warning(
                    f"Query attempt {attempt} failed: {e}. "
                    f"Retrying in {self.RETRY_DELAY_SECONDS}s..."
                )
                time.sleep(self.RETRY_DELAY_SECONDS)

        # Rename WRDS columns → canonical names
        rename_map = dataset.wrds_to_canonical
        # Only rename columns that are actually present
        rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
        df = df.rename(columns=rename_map)

        logger.debug(
            f"Fetched {len(df):,} rows from {dataset.wrds_table} "
            f"({len(df.columns)} columns)"
        )
        return df

    def is_available(self, dataset: DatasetDefinition) -> bool:
        """Check if we can connect and the table exists."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    text(f"SELECT 1 FROM {dataset.wrds_table} LIMIT 1")
                )
                result.fetchone()
            return True
        except Exception:
            return False

    def close(self) -> None:
        """Dispose of the connection pool."""
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
            logger.debug("WRDS connection closed")
