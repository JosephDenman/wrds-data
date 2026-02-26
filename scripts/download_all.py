#!/usr/bin/env python3
"""
Download ALL accessible WRDS datasets to an external drive.

This script bypasses the DatasetDefinition registry and instead discovers
every schema and table you have access to via PostgreSQL's information_schema.
It downloads each table to Parquet, with automatic date-chunking for large
time-series tables and full-table downloads for smaller reference tables.

Usage:
    # Discover what's available (dry run — no downloads)
    python scripts/download_all.py --discover-only

    # Download everything to an external drive
    python scripts/download_all.py --output-dir /Volumes/MyDrive/wrds

    # Download specific libraries only
    python scripts/download_all.py --output-dir /Volumes/MyDrive/wrds --libraries crsp comp ibes

    # Resume a previous download (skips existing files)
    python scripts/download_all.py --output-dir /Volumes/MyDrive/wrds

    # Force re-download everything
    python scripts/download_all.py --output-dir /Volumes/MyDrive/wrds --force

    # Skip enormous tables (e.g. TAQ tick data)
    python scripts/download_all.py --output-dir /Volumes/MyDrive/wrds --max-rows 500_000_000

    # Use row-count estimation to skip tables over a threshold
    python scripts/download_all.py --output-dir /Volumes/MyDrive/wrds --skip-libraries taq taqmsec

Environment:
    WRDS_USERNAME  — your WRDS username
    WRDS_PASSWORD  — your WRDS password
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import pandas as pd
import pyarrow.parquet as pq
from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from tqdm import tqdm


# ── Configuration ────────────────────────────────────────────────────────────

WRDS_HOST = "wrds-pgdata.wharton.upenn.edu"
WRDS_PORT = 9737
WRDS_DB = "wrds"

# Schemas to always skip (PostgreSQL internals, WRDS system schemas)
SKIP_SCHEMAS = frozenset({
    "pg_catalog",
    "information_schema",
    "pg_toast",
    "pg_temp_1",
    "pg_toast_temp_1",
    "columnar",       # PostgreSQL/Citus internal columnar storage metadata
})

# Schemas known to be enormous (TAQ tick data can be 10+ TB).
# These are NOT skipped by default but flagged in discovery output.
# taqmsec alone has 36,000+ tables and takes hours just to enumerate.
WARN_SCHEMAS = frozenset({
    "taq", "taqmsec", "taqmsamp", "taqmsamp_all", "taqsamp", "taqsamp_all", "nastraq",
})

# Maximum rows to fetch in a single query before we chunk by date
CHUNK_THRESHOLD = 50_000_000  # 50M rows

# Default chunk size in years for date-based tables
DEFAULT_CHUNK_YEARS = 1

# For tables without a usable date column, download in row-offset chunks
ROW_CHUNK_SIZE = 5_000_000  # 5M rows per chunk

# Per-query statement timeout (milliseconds). Prevents multi-hour hangs on
# tables that are extremely slow to read (e.g., columnar metadata, huge views).
# 30 minutes should be plenty for any single year-chunk or offset-chunk query.
STATEMENT_TIMEOUT_MS = 30 * 60 * 1000  # 30 minutes


# ── Size estimation ──────────────────────────────────────────────────────────

# Average bytes per value by PostgreSQL data type, after Parquet compression.
# Parquet uses dictionary encoding, run-length encoding, and snappy/zstd
# compression. These are conservative (slightly high) estimates based on
# typical WRDS financial data.
_BYTES_PER_VALUE: dict[str, int] = {
    # Numeric types
    "integer":                       4,
    "smallint":                      2,
    "bigint":                        8,
    "real":                          4,
    "double precision":              8,
    "numeric":                       8,
    # Date/time
    "date":                          4,
    "timestamp without time zone":   8,
    "timestamp with time zone":      8,
    # Text (highly variable — WRDS columns are mostly short codes/tickers)
    "character":                     4,
    "character varying":            12,
    "text":                         20,
    # Boolean
    "boolean":                       1,
}

# Fallback for unknown types
_DEFAULT_BYTES_PER_VALUE = 8

# Parquet overhead factor (file metadata, row group headers, page headers).
# Adds ~5–10% on top of raw column data.
_PARQUET_OVERHEAD = 1.08


def estimate_bytes_per_row(columns: list[dict[str, str]]) -> float:
    """
    Estimate compressed Parquet bytes per row based on column data types.

    This uses per-type averages that account for Parquet's dictionary encoding
    and compression. Real sizes will vary ±30% depending on data cardinality
    and null density, but this gives a useful planning estimate.
    """
    total = 0.0
    for col in columns:
        total += _BYTES_PER_VALUE.get(col["type"], _DEFAULT_BYTES_PER_VALUE)
    return total * _PARQUET_OVERHEAD


def estimate_table_bytes(info: "TableInfo") -> int:
    """Estimate total Parquet size in bytes for a table."""
    if info.estimated_rows == 0 or not info.columns:
        return 0
    bytes_per_row = estimate_bytes_per_row(info.columns)
    return int(info.estimated_rows * bytes_per_row)


def format_bytes(n: int | float) -> str:
    """Human-readable byte size (e.g. '1.23 GB')."""
    if n < 1024:
        return f"{n} B"
    elif n < 1024 ** 2:
        return f"{n / 1024:.1f} KB"
    elif n < 1024 ** 3:
        return f"{n / 1024**2:.1f} MB"
    elif n < 1024 ** 4:
        return f"{n / 1024**3:.2f} GB"
    else:
        return f"{n / 1024**4:.2f} TB"


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class TableInfo:
    """Metadata for a single WRDS table."""
    schema: str
    table: str
    columns: list[dict[str, str]] = field(default_factory=list)
    date_column: str | None = None           # Best date column for chunking
    estimated_rows: int = 0
    accessible: bool = True
    error: str = ""

    @property
    def fqn(self) -> str:
        """Fully qualified name: schema.table"""
        return f"{self.schema}.{self.table}"

    @property
    def safe_name(self) -> str:
        """Filesystem-safe name: schema__table"""
        return f"{self.schema}__{self.table}"

    @property
    def estimated_bytes(self) -> int:
        """Estimated Parquet size in bytes."""
        return estimate_table_bytes(self)


# ── Connection ───────────────────────────────────────────────────────────────

def build_connection_string(username: str, password: str) -> str:
    user = quote_plus(username)
    pwd = quote_plus(password)
    return f"postgresql+psycopg2://{user}:{pwd}@{WRDS_HOST}:{WRDS_PORT}/{WRDS_DB}"


def connect(username: str, password: str) -> Engine:
    """Create and test a SQLAlchemy engine to WRDS."""
    conn_str = build_connection_string(username, password)
    engine = create_engine(
        conn_str,
        pool_pre_ping=True,
        pool_size=2,
        max_overflow=3,
        connect_args={
            "connect_timeout": 30,
            "options": f"-c statement_timeout={STATEMENT_TIMEOUT_MS}",
        },
    )
    # Test connection
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    logger.info("Connected to WRDS PostgreSQL database")
    return engine


# ── Discovery ────────────────────────────────────────────────────────────────

def discover_schemas(engine: Engine, filter_libraries: list[str] | None = None) -> list[str]:
    """List all accessible schemas (libraries) on WRDS."""
    sql = """
        SELECT schema_name
        FROM information_schema.schemata
        ORDER BY schema_name
    """
    with engine.connect() as conn:
        result = conn.execute(text(sql))
        all_schemas = [row[0] for row in result]

    # Filter out system schemas
    schemas = [s for s in all_schemas if s not in SKIP_SCHEMAS]

    # Apply user filter
    if filter_libraries:
        requested = {lib.lower() for lib in filter_libraries}
        schemas = [s for s in schemas if s in requested]
        missing = requested - set(schemas)
        if missing:
            logger.warning(f"Requested libraries not found: {missing}")

    logger.info(f"Found {len(schemas)} accessible schemas (of {len(all_schemas)} total)")
    return schemas


def discover_tables(engine: Engine, schema: str) -> list[str]:
    """List all tables (BASE TABLE + VIEW) in a schema."""
    sql = """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = :schema
          AND table_type IN ('BASE TABLE', 'VIEW')
        ORDER BY table_name
    """
    with engine.connect() as conn:
        result = conn.execute(text(sql), {"schema": schema})
        return [row[0] for row in result]


def discover_columns(engine: Engine, schema: str, table: str) -> list[dict[str, str]]:
    """Get column names and data types for a table."""
    sql = """
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = :schema
          AND table_name = :table
        ORDER BY ordinal_position
    """
    with engine.connect() as conn:
        result = conn.execute(text(sql), {"schema": schema, "table": table})
        return [{"name": row[0], "type": row[1]} for row in result]


def estimate_row_count(engine: Engine, schema: str, table: str) -> int:
    """
    Fast row-count estimate from pg_catalog (uses table statistics, not COUNT(*)).

    This is orders of magnitude faster than SELECT COUNT(*) for large tables.
    Returns 0 if the table has never been analyzed.
    """
    sql = """
        SELECT reltuples::bigint
        FROM pg_catalog.pg_class c
        JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = :schema AND c.relname = :table
    """
    try:
        with engine.connect() as conn:
            result = conn.execute(text(sql), {"schema": schema, "table": table})
            row = result.fetchone()
            return max(0, row[0]) if row else 0
    except Exception:
        return 0


def find_best_date_column(columns: list[dict[str, str]]) -> str | None:
    """
    Heuristically pick the best date column for time-based chunking.

    Prefers columns named 'date', 'datadate', then any date-type column
    with common temporal names.
    """
    date_type_cols = [
        c for c in columns
        if c["type"] in ("date", "timestamp without time zone", "timestamp with time zone")
    ]

    if not date_type_cols:
        return None

    # Priority order for common WRDS date column names
    preferred_names = [
        "date", "datadate", "dlycaldt", "mthcaldt", "caldt",
        "fpedats", "statpers", "anndats", "actdats",
        "effdate", "fdate", "rdate", "reportdt", "rdq",
        "linkdt", "namedt", "dlstdt",
        "trd_exctn_dt", "trade_dt", "tdate",
    ]

    for name in preferred_names:
        for c in date_type_cols:
            if c["name"] == name:
                return name

    # Fall back to any date column containing common substrings
    for substr in ["date", "dt", "caldt"]:
        for c in date_type_cols:
            if substr in c["name"]:
                return c["name"]

    # Last resort: first date-type column
    return date_type_cols[0]["name"]


def check_table_access(engine: Engine, fqn: str) -> tuple[bool, str]:
    """Quick check: can we SELECT from this table?"""
    try:
        with engine.connect() as conn:
            conn.execute(text(f"SELECT * FROM {fqn} LIMIT 1"))
        return True, ""
    except Exception as e:
        return False, str(e)[:200]


def discover_all(
    engine: Engine,
    schemas: list[str],
    check_access: bool = True,
) -> list[TableInfo]:
    """
    Full discovery: enumerate all tables across all schemas.

    Returns a list of TableInfo objects with column metadata,
    date column detection, and row-count estimates.
    """
    all_tables: list[TableInfo] = []

    for schema in tqdm(schemas, desc="Discovering schemas", unit="schema"):
        tables = discover_tables(engine, schema)
        logger.info(f"  {schema}: {len(tables)} tables")

        for table in tables:
            info = TableInfo(schema=schema, table=table)

            # Get columns
            try:
                info.columns = discover_columns(engine, schema, table)
            except Exception as e:
                info.accessible = False
                info.error = str(e)[:200]
                all_tables.append(info)
                continue

            # Detect best date column
            info.date_column = find_best_date_column(info.columns)

            # Estimate rows
            info.estimated_rows = estimate_row_count(engine, schema, table)

            # Access check
            if check_access:
                info.accessible, info.error = check_table_access(engine, info.fqn)

            all_tables.append(info)

    return all_tables


# ── Column list helpers ──────────────────────────────────────────────────────

def _parquet_row_count(path: Path) -> int:
    """Read row count from Parquet metadata without loading the file."""
    try:
        return pq.read_metadata(path).num_rows
    except Exception:
        return -1  # Corrupt or unreadable


def _build_col_list(info: "TableInfo") -> str:
    """Build a quoted column list from cached metadata."""
    return ", ".join(f'"{c["name"]}"' for c in info.columns)


def _is_column_error(exc: Exception) -> bool:
    """Check if an exception is caused by a missing/undefined column."""
    msg = str(exc).lower()
    return "undefinedcolumn" in msg or "undefined column" in msg or "does not exist" in msg


def _is_timeout_error(exc: Exception) -> bool:
    """Check if an exception is a statement timeout cancellation."""
    msg = str(exc).lower()
    return "statement timeout" in msg or "querycanceled" in msg or "query canceled" in msg


# ── Download ─────────────────────────────────────────────────────────────────

def compute_year_ranges(
    start_year: int, end_year: int, chunk_years: int = 1
) -> list[tuple[int, int]]:
    """Split a year range into chunks."""
    ranges = []
    yr = start_year
    while yr <= end_year:
        chunk_end = min(yr + chunk_years - 1, end_year)
        ranges.append((yr, chunk_end))
        yr = chunk_end + 1
    return ranges


def download_table_with_dates(
    engine: Engine,
    info: TableInfo,
    output_dir: Path,
    start_year: int,
    end_year: int,
    chunk_years: int,
    force: bool,
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> int:
    """
    Download a table with a date column in year-based chunks.

    Returns total rows downloaded.
    """
    year_ranges = compute_year_ranges(start_year, end_year, chunk_years)
    total_rows = 0
    col_list = _build_col_list(info)
    use_star = False  # Fall back to SELECT * if column metadata is stale

    for start_yr, end_yr in year_ranges:
        if start_yr == end_yr:
            fname = f"{info.safe_name}_{start_yr}.parquet"
        else:
            fname = f"{info.safe_name}_{start_yr}_{end_yr}.parquet"

        output_path = output_dir / fname
        failed_marker = output_dir / f"{fname}.FAILED"

        # Resume: skip existing files (read row count from metadata, not full file)
        if output_path.exists() and not force:
            n = _parquet_row_count(output_path)
            if n >= 0:
                total_rows += n
                continue
            else:
                logger.warning(f"  Corrupt file {fname}, re-downloading")

        # Clean up old failed marker
        if failed_marker.exists():
            failed_marker.unlink()

        select_cols = "*" if use_star else col_list
        sql = f"""
            SELECT {select_cols}
            FROM {info.fqn}
            WHERE "{info.date_column}" >= :start_date
              AND "{info.date_column}" <= :end_date
        """

        start_date = date(start_yr, 1, 1)
        end_date = date(end_yr, 12, 31)

        for attempt in range(1, max_retries + 1):
            try:
                df = pd.read_sql(
                    text(sql), engine,
                    params={"start_date": start_date, "end_date": end_date},
                )
                break
            except Exception as e:
                # If column metadata is stale (view changed), fall back to SELECT *
                if _is_column_error(e) and not use_star:
                    logger.warning(f"  Column metadata stale for {info.fqn}, falling back to SELECT *")
                    use_star = True
                    sql = f"""
                        SELECT *
                        FROM {info.fqn}
                        WHERE "{info.date_column}" >= :start_date
                          AND "{info.date_column}" <= :end_date
                    """
                    try:
                        df = pd.read_sql(
                            text(sql), engine,
                            params={"start_date": start_date, "end_date": end_date},
                        )
                        break
                    except Exception as e2:
                        if attempt == max_retries:
                            failed_marker.touch()
                            logger.error(f"  FAILED {fname} after {max_retries} attempts: {e2}")
                            return total_rows
                        logger.warning(f"  Attempt {attempt} failed for {fname}: {e2}")
                        time.sleep(retry_delay)
                        continue

                if attempt == max_retries:
                    failed_marker.touch()
                    logger.error(f"  FAILED {fname} after {max_retries} attempts: {e}")
                    return total_rows
                logger.warning(f"  Attempt {attempt} failed for {fname}: {e}")
                time.sleep(retry_delay)

        if len(df) > 0:
            df.to_parquet(output_path, index=False)
            total_rows += len(df)

    return total_rows


def download_table_full(
    engine: Engine,
    info: TableInfo,
    output_dir: Path,
    force: bool,
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> int:
    """
    Download a table without date chunking.

    Always uses LIMIT/OFFSET chunking to avoid statement timeouts on
    tables whose actual size doesn't match the pg_catalog estimate.
    """
    col_list = _build_col_list(info)
    return _download_table_offset_chunks(
        engine, info, output_dir, col_list, False, force, max_retries, retry_delay
    )


def _download_table_offset_chunks(
    engine: Engine,
    info: TableInfo,
    output_dir: Path,
    col_list: str,
    use_star: bool,
    force: bool,
    max_retries: int,
    retry_delay: float,
) -> int:
    """Download a large table without a date column using LIMIT/OFFSET."""
    total_rows = 0
    chunk_idx = 0

    while True:
        fname = f"{info.safe_name}_chunk{chunk_idx:04d}.parquet"
        output_path = output_dir / fname
        failed_marker = output_dir / f"{fname}.FAILED"

        # Resume (read row count from parquet metadata, not full file)
        if output_path.exists() and not force:
            n = _parquet_row_count(output_path)
            if n >= 0:
                total_rows += n
                if n < ROW_CHUNK_SIZE:
                    break  # Last chunk was partial → done
                chunk_idx += 1
                continue
            else:
                logger.warning(f"  Corrupt chunk file {fname}, re-downloading")
                output_path.unlink()

        offset = chunk_idx * ROW_CHUNK_SIZE
        select_cols = "*" if use_star else col_list
        sql = f"SELECT {select_cols} FROM {info.fqn} LIMIT {ROW_CHUNK_SIZE} OFFSET {offset}"

        chunk_t0 = time.time()
        for attempt in range(1, max_retries + 1):
            try:
                df = pd.read_sql(text(sql), engine)
                break
            except Exception as e:
                # If column metadata is stale, fall back to SELECT *
                if _is_column_error(e) and not use_star:
                    logger.warning(f"  Column metadata stale for {info.fqn}, falling back to SELECT *")
                    use_star = True
                    sql = f"SELECT * FROM {info.fqn} LIMIT {ROW_CHUNK_SIZE} OFFSET {offset}"
                    try:
                        df = pd.read_sql(text(sql), engine)
                        break
                    except Exception as e2:
                        if attempt == max_retries:
                            failed_marker.touch()
                            logger.error(f"  FAILED {fname}: {e2}")
                            return total_rows
                        time.sleep(retry_delay)
                        continue

                if attempt == max_retries:
                    failed_marker.touch()
                    logger.error(f"  FAILED {fname}: {e}")
                    return total_rows
                time.sleep(retry_delay)

        if len(df) == 0:
            break

        df.to_parquet(output_path, index=False)
        total_rows += len(df)
        chunk_elapsed = time.time() - chunk_t0
        logger.info(
            f"    {info.fqn} chunk {chunk_idx}: {len(df):,} rows "
            f"({chunk_elapsed:.1f}s, {total_rows:,} total so far)"
        )

        if len(df) < ROW_CHUNK_SIZE:
            break  # Last chunk

        chunk_idx += 1

    return total_rows


def download_all_tables(
    engine: Engine,
    tables: list[TableInfo],
    output_dir: Path,
    start_year: int = 1960,
    end_year: int | None = None,
    force: bool = False,
    max_rows: int | None = None,
    skip_schemas: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Download all discovered tables to Parquet files.

    Args:
        engine: SQLAlchemy engine connected to WRDS
        tables: List of TableInfo from discovery
        output_dir: Root output directory (e.g., /Volumes/MyDrive/wrds)
        start_year: Earliest year for date-based chunking
        end_year: Latest year (default: current year)
        force: Re-download even if files exist
        max_rows: Skip tables with estimated rows above this threshold
        skip_schemas: Additional schemas to skip

    Returns:
        Results dict: table FQN → {rows, status, error, elapsed_s}
    """
    if end_year is None:
        end_year = date.today().year

    skip = skip_schemas or set()
    accessible = [t for t in tables if t.accessible and t.schema not in skip]

    if max_rows is not None:
        skipped_large = [t for t in accessible if t.estimated_rows > max_rows]
        if skipped_large:
            logger.info(
                f"Skipping {len(skipped_large)} tables with >{max_rows:,} estimated rows: "
                f"{[t.fqn for t in skipped_large[:10]]}"
            )
        accessible = [t for t in accessible if t.estimated_rows <= max_rows]

    logger.info(f"Downloading {len(accessible)} accessible tables to {output_dir}")

    results: dict[str, dict[str, Any]] = {}

    for info in tqdm(accessible, desc="Downloading", unit="table"):
        # Create schema subdirectory
        table_dir = output_dir / info.schema
        table_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        est_size = format_bytes(info.estimated_bytes) if info.estimated_bytes else "unknown"
        logger.info(
            f"  ▶ {info.fqn} (~{info.estimated_rows:,} rows, ~{est_size}, "
            f"{len(info.columns)} cols, "
            f"{'date:' + info.date_column if info.date_column else 'no date col'})"
        )
        try:
            if info.date_column:
                # Determine chunk size based on estimated table size
                if info.estimated_rows > 100_000_000:
                    chunk_years = 1
                elif info.estimated_rows > 10_000_000:
                    chunk_years = 2
                else:
                    chunk_years = 5

                rows = download_table_with_dates(
                    engine, info, table_dir,
                    start_year, end_year, chunk_years,
                    force=force,
                )
            else:
                rows = download_table_full(
                    engine, info, table_dir, force=force,
                )

            elapsed = time.time() - t0
            results[info.fqn] = {
                "rows": rows,
                "status": "ok",
                "error": "",
                "elapsed_s": round(elapsed, 1),
                "date_column": info.date_column or "",
                "n_columns": len(info.columns),
            }

            if rows > 0:
                logger.info(
                    f"  {info.fqn}: {rows:,} rows "
                    f"({elapsed:.1f}s, {'date-chunked' if info.date_column else 'full'})"
                )

        except Exception as e:
            elapsed = time.time() - t0
            results[info.fqn] = {
                "rows": 0,
                "status": "error",
                "error": str(e)[:300],
                "elapsed_s": round(elapsed, 1),
                "date_column": info.date_column or "",
                "n_columns": len(info.columns),
            }
            logger.error(f"  {info.fqn}: FAILED — {e}")

    return results


# ── Reporting ────────────────────────────────────────────────────────────────

DISCOVERY_CACHE_FILENAME = "wrds_discovery_cache.json"


def save_discovery_cache(tables: list[TableInfo], output_path: Path) -> None:
    """Save discovery results to a JSON cache file for fast resume."""
    data = []
    for t in tables:
        data.append({
            "schema": t.schema,
            "table": t.table,
            "columns": t.columns,
            "date_column": t.date_column,
            "estimated_rows": t.estimated_rows,
            "accessible": t.accessible,
            "error": t.error,
        })
    with open(output_path, "w") as f:
        json.dump(data, f)
    logger.info(f"Discovery cache saved to {output_path} ({len(data)} tables)")


def load_discovery_cache(cache_path: Path) -> list[TableInfo]:
    """Load discovery results from a JSON cache file."""
    with open(cache_path) as f:
        data = json.load(f)
    tables = []
    for entry in data:
        tables.append(TableInfo(
            schema=entry["schema"],
            table=entry["table"],
            columns=entry.get("columns", []),
            date_column=entry.get("date_column"),
            estimated_rows=entry.get("estimated_rows", 0),
            accessible=entry.get("accessible", True),
            error=entry.get("error", ""),
        ))
    logger.info(f"Loaded discovery cache: {len(tables)} tables from {cache_path}")
    return tables


def save_catalog(tables: list[TableInfo], output_path: Path) -> None:
    """Save discovery results as a CSV catalog."""
    rows = []
    for t in tables:
        rows.append({
            "schema": t.schema,
            "table": t.table,
            "fqn": t.fqn,
            "accessible": t.accessible,
            "date_column": t.date_column or "",
            "estimated_rows": t.estimated_rows,
            "estimated_bytes": t.estimated_bytes,
            "estimated_size": format_bytes(t.estimated_bytes),
            "n_columns": len(t.columns),
            "column_names": "; ".join(c["name"] for c in t.columns),
            "error": t.error,
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    logger.info(f"Catalog saved to {output_path}")


def save_results(results: dict[str, dict[str, Any]], output_path: Path) -> None:
    """Save download results as JSON."""
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")


def print_discovery_summary(tables: list[TableInfo]) -> None:
    """Print a human-readable summary of discovered tables with size estimates."""
    schemas = {}
    for t in tables:
        if t.schema not in schemas:
            schemas[t.schema] = {"total": 0, "accessible": 0, "rows": 0, "bytes": 0}
        schemas[t.schema]["total"] += 1
        if t.accessible:
            schemas[t.schema]["accessible"] += 1
            schemas[t.schema]["rows"] += t.estimated_rows
            schemas[t.schema]["bytes"] += t.estimated_bytes

    print("\n" + "=" * 95)
    print("WRDS DISCOVERY SUMMARY")
    print("=" * 95)
    print(
        f"\n{'Schema':<25} {'Tables':>8} {'Accessible':>12} "
        f"{'Est. Rows':>15} {'Est. Size':>12}"
    )
    print("-" * 75)

    total_tables = 0
    total_accessible = 0
    total_rows = 0
    total_bytes = 0

    for schema in sorted(schemas):
        s = schemas[schema]
        total_tables += s["total"]
        total_accessible += s["accessible"]
        total_rows += s["rows"]
        total_bytes += s["bytes"]

        warn = " ⚠️ LARGE" if schema in WARN_SCHEMAS else ""
        print(
            f"  {schema:<23} {s['total']:>8} {s['accessible']:>12} "
            f"{s['rows']:>15,} {format_bytes(s['bytes']):>12}{warn}"
        )

    print("-" * 75)
    print(
        f"  {'TOTAL':<23} {total_tables:>8} {total_accessible:>12} "
        f"{total_rows:>15,} {format_bytes(total_bytes):>12}"
    )
    print()

    # Top 10 largest tables
    accessible_tables = [t for t in tables if t.accessible and t.estimated_bytes > 0]
    if accessible_tables:
        accessible_tables.sort(key=lambda t: t.estimated_bytes, reverse=True)
        print("Top 10 largest tables:")
        for t in accessible_tables[:10]:
            print(f"  {t.fqn:<45} {format_bytes(t.estimated_bytes):>12} ({t.estimated_rows:>15,} rows)")
        print()

    inaccessible = [t for t in tables if not t.accessible]
    if inaccessible:
        print(f"{len(inaccessible)} tables are NOT accessible (no subscription):")
        for t in inaccessible[:20]:
            print(f"  {t.fqn}: {t.error[:80]}")
        if len(inaccessible) > 20:
            print(f"  ... and {len(inaccessible) - 20} more")

    print()


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download ALL accessible WRDS datasets to local Parquet files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=None,
        help="Root output directory (e.g., /Volumes/MyDrive/wrds). "
             "Required unless --discover-only.",
    )
    parser.add_argument(
        "--libraries", "-l",
        nargs="+",
        default=None,
        help="Only download these WRDS libraries/schemas (e.g., crsp comp ibes).",
    )
    default_skip = [
        # TAQ tick data (enormous, 10+ TB)
        "taq", "taqmsec", "taqsamp", "taqsamp_all", "taqmsamp", "taqmsamp_all",
        # Contributor datasets (niche academic datasets, ~175 GB total)
        "contrib", "contrib_as_filed_financials", "contrib_bond_firm_link",
        "contrib_bond_firm_link_old", "contrib_char_returns",
        "contrib_corporate_culture", "contrib_general", "contrib_global_factor",
        "contrib_intangible_value", "contrib_kpss", "contrib_liva",
        # TRACE bond transaction data (~600 GB)
        "trace", "trace_enhanced", "trace_standard", "trace_standard_old",
    ]
    parser.add_argument(
        "--skip-libraries",
        nargs="+",
        default=default_skip,
        help="Skip these libraries. Defaults to TAQ and contrib schemas. "
             "Use --skip-libraries=none to skip nothing.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=1960,
        help="Earliest year for date-based chunking (default: 1960).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Skip tables with more estimated rows than this threshold.",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Re-download even if local files exist.",
    )
    parser.add_argument(
        "--discover-only", "-d",
        action="store_true",
        help="Only discover tables, don't download. Saves catalog CSV.",
    )
    parser.add_argument(
        "--no-access-check",
        action="store_true",
        help="Skip per-table access checks during discovery (faster but less accurate).",
    )
    parser.add_argument(
        "--skip-discovery",
        action="store_true",
        help="Skip the discovery phase and load from cache. "
             "Requires a previous run that saved wrds_discovery_cache.json "
             "in the output directory.",
    )
    parser.add_argument(
        "--rediscover",
        action="store_true",
        help="Force fresh discovery even if a cache exists.",
    )
    parser.add_argument(
        "--username",
        default=None,
        help="WRDS username (or set WRDS_USERNAME env var).",
    )
    parser.add_argument(
        "--password",
        default=None,
        help="WRDS password (or set WRDS_PASSWORD env var).",
    )

    args = parser.parse_args()

    # Allow --skip-libraries none to clear defaults
    if args.skip_libraries and len(args.skip_libraries) == 1 and args.skip_libraries[0].lower() == "none":
        args.skip_libraries = None

    return args


def main() -> None:
    args = parse_args()

    # Set up logging
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")

    # Resolve credentials
    username="YOUR_USERNAME"
    password="YOUR_PASSWORD"

    # Validate output dir
    if not args.discover_only and args.output_dir is None:
        logger.error("--output-dir is required (unless using --discover-only)")
        sys.exit(1)

    # Connect
    logger.info("Connecting to WRDS...")
    engine = connect(username, password)

    try:
        # ── Phase 1: Discovery (or load from cache) ──────────────────────
        catalog_dir = args.output_dir or Path(".")
        catalog_dir.mkdir(parents=True, exist_ok=True)
        cache_path = catalog_dir / DISCOVERY_CACHE_FILENAME

        # Decide whether to use cache or run fresh discovery
        use_cache = False
        if args.skip_discovery:
            if not cache_path.exists():
                logger.error(
                    f"--skip-discovery specified but no cache found at {cache_path}. "
                    f"Run without --skip-discovery first to build the cache."
                )
                sys.exit(1)
            use_cache = True
        elif not args.rediscover and cache_path.exists() and not args.discover_only:
            use_cache = True
            logger.info(f"Found discovery cache at {cache_path}, skipping rediscovery. "
                        f"Use --rediscover to force fresh discovery.")

        if use_cache:
            tables = load_discovery_cache(cache_path)
            # Apply --skip-libraries filter to cached tables
            if args.skip_libraries:
                skip_set = {lib.lower() for lib in args.skip_libraries}
                before = len(tables)
                tables = [t for t in tables if t.schema not in skip_set]
                if before != len(tables):
                    logger.info(f"Filtered out {before - len(tables)} tables from skipped libraries")
            # Apply --libraries filter to cached tables
            if args.libraries:
                requested = {lib.lower() for lib in args.libraries}
                tables = [t for t in tables if t.schema in requested]
            print_discovery_summary(tables)
        else:
            logger.info("Phase 1: Discovering all schemas and tables...")

            schemas = discover_schemas(engine, filter_libraries=args.libraries)

            # Apply --skip-libraries to discovery too (not just download).
            # Without this, schemas like taqmsec (36,000+ tables) would take
            # hours to enumerate even if they'll be skipped during download.
            if args.skip_libraries:
                skip_set = {lib.lower() for lib in args.skip_libraries}
                before = len(schemas)
                schemas = [s for s in schemas if s not in skip_set]
                if before != len(schemas):
                    logger.info(
                        f"Skipping {before - len(schemas)} libraries from discovery: "
                        f"{sorted(skip_set & {s for s in schemas} | skip_set)}"
                    )

            tables = discover_all(
                engine, schemas,
                check_access=not args.no_access_check,
            )

            # Print summary
            print_discovery_summary(tables)

            # Save discovery cache and catalog
            save_discovery_cache(tables, cache_path)
            save_catalog(tables, catalog_dir / "wrds_catalog.csv")

        if args.discover_only:
            accessible = [t for t in tables if t.accessible]
            total_est_rows = sum(t.estimated_rows for t in accessible)
            total_est_bytes = sum(t.estimated_bytes for t in accessible)
            print(f"Discovery complete. {len(accessible)} accessible tables, "
                  f"~{total_est_rows:,} estimated total rows, "
                  f"~{format_bytes(total_est_bytes)} estimated on disk.")
            print(f"Catalog saved to {catalog_dir / 'wrds_catalog.csv'}")
            return

        # ── Phase 2: Download ────────────────────────────────────────────
        logger.info("Phase 2: Downloading all accessible tables...")

        skip = set(args.skip_libraries) if args.skip_libraries else set()

        results = download_all_tables(
            engine=engine,
            tables=tables,
            output_dir=args.output_dir,
            start_year=args.start_year,
            force=args.force,
            max_rows=args.max_rows,
            skip_schemas=skip,
        )

        # Save results
        save_results(results, args.output_dir / "download_results.json")

        # Final summary
        ok = sum(1 for r in results.values() if r["status"] == "ok" and r["rows"] > 0)
        empty = sum(1 for r in results.values() if r["status"] == "ok" and r["rows"] == 0)
        failed = sum(1 for r in results.values() if r["status"] == "error")
        total_rows = sum(r["rows"] for r in results.values())
        total_time = sum(r["elapsed_s"] for r in results.values())

        print("\n" + "=" * 60)
        print("DOWNLOAD COMPLETE")
        print("=" * 60)
        print(f"  Downloaded:  {ok} tables ({total_rows:,} total rows)")
        print(f"  Empty:       {empty} tables (no data)")
        print(f"  Failed:      {failed} tables")
        print(f"  Total time:  {total_time/60:.1f} minutes")
        print(f"  Output:      {args.output_dir}")
        print()

    finally:
        engine.dispose()
        logger.info("Connection closed")


if __name__ == "__main__":
    main()
