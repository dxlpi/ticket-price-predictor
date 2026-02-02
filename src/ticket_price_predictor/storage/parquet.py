"""Low-level Parquet I/O utilities."""

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel


def _model_to_dict(model: BaseModel) -> dict[str, Any]:
    """Convert a Pydantic model to a dictionary suitable for PyArrow."""
    data = model.model_dump()
    # Convert enums to their values
    for key, value in data.items():
        if hasattr(value, "value"):
            data[key] = value.value
        # Ensure datetime is timezone-aware for PyArrow
        if isinstance(value, datetime) and value.tzinfo is None:
            data[key] = value.replace(tzinfo=UTC)
    return data


def write_parquet(
    data: Sequence[BaseModel],
    path: Path,
    schema: pa.Schema,
) -> int:
    """Write a list of Pydantic models to a Parquet file.

    Args:
        data: List of Pydantic models to write
        path: Path to the Parquet file
        schema: PyArrow schema for the data

    Returns:
        Number of rows written
    """
    if not data:
        return 0

    # Convert models to dictionaries
    records = [_model_to_dict(item) for item in data]

    # Create PyArrow table
    table = pa.Table.from_pylist(records, schema=schema)

    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write to Parquet
    pq.write_table(table, path)

    return len(data)


def read_parquet(
    path: Path,
    filters: list[tuple[str, str, Any]] | None = None,
) -> pa.Table | None:
    """Read a Parquet file or directory.

    Args:
        path: Path to Parquet file or directory
        filters: Optional list of filters as (column, op, value) tuples

    Returns:
        PyArrow Table or None if file doesn't exist
    """
    if not path.exists():
        return None

    # Convert filter tuples to PyArrow filter expressions
    if filters:
        return pq.read_table(path, filters=filters)
    return pq.read_table(path)


def append_parquet(
    data: Sequence[BaseModel],
    path: Path,
    schema: pa.Schema,
) -> int:
    """Append data to an existing Parquet file.

    If the file doesn't exist, creates a new one.

    Args:
        data: List of Pydantic models to append
        path: Path to the Parquet file
        schema: PyArrow schema for the data

    Returns:
        Total number of rows after append
    """
    if not data:
        existing = read_parquet(path)
        return existing.num_rows if existing else 0

    # Convert new data to table
    records = [_model_to_dict(item) for item in data]
    new_table = pa.Table.from_pylist(records, schema=schema)

    # Read existing data if present
    existing = read_parquet(path)

    if existing is not None:
        # Select only the columns that match the schema to handle partition columns
        existing_cols = set(existing.schema.names)
        new_cols = set(new_table.schema.names)
        common_cols = list(existing_cols & new_cols)

        # If schemas differ, use only common columns
        if existing.schema != new_table.schema:
            existing = existing.select(common_cols)
            new_table = new_table.select(common_cols)

        combined = pa.concat_tables([existing, new_table], promote_options="default")
    else:
        combined = new_table

    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write combined data
    pq.write_table(combined, path)

    return cast(int, combined.num_rows)


def table_to_dicts(table: pa.Table) -> list[dict[str, Any]]:
    """Convert a PyArrow table to a list of dictionaries.

    Args:
        table: PyArrow table

    Returns:
        List of dictionaries
    """
    return cast(list[dict[str, Any]], table.to_pylist())
