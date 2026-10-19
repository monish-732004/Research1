"""Tests for src.ingest.schema — schema_diff correctness."""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.ingest.schema import SCHEMAS, schema_diff, SchemaMismatch


# ── Helpers ───────────────────────────────────────────────────────────────────

def _write_parquet(schema: pa.Schema, n_rows: int = 10) -> Path:
    """Write a minimal valid Parquet file with *schema* to a temp path."""
    arrays = []
    for field in schema:
        if pa.types.is_integer(field.type):
            arr = pa.array(list(range(n_rows)), type=field.type)
        elif pa.types.is_floating(field.type):
            arr = pa.array([float(i) * 0.1 for i in range(n_rows)], type=field.type)
        elif pa.types.is_string(field.type) or pa.types.is_large_string(field.type):
            arr = pa.array([f"val_{i}" for i in range(n_rows)], type=field.type)
        elif pa.types.is_dictionary(field.type):
            # Build a small dictionary array
            indices = pa.array([i % 3 for i in range(n_rows)], type=field.type.index_type)
            dictionary = pa.array(["a", "b", "c"], type=field.type.value_type)
            arr = pa.DictionaryArray.from_arrays(indices, dictionary)
        else:
            arr = pa.array([None] * n_rows, type=field.type)
        arrays.append(arr)

    table = pa.table(dict(zip([f.name for f in schema], arrays)))
    tmp = Path(tempfile.mktemp(suffix=".parquet"))
    pq.write_table(table, tmp)
    return tmp


# ── Tests: schema_diff ────────────────────────────────────────────────────────

class TestSchemaDiffPassingSchemas:
    """Parquet files whose schema exactly matches the spec → zero mismatches."""

    @pytest.mark.parametrize("table_name", list(SCHEMAS.keys()))
    def test_exact_match(self, table_name: str) -> None:
        expected = SCHEMAS[table_name]
        path = _write_parquet(expected.pa_schema())
        actual_schema = pq.read_schema(path)
        mismatches = schema_diff(table_name, actual_schema)
        assert mismatches == [], (
            f"{table_name}: expected no mismatches, got:\n"
            + "\n".join(str(m) for m in mismatches)
        )
        path.unlink(missing_ok=True)

    def test_extra_columns_not_flagged_by_default(self) -> None:
        """Extra columns in actual are silently ignored unless strict_extra=True."""
        base = SCHEMAS["machine_events"].pa_schema()
        extra = pa.schema(list(base) + [pa.field("extra_col", pa.string())])
        path = _write_parquet(extra)
        actual_schema = pq.read_schema(path)
        mismatches = schema_diff("machine_events", actual_schema, strict_extra=False)
        assert mismatches == []
        path.unlink(missing_ok=True)

    def test_extra_columns_flagged_in_strict_mode(self) -> None:
        base = SCHEMAS["machine_events"].pa_schema()
        extra = pa.schema(list(base) + [pa.field("extra_col", pa.string())])
        path = _write_parquet(extra)
        actual_schema = pq.read_schema(path)
        mismatches = schema_diff("machine_events", actual_schema, strict_extra=True)
        kinds = [m.kind for m in mismatches]
        assert "extra_column" in kinds
        assert any(m.column == "extra_col" for m in mismatches)
        path.unlink(missing_ok=True)

    def test_dictionary_value_type_compatible(self) -> None:
        """Columns declared as dictionary(int8, string) are compatible with plain string."""
        plain_schema = pa.schema([
            pa.field("machine_id",  pa.string()),
            pa.field("event_type",  pa.string()),   # no dictionary encoding
            pa.field("timestamp",   pa.int64()),
            pa.field("region",      pa.string()),
            pa.field("gpu_count",   pa.int32()),
            pa.field("gpu_model",   pa.string()),
            pa.field("cpu_count",   pa.int32()),
            pa.field("memory_gb",   pa.float32()),
        ])
        path = _write_parquet(plain_schema)
        actual_schema = pq.read_schema(path)
        mismatches = schema_diff("machine_events", actual_schema)
        assert mismatches == []
        path.unlink(missing_ok=True)


class TestSchemaDiffFailingSchemas:
    """Parquet files with schema violations → expected mismatches reported."""

    def test_missing_required_column(self) -> None:
        """Dropping a required column → 'missing_column' mismatch."""
        full = SCHEMAS["resource_usage"].pa_schema()
        # Remove 'gpu_util'
        reduced = pa.schema([f for f in full if f.name != "gpu_util"])
        path = _write_parquet(reduced)
        actual_schema = pq.read_schema(path)
        mismatches = schema_diff("resource_usage", actual_schema)
        assert any(
            m.kind == "missing_column" and m.column == "gpu_util"
            for m in mismatches
        ), f"Expected missing_column for gpu_util, got: {mismatches}"
        path.unlink(missing_ok=True)

    def test_wrong_type_integer_instead_of_float(self) -> None:
        """Replacing a float32 column with int32 → 'type_mismatch'."""
        expected = SCHEMAS["resource_usage"]
        wrong_schema = pa.schema([
            pa.field(c.name, pa.int32() if c.name == "gpu_util" else c.dtype)
            for c in expected.columns
        ])
        path = _write_parquet(wrong_schema)
        actual_schema = pq.read_schema(path)
        mismatches = schema_diff("resource_usage", actual_schema)
        assert any(
            m.kind == "type_mismatch" and m.column == "gpu_util"
            for m in mismatches
        ), f"Expected type_mismatch for gpu_util, got: {mismatches}"
        path.unlink(missing_ok=True)

    def test_multiple_missing_columns(self) -> None:
        """Dropping two columns → two 'missing_column' mismatches."""
        full = SCHEMAS["instance_events"].pa_schema()
        reduced = pa.schema([f for f in full if f.name not in ("machine_id", "status")])
        path = _write_parquet(reduced)
        actual_schema = pq.read_schema(path)
        mismatches = schema_diff("instance_events", actual_schema)
        missing = {m.column for m in mismatches if m.kind == "missing_column"}
        assert "machine_id" in missing
        assert "status" in missing
        path.unlink(missing_ok=True)

    def test_wrong_type_for_timestamp(self) -> None:
        """Using float64 for a timestamp column → type_mismatch."""
        expected = SCHEMAS["resource_request"]
        wrong_schema = pa.schema([
            pa.field(c.name, pa.float64() if c.name == "submit_time" else c.dtype)
            for c in expected.columns
        ])
        path = _write_parquet(wrong_schema)
        actual_schema = pq.read_schema(path)
        mismatches = schema_diff("resource_request", actual_schema)
        assert any(
            m.kind == "type_mismatch" and m.column == "submit_time"
            for m in mismatches
        )
        path.unlink(missing_ok=True)


class TestSchemaDiffEdgeCases:
    def test_unknown_table_raises(self) -> None:
        with pytest.raises(KeyError, match="no_such_table"):
            schema_diff("no_such_table", pa.schema([]))

    def test_empty_actual_schema_reports_all_missing(self) -> None:
        mismatches = schema_diff("task_metadata", pa.schema([]))
        expected_cols = set(SCHEMAS["task_metadata"].column_map().keys())
        missing_cols = {m.column for m in mismatches if m.kind == "missing_column"}
        assert expected_cols == missing_cols
