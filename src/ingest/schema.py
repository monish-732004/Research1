"""Expected schemas for every Alibaba cluster-trace table.

Usage
-----
    from src.ingest.schema import SCHEMAS, schema_diff
    mismatches = schema_diff("resource_usage", actual_pyarrow_schema)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pyarrow as pa


# ── Column-level spec ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ColumnSpec:
    name: str
    dtype: pa.DataType          # expected PyArrow type
    nullable: bool = True
    is_pk: bool = False         # part of logical primary key


# ── Table-level spec ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TableSchema:
    name: str
    description: str
    columns: list[ColumnSpec] = field(default_factory=list)

    def pa_schema(self) -> pa.Schema:
        return pa.schema(
            [(c.name, c.dtype) for c in self.columns],
            metadata={"table": self.name},
        )

    def column_map(self) -> dict[str, ColumnSpec]:
        return {c.name: c for c in self.columns}


# ── Table definitions ─────────────────────────────────────────────────────────

SCHEMAS: dict[str, TableSchema] = {

    "instance_events": TableSchema(
        name="instance_events",
        description="Job lifecycle events: submitted, scheduled, started, finished, failed, killed.",
        columns=[
            ColumnSpec("job_id",       pa.string(),              nullable=False, is_pk=True),
            ColumnSpec("task_id",      pa.string(),              nullable=False, is_pk=True),
            ColumnSpec("instance_id",  pa.string(),              nullable=False, is_pk=True),
            ColumnSpec("event_type",   pa.dictionary(pa.int8(), pa.string()), nullable=False, is_pk=True),
            ColumnSpec("timestamp",    pa.int64(),               nullable=False, is_pk=True),  # Unix seconds
            ColumnSpec("machine_id",   pa.string(),              nullable=True),
            ColumnSpec("status",       pa.dictionary(pa.int8(), pa.string()), nullable=True),
        ],
    ),

    "machine_events": TableSchema(
        name="machine_events",
        description="Server availability events: ADD, REMOVE, UPDATE (hardware repair / capacity change).",
        columns=[
            ColumnSpec("machine_id",   pa.string(),              nullable=False, is_pk=True),
            ColumnSpec("event_type",   pa.dictionary(pa.int8(), pa.string()), nullable=False, is_pk=True),
            ColumnSpec("timestamp",    pa.int64(),               nullable=False, is_pk=True),
            ColumnSpec("region",       pa.string(),              nullable=True),
            ColumnSpec("gpu_count",    pa.int32(),               nullable=True),
            ColumnSpec("gpu_model",    pa.string(),              nullable=True),
            ColumnSpec("cpu_count",    pa.int32(),               nullable=True),
            ColumnSpec("memory_gb",    pa.float32(),             nullable=True),
        ],
    ),

    "resource_usage": TableSchema(
        name="resource_usage",
        description="5-minute sampled GPU/CPU/memory utilisation per instance.",
        columns=[
            ColumnSpec("instance_id",    pa.string(),   nullable=False, is_pk=True),
            ColumnSpec("timestamp",      pa.int64(),    nullable=False, is_pk=True),
            ColumnSpec("gpu_util",       pa.float32(),  nullable=True),   # 0.0–1.0
            ColumnSpec("gpu_mem_util",   pa.float32(),  nullable=True),
            ColumnSpec("cpu_util",       pa.float32(),  nullable=True),
            ColumnSpec("mem_util",       pa.float32(),  nullable=True),
            ColumnSpec("net_in_mbps",    pa.float32(),  nullable=True),
            ColumnSpec("net_out_mbps",   pa.float32(),  nullable=True),
        ],
    ),

    "resource_request": TableSchema(
        name="resource_request",
        description="Resource requests made at job submit time.",
        columns=[
            ColumnSpec("job_id",          pa.string(),   nullable=False, is_pk=True),
            ColumnSpec("task_id",         pa.string(),   nullable=False, is_pk=True),
            ColumnSpec("submit_time",     pa.int64(),    nullable=False),
            ColumnSpec("gpu_request",     pa.int32(),    nullable=False),
            ColumnSpec("cpu_request",     pa.float32(),  nullable=True),
            ColumnSpec("memory_request",  pa.float32(),  nullable=True),  # GiB
            ColumnSpec("gpu_memory_request", pa.float32(), nullable=True),  # GiB
            ColumnSpec("duration_hint",   pa.int64(),    nullable=True),   # seconds, may be absent
        ],
    ),

    "task_metadata": TableSchema(
        name="task_metadata",
        description="Per-job metadata: workload type, framework, priority.",
        columns=[
            ColumnSpec("job_id",       pa.string(),              nullable=False, is_pk=True),
            ColumnSpec("task_id",      pa.string(),              nullable=False, is_pk=True),
            ColumnSpec("job_type",     pa.dictionary(pa.int8(), pa.string()), nullable=False),
            # job_type values: "training", "inference", "data-processing", "other"
            ColumnSpec("framework",    pa.string(),              nullable=True),
            ColumnSpec("priority",     pa.int32(),               nullable=True),
            ColumnSpec("user_id",      pa.string(),              nullable=True),
            ColumnSpec("submit_time",  pa.int64(),               nullable=True),
            ColumnSpec("region",       pa.string(),              nullable=True),
        ],
    ),
}


# ── Schema diff ───────────────────────────────────────────────────────────────

@dataclass
class SchemaMismatch:
    kind: str       # "missing_column" | "extra_column" | "type_mismatch" | "nullability"
    column: str
    expected: str | None = None
    actual: str | None = None

    def __str__(self) -> str:
        if self.kind == "missing_column":
            return f"[missing_column] '{self.column}' expected {self.expected} but not found in actual"
        if self.kind == "extra_column":
            return f"[extra_column]   '{self.column}' present in actual but not in expected schema"
        if self.kind == "type_mismatch":
            return f"[type_mismatch]  '{self.column}' expected {self.expected}, got {self.actual}"
        if self.kind == "nullability":
            return f"[nullability]    '{self.column}' expected nullable={self.expected}, got nullable={self.actual}"
        return f"[{self.kind}] {self.column}"


def _type_compatible(expected: pa.DataType, actual: pa.DataType) -> bool:
    """True if actual satisfies the expected type.

    Dictionary-encoded columns are compatible with their value type
    (Parquet producers sometimes skip dictionary encoding).
    """
    if pa.types.is_dictionary(expected):
        expected = expected.value_type
    if pa.types.is_dictionary(actual):
        actual = actual.value_type
    return expected == actual


def schema_diff(
    table_name: str,
    actual: pa.Schema,
    *,
    strict_extra: bool = False,
) -> list[SchemaMismatch]:
    """Compare *actual* PyArrow schema against the expected spec for *table_name*.

    Parameters
    ----------
    table_name:
        Key into SCHEMAS.
    actual:
        PyArrow schema read from the real Parquet file.
    strict_extra:
        If True, columns present in *actual* but absent from the spec are
        reported as ``extra_column`` mismatches (useful for auditing surprises).

    Returns
    -------
    List of :class:`SchemaMismatch` objects; empty means the schemas match.
    """
    if table_name not in SCHEMAS:
        raise KeyError(f"No schema defined for table '{table_name}'. "
                       f"Known tables: {list(SCHEMAS)}")

    expected_schema = SCHEMAS[table_name]
    expected_cols = expected_schema.column_map()
    actual_fields = {f.name: f for f in actual}

    mismatches: list[SchemaMismatch] = []

    # Required columns present and typed correctly?
    for col_name, spec in expected_cols.items():
        if col_name not in actual_fields:
            mismatches.append(SchemaMismatch(
                kind="missing_column",
                column=col_name,
                expected=str(spec.dtype),
            ))
            continue

        af = actual_fields[col_name]
        if not _type_compatible(spec.dtype, af.type):
            mismatches.append(SchemaMismatch(
                kind="type_mismatch",
                column=col_name,
                expected=str(spec.dtype),
                actual=str(af.type),
            ))

    # Extra columns in actual?
    if strict_extra:
        for col_name in actual_fields:
            if col_name not in expected_cols:
                mismatches.append(SchemaMismatch(
                    kind="extra_column",
                    column=col_name,
                    actual=str(actual_fields[col_name].type),
                ))

    return mismatches
