"""DuckDB schema definitions for each Alibaba trace table.

These are used both for CREATE TABLE statements and for read_parquet()
column casting so that DuckDB infers the right types even when Parquet
metadata is absent or mismatched.
"""

# Each entry maps table_name → list of (column_name, duckdb_type)
SCHEMAS: dict[str, list[tuple[str, str]]] = {
    "machine_spec": [
        ("machine_id",   "VARCHAR"),
        ("region",       "VARCHAR"),
        ("gpu_count",    "INTEGER"),
        ("gpu_model",    "VARCHAR"),
        ("cpu_count",    "INTEGER"),
        ("memory_gb",    "FLOAT"),
    ],
    "job_table": [
        ("job_id",          "VARCHAR"),
        ("user_id",         "VARCHAR"),
        ("submit_time",     "BIGINT"),   # Unix seconds
        ("start_time",      "BIGINT"),
        ("end_time",        "BIGINT"),
        ("status",          "VARCHAR"),
        ("gpu_request",     "INTEGER"),
        ("cpu_request",     "FLOAT"),
        ("memory_request",  "FLOAT"),
    ],
    "task_table": [
        ("task_id",         "VARCHAR"),
        ("job_id",          "VARCHAR"),
        ("machine_id",      "VARCHAR"),
        ("start_time",      "BIGINT"),
        ("end_time",        "BIGINT"),
        ("status",          "VARCHAR"),
        ("gpu_assigned",    "INTEGER"),
    ],
    "instance_table": [
        ("instance_id",     "VARCHAR"),
        ("task_id",         "VARCHAR"),
        ("machine_id",      "VARCHAR"),
        ("start_time",      "BIGINT"),
        ("end_time",        "BIGINT"),
    ],
    "machine_metric": [
        ("machine_id",      "VARCHAR"),
        ("timestamp",       "BIGINT"),   # Unix seconds
        ("gpu_util",        "FLOAT"),    # 0.0–1.0
        ("gpu_mem_util",    "FLOAT"),
        ("cpu_util",        "FLOAT"),
        ("mem_util",        "FLOAT"),
    ],
}


def create_ddl(table: str, schema: list[tuple[str, str]]) -> str:
    cols = ",\n    ".join(f"{name} {dtype}" for name, dtype in schema)
    return f"CREATE TABLE IF NOT EXISTS {table} (\n    {cols}\n);"


def all_ddl() -> str:
    return "\n\n".join(create_ddl(t, s) for t, s in SCHEMAS.items())
