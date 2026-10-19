import duckdb, json, pathlib
from src.config import table_glob

con = duckdb.connect()
tables = [
    "asi_opensource_server_hourly",
    "asi_opensource_job_execution_summary",
    "asi_opensource_network_hourly",
]

schema = {}
for table in tables:
    glob = table_glob(table)
    try:
        cols = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{glob}') LIMIT 1").fetchall()
        count = con.execute(f"SELECT COUNT(*) FROM read_parquet('{glob}')").fetchone()[0]
        sample = con.execute(f"SELECT * FROM read_parquet('{glob}') LIMIT 3").fetchdf()
        schema[table] = {
            "columns": [{"name": r[0], "type": r[1]} for r in cols],
            "row_count": count,
            "sample_rows": sample.to_dict(orient="records")
        }
        print(f"\n{'='*60}")
        print(f"TABLE: {table}")
        print(f"Rows: {count:,}")
        print(f"Columns ({len(cols)}):")
        for r in cols:
            print(f"  {r[0]:<40} {r[1]}")
        print(f"Sample row:")
        print(sample.iloc[0].to_string())
    except Exception as e:
        schema[table] = {"error": str(e)}
        print(f"{table}: FAILED — {e}")

pathlib.Path("outputs/logs").mkdir(parents=True, exist_ok=True)
with open("outputs/logs/discovered_schema.json", "w") as f:
    json.dump(schema, f, indent=2, default=str)
print(f"\nWritten to outputs/logs/discovered_schema.json")
