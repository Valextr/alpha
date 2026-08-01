from pathlib import Path

import duckdb

DB_PATH = Path(__file__).resolve().parent / "data" / "alpha.duckdb"
conn = duckdb.connect(str(DB_PATH))

# Check stats
for view in ["bronze_daily", "bronze_dividends", "silver_daily", "gold_daily", "universe_daily"]:
    try:
        result = conn.execute(f"SELECT COUNT(*) FROM {view}").fetchone()
        print(f"{view}: {result[0]} rows")
    except Exception as e:
        print(f"{view}: {e}")

print()
print("=== Gold layer sample ===")
print(conn.execute("SELECT * FROM gold_daily WHERE ticker = 'AAPL' LIMIT 5").fetchdf().to_string())

print()
print("=== Tickers ===")
print(conn.execute("SELECT DISTINCT ticker FROM gold_daily ORDER BY ticker").fetchdf().to_string())

print()
print("=== Date range ===")
print(conn.execute("SELECT ticker, MIN(date) as first, MAX(date) as last FROM gold_daily GROUP BY ticker").fetchdf().to_string())

conn.close()
