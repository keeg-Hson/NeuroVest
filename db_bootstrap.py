#!/usr/bin/env python3
# db_bootstrap.py - build a tiny DuckDB from CSVs

import pathlib

import duckdb

print("DuckDB version:", duckdb.__version__)
DB = "neurovest.duckdb"
DATA = pathlib.Path("data")
LOGS = pathlib.Path("logs")

con = duckdb.connect(DB)


def load_prices(name: str):
    csv = DATA / f"{name}.csv"
    if not csv.exists():
        print(f"[skip] {csv} not found")
        return
    con.execute(
        f"""
        CREATE OR REPLACE TABLE {name}_prices AS
        SELECT
          CAST(Date AS DATE)                AS Date,
          CAST(Open AS DOUBLE)              AS Open,
          CAST(High AS DOUBLE)              AS High,
          CAST(Low  AS DOUBLE)              AS Low,
          CAST("Close" AS DOUBLE)           AS Close,
          CAST("Adj Close" AS DOUBLE)       AS AdjClose,
          CAST(Volume AS BIGINT)            AS Volume
        FROM read_csv_auto('{csv}', DATEFORMAT='%Y-%m-%d', HEADER=TRUE);
    """
    )
    con.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS idx_{name}_prices_date ON {name}_prices(Date);")
    print(f"✅ loaded {csv}")


def _clean_to_temp(path: pathlib.Path) -> pathlib.Path | None:
    """Return a temp path with BOM/nulls stripped if needed, else None."""
    b = path.read_bytes()
    # empty or whitespace-only
    if not b or b.strip() == b"":
        return None
    changed = False
    # strip UTF-8 BOM
    if b.startswith(b"\xef\xbb\xbf"):
        b = b[3:]
        changed = True
    # remove NUL bytes
    if b.find(b"\x00") != -1:
        b = b.replace(b"\x00", b"")
        changed = True
    if not changed:
        return None
    tmp = path.with_suffix(path.suffix + ".tmp_clean")
    tmp.write_bytes(b)
    return tmp


def load_predictions():
    import csv

    pred_path = LOGS / "daily_predictions.csv"
    if not pred_path.exists():
        print(f"[skip] {pred_path} not found")
        return
    if pred_path.stat().st_size == 0:
        print(f"[skip] {pred_path} is empty")
        return

    def _clean_to_temp(path: pathlib.Path) -> pathlib.Path | None:
        b = path.read_bytes()
        if not b or b.strip() == b"":
            return None
        changed = False
        # strip UTF-8 BOM
        if b.startswith(b"\xef\xbb\xbf"):
            b = b[3:]
            changed = True
        # strip NULLs
        if b.find(b"\x00") != -1:
            b = b.replace(b"\x00", b"")
            changed = True
        if not changed:
            return None
        tmp = path.with_suffix(path.suffix + ".tmp_clean")
        tmp.write_bytes(b)
        return tmp

    def _sniff_delim(path: pathlib.Path) -> str:
        # default to comma if sniff fails
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                sample = f.read(4096)
            d = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
            return d.delimiter
        except Exception:
            return ","

    cleaned = _clean_to_temp(pred_path) or pred_path

    # First: try DuckDB read_csv with forgiving settings
    try:
        con.execute(
            f"""
            CREATE OR REPLACE TABLE predictions AS
            SELECT
              CAST(COALESCE(Date, Timestamp) AS DATE) AS Date,
              CAST(Prediction AS BIGINT)              AS Prediction,
              CAST(Spike_Conf AS DOUBLE)              AS Spike_Conf,
              CAST(Crash_Conf AS DOUBLE)              AS Crash_Conf,
              Timestamp
            FROM read_csv(
              '{cleaned}',
              auto_detect=true,
              header=true,
              delim='{_sniff_delim(cleaned)}',
              quote='\"',
              escape='\"',
              ignore_errors=true,
              null_padding=true,
              sample_size=-1
            );
        """
        )
        print("✅ loaded predictions via DuckDB CSV reader")
    except Exception as e1:
        print(f"[warn] DuckDB read_csv failed: {e1}\n→ falling back to pandas parser")
        # Fallback: pandas engine='python' (DON'T pass low_memory)
        import pandas as pd

        try:
            df = pd.read_csv(
                cleaned,
                engine="python",
                sep=None,  # sniff delimiter
                on_bad_lines="skip",
                encoding_errors="replace",
            )
            if df.empty:
                print(f"[skip] {pred_path} parsed empty after cleaning")
                return

            # Normalize expected columns
            lower = {c.lower(): c for c in df.columns}
            ren = {}
            for want, alts in {
                "Date": ["date"],
                "Timestamp": ["timestamp"],
                "Prediction": ["prediction", "pred"],
                "Spike_Conf": ["spike_conf", "spikeprob", "trade_conf"],
                "Crash_Conf": ["crash_conf", "crashprob"],
            }.items():
                if want not in df.columns:
                    for a in alts:
                        if a in lower:
                            ren[lower[a]] = want
                            break
            if ren:
                df.rename(columns=ren, inplace=True)

            # Coerce types
            if "Date" in df.columns:
                df["Date"] = (
                    pd.to_datetime(df["Date"], errors="coerce").dt.tz_localize(None).dt.date
                )
            elif "Timestamp" in df.columns:
                df["Date"] = (
                    pd.to_datetime(df["Timestamp"], errors="coerce").dt.tz_localize(None).dt.date
                )
            else:
                raise ValueError("No Date or Timestamp column in predictions CSV")

            if "Timestamp" in df.columns:
                df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce").dt.tz_localize(
                    None
                )

            for c in ("Prediction", "Spike_Conf", "Crash_Conf"):
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")

            keep = [
                c
                for c in ["Date", "Prediction", "Spike_Conf", "Crash_Conf", "Timestamp"]
                if c in df.columns
            ]
            df = df[keep].dropna(subset=["Date"])
            if "Timestamp" in df.columns:
                df = df.sort_values(["Date", "Timestamp"]).drop_duplicates(
                    subset=["Date"], keep="last"
                )
            else:
                df = df.drop_duplicates(subset=["Date"], keep="last")

            # Register → materialize to DuckDB
            con.register("pred_df", df)
            con.execute(
                """
                CREATE OR REPLACE TABLE predictions AS
                SELECT
                  CAST(Date AS DATE)           AS Date,
                  CAST(Prediction AS BIGINT)   AS Prediction,
                  CAST(Spike_Conf AS DOUBLE)   AS Spike_Conf,
                  CAST(Crash_Conf AS DOUBLE)   AS Crash_Conf,
                  Timestamp
                FROM pred_df;
            """
            )
            con.unregister("pred_df")
            print("✅ loaded predictions via pandas → DuckDB")
        except Exception as e2:
            print(f"❌ Could not load predictions with pandas either: {e2}")
            return
    finally:
        if cleaned is not pred_path and isinstance(cleaned, pathlib.Path) and cleaned.exists():
            try:
                cleaned.unlink()
            except Exception:
                pass

    # one row per day constraint via unique index
    try:
        con.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_predictions_date ON predictions(Date);")
    except Exception as e:
        print(f"[warn] index on predictions(Date): {e}")


# ---- run loads ----
for sym in ["SPY", "HYG", "LQD", "UUP", "TNX", "DXY"]:
    load_prices(sym)

load_predictions()

# ---- handy views ----
con.execute(
    """
CREATE OR REPLACE VIEW spy_labels AS
WITH b AS (
  SELECT Date, COALESCE(AdjClose, Close) AS Close
  FROM SPY_prices
  WHERE Date IS NOT NULL
  ORDER BY Date
),
f AS (
  SELECT
    Date,
    Close,
    LEAD(Close, 1) OVER (ORDER BY Date) AS NextClose,
    LEAD(Close, 5) OVER (ORDER BY Date) AS NextClose5
  FROM b
)
SELECT
  Date,
  Close,
  (NextClose  / Close - 1.0) AS FwdRet_1d,
  (NextClose5 / Close - 1.0) AS FwdRet_5d,
  CASE WHEN (NextClose / Close - 1.0) > 0.005 THEN 1 ELSE 0 END AS Actual_Event_1d
FROM f;
"""
)

con.execute(
    """
CREATE OR REPLACE VIEW eval_join AS
SELECT
  p.Date,
  p.Prediction,
  p.Spike_Conf,
  p.Crash_Conf,
  CASE
    WHEN p.Prediction = 2 THEN 1
    WHEN p.Prediction IN (0,1) THEN 0
    ELSE NULL
  END AS PredLong,
  s.FwdRet_1d,
  s.Actual_Event_1d,
  CASE
    WHEN PredLong = 1 AND s.Actual_Event_1d = 1 THEN 'TP'
    WHEN PredLong = 1 AND s.Actual_Event_1d = 0 THEN 'FP'
    WHEN PredLong = 0 AND s.Actual_Event_1d = 0 THEN 'TN'
    WHEN PredLong = 0 AND s.Actual_Event_1d = 1 THEN 'FN'
    ELSE '?'
  END AS Outcome
FROM predictions p
LEFT JOIN spy_labels s USING(Date)
ORDER BY Date;
"""
)

con.close()
print("🎉 neurovest.duckdb ready")
