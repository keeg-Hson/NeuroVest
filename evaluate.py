#!/usr/bin/env python3
# evaluate.py — prefer DuckDB eval_join, fallback to CSV merge

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report

LABELED_LOG = Path("logs/labeled_predictions.csv")
METRIC_OUTPUT_CSV = Path("logs/model_performance.csv")
DB_PATH = Path("neurovest.duckdb")


def _from_duckdb() -> pd.DataFrame:
    import duckdb  # will raise ImportError if missing

    if not DB_PATH.exists():
        raise FileNotFoundError("DuckDB file not found")
    con = duckdb.connect(str(DB_PATH))
    # eval_join has: Date, Prediction, Spike_Conf, Crash_Conf, PredLong, FwdRet_1d, Actual_Event_1d, Outcome
    df = con.execute("SELECT * FROM eval_join ORDER BY Date").df()
    con.close()
    if df.empty:
        raise ValueError("eval_join returned no rows")
    # Normalize column names to what the old evaluator expects
    if "Actual_Event_1d" in df.columns and "Actual_Event" not in df.columns:
        df = df.rename(columns={"Actual_Event_1d": "Actual_Event"})
    return df


def _from_csv_pipeline() -> pd.DataFrame:
    # Fallback = reuse your previous CSV-based labeled log if it exists
    p = LABELED_LOG
    if not p.exists():
        raise FileNotFoundError(
            "No DuckDB and no labeled CSV; run the pipeline to create logs/daily_predictions.csv first."
        )
    return pd.read_csv(p, low_memory=False)


def main():
    print("📊 Evaluating predictions...")
    # 1) Source labeled / joined data
    try:
        df = _from_duckdb()
        source = "duckdb"
    except Exception as e:
        print(f"[ℹ️] DuckDB path unavailable: {e} → falling back to CSV")
        df = _from_csv_pipeline()
        source = "csv"

    # 2) Persist a labeled CSV for consistency (even when coming from DB)
    # If coming from DuckDB, ensure we have Outcome & PredLong; if not, derive minimal fields
    if "Outcome" not in df.columns:

        def _outcome_row(r):
            p = int(r.get("PredLong", 0)) if pd.notna(r.get("PredLong", pd.NA)) else 0
            a = int(r.get("Actual_Event", 0)) if pd.notna(r.get("Actual_Event", pd.NA)) else 0
            if p == 1 and a == 1:
                return "TP"
            if p == 1 and a == 0:
                return "FP"
            if p == 0 and a == 0:
                return "TN"
            if p == 0 and a == 1:
                return "FN"
            return "?"

        df["Outcome"] = df.apply(_outcome_row, axis=1)

    LABELED_LOG.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(LABELED_LOG, index=False)
    print(f"✅ Labeled data ({source}) → {LABELED_LOG}")

    # 3) Metrics
    need = {"PredLong", "Actual_Event"}
    if not need.issubset(df.columns):
        missing = sorted(need - set(df.columns))
        raise SystemExit(f"Missing columns for evaluation: {missing}")

    sub = df.dropna(subset=["PredLong", "Actual_Event"]).copy()
    sub["PredLong"] = sub["PredLong"].astype(int)
    sub["Actual_Event"] = sub["Actual_Event"].astype(int)

    total = len(sub)
    acc = (sub["PredLong"] == sub["Actual_Event"]).mean() if total else float("nan")
    print("\n📋 Summary")
    print(f"Rows evaluated: {total}")
    print(f"Accuracy:       {acc:.2%}")

    report = classification_report(
        sub["Actual_Event"], sub["PredLong"], output_dict=True, zero_division=0
    )
    rep_df = pd.DataFrame(report).transpose()
    print("\n🔍 Classification Report:")
    print(rep_df)

    METRIC_OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rep_df.to_csv(METRIC_OUTPUT_CSV)
    print(f"\n💾 Saved detailed report → {METRIC_OUTPUT_CSV}")


if __name__ == "__main__":
    main()
