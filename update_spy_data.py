#!/usr/bin/env python3
"""
update_spy_data.py

Refreshes data/SPY.csv in a safe, schema-consistent way.

Behavior:
- Always canonicalizes the CSV into Yahoo-style columns:
  ["Date","Open","High","Low","Close","Adj Close","Volume"]
- If OFFLINE_MODE != 1, attempts to append new daily rows from yfinance.
- Uses exclusive end date (+1 day) and starts from next business day after the
  last available date.
- Never hard-fails on network errors; if download fails, it still cleans the
  existing file and exits 0 if a usable CSV remains.

Exit codes:
- 0 on success (file exists and is schema-clean after run)
- 1 if file is missing and bootstrap failed
"""

from __future__ import annotations

import os
from datetime import date, timedelta

import pandas as pd
import yfinance as yf
from pandas.tseries.offsets import BDay

# Single source of truth path from utils.py
from utils import CSV_PATH  # points to "data/SPY.csv"

CANON_COLS = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]


def _ensure_dirs() -> None:
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)


def _exclusive_end_today() -> str:
    # yfinance end is exclusive; add +1 day to include today's bar when available
    return (date.today() + timedelta(days=1)).isoformat()


def _canonicalize(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce to canonical schema, types, sorted, deduped."""
    d = df.copy()

    # Robust Date detection/rename
    if "Date" not in d.columns:
        for c in list(d.columns):
            if str(c).strip().lower() == "date":
                d = d.rename(columns={c: "Date"})
                break

    if "Date" not in d.columns:
        raise ValueError("No 'Date' column found to canonicalize.")

    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    d = d.dropna(subset=["Date"])
    d["Date"] = d["Date"].dt.tz_localize(None)

    # If columns missing, create placeholders so downstream is stable
    for c in CANON_COLS:
        if c not in d.columns:
            d[c] = pd.NA

    # Coerce numerics
    for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    # Drop rows without OHLC; Volume can be 0
    d = d.dropna(subset=["Open", "High", "Low", "Close", "Adj Close"])

    # Deduplicate & sort
    d = d.drop_duplicates(subset=["Date"], keep="last").sort_values("Date")

    # Reorder columns canonically
    d = d[CANON_COLS]
    return d.reset_index(drop=True)


def _read_existing() -> pd.DataFrame | None:
    if not os.path.exists(CSV_PATH):
        return None
    try:
        # Parse Date early; tolerate messy CSVs
        df = pd.read_csv(CSV_PATH, low_memory=False)
        return _canonicalize(df)
    except Exception as e:
        print(f"⚠️  Failed to read/canonicalize existing CSV: {e}")
        return None


def _write(df: pd.DataFrame, msg_prefix: str) -> None:
    df.to_csv(CSV_PATH, index=False)
    print(
        f"{msg_prefix} → {CSV_PATH}  rows={len(df)}  range={df['Date'].min().date()} → {df['Date'].max().date()}"
    )


def _bootstrap() -> pd.DataFrame | None:
    base = "2010-01-01"
    end = _exclusive_end_today()
    try:
        df = yf.download(
            "SPY", start=base, end=end, interval="1d", auto_adjust=False, progress=False
        )
    except Exception as e:
        print(f"❌ yfinance bootstrap failed: {e}")
        return None
    if df is None or df.empty:
        print("❌ Could not download initial SPY history.")
        return None
    df = df.reset_index()
    df = df.rename(columns={"Adj Close": "Adj Close"})  # explicit for clarity
    df = df[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    df = _canonicalize(df)
    _write(df, "✅ Created")
    return df


def _append_new_rows(base_df: pd.DataFrame) -> pd.DataFrame:
    # Last available date
    last_date = pd.to_datetime(base_df["Date"]).max().date()
    # Start next business day after last_date
    start = (pd.Timestamp(last_date) + BDay(1)).date()
    end = date.today() + timedelta(days=1)  # exclusive end

    if start >= end:
        print("ℹ️ SPY.csv is already up to date — no new days to fetch.")
        return base_df

    try:
        newdf = yf.download(
            "SPY",
            start=start.isoformat(),
            end=end.isoformat(),
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
    except Exception as e:
        print(f"⚠️  yfinance append failed: {e}")
        return base_df  # keep existing, still success after canonicalize

    if newdf is None or newdf.empty:
        print("ℹ️ No new SPY data returned by yfinance.")
        return base_df

    newdf = newdf.reset_index()
    newdf = newdf[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    newdf = _canonicalize(newdf)

    merged = pd.concat([base_df, newdf], ignore_index=True)
    merged = _canonicalize(merged)
    _write(merged, f"✅ Appended {len(newdf)} new rows")
    return merged


def main() -> int:
    _ensure_dirs()
    offline = os.getenv("OFFLINE_MODE", "0").strip() == "1"

    base_df = _read_existing()

    if base_df is None:
        if offline:
            print("❌ OFFLINE_MODE=1 and data/SPY.csv is missing — cannot bootstrap.")
            return 1
        base_df = _bootstrap()
        if base_df is None:
            return 1  # bootstrap failed
        return 0

    # Always canonicalize existing file first
    base_df = _canonicalize(base_df)
    _write(base_df, "ℹ️ Canonicalized existing file")

    if offline:
        print("ℹ️ OFFLINE_MODE=1 — skipping download step.")
        return 0

    # Try to append fresh rows; never hard-fail
    _append_new_rows(base_df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
