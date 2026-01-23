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
<<<<<<< HEAD

    # Only drop NaN dates if there are any valid dates
    if not d["Date"].isna().all():
        d = d.dropna(subset=["Date"])
        d["Date"] = d["Date"].dt.tz_localize(None)
    else:
        # If all dates are NaN, return empty dataframe with correct structure
        return pd.DataFrame(columns=CANON_COLS)
=======
    d = d.dropna(subset=["Date"])
    d["Date"] = d["Date"].dt.tz_localize(None)
>>>>>>> f1644007

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
<<<<<<< HEAD
    if len(df) == 0:
        print(f"{msg_prefix} → {CSV_PATH}  rows=0  (empty)")
    else:
        print(
            f"{msg_prefix} → {CSV_PATH}  rows={len(df)}  range={df['Date'].min().date()} → {df['Date'].max().date()}"
        )
=======
    print(
        f"{msg_prefix} → {CSV_PATH}  rows={len(df)}  range={df['Date'].min().date()} → {df['Date'].max().date()}"
    )
>>>>>>> f1644007


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
<<<<<<< HEAD

    # Reset index and ensure it's named 'Date'
    df = df.reset_index()

    # Rename index column to 'Date' if it has a different name
    if df.columns[0] != 'Date':
        df = df.rename(columns={df.columns[0]: 'Date'})

    # Select and order columns (only if they exist)
    available_cols = []
    for col in ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if col in df.columns:
            available_cols.append(col)

    if available_cols:
        df = df[available_cols]

=======
    df = df.reset_index()
    df = df.rename(columns={"Adj Close": "Adj Close"})  # explicit for clarity
    df = df[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
>>>>>>> f1644007
    df = _canonicalize(df)
    _write(df, "✅ Created")
    return df


def _append_new_rows(base_df: pd.DataFrame) -> pd.DataFrame:
<<<<<<< HEAD
    # Handle empty dataframe - need to bootstrap instead
    if len(base_df) == 0:
        print("⚠️  Base dataframe is empty - bootstrapping instead...")
        return _bootstrap() or base_df

    # Last available date
    last_date_ts = pd.to_datetime(base_df["Date"]).max()

    # Handle NaT (Not a Time) - dataframe has no valid dates
    if pd.isna(last_date_ts):
        print("⚠️  No valid dates in base dataframe - bootstrapping instead...")
        return _bootstrap() or base_df

    last_date = last_date_ts.date()
=======
    # Last available date
    last_date = pd.to_datetime(base_df["Date"]).max().date()
>>>>>>> f1644007
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

<<<<<<< HEAD
    # Handle MultiIndex columns from newer yfinance versions
    if isinstance(newdf.columns, pd.MultiIndex):
        newdf.columns = [col[0] if isinstance(col, tuple) else col for col in newdf.columns]

    newdf = newdf.reset_index()

    # Handle case where index becomes 'Datetime' instead of 'Date'
    if 'Datetime' in newdf.columns and 'Date' not in newdf.columns:
        newdf = newdf.rename(columns={'Datetime': 'Date'})

    # Check if required columns exist before selecting
    required_cols = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    if not all(col in newdf.columns for col in required_cols):
        print(f"⚠️  yfinance data missing required columns: {[c for c in required_cols if c not in newdf.columns]}")
        print(f"   Available columns: {list(newdf.columns)}")
        return base_df

    newdf = newdf[required_cols]
=======
    newdf = newdf.reset_index()
    newdf = newdf[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
>>>>>>> f1644007
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
<<<<<<< HEAD
    updated_df = _append_new_rows(base_df)
    # If we got a different dataframe back (e.g., from bootstrap), write it
    if updated_df is not base_df and updated_df is not None:
        _write(updated_df, "✅ Updated")
=======
    _append_new_rows(base_df)
>>>>>>> f1644007
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
