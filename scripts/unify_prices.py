#unify_prices.py
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT_MAIN = DATA / "spy_daily.csv"
OUT_ALT  = DATA / "SPY.csv"

def _read_one(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p, low_memory=False)
    df.columns = df.columns.map(str).str.strip()
    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    # choose a date-like column by priority and coverage
    candidates = [c for c in df.columns if c.lower() in ("date","datetime","timestamp")]
    if not candidates:
        return pd.DataFrame()
    dcol = max(candidates, key=lambda c: df[c].notna().sum())

    raw = df[dcol].astype(str).str.strip()

    # 1) ISO-8601 first
    parsed = pd.to_datetime(raw, errors="coerce", format="ISO8601")

    # 2) fallback common US format
    if parsed.isna().all():
        parsed = pd.to_datetime(raw, errors="coerce", format="%m/%d/%Y")

    # 3) fallback numeric epoch (s, then ms)
    if parsed.isna().all():
        num = pd.to_numeric(raw, errors="coerce")
        parsed = pd.to_datetime(num, errors="coerce", unit="s")
        if parsed.isna().all():
            parsed = pd.to_datetime(num, errors="coerce", unit="ms")

    df["__dt"] = parsed
    df = df[df["__dt"].notna()].drop_duplicates(subset=["__dt"]).set_index("__dt").sort_index()

    def pick(*names):
        for n in names:
            cands = [c for c in df.columns if c == n or c.startswith(n + ".")]
            for c in cands:
                s = pd.to_numeric(df[c], errors="coerce")
                if s.notna().any():
                    return s
        return pd.Series(index=df.index, dtype="float64")

    out = pd.DataFrame(index=df.index)
    out["Open"]      = pick("Open","1. open")
    out["High"]      = pick("High","2. high")
    out["Low"]       = pick("Low","3. low")
    out["Close"]     = pick("Close","4. close","Adj Close","adjclose","close")
    out["Adj Close"] = pick("Adj Close","adjclose","close")
    out["Volume"]    = pick("Volume","5. volume","volume")
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


paths = [DATA / "spy_daily.csv", DATA / "SPY.csv"]
paths = [p for p in paths if p.exists()]

base = None
for p in paths:
    d = _read_one(p)
    if not d.empty:
        print(f"[src] {p.name} → {d.index.min().date()} → {d.index.max().date()} rows:{len(d)}")
        base = d if base is None else base.combine_first(d)

if base is None or base.empty:
    raise SystemExit("No input price CSVs found.")

base = base.sort_index()
base.to_csv(OUT_MAIN, index_label="Date")
base.to_csv(OUT_ALT,  index_label="Date")
print(f"Unified → {base.index.min().date()} → {base.index.max().date()} rows: {len(base)}")
