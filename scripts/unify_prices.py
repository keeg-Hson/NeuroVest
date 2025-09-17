import pandas as pd, pathlib as p

def load_any(path: p.Path):
    if not path.exists():
        return None
    df = pd.read_csv(path, low_memory=False)
    if "Date" not in df.columns:
        df = pd.read_csv(path, index_col=0, low_memory=False)
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[~df.index.isna()].copy()
        df.index.name = "Date"
        df = df.reset_index()
    else:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df[~df["Date"].isna()].copy()
    return df

paths = [p.Path("data/SPY.csv"), p.Path("data/spy_daily.csv")]
dfs = [d for d in (load_any(x) for x in paths) if d is not None]
if not dfs:
    raise SystemExit("No SPY file found to unify")

df = pd.concat(dfs, ignore_index=True)

for c in ["Open","High","Low","Close","Adj Close","Volume"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
df.to_csv("data/spy_daily.csv", index=False)
df.to_csv("data/SPY.csv", index=False)
print("Unified →", df["Date"].min().date(), "→", df["Date"].max().date(), "rows:", len(df))
