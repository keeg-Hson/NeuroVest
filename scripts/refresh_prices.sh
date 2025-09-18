#!/bin/zsh
set -euo pipefail
cd "$(dirname "$0")/.."

python3 fetch_spy_history.py || true
python3 update_spy_data.py   || true

python3 scripts/unify_prices.py

python3 - <<'PY'
from utils import load_SPY_data
df = load_SPY_data()
print("OK unified →", df.index.min().date(), "→", df.index.max().date(), "rows:", len(df))
PY
