#!/usr/bin/env bash
set -euo pipefail
python3 -V
echo "== .env =="
python3 scripts/test_env.py || true
echo "== NewsAPI ==";        python3 scripts/test_newsapi.py || true
echo "== Reddit ==";         python3 scripts/test_reddit.py  || true
echo "== FRED ==";           python3 scripts/test_fred.py    || true
echo "== AlphaVantage ==";   python3 scripts/test_alpha_vantage.py || true
echo "== Telegram ==";       python3 scripts/test_telegram.py || true
