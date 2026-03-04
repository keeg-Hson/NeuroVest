# Model Change Log

Record every intentional model or configuration change here.
Format: `YYYY-MM-DD | who | what changed | why | metric impact`.

---

## 2026-02-27 | claude/codebase-assessment | Documentation sync & metrics consolidation

**What changed:**
- Created `docs/METRICS_SUMMARY.md` as single source of truth for all model metrics
- Updated `README.md` with current production metrics from latest evaluation run
- Documented locked configuration: 1-day horizon, 0.5% threshold, 0.45 prediction threshold

**Current Production Metrics:**
- Accuracy: 78.62%, AUC: 0.7792
- Sharpe Ratio: 2.55 (vs 0.42 buy-and-hold)
- Max Drawdown: -5.4% (vs -55% buy-and-hold)
- Precision: 86.99% when model signals
- Win Rate: 54.0%, Profit Factor: 1.87

**Feature Configuration:**
- 164 total features (153 base + 11 regime)
- Top regime features: ADX (280), Plus_DI (155), MA200_Distance_Vol (125)

**Why:** Sync documentation with actual model performance; establish single source of truth.

**Metric impact:** Documentation only; no model changes.

---

## 2026-02-27 | claude/codebase-assessment | Pipeline hardening (Days 2–7)

**What changed:**
- `train.py`: Added `--asof YYYY-MM-DD` flag for point-in-time reproducible training.
- `train.py`: Save `models/artifact_manifest.json` after every training run with
  `model_hash_sha256`, `code_git_sha`, `trained_at`, `data_window`, `config_hash`, `val_metrics`.
- `predict.py`: Added `--asof YYYY-MM-DD` flag; signal rows now include signal contract fields:
  `symbol`, `horizon`, `side`, `ev`, `model_version`, `config_hash`.
- `predict.py`: Added `--fill-realized` flag + `fill_realized_pnl()` to annotate past signals
  with realized returns once the horizon has elapsed.
- `backtest.py`: Saves `outputs/metrics_YYYYMMDD_HHMMSS.json` (CAGR, Sharpe, Sortino, max DD,
  hit-rate) and `outputs/plots/equity_drawdown_*.png` on every run.
- `run_all.py`: Added `--daily` mode (refresh → predict → fill-realized → backtest snapshot,
  skips train/analyze/tearsheet). Added structured JSON logging to `logs/run_all_*.jsonl`.
  Pipeline summary saved to `outputs/run_summary_*.json`.

**Why:** Week-1 hardening plan — make the pipeline reproducible, observable, and safe to run
daily without manual intervention.

**Metric impact:** No model weights changed; pipeline plumbing only.

---

## 2026-02-XX | prior work | Regime-adaptive thresholds

**What changed:** Regime-based per-row decision thresholds using volatility/trend/risk-appetite
signals. Position sizing scaled by regime confidence.

**Why:** Static threshold (0.45) applied uniformly regardless of market regime.

**Metric impact:** Improved OOS Sharpe (see `outputs/metrics_*.json` files going forward).

---

_Add entries above this line (newest first)._
