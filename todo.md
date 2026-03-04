## Goal
By the end of this week, `python run_all.py --daily` runs end-to-end on a clean machine:
refresh data → (re)train if scheduled → predict → append to point-in-time store → backtest snapshot → logs & artifacts.

## Day 1 — Environment & skeleton
- [x] Add pinned `requirements.txt` and verify `pip install -r requirements.txt` on a clean venv.
- [x] Commit `.env.example` (this issue includes a template). Ensure `.env` is gitignored.
- [x] Create folders: `data/`, `data_cache/`, `models/`, `logs/`, `outputs/`.
- [x] Add `pyproject.toml` (project metadata + ruff/black config).
- [x] Lint format baseline: `ruff .` and `black .` pass.

## Day 2 — Fix broken/truncated code & imports
- [x] Remove/replace ellipses and cut-offs in: `train.py`, `predict.py`, `run_all.py`, `backtest.py`, `main.py`.
- [x] Ensure all modules import: `python -m compileall .` passes.
- [x] Centralize config/env loading (one `load_settings()` used everywhere).

## Day 3 — Deterministic training & artifacts
- [x] Time-series CV uses purge/embargo; fixed random seeds.
- [x] Save **artifact manifest** (JSON) with: model_hash, code_git_sha, data_window, params.
- [x] Persist scalers/encoders with model (joblib bundle) and load in `predict.py`.
- [x] CLI: `python train.py --asof 2025-01-31` reproduces the same model/hash.

## Day 4 — Prediction & point-in-time store
- [x] Define signal contract (fields): `ts`, `symbol`, `horizon`, `side`, `prob`, `ev`, `model_version`, `config_hash`.
- [x] Implement append-only signal history (SQLite or CSV) — no overwrites.
- [x] `python predict.py --asof today` appends one row per symbol with the above fields.
- [x] Add realized PnL filler that annotates signals after horizon elapses.

## Day 5 — Backtest integrity & metrics
- [x] Backtest has slippage/fees; sweep separated from test data (no peeking).
- [x] Output standard metrics (CAGR, Sharpe, Sortino, max DD, hit-rate) to `outputs/metrics_*.json`.
- [x] Plot equity curve & drawdown to `outputs/plots/`.
- [ ] Regression test: metrics remain within tolerance across code changes.

## Day 6 — Orchestration & observability
- [x] `run_all.py --daily` pipeline: refresh → train (if schedule) → predict → update realized → backtest snapshot.
- [x] Structured logging (JSON) with a run_id; rotating file handler in `logs/`.
- [x] Basic incident log (`docs/incidents.md`) + **model change log** (`docs/model_changelog.md`).

## Day 7 — Smoke test & docs
- [ ] Fresh machine smoke test: clone → `pip install -r requirements.txt` → copy `.env` → `make run` (or single command) works.
- [x] README updates: quickstart, env vars table, signal contract, run commands.
- [x] Create a short `DEMO.ipynb` that loads the latest signals DB and plots hit-rate/DD.

## Stretch (if time allows)
- [x] Dockerfile + healthcheck; containerized `run_all.py --daily`.
- [x] Minimal FastAPI read-only endpoints: `/predict`, `/history`, `/metrics` backed by the signal DB.
- [ ] Postman collection + example Python client.

## Acceptance Criteria
- [x] A single command runs the full daily flow without manual steps.
- [ ] Two consecutive days of timestamped signals stored and visible in a simple chart.
- [ ] Re-running with the same `--asof` and seed reproduces the same model hash & predictions.
