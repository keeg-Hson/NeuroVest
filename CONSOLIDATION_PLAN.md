# NeuroVest Codebase Consolidation Plan

## Executive Summary

This document outlines the consolidation strategy for the NeuroVest codebase to establish
single sources of truth, eliminate redundancy, and fix broken functionality.

---

## 1. Identified Issues

### 1.1 Broken Menu System (main.py)

The following scripts are referenced in the menu but do NOT exist at the root level:

| Menu Item | Script Referenced | Actual Location |
|-----------|------------------|-----------------|
| Training Option 1-5 | `train_multi_asset.py` | `archive/train_scripts/` |
| Training Option 6-7 | `train_multi_horizon_signals.py` | `archive/train_scripts/` |
| Training Option 8 | `train_per_asset.py` | `archive/train_scripts/` |
| Prediction Option 1 | `predict_multi_asset_ensemble.py` | `archive/predict_scripts/` |
| Prediction Option 2-4 | `predict_per_asset.py` | `archive/predict_scripts/` |

**Resolution:** Update menu to use existing scripts (`train.py`, `predict.py`) or stub the missing scripts.

### 1.2 Missing Directories

| Directory | Purpose | Status |
|-----------|---------|--------|
| `data_cache/` | Asset data cache for multi-asset support | Created |
| `models/` | Trained model storage | Created |

### 1.3 Redundant Files

#### Data Managers (3 versions - use `data_manager_postgres.py`)
- `core/data_manager.py` - SQLite only (LEGACY)
- `core/data_manager_postgres.py` - PostgreSQL/SQLite hybrid (CANONICAL)
- `core/data_manager_sqlite_backup.py` - Backup (DEPRECATED)

#### Training Scripts (24 in archive, 2 at root)
- `train.py` - CANONICAL (root level)
- `train_unified.py` - Alternative (root level)
- 22 legacy variants in `archive/train_scripts/`

#### Prediction Scripts (4 variants)
- `predict.py` - CANONICAL (root level)
- `predict_all_assets.py` - Multi-asset variant (root level)
- 3 legacy variants in `archive/predict_scripts/`

#### Dashboard Scripts (4 variants)
- `dashboard_comprehensive.py` - CANONICAL (166 KB, full-featured)
- `dashboard.py` - Basic version
- `dashboard_demo.py` - Demo version
- `dashboard_improvements.py` - Legacy

#### Download Scripts (CONSOLIDATED)
- `update_data.py` - CANONICAL (CLI interface for all data updates)
- `download_crypto_comprehensive.py` - Multi-source crypto data (CryptoCompare, Binance, CoinGecko)
- 9 legacy variants archived in `archive/download_scripts/`

---

## 2. Source of Truth Designations

### 2.1 Configuration
| Component | Canonical File |
|-----------|---------------|
| Global config | `config.py` |
| Trading profiles | `configs/*.json` |
| Asset definitions | `config/assets.yaml` |

### 2.2 Core Modules
| Component | Canonical File |
|-----------|---------------|
| Data Management | `core/data_manager_postgres.py` |
| Model Architectures | `core/models/base_models.py` |
| Training Pipeline | `core/train_models.py` |
| Prediction Engine | `core/prediction_engine.py` |
| Feature Engineering | `build_feature_table.py` |
| Data Pipeline | `core/data_pipeline.py` |

### 2.3 Entry Points
| Function | Canonical Script |
|----------|-----------------|
| Training | `train.py` |
| Prediction | `predict.py` |
| Backtesting | `backtest.py` |
| Web Dashboard | `dashboard_comprehensive.py` |
| CLI | `neurovest_cli.py` |
| Interactive Menu | `main.py` |

---

## 3. Menu System Fix Strategy

### Option A: Route to Existing Scripts (IMPLEMENTED)
Update `main.py` to use the canonical scripts that exist:
- Training: Route all options to `train.py` with appropriate flags
- Prediction: Route all options to `predict.py` with appropriate flags

### Option B: Create Stub Scripts (NOT IMPLEMENTED)
Create minimal wrapper scripts that delegate to the core functionality.

---

## 4. Model Improvements

### 4.1 Feature Pruning
- Add recursive feature elimination (RFE) support
- Add SHAP-based feature importance filtering
- Integrate with `core/feature_selection.py`

### 4.2 Ensemble Improvements
- Add calibrated probability outputs
- Add cross-validation consistency checks
- Add model averaging with uncertainty estimates

### 4.3 Validation Improvements
- Add walk-forward validation
- Add regime-aware splits
- Add out-of-sample metrics tracking

---

## 5. Implementation Checklist

- [x] Create missing directories (`data_cache/`, `models/`)
- [x] Fix broken menu items in `main.py`
- [x] Add model improvements to `core/models/base_models.py`
- [x] Archive deprecated data managers
- [x] Consolidate download scripts
- [x] Update documentation

---

## 6. File Disposition

### Keep at Root (Canonical)
```
train.py              - Model training entry point
predict.py            - Prediction entry point
backtest.py           - Backtesting entry point
main.py               - Interactive menu (FIXED)
config.py             - Global configuration
neurovest_cli.py      - CLI interface
build_feature_table.py - Feature engineering
```

### Archive (Legacy - in archive/)
```
archive/train_scripts/     - 22 legacy training variants
archive/predict_scripts/   - 3 legacy prediction variants
```

### Archived Core (in archive/core/)
```
data_manager.py              - Archived; use data_manager_postgres.py
data_manager_sqlite_backup.py - Archived; use data_manager_postgres.py
```

### Archived Download Scripts (in archive/download_scripts/)
```
download_assets_simple.py, download_cross_asset_simple.py, download_crypto_data.py,
download_crypto_enhanced.py, download_equity_etfs.py, download_equity_etfs_alternative.py,
download_multi_asset_data.py, download_spy_data.py, update_spy_data.py
```

### Deprecate (Mark for removal)
```
dashboard_demo.py                  - Use dashboard_comprehensive.py
dashboard_improvements.py          - Use dashboard_comprehensive.py
```

---

## 7. Migration Notes

### For Developers
1. Always use `config.py` constants for paths (DATA_DIR, MODELS_DIR, etc.)
2. Use `core/models/base_models.py` for model creation via `create_model()`
3. Use `core/data_manager_postgres.py` for all data operations
4. Run training via `train.py`, not archived scripts

### For Users
1. Use `main.py` for interactive menu
2. Use `neurovest_cli.py` for command-line operations
3. Use `dashboard_comprehensive.py` for web interface

---

*Document generated: 2026-02-10*
*Last updated: 2026-02-11*
