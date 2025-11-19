# Archived Documentation

This directory contains historical development logs and planning documents from the NeuroVest project evolution.

---

## Directory Structure

```
docs/archive/
├── development_logs/       # Historical completion summaries
├── historical_plans/       # Planning and improvement documents
├── KNOWN_ISSUES.md        # Outdated issues list
└── QUICKSTART.md          # Old trading-focused quickstart
```

---

## Development Logs

Historical summaries of completed work phases:

- **`DAY_2_3_COMPLETE.md`** - Day 2-3 completion summary
- **`PHASE_1_COMPLETE.md`** - Phase 1 completion log
- **`PHASE_1_POST_MORTEM.md`** - Phase 1 retrospective
- **`IMPROVEMENTS_SUMMARY.md`** - General improvements summary
- **`LESSONS_LEARNED.md`** - Key learnings from development

These documents capture the iterative development process but are superseded by:
- **`IMPLEMENTATION_SUMMARY.md`** (root) - Comprehensive Nov 2025 improvements

---

## Historical Plans

Planning documents for various improvement initiatives:

### Accuracy Improvements
- **`ACCURACY_IMPROVEMENT_PLAN.md`** - Original accuracy improvement strategy
- **`ACCURACY_IMPROVEMENT_ACTIONABLE_PLAN.md`** - Actionable steps
- **`ACCURACY_CORRECTION.md`** - Accuracy metric corrections
- **`README_ACCURACY_FIXES.md`** - README accuracy updates

### Feature Engineering
- **`FEATURE_REDUCTION_PLAN.md`** - Plan to reduce feature count
- **`PHASE_1_DATA_INTEGRATION.md`** - Data integration plan
- **`DATA_INTEGRATION_PLAN.md`** - Broader data integration strategy

### Technical Fixes
- **`FIX_DATA_LEAKAGE.md`** - Data leakage fix planning
- **`FIX_SAMPLE_WEIGHTING.md`** - Sample weight fix planning
- **`INCREASE_TRAINING_DATA.md`** - Training data expansion plan

### Multi-Asset
- **`MULTI_ASSET_IMPROVEMENT.md`** - Multi-asset modeling improvements

These plans are now complete or superseded. Current status tracked in:
- **`IMPROVEMENT_TRACKER.md`** (root) - Current tracking document
- **`ACCURACY_IMPROVEMENT_ANALYSIS.md`** (root) - Data-driven analysis

---

## Outdated Documentation

### `KNOWN_ISSUES.md`
Original issues list from earlier development phases. Many issues have been resolved:
- ✅ Model overfitting → Fixed with walk-forward CV
- ✅ Zero-importance features → Removed in Nov 2025
- ✅ Transaction cost modeling → Improved in backtest
- ✅ Threshold optimization → Completed (0.55 → 0.45)

### `QUICKSTART.md`
Original quickstart guide focused on trading system usage. Replaced with economic modeling-focused version in root directory.

---

## Why These Files Were Archived

The main README and documentation were refocused (Nov 2025) to emphasize:

1. **Economic Modeling Primary Purpose**
   - Original docs emphasized trading performance
   - New docs emphasize regime forecasting and economic analysis
   - Backtest repositioned as validation tool, not main feature

2. **Completed Work Consolidation**
   - Many incremental plans and logs scattered across root
   - Consolidated into comprehensive `IMPLEMENTATION_SUMMARY.md`
   - Cleaner project structure for new users

3. **Historical Context Preservation**
   - Development logs show project evolution
   - Planning documents capture decision-making process
   - Useful for understanding why certain approaches were chosen

---

## Current Documentation (Root Directory)

Use these instead of archived documents:

| File | Purpose |
|------|---------|
| **`README.md`** | Main documentation - economic modeling focus |
| **`QUICKSTART.md`** | 5-minute setup guide |
| **`IMPLEMENTATION_SUMMARY.md`** | Nov 2025 comprehensive improvements |
| **`ACCURACY_IMPROVEMENT_ANALYSIS.md`** | Data-driven feature/threshold analysis |
| **`IMPROVEMENT_TRACKER.md`** | Current status tracking |

---

## When to Reference Archived Docs

### Useful for:
- Understanding project history and evolution
- Learning why certain technical decisions were made
- Reviewing lessons learned during development
- Seeing iterative improvement process

### Not useful for:
- Current system capabilities (see main README)
- Running the models (see QUICKSTART)
- Understanding recent improvements (see IMPLEMENTATION_SUMMARY)
- Current known issues (open GitHub issues instead)

---

## Archive Date

**Archived**: 2025-11-16
**Reason**: Documentation reorganization to emphasize economic modeling focus

**Archived by**: Documentation cleanup (Claude session `01HmCRFQaz3HcUVK4VP1KrmK`)

---

## Questions?

If you need information from these archived documents:

1. Check current documentation first (it's more comprehensive)
2. Search GitHub commit history for context
3. Open a GitHub issue if something important is missing

Most content from these archived docs has been incorporated into the current documentation with better organization and clarity.
