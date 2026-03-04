# Archived Scripts

These scripts have been archived as part of the codebase consolidation effort.
They have been replaced by unified modules in `core/`.

## Archived Date: 2026-01-23

## Replacement Modules

### Training Scripts
All 21 training scripts have been consolidated into:
- `core/data_pipeline.py` - Unified data loading and preprocessing
- `core/models/base_models.py` - Consolidated model architectures
- `train_unified.py` - Single entry point for all training

### Prediction Scripts
All prediction scripts have been consolidated into:
- `core/prediction_engine.py` - Unified prediction engine

## Migration Guide

### Old Way (multiple scripts)
```bash
python train_lstm_model.py
python train_xgboost.py
python train_lightgbm_catboost.py
```

### New Way (unified script)
```bash
python train_unified.py --model lstm
python train_unified.py --model xgboost
python train_unified.py --model ensemble  # Trains XGBoost, LightGBM, CatBoost
```

### Old Way (prediction)
```python
from predict_multi_asset_ensemble import predict_spy
```

### New Way (unified engine)
```python
from core.prediction_engine import PredictionEngine

engine = PredictionEngine()
engine.load_models()
result = engine.predict_latest(df, 'SPY')
```

## Why Archive Instead of Delete?

1. **Reference**: Archived scripts contain domain-specific logic that may be useful
2. **Rollback**: If issues are found with consolidated modules, originals are available
3. **Documentation**: Shows the evolution of the codebase

## Files Archived

### train_scripts/ (21 files)
- train_attention_lstm.py → LSTMModel(use_attention=True)
- train_cnn_lstm.py → LSTMModel(use_cnn=True)
- train_ensemble_stacking.py → MetaLearnerModel
- train_from_labels.py → DataPipeline + TreeEnsembleModel
- train_improved.py → TreeEnsembleModel with config
- train_lightgbm_catboost.py → TreeEnsembleModel(model_type='lightgbm'|'catboost')
- train_lstm_focal_loss.py → LSTMModel(loss_type='focal')
- train_lstm_model.py → LSTMModel
- train_lstm_v2_focal.py → LSTMModel with config
- train_multi_asset.py → DataPipeline(multi_asset=True)
- train_multi_horizon.py → DataPipeline(horizon=X)
- train_multi_horizon_signals.py → DataPipeline with 3-class
- train_neural_meta_learner.py → MetaLearnerModel
- train_per_asset.py → DataPipeline per asset
- train_profit_optimized.py → TreeEnsembleModel with custom scorer
- train_regime_switching_improved.py → Custom implementation (preserved)
- train_regime_switching_model.py → Custom implementation (preserved)
- train_simple_meta_learner.py → MetaLearnerModel
- train_transformer_model.py → TransformerModel
- train_with_options_flow.py → DataPipeline with feature config
- train_with_selected_features.py → DataPipeline with feature selection

### predict_scripts/ (3 files)
- predict_all_assets.py → PredictionEngine.run_batch_predictions()
- predict_multi_asset_ensemble.py → PredictionEngine
- predict_per_asset.py → PredictionEngine with per-asset models
