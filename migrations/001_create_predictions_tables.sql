-- NeuroVest Predictions Schema
-- Stores model predictions and metadata in PostgreSQL for cross-container access

-- Predictions table: stores ensemble predictions for all assets
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(20) NOT NULL,
    prediction_date DATE NOT NULL,
    prediction_timestamp TIMESTAMP DEFAULT NOW(),

    -- Individual model probabilities
    xgboost_prob FLOAT,
    lightgbm_prob FLOAT,
    catboost_prob FLOAT,

    -- Ensemble prediction
    ensemble_prob FLOAT NOT NULL,
    prediction_label VARCHAR(20) NOT NULL,  -- CRASH, NORMAL, SPIKE

    -- Metadata
    model_agreement BOOLEAN,  -- All 3 models agree?
    confidence_score FLOAT,

    -- Constraints
    UNIQUE(ticker, prediction_date),
    CHECK (ensemble_prob >= 0 AND ensemble_prob <= 1)
);

-- Index for fast lookups
CREATE INDEX IF NOT EXISTS idx_predictions_ticker_date ON predictions(ticker, prediction_date DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(prediction_timestamp DESC);

-- Model metadata: tracks trained models
CREATE TABLE IF NOT EXISTS model_metadata (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(50) NOT NULL,
    model_type VARCHAR(20) NOT NULL,  -- xgboost, lightgbm, catboost, ensemble
    trained_at TIMESTAMP DEFAULT NOW(),

    -- Training info
    feature_count INTEGER,
    training_samples INTEGER,
    assets_used TEXT[],  -- Array of tickers

    -- Performance metrics (JSON)
    metrics JSONB,

    -- Model config
    hyperparameters JSONB,

    UNIQUE(model_name, trained_at)
);

CREATE INDEX IF NOT EXISTS idx_model_metadata_type ON model_metadata(model_type, trained_at DESC);

-- Prediction summary view: latest prediction per asset
CREATE OR REPLACE VIEW latest_predictions AS
SELECT DISTINCT ON (ticker)
    ticker,
    prediction_date,
    prediction_timestamp,
    ensemble_prob,
    prediction_label,
    model_agreement,
    confidence_score
FROM predictions
ORDER BY ticker, prediction_date DESC, prediction_timestamp DESC;

-- Model performance view: latest models
CREATE OR REPLACE VIEW latest_models AS
SELECT DISTINCT ON (model_type)
    model_name,
    model_type,
    trained_at,
    feature_count,
    training_samples,
    metrics
FROM model_metadata
ORDER BY model_type, trained_at DESC;

COMMENT ON TABLE predictions IS 'Ensemble predictions for market forecasting';
COMMENT ON TABLE model_metadata IS 'Trained model tracking and performance metrics';
COMMENT ON VIEW latest_predictions IS 'Most recent prediction for each asset';
COMMENT ON VIEW latest_models IS 'Most recent trained model of each type';
