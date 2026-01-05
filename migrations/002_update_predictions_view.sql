-- Update latest_predictions view to include all needed columns for API
CREATE OR REPLACE VIEW latest_predictions AS
SELECT DISTINCT ON (ticker)
    ticker,
    prediction_date,
    prediction_timestamp,
    ensemble_prob,
    prediction_label,
    xgboost_prob,
    lightgbm_prob,
    catboost_prob,
    model_agreement,
    confidence_score,
    -- Derive confidence level from score
    CASE
        WHEN confidence_score >= 0.7 THEN 'high'
        WHEN confidence_score >= 0.5 THEN 'medium'
        ELSE 'low'
    END as confidence
FROM predictions
ORDER BY ticker, prediction_date DESC, prediction_timestamp DESC;
