#!/bin/bash
# Combined Worker + Dashboard Startup Script
# Runs both data collection worker and Streamlit dashboard in the same container

echo "======================================================================="
echo "🚀 NEUROVEST COMBINED SERVICE - STARTING"
echo "======================================================================="
echo "  Platform: Railway"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Components: Worker + Dashboard"
echo "======================================================================="

# Start the data worker in the background
echo ""
echo "📊 Starting Data Worker (background)..."
python3 worker_data_scheduler.py &
WORKER_PID=$!
echo "  Worker PID: $WORKER_PID"

# Give worker a moment to initialize
sleep 3

# Start Streamlit dashboard in the foreground
echo ""
echo "🌐 Starting Streamlit Dashboard..."
echo "  Dashboard will be available on port $PORT"
echo "======================================================================="
echo ""

# Streamlit reads PORT from environment automatically
streamlit run dashboard_comprehensive.py \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false

# If Streamlit exits, kill the worker too
echo ""
echo "Streamlit stopped, shutting down worker..."
kill $WORKER_PID 2>/dev/null
wait $WORKER_PID 2>/dev/null
echo "Shutdown complete."
