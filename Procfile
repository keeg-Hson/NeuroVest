# Railway Procfile - Tells Railway how to run services
# NOTE: Railway automatically sets PORT env var, Streamlit reads it

web: streamlit run dashboard_comprehensive.py --server.address=0.0.0.0 --server.headless=true
worker: python3 worker_data_scheduler.py
