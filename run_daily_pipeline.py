# run_daily_pipeline.py

import subprocess
from datetime import datetime

from apscheduler.schedulers.blocking import BlockingScheduler


# Define jobs
def update_data():
    print(f"[⏳] {datetime.now()} - Running download_spy_data.py...")
    subprocess.run(["python3", "download_spy_data.py"], check=True)
    print(f"[✅] {datetime.now()} - SPY data updated.")


def run_prediction():
    print(f"[⏳] {datetime.now()} - Running predict.py...")
    subprocess.run(["python3", "predict.py"], check=True)
    print(f"[✅] {datetime.now()} - Prediction complete.")


# Setup scheduler
scheduler = BlockingScheduler()

# Run every weekday at 16:30 (market close EST)
scheduler.add_job(update_data, "cron", day_of_week="mon-fri", hour=16, minute=30)
scheduler.add_job(run_prediction, "cron", day_of_week="mon-fri", hour=16, minute=35)

print("🚀 Scheduler started. Waiting for jobs...")
scheduler.start()
