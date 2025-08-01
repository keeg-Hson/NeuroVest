# run_all.py

import os
import pandas as pd

# === Step 1: Fetch or update SPY data ===
print("📡 Step 1: Updating SPY data...")
os.system("python3 update_spy_data.py")

# === Step 2: Run live predictions ===
print("🧠 Step 2: Running predictions...")
os.system("python3 predict.py")

# === Step 3: Log signals (spike/crash as BUY/SELL) ===
print("📊 Step 3: Logging signals...")
os.system("python3 signal_logger.py")

# === Step 4: Simulate trades ===
print("📈 Step 4: Simulating trades...")
os.system("python3 trade_simulator.py")

# === Step 5: Evaluate model performance ===
print("🧪 Step 5: Evaluating model performance...")
os.system("python3 evaluate.py")

# === Step 6: Visualize prediction overlays ===
print("🖼️ Step 6: Visualizing overlays...")
os.system("python3 viz.py")

print("✅ All pipeline steps complete.")
