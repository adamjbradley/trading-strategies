import os
import json
from datetime import datetime

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "signals.log")

def log_signal(symbol, timestamp, signal, confidence=None, price=None):
    entry = {
        "symbol": symbol,
        "timestamp": str(timestamp),
        "signal": signal,
        "confidence": confidence,
        "price": price
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
