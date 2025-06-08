
# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy source code
COPY grid_backtester/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY grid_backtester/ ./grid_backtester/

# Set default command
CMD ["python", "grid_backtester/src/run_backtest.py"]
