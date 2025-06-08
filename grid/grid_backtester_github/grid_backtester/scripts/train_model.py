
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnx

# Initialize connection to MetaTrader 5
if not mt5.initialize():
    print("Failed to initialize MT5")
    quit()

symbol = "EURUSD"
rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 10000)
df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')

# Resample to 1-min OHLC
df.set_index('time', inplace=True)
ohlc = df['close'].resample('1min').ohlc().dropna()
ohlc['return'] = ohlc['close'].pct_change()
ohlc['volatility'] = ohlc['high'] - ohlc['low']
ohlc['rolling_std'] = ohlc['return'].rolling(5).std()
ohlc = ohlc.dropna()

# Target: high volatility in next bar
threshold = ohlc['volatility'].quantile(0.75)
ohlc['target'] = (ohlc['volatility'].shift(-1) > threshold).astype(int)

X = ohlc[['return', 'rolling_std']]
y = ohlc['target']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)

# Normalize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train_scaled, y_train)

# Export to ONNX
initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)
with open("volatility_predictor.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("✅ ONNX model exported.")
mt5.shutdown()
