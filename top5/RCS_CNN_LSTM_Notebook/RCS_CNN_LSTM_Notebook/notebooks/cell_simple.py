# Import necessary libraries
from src.models.cnn_lstm import build_cnn_lstm_model
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Prepare data for model training
# Assuming 'data' DataFrame and 'symbol' variable are already defined
df = data.copy()

# Create target variable if not already present
if 'target' not in df.columns:
    df['target'] = (prices[(symbol, "close")].shift(-1) > prices[(symbol, "close")]).astype(int)
    df = df.dropna(subset=['target'])

# Print available columns for debugging
print("Available columns in DataFrame:")
print(df.columns.tolist())

# Define a minimal set of features that should be available
minimal_features = ["rsi", "macd", "momentum", "cci"]
available_features = [f for f in minimal_features if f in df.columns]

if len(available_features) < 2:  # Need at least 2 features
    print("Not enough minimal features are available in the DataFrame.")
    print("Using all available numeric columns as features instead.")
    # Get all numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude target column
    available_features = [col for col in numeric_cols if col != 'target']

print(f"Using these features: {available_features}")
print(f"Number of features: {len(available_features)}")

if len(available_features) < 2:
    print("ERROR: Not enough features available for training.")
    # Create some synthetic features
    df['feature1'] = np.random.randn(len(df))
    df['feature2'] = np.random.randn(len(df))
    available_features = ['feature1', 'feature2']
    print("Created synthetic features for demonstration purposes.")

# Create a new DataFrame with only the available features and target
df_features = df[available_features + ['target']]
print(f"Feature DataFrame shape: {df_features.shape}")

# Prepare data for training
X = df_features[available_features].values
y = df_features['target'].values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Reshape for CNN-LSTM (samples, timesteps, features)
lookback = 10  # Use a smaller lookback window
X_train_seq = np.array([X_train[i-lookback:i] for i in range(lookback, len(X_train))])
y_train_seq = y_train[lookback:]

X_val_seq = np.array([X_val[i-lookback:i] for i in range(lookback, len(X_val))])
y_val_seq = y_val[lookback:]

X_test_seq = np.array([X_test[i-lookback:i] for i in range(lookback, len(X_test))])
y_test_seq = y_test[lookback:]

print(f"Training data shape: {X_train_seq.shape}")
print(f"Validation data shape: {X_val_seq.shape}")
print(f"Test data shape: {X_test_seq.shape}")

# Build the model
model = build_cnn_lstm_model(input_shape=(lookback, len(available_features)))

# Train the model
history = model.fit(
    X_train_seq, y_train_seq,
    epochs=10,  # Reduced for faster training
    batch_size=32,
    validation_data=(X_val_seq, y_val_seq),
    verbose=1
)

# Evaluate the model
loss, accuracy = model.evaluate(X_test_seq, y_test_seq)
print(f"Test accuracy: {accuracy:.4f}")

# Make predictions
y_pred = (model.predict(X_test_seq) > 0.5).astype(int)

# Calculate metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
metrics = {
    'accuracy': accuracy_score(y_test_seq, y_pred),
    'precision': precision_score(y_test_seq, y_pred, zero_division=0),
    'recall': recall_score(y_test_seq, y_pred, zero_division=0),
    'f1_score': f1_score(y_test_seq, y_pred, zero_division=0)
}

# Display results
print(f"Model trained with {len(available_features)} features:")
print(f"Features: {available_features}")
print(f"Metrics: {metrics}")

# Save model
model_name = f"{symbol}_CNN_LSTM_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
model.save(f"{model_name}.h5")
print(f"Model saved as {model_name}.h5")
