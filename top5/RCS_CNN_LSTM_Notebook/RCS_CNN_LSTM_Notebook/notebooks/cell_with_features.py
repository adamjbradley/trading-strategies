# Import the build_cnn_lstm_model function and training utilities
from src.models.cnn_lstm import build_cnn_lstm_model
from src.models.training import train_model_with_best_features, evaluate_model
import pandas as pd

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

# Train model using best features
try:
    model, feature_names, X_train, y_train, X_val, y_val, X_test, y_test = train_model_with_best_features(
        symbol=symbol,
        data=df,
        model_builder_func=build_cnn_lstm_model,
        n_features=15,
        use_saved_features=False,  # Set to False to use feature importance instead of saved features
        epochs=10,  # Reduced for faster training
        batch_size=32
    )

    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)

    # Display results
    print(f"Model trained with {len(feature_names)} features:")
    print(f"Features: {feature_names}")
    print(f"Metrics: {metrics}")

    # Save model
    model_name = f"{symbol}_CNN_LSTM_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    model.save(f"{model_name}.h5")
    print(f"Model saved as {model_name}.h5")
    
except Exception as e:
    print(f"Error during model training: {str(e)}")
    
    # Print more detailed error information
    import traceback
    traceback.print_exc()
