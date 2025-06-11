"""
Train with Best Features

This script provides a function to train a model using the best features.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, BatchNormalization, Flatten
from model_training_utils import train_model_with_best_features, evaluate_model
from model_export_utils import run_backtest_and_export

def create_cnn_lstm_model(input_shape, X_train=None, y_train=None, X_val=None, y_val=None, epochs=50, batch_size=32):
    """
    Create and train a CNN-LSTM model.
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of the input data (timesteps, features)
    X_train : numpy.ndarray, optional
        Training features
    y_train : numpy.ndarray, optional
        Training targets
    X_val : numpy.ndarray, optional
        Validation features
    y_val : numpy.ndarray, optional
        Validation targets
    epochs : int, default=50
        Number of epochs to train for
    batch_size : int, default=32
        Batch size for training
        
    Returns:
    --------
    keras.Model
        Trained model
    """
    # Adapt kernel size based on sequence length (first dimension of input_shape)
    sequence_length = input_shape[0]
    kernel_size = min(3, sequence_length)
    
    model = Sequential([
        # CNN layers with adaptive kernel size
        Conv1D(filters=64, kernel_size=kernel_size, padding='same', activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.2),
        
        # LSTM layers
        LSTM(units=50, return_sequences=True),
        BatchNormalization(),
        Dropout(0.2),
        
        LSTM(units=50),
        BatchNormalization(),
        Dropout(0.2),
        
        # Output layer
        Dense(1, activation='sigmoid')
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train the model if training data is provided
    if X_train is not None and y_train is not None:
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val) if X_val is not None and y_val is not None else None,
            verbose=1
        )
    
    return model

def train_and_export_model(symbol, data, lookback_window=20, epochs=50, batch_size=32, export_dir="exported_models"):
    """
    Train a model using the best features and export it.
    
    Parameters:
    -----------
    symbol : str
        Trading symbol (e.g., 'EURUSD')
    data : pandas.DataFrame
        DataFrame containing the data for training
    lookback_window : int, default=20
        Number of time steps to use for sequence data
    epochs : int, default=50
        Number of epochs to train for
    batch_size : int, default=32
        Batch size for training
    export_dir : str, default="exported_models"
        Directory to export the model to
        
    Returns:
    --------
    dict
        Dictionary containing export results
    """
    print(f"🔍 Training model for {symbol} using best features with lookback window {lookback_window}")
    
    # Prepare the data
    feature_matrix = data.copy()
    
    # Create target column if it doesn't exist
    if 'target' not in feature_matrix.columns:
        if (symbol, 'close') in feature_matrix.columns:
            feature_matrix['target'] = (feature_matrix[(symbol, 'close')].shift(-1) > feature_matrix[(symbol, 'close')]).astype(int)
        elif symbol in feature_matrix.columns:
            feature_matrix['target'] = (feature_matrix[symbol].shift(-1) > feature_matrix[symbol]).astype(int)
        else:
            # Try to find a price column
            price_cols = [col for col in feature_matrix.columns if 'close' in str(col).lower() or 'price' in str(col).lower()]
            if price_cols:
                feature_matrix['target'] = (feature_matrix[price_cols[0]].shift(-1) > feature_matrix[price_cols[0]]).astype(int)
            else:
                raise ValueError("Cannot create target column: no price column found")
    
    # Train the model using the best features
    model, feature_names, X_train, y_train, X_val, y_val, X_test, y_test = train_model_with_best_features(
        symbol=symbol,
        data=feature_matrix,
        model_builder_func=lambda X_train, y_train, X_val, y_val: create_cnn_lstm_model(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=epochs,
            batch_size=batch_size
        ),
        use_saved_features=True
    )
    
    # Evaluate the model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Get the price data for backtesting
    if (symbol, 'close') in data.columns:
        price_data = data[(symbol, 'close')]
    elif symbol in data.columns:
        price_data = data[symbol]
    else:
        # Try to find a price column
        price_cols = [col for col in data.columns if 'close' in str(col).lower() or 'price' in str(col).lower()]
        if price_cols:
            price_data = data[price_cols[0]]
        else:
            raise ValueError("Cannot find price data for backtesting")
    
    # Run backtest and export the model
    export_data = run_backtest_and_export(
        model=model,
        X_test=X_test,
        y_test=y_test,
        price_data=price_data,
        symbol=symbol,
        feature_names=feature_names,
        lookback=1,
        export_dir=export_dir
    )
    
    print(f"✅ Model trained and exported successfully!")
    print(f"Model name: {export_data['model_name']}")
    print(f"Accuracy: {export_data['accuracy']:.4f}")
    print(f"Sharpe ratio: {export_data['sharpe_ratio']:.4f}")
    print(f"Total return: {export_data['total_return']:.4f}")
    
    return export_data

def notebook_code_snippet():
    """
    Print a code snippet to use in the notebook.
    """
    code = """
# --- Train model using best features ---
from train_with_best_features import train_and_export_model

# Train and export the model
export_data = train_and_export_model(
    symbol=symbol_to_predict,
    data=indicators,
    lookback_window=lookback_window,
    epochs=50,
    batch_size=32,
    export_dir="exported_models"
)

print("✅ Model trained and exported successfully!")
print(f"Model name: {export_data['model_name']}")
print(f"Accuracy: {export_data['accuracy']:.4f}")
print(f"Sharpe ratio: {export_data['sharpe_ratio']:.4f}")
print(f"Total return: {export_data['total_return']:.4f}")
"""
    print("Copy and paste this code into your notebook:")
    print(code)

if __name__ == "__main__":
    notebook_code_snippet()
