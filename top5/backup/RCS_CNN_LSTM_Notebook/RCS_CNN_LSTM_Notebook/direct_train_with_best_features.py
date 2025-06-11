"""
Direct Train with Best Features

This script provides a function to directly train a model using the best features.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, BatchNormalization, Flatten
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

def direct_train_with_best_features(symbol, data, best_features, lookback_window=20, epochs=50, batch_size=32, export_dir="exported_models"):
    """
    Directly train a model using the provided best features.
    
    Parameters:
    -----------
    symbol : str
        Trading symbol (e.g., 'EURUSD')
    data : pandas.DataFrame
        DataFrame containing the data for training
    best_features : list
        List of feature names to use for training
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
    print(f"🔍 Training model for {symbol} using {len(best_features)} best features with lookback window {lookback_window}")
    print(f"Best features: {best_features}")
    
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
    
    # Filter the features to only include those available in the data
    available_features = [f for f in best_features if f in feature_matrix.columns]
    
    if not available_features:
        # If no features are available, use all numeric columns except target
        print("⚠️ No valid features found, using all numeric columns")
        numeric_cols = feature_matrix.select_dtypes(include=[np.number]).columns.tolist()
        available_features = [col for col in numeric_cols if col != 'target']
        
        if not available_features:
            raise ValueError(f"No numeric features found in data")
    
    print(f"✅ Using {len(available_features)} features for model training: {available_features}")
    
    # Check for NaN values in features and target
    feature_data = feature_matrix[available_features]
    nan_counts = feature_data.isna().sum()
    
    if nan_counts.sum() > 0:
        print("⚠️ NaN values found in features:")
        for col, count in nan_counts[nan_counts > 0].items():
            print(f"  - {col}: {count} NaN values")
        
        print("Filling NaN values with appropriate methods...")
        # Fill NaN values with appropriate methods
        for col in feature_data.columns:
            if nan_counts[col] > 0:
                # Use forward fill first
                feature_data[col] = feature_data[col].ffill()
                # Then use backward fill for any remaining NaNs
                feature_data[col] = feature_data[col].bfill()
                # If still NaN (e.g., all NaN column), fill with 0
                feature_data[col] = feature_data[col].fillna(0)
    
    # Check for NaN values in target
    target_nan_count = feature_matrix['target'].isna().sum()
    if target_nan_count > 0:
        print(f"⚠️ {target_nan_count} NaN values found in target, filling with forward fill")
        feature_matrix['target'] = feature_matrix['target'].ffill().bfill().fillna(0)
    
    # Drop any remaining rows with NaN values
    valid_rows = ~(feature_data.isna().any(axis=1) | feature_matrix['target'].isna())
    if valid_rows.sum() < len(feature_matrix):
        print(f"⚠️ Dropping {len(feature_matrix) - valid_rows.sum()} rows with NaN values")
        feature_data = feature_data[valid_rows]
        target_data = feature_matrix['target'][valid_rows]
    else:
        target_data = feature_matrix['target']
    
    # Scale the features
    scaler = StandardScaler()
    feature_data_scaled = scaler.fit_transform(feature_data)
    
    # Create sequences for LSTM
    X_seq = []
    y_seq = []
    
    for i in range(lookback_window, len(feature_data_scaled)):
        X_seq.append(feature_data_scaled[i-lookback_window:i])
        y_seq.append(target_data.iloc[i])
    
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    # Verify data shapes
    print(f"X_seq shape: {X_seq.shape}")
    print(f"y_seq shape: {y_seq.shape}")
    
    # Split the data into train, validation, and test sets
    # Assuming a 70/15/15 split
    train_size = int(0.7 * len(X_seq))
    val_size = int(0.15 * len(X_seq))
    
    X_train = X_seq[:train_size]
    y_train = y_seq[:train_size]
    
    X_val = X_seq[train_size:train_size+val_size]
    y_val = y_seq[train_size:train_size+val_size]
    
    X_test = X_seq[train_size+val_size:]
    y_test = y_seq[train_size+val_size:]
    
    # Create and train the model
    model = create_cnn_lstm_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=epochs,
        batch_size=batch_size
    )
    
    # Evaluate the model
    y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Print metrics
    print(f"✅ Model evaluation metrics:")
    print(f"  - accuracy: {accuracy:.4f}")
    print(f"  - precision: {precision:.4f}")
    print(f"  - recall: {recall:.4f}")
    print(f"  - f1_score: {f1:.4f}")
    
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
        feature_names=available_features,
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
# --- Train model using best features directly ---
from direct_train_with_best_features import direct_train_with_best_features

# Use the best features we found earlier
export_data = direct_train_with_best_features(
    symbol=symbol_to_predict,
    data=indicators,
    best_features=best_features,  # Use the best_features variable from the feature evaluation
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
