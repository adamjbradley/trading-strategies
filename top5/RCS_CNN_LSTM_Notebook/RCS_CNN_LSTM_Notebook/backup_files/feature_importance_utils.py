"""
Feature Importance Utilities

This module provides functions for computing feature importance using permutation importance
and visualizing the results. It includes robust handling of different input shapes for CNN models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def compute_permutation_importance(model, X_val, y_val, feature_names, n_repeats=3):
    """
    Compute permutation importance for features in a model.
    
    This function handles both 3D and 4D input shapes for CNN models.
    
    Parameters:
    -----------
    model : keras.Model
        Trained model to evaluate
    X_val : numpy.ndarray
        Validation data with shape (samples, time_steps, features) or (samples, time_steps, features, channels)
    y_val : numpy.ndarray
        Validation labels
    feature_names : list
        List of feature names
    n_repeats : int, default=3
        Number of times to permute each feature
        
    Returns:
    --------
    list
        List of tuples (feature_name, importance_score)
    """
    print("Model input shape:", model.input_shape)
    print("X_val shape:", X_val.shape)
    
    # Handle different input shapes for CNN vs non-CNN models
    if hasattr(model, 'input_shape'):
        # Check if model expects 4D input (CNN) but X_val is 3D
        if len(model.input_shape) == 4 and len(X_val.shape) == 3:
            print("Reshaping 3D input to 4D for CNN model...")
            # Reshape from (batch, time_steps, features) to (batch, time_steps, features, 1)
            X_val_reshaped = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], 1)
            print("Reshaped X_val shape:", X_val_reshaped.shape)
            base_preds = (model.predict(X_val_reshaped) > 0.5).astype(int).flatten()
        # Check if model expects 3D input but X_val is 4D
        elif len(model.input_shape) == 3 and len(X_val.shape) == 4:
            print("Reshaping 4D input to 3D for non-CNN model...")
            # Reshape from (batch, time_steps, features, 1) to (batch, time_steps, features)
            X_val_reshaped = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2])
            print("Reshaped X_val shape:", X_val_reshaped.shape)
            base_preds = (model.predict(X_val_reshaped) > 0.5).astype(int).flatten()
        else:
            print("Using original shape...")
            base_preds = (model.predict(X_val) > 0.5).astype(int).flatten()
    else:
        print("Model has no input_shape attribute, using original shape...")
        base_preds = (model.predict(X_val) > 0.5).astype(int).flatten()
    
    # Ensure y_val and base_preds have the same length
    min_len = min(len(y_val), len(base_preds))
    y_val_aligned = y_val[:min_len]
    base_preds = base_preds[:min_len]
    
    # Calculate baseline accuracy
    base_score = accuracy_score(y_val_aligned, base_preds)
    print(f"Baseline accuracy: {base_score:.4f}")
    
    # Initialize importance scores
    importances = []
    
    # For each feature
    for i, feature_name in enumerate(feature_names):
        # Initialize scores for this feature
        feature_scores = []
        
        # Repeat permutation n_repeats times
        for _ in range(n_repeats):
            # Create a copy of the validation data
            if len(X_val.shape) == 4:  # 4D input (batch, time_steps, features, channels)
                X_permuted = X_val.copy()
                # Permute the feature across all samples
                perm_idx = np.random.permutation(X_permuted.shape[0])
                X_permuted[:, :, i, :] = X_permuted[perm_idx, :, i, :]
            else:  # 3D input (batch, time_steps, features)
                X_permuted = X_val.copy()
                # Permute the feature across all samples
                perm_idx = np.random.permutation(X_permuted.shape[0])
                X_permuted[:, :, i] = X_permuted[perm_idx, :, i]
            
            # Make predictions with permuted data
            if hasattr(model, 'input_shape'):
                # Check if model expects 4D input (CNN) but X_permuted is 3D
                if len(model.input_shape) == 4 and len(X_permuted.shape) == 3:
                    # Reshape from (batch, time_steps, features) to (batch, time_steps, features, 1)
                    X_permuted_reshaped = X_permuted.reshape(X_permuted.shape[0], X_permuted.shape[1], X_permuted.shape[2], 1)
                    perm_preds = (model.predict(X_permuted_reshaped) > 0.5).astype(int).flatten()
                # Check if model expects 3D input but X_permuted is 4D
                elif len(model.input_shape) == 3 and len(X_permuted.shape) == 4:
                    # Reshape from (batch, time_steps, features, 1) to (batch, time_steps, features)
                    X_permuted_reshaped = X_permuted.reshape(X_permuted.shape[0], X_permuted.shape[1], X_permuted.shape[2])
                    perm_preds = (model.predict(X_permuted_reshaped) > 0.5).astype(int).flatten()
                else:
                    perm_preds = (model.predict(X_permuted) > 0.5).astype(int).flatten()
            else:
                perm_preds = (model.predict(X_permuted) > 0.5).astype(int).flatten()
            
            # Ensure perm_preds has the same length as y_val_aligned
            perm_preds = perm_preds[:min_len]
            
            # Calculate permuted accuracy
            perm_score = accuracy_score(y_val_aligned, perm_preds)
            
            # Calculate importance (decrease in accuracy)
            importance = base_score - perm_score
            feature_scores.append(importance)
        
        # Average importance across repeats
        mean_importance = np.mean(feature_scores)
        importances.append((feature_name, mean_importance))
        print(f"Feature {feature_name}: importance = {mean_importance:.4f}")
    
    # Sort by importance (descending)
    importances.sort(key=lambda x: x[1], reverse=True)
    
    return importances

def prepare_features_for_importance(data, minimal_features, target, lookback=20):
    """
    Prepare features for importance analysis with robust feature_names alignment.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing the features
    minimal_features : list
        List of feature names to use
    target : pandas.Series
        Target variable
    lookback : int, default=20
        Number of periods to look back for creating rolling windows
        
    Returns:
    --------
    tuple
        (X, y, X_train, X_test, y_train, y_test, feature_names)
    """
    # Filter features that exist in the data
    minimal_features = [f for f in minimal_features if f in data.columns]
    
    # Extract features and drop NaN values
    features = data[minimal_features].dropna().reset_index(drop=True)
    feature_names = features.columns.tolist()
    print("feature_names:", feature_names)
    
    # Align target with features
    y = target.reset_index(drop=True).values
    
    # Standardize features
    features_scaled = StandardScaler().fit_transform(features)
    
    # Create rolling windows
    X = np.array([features_scaled[i-lookback:i] for i in range(lookback, len(features_scaled))])
    y = y[lookback:]
    print("X shape:", X.shape, "y shape:", y.shape)
    
    # Train/test split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    return X, y, X_train, X_test, y_train, y_test, feature_names

def plot_feature_importance(importances, symbol=None):
    """
    Plot feature importance using a bar chart.
    
    Parameters:
    -----------
    importances : list or pandas.DataFrame
        List of tuples (feature_name, importance_score) or DataFrame with columns ["Feature", "Importance"]
    symbol : str, optional
        Symbol to include in the plot title
    """
    if not isinstance(importances, pd.DataFrame):
        importance_df = pd.DataFrame(importances, columns=["Feature", "Importance"])
    else:
        importance_df = importances
    
    plt.figure(figsize=(10, 5))
    sns.barplot(data=importance_df, x="Importance", y="Feature")
    
    title = "Feature Importance via Permutation"
    if symbol:
        title += f": {symbol}"
    
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def evaluate_predictions(model, X_test, y_test):
    """
    Evaluate model predictions with proper alignment.
    
    Parameters:
    -----------
    model : keras.Model
        Trained model to evaluate
    X_test : numpy.ndarray
        Test data
    y_test : numpy.ndarray
        Test labels
        
    Returns:
    --------
    tuple
        (y_test_aligned, y_pred_aligned, accuracy)
    """
    # Handle different input shapes for CNN vs non-CNN models
    if hasattr(model, 'input_shape'):
        # Check if model expects 4D input (CNN) but X_test is 3D
        if len(model.input_shape) == 4 and len(X_test.shape) == 3:
            print("Reshaping 3D input to 4D for CNN model...")
            # Reshape from (batch, time_steps, features) to (batch, time_steps, features, 1)
            X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
            print("Reshaped X_test shape:", X_test_reshaped.shape)
            y_pred = (model.predict(X_test_reshaped) > 0.5).astype(int).flatten()
        # Check if model expects 3D input but X_test is 4D
        elif len(model.input_shape) == 3 and len(X_test.shape) == 4:
            print("Reshaping 4D input to 3D for non-CNN model...")
            # Reshape from (batch, time_steps, features, 1) to (batch, time_steps, features)
            X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])
            print("Reshaped X_test shape:", X_test_reshaped.shape)
            y_pred = (model.predict(X_test_reshaped) > 0.5).astype(int).flatten()
        else:
            print("Using original shape...")
            y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
    else:
        print("Model has no input_shape attribute, using original shape...")
        y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
    
    # Ensure y_test and y_pred have the same length
    min_len = min(len(y_test), len(y_pred))
    y_test_aligned = y_test[:min_len]
    y_pred_aligned = y_pred[:min_len]
    
    print("y_test_aligned shape:", y_test_aligned.shape)
    print("y_pred_aligned shape:", y_pred_aligned.shape)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test_aligned, y_pred_aligned)
    print(f"Accuracy: {accuracy:.4f}")
    
    return y_test_aligned, y_pred_aligned, accuracy

# Example usage
"""
# Define minimal features
minimal_features = ["rsi", "macd", "momentum", "cci", "atr", "adx", "stoch_k", "stoch_d", 
                   "roc", "bbw", "return_1d", "return_3d", "rolling_mean_5", 
                   "rolling_std_5", "momentum_slope"]

# Prepare features
X, y, X_train, X_test, y_train, y_test, feature_names = prepare_features_for_importance(
    data, minimal_features, target, lookback=20
)

# Verify feature alignment
print("feature_names:", feature_names)
print("X_test shape:", X_test.shape)
assert X_test.shape[2] == len(feature_names), "Mismatch between X_test features and feature_names"

# Compute feature importance
importances = compute_permutation_importance(model, X_test, y_test, feature_names)

# Plot feature importance
plot_feature_importance(importances, symbol="EURUSD")
"""

print("✅ Feature importance utilities loaded")
