import numpy as np
import pandas as pd
import copy
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

"""
Fixed Code Block for Permutation Importance

This script provides a fixed version of the permutation importance code block
that properly handles the CNN model's input shape requirements.
"""

import copy
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def compute_permutation_importance(model, X_val, y_val, feature_names, threshold=0.5):
    """
    Compute feature importance via permutation method.
    
    This version handles both 3D input (batch, time_steps, features) and
    4D input (batch, time_steps, features, channels) for CNN models.
    """
    print("Input shapes:")
    print("X_val shape:", X_val.shape)
    print("y_val shape:", y_val.shape)
    print("feature_names:", feature_names)
    
    # Check if we need to reshape the input for CNN models (4D input)
    if hasattr(model, 'input_shape') and len(model.input_shape) == 4 and len(X_val.shape) == 3:
        print("Reshaping input from 3D to 4D...")
        # Reshape from (batch, time_steps, features) to (batch, time_steps, features, 1)
        X_val_reshaped = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], 1)
        print("Reshaped X_val shape:", X_val_reshaped.shape)
        base_preds = (model.predict(X_val_reshaped) > threshold).astype(int).flatten()
    else:
        print("Using original shape...")
        # Use original shape for non-CNN models
        base_preds = (model.predict(X_val) > threshold).astype(int).flatten()
    
    print("base_preds shape:", base_preds.shape)
    
    min_len = min(len(y_val), len(base_preds))
    y_val_aligned = y_val[:min_len]
    base_preds_aligned = base_preds[:min_len]
    base_acc = accuracy_score(y_val_aligned, base_preds_aligned)
    print("Base accuracy:", base_acc)
    
    importances = []
    
    for i in range(X_val.shape[2]):
        print(f"Processing feature {i}: {feature_names[i]}")
        X_permuted = copy.deepcopy(X_val)
        np.random.shuffle(X_permuted[:, :, i])
        
        # Apply the same reshaping logic for prediction
        if hasattr(model, 'input_shape') and len(model.input_shape) == 4 and len(X_permuted.shape) == 3:
            X_permuted_reshaped = X_permuted.reshape(X_permuted.shape[0], X_permuted.shape[1], X_permuted.shape[2], 1)
            perm_preds = (model.predict(X_permuted_reshaped) > threshold).astype(int).flatten()
        else:
            perm_preds = (model.predict(X_permuted) > threshold).astype(int).flatten()
        
        min_len_perm = min(len(y_val), len(perm_preds))
        y_val_perm = y_val[:min_len_perm]
        perm_preds_aligned = perm_preds[:min_len_perm]
        perm_acc = accuracy_score(y_val_perm, perm_preds_aligned)
        importance = base_acc - perm_acc
        importances.append((feature_names[i], importance))
        print(f"Feature {feature_names[i]} importance: {importance}")
    
    return sorted(importances, key=lambda x: x[1], reverse=True)

# Example usage (replace with your actual code):
def run_permutation_importance(model, data, target, symbol, minimal_features=None):
    """
    Run permutation importance analysis on the given model and data.
    
    Parameters:
    -----------
    model : keras.Model
        Trained model to evaluate
    data : pandas.DataFrame
        DataFrame containing features
    target : pandas.Series
        Target variable
    symbol : str
        Symbol name for plot title
    minimal_features : list, optional
        List of minimal features to use. If None, uses ["rsi", "macd", "momentum", "cci"]
    """
    # Define minimal features if not provided
    if minimal_features is None:
        minimal_features = ["rsi", "macd", "momentum", "cci"]
    
    # Filter features that exist in data
    minimal_features = [f for f in minimal_features if f in data.columns]
    features = data[minimal_features].dropna()
    feature_names = features.columns.tolist()
    print("feature_names:", feature_names)
    
    # Reset index to align features and target
    features = features.reset_index(drop=True)
    y = target.reset_index(drop=True).values
    
    # Standardize and create rolling windows
    features_scaled = StandardScaler().fit_transform(features)
    lookback = 20
    X = np.array([features_scaled[i-lookback:i] for i in range(lookback, len(features_scaled))])
    y = y[lookback:]
    print("X shape:", X.shape, "y shape:", y.shape)
    
    # Train/test split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Check if X_test and feature_names match
    print("X_test shape:", X_test.shape)
    print("feature_names:", feature_names)
    assert X_test.shape[2] == len(feature_names), "Mismatch between X_test features and feature_names"
    
    # Compute permutation importance
    importances = compute_permutation_importance(model, X_test, y_test, feature_names)
    importance_df = pd.DataFrame(importances, columns=["Feature", "Importance"])
    
    # Plot results
    plt.figure(figsize=(10,5))
    sns.barplot(data=importance_df, x="Importance", y="Feature")
    plt.title(f"Feature Importance via Permutation: {symbol}")
    plt.grid(True)
    plt.show()
    
    return importance_df

def evaluate_and_save_feature_sets(feature_matrix, y, feature_sets, symbol_to_predict, lookback_window=20, model_fn=None, epochs=10, batch_size=32):
    """
    Evaluate feature sets, build results_df, save results to CSV, and return results_df and best_row.
    """
    results_df = compare_feature_sets(
        feature_matrix, y, feature_sets,
        lookback_window=lookback_window,
        model_fn=model_fn,
        epochs=epochs,
        batch_size=batch_size
    )
    best_row = results_df.sort_values(by='Accuracy', ascending=False).iloc[0]
    results_df.to_csv(f'feature_set_results_{symbol_to_predict}.csv', index=False)
    best_row.to_frame().T.to_csv(f'best_feature_set_{symbol_to_predict}.csv', index=False)
    print(f"📁 Saved results to feature_set_results_{symbol_to_predict}.csv")
    print(f"🏆 Best set saved to best_feature_set_{symbol_to_predict}.csv")
    return results_df, best_row
