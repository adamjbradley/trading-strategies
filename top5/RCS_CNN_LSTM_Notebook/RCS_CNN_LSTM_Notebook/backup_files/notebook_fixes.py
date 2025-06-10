"""
Notebook Fixes

This script contains fixes for common issues in the RCS_CNN_LSTM notebook.
Copy and paste the relevant sections directly into your notebook cells.
"""

#####################################################################
# FIX 1: Define core_features variable
#####################################################################

# Copy and paste this code into a cell before any cell that uses core_features
"""
# Define the core_features variable
core_features = [
    'rsi', 
    'macd', 
    'momentum', 
    'cci', 
    'atr', 
    'adx', 
    'stoch_k', 
    'stoch_d', 
    'roc', 
    'bbw', 
    'return_1d', 
    'return_3d', 
    'rolling_mean_5', 
    'rolling_std_5', 
    'momentum_slope'
]
print("✅ core_features variable defined with", len(core_features), "features")
"""

#####################################################################
# FIX 2: Fixed compute_permutation_importance function
#####################################################################

# Copy and paste this code into a cell before any cell that uses compute_permutation_importance
"""
import copy
import numpy as np
from sklearn.metrics import accuracy_score

def compute_permutation_importance(model, X_val, y_val, feature_names, threshold=0.5):
    \"\"\"
    Compute feature importance via permutation method.
    
    This version handles both 3D input (batch, time_steps, features) and
    4D input (batch, time_steps, features, channels) for CNN models.
    \"\"\"
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

print("✅ Fixed compute_permutation_importance function is ready to use")
"""

print("✅ Notebook fixes are ready to be copied into your notebook")
print("Copy and paste the relevant sections directly into your notebook cells")
