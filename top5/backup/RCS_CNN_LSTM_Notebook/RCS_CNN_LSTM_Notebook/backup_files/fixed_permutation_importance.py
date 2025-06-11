"""
Fixed Permutation Importance

This module provides a fixed version of the compute_permutation_importance function
that properly handles different input shapes for CNN models.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

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

print("✅ Fixed permutation importance function loaded")
