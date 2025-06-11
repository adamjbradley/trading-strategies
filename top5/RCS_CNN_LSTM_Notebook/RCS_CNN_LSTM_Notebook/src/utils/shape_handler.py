"""
Input Shape Handler

This module provides utilities for handling input shape mismatches in deep learning models.
"""

import numpy as np
from sklearn.metrics import accuracy_score

def ensure_compatible_input_shape(X, expected_shape):
    """
    Ensure that the input shape is compatible with the expected shape.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Input data
    expected_shape : tuple
        Expected shape (excluding batch dimension)
        
    Returns:
    --------
    numpy.ndarray
        Reshaped input data
    """
    # Get current shape (excluding batch dimension)
    current_shape = X.shape[1:]
    
    # If shapes match, return original data
    if current_shape == expected_shape:
        return X
    
    # If dimensions match but sizes differ, reshape
    if len(current_shape) == len(expected_shape):
        # Handle 2D case (samples, features)
        if len(current_shape) == 1:
            # Pad or truncate features
            if current_shape[0] < expected_shape[0]:
                # Pad with zeros
                padding = np.zeros((X.shape[0], expected_shape[0] - current_shape[0]))
                return np.hstack((X, padding))
            else:
                # Truncate
                return X[:, :expected_shape[0]]
        
        # Handle 3D case (samples, timesteps, features)
        elif len(current_shape) == 2:
            result = X.copy()
            
            # Handle timestep dimension
            if current_shape[0] < expected_shape[0]:
                # Pad with zeros
                padding = np.zeros((X.shape[0], expected_shape[0] - current_shape[0], current_shape[1]))
                result = np.concatenate((result, padding), axis=1)
            elif current_shape[0] > expected_shape[0]:
                # Truncate
                result = result[:, :expected_shape[0], :]
            
            # Handle feature dimension
            if current_shape[1] < expected_shape[1]:
                # Pad with zeros
                padding = np.zeros((result.shape[0], result.shape[1], expected_shape[1] - current_shape[1]))
                result = np.concatenate((result, padding), axis=2)
            elif current_shape[1] > expected_shape[1]:
                # Truncate
                result = result[:, :, :expected_shape[1]]
            
            return result
    
    # If dimensions don't match, try to reshape
    try:
        # Flatten and reshape
        flattened = X.reshape(X.shape[0], -1)
        expected_size = np.prod(expected_shape)
        
        if flattened.shape[1] < expected_size:
            # Pad with zeros
            padding = np.zeros((flattened.shape[0], expected_size - flattened.shape[1]))
            flattened = np.hstack((flattened, padding))
        elif flattened.shape[1] > expected_size:
            # Truncate
            flattened = flattened[:, :expected_size]
        
        # Reshape to expected shape
        return flattened.reshape((X.shape[0],) + expected_shape)
    except:
        raise ValueError(f"Cannot reshape input with shape {X.shape} to match expected shape {expected_shape}")

def compute_permutation_importance_with_shape_handling(model, X_val, y_val, feature_names, n_repeats=3, verbose=True):
    """
    Compute permutation importance with shape handling.
    
    Parameters:
    -----------
    model : keras.Model
        Trained model
    X_val : numpy.ndarray
        Validation features
    y_val : numpy.ndarray
        Validation targets
    feature_names : list
        List of feature names
    n_repeats : int, default=3
        Number of times to permute each feature
    verbose : bool, default=True
        Whether to print progress
        
    Returns:
    --------
    list
        List of (feature, importance) tuples
    """
    # Get baseline predictions
    baseline_preds = model.predict(X_val)
    baseline_preds_binary = (baseline_preds > 0.5).astype(int)
    
    # Calculate baseline accuracy
    baseline_accuracy = accuracy_score(y_val, baseline_preds_binary)
    
    if verbose:
        print(f"Baseline accuracy: {baseline_accuracy:.4f}")
    
    # Initialize importance scores
    importances = []
    
    # Calculate importance for each feature
    for i, feature_name in enumerate(feature_names):
        if verbose:
            print(f"Processing feature {i+1}/{len(feature_names)}: {feature_name}")
        
        feature_importance = 0
        
        for j in range(n_repeats):
            # Create a copy of the validation data
            X_permuted = X_val.copy()
            
            # Permute the feature across all timesteps
            if len(X_val.shape) == 3:  # (samples, timesteps, features)
                for t in range(X_permuted.shape[1]):
                    X_permuted[:, t, i] = np.random.permutation(X_permuted[:, t, i])
            else:  # (samples, features)
                X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
            
            # Get predictions with permuted feature
            permuted_preds = model.predict(X_permuted)
            permuted_preds_binary = (permuted_preds > 0.5).astype(int)
            
            # Calculate accuracy with permuted feature
            permuted_accuracy = accuracy_score(y_val, permuted_preds_binary)
            
            # Calculate importance (decrease in accuracy)
            importance = baseline_accuracy - permuted_accuracy
            feature_importance += importance
        
        # Average importance over repeats
        feature_importance /= n_repeats
        
        if verbose:
            print(f"  Importance: {feature_importance:.4f}")
        
        importances.append((feature_name, feature_importance))
    
    # Sort by importance
    importances.sort(key=lambda x: x[1], reverse=True)
    
    return importances

print("✅ Input shape handler loaded")
