"""
Feature Importance Utilities

This module provides utilities for calculating feature importance.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def prepare_features_for_importance(data, feature_list, target, lookback=20):
    """
    Prepare features for importance calculation.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing the features
    feature_list : list
        List of feature names to use
    target : pandas.Series
        Target variable
    lookback : int, default=20
        Number of timesteps to use for sequence data
        
    Returns:
    --------
    tuple
        (X, y, X_train, X_test, y_train, y_test, feature_names)
    """
    # Filter features that exist in the data
    available_features = [f for f in feature_list if f in data.columns]
    
    if len(available_features) == 0:
        raise ValueError("None of the specified features are available in the data")
    
    print(f"Using {len(available_features)} features: {available_features}")
    
    # Extract features and target
    features = data[available_features].values
    y = target.values
    
    # Ensure features and target are aligned
    min_len = min(len(features), len(y))
    features = features[:min_len]
    y = y[:min_len]
    
    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Create sequences
    X = np.array([features_scaled[i-lookback:i] for i in range(lookback, len(features_scaled))])
    y_seq = y[lookback:]
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_seq, test_size=0.2, shuffle=False)
    
    return X, y_seq, X_train, X_test, y_train, y_test, available_features

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
    from sklearn.metrics import accuracy_score
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
            for t in range(X_permuted.shape[1]):
                X_permuted[:, t, i] = np.random.permutation(X_permuted[:, t, i])
            
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

print("✅ Feature importance utilities loaded")
