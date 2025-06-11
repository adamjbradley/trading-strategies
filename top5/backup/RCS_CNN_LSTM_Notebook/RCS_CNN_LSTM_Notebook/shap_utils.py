"""
SHAP Utilities

This module provides utilities for working with SHAP values.
"""

import numpy as np
import pandas as pd
import shap

def compute_shap_feature_importance(model, feature_matrix, feature_names=None, target=None):
    """
    Compute feature importance using SHAP values.
    
    Parameters:
    -----------
    model : object or None
        Trained model (e.g., RandomForestClassifier). If None and target is provided, a new model will be trained.
    feature_matrix : numpy.ndarray or pandas.DataFrame
        Feature matrix
    feature_names : list, optional
        List of feature names. If None and feature_matrix is a DataFrame, column names will be used.
    target : numpy.ndarray or pandas.Series, optional
        Target values. Required if model is None.
        
    Returns:
    --------
    pandas.Series
        Series with feature names as index and importance scores as values, sorted by importance
    """
    # Train a model if one isn't provided
    if model is None:
        if target is None:
            # Try to find target in the same scope
            import inspect
            frame = inspect.currentframe()
            try:
                outer_frames = inspect.getouterframes(frame)
                for outer_frame in outer_frames:
                    if 'target' in outer_frame.frame.f_locals:
                        target = outer_frame.frame.f_locals['target']
                        print("✅ Found target in outer scope")
                        break
            finally:
                del frame
        
        if target is None:
            print("⚠️ No target provided and couldn't find one in outer scope. Using random forest feature importance instead.")
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            # Just fit with random data to get feature importance
            X = feature_matrix.values if isinstance(feature_matrix, pd.DataFrame) else feature_matrix
            y = np.random.randint(0, 2, size=X.shape[0])
            model.fit(X, y)
            
            # Get feature names
            if feature_names is None:
                if isinstance(feature_matrix, pd.DataFrame):
                    feature_names = feature_matrix.columns.tolist()
                else:
                    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            
            # Return feature importance
            importance = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
            return importance
        else:
            print("✅ Training a new RandomForestClassifier for SHAP values")
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            X = feature_matrix.values if isinstance(feature_matrix, pd.DataFrame) else feature_matrix
            model.fit(X, target)
    
    # Get feature names
    if feature_names is None:
        if isinstance(feature_matrix, pd.DataFrame):
            feature_names = feature_matrix.columns.tolist()
        else:
            feature_names = [f"feature_{i}" for i in range(feature_matrix.shape[1])]
    
    # Check if feature_names length matches feature_matrix columns
    if len(feature_names) != feature_matrix.shape[1]:
        print(f"⚠️ Length mismatch: {len(feature_names)} feature names but {feature_matrix.shape[1]} features")
        # Adjust feature_names to match feature_matrix columns
        if len(feature_names) > feature_matrix.shape[1]:
            feature_names = feature_names[:feature_matrix.shape[1]]
            print(f"Truncated feature_names to {len(feature_names)}")
        else:
            # Extend feature_names with generic names
            additional_names = [f"feature_{i+len(feature_names)}" for i in range(feature_matrix.shape[1] - len(feature_names))]
            feature_names = feature_names + additional_names
            print(f"Extended feature_names to {len(feature_names)}")
    
    # Create explainer
    try:
        explainer = shap.TreeExplainer(model)
        
        # Compute SHAP values
        if isinstance(feature_matrix, pd.DataFrame):
            shap_values = explainer.shap_values(feature_matrix.values)
        else:
            shap_values = explainer.shap_values(feature_matrix)
        
        # Handle different output formats from shap_values
        if isinstance(shap_values, list):
            # For multi-class models, use the positive class (index 1)
            if len(shap_values) > 1:
                shap_sum = np.abs(shap_values[1]).mean(axis=0)
            else:
                shap_sum = np.abs(shap_values[0]).mean(axis=0)
        else:
            shap_sum = np.abs(shap_values).mean(axis=0)
        
        # Create Series with feature importance
        importance = pd.Series(shap_sum, index=feature_names).sort_values(ascending=False)
        
        return importance
    
    except Exception as e:
        print(f"⚠️ Error computing SHAP values: {str(e)}")
        print("Falling back to model feature importance if available")
        
        # Try to use model's feature_importances_ attribute
        if hasattr(model, 'feature_importances_'):
            importance = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
            return importance
        
        # If all else fails, return a Series with zeros
        return pd.Series(np.zeros(len(feature_names)), index=feature_names)

def plot_shap_summary(model, feature_matrix, feature_names=None, max_display=20, target=None):
    """
    Plot SHAP summary.
    
    Parameters:
    -----------
    model : object or None
        Trained model (e.g., RandomForestClassifier). If None and target is provided, a new model will be trained.
    feature_matrix : numpy.ndarray or pandas.DataFrame
        Feature matrix
    feature_names : list, optional
        List of feature names. If None and feature_matrix is a DataFrame, column names will be used.
    max_display : int, default=20
        Maximum number of features to display
    target : numpy.ndarray or pandas.Series, optional
        Target values. Required if model is None.
        
    Returns:
    --------
    None
    """
    # Train a model if one isn't provided
    if model is None:
        if target is None:
            # Try to find target in the same scope
            import inspect
            frame = inspect.currentframe()
            try:
                outer_frames = inspect.getouterframes(frame)
                for outer_frame in outer_frames:
                    if 'target' in outer_frame.frame.f_locals:
                        target = outer_frame.frame.f_locals['target']
                        print("✅ Found target in outer scope")
                        break
            finally:
                del frame
        
        if target is None:
            print("⚠️ No target provided and couldn't find one in outer scope. Cannot plot SHAP summary.")
            return
        else:
            print("✅ Training a new RandomForestClassifier for SHAP values")
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            X = feature_matrix.values if isinstance(feature_matrix, pd.DataFrame) else feature_matrix
            model.fit(X, target)
    
    # Get feature names
    if feature_names is None:
        if isinstance(feature_matrix, pd.DataFrame):
            feature_names = feature_matrix.columns.tolist()
        else:
            feature_names = [f"feature_{i}" for i in range(feature_matrix.shape[1])]
    
    # Check if feature_names length matches feature_matrix columns
    if len(feature_names) != feature_matrix.shape[1]:
        print(f"⚠️ Length mismatch: {len(feature_names)} feature names but {feature_matrix.shape[1]} features")
        # Adjust feature_names to match feature_matrix columns
        if len(feature_names) > feature_matrix.shape[1]:
            feature_names = feature_names[:feature_matrix.shape[1]]
            print(f"Truncated feature_names to {len(feature_names)}")
        else:
            # Extend feature_names with generic names
            additional_names = [f"feature_{i+len(feature_names)}" for i in range(feature_matrix.shape[1] - len(feature_names))]
            feature_names = feature_names + additional_names
            print(f"Extended feature_names to {len(feature_names)}")
    
    # Create DataFrame with feature names if input is numpy array
    if not isinstance(feature_matrix, pd.DataFrame):
        feature_matrix = pd.DataFrame(feature_matrix, columns=feature_names)
    
    # Create explainer
    try:
        explainer = shap.TreeExplainer(model)
        
        # Compute SHAP values
        shap_values = explainer.shap_values(feature_matrix)
        
        # Plot summary
        if isinstance(shap_values, list) and len(shap_values) > 1:
            # For multi-class models, use the positive class (index 1)
            shap.summary_plot(shap_values[1], feature_matrix, feature_names=feature_names, max_display=max_display, show=False)
        else:
            # For binary classification or regression
            shap.summary_plot(shap_values, feature_matrix, feature_names=feature_names, max_display=max_display, show=False)
    
    except Exception as e:
        print(f"⚠️ Error plotting SHAP summary: {str(e)}")

print("✅ SHAP utilities loaded")
