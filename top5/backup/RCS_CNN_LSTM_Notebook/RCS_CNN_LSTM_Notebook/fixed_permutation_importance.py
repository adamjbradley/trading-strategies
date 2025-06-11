"""
Fixed Permutation Importance Module

This module provides functions for calculating permutation importance for machine learning models.
"""

import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_permutation_importance(model, X, y, feature_names, n_repeats=10, random_state=42):
    """
    Calculate permutation importance for a model.
    
    Parameters:
    -----------
    model : object
        Model with a predict method
    X : numpy.ndarray
        Feature matrix
    y : numpy.ndarray
        Target vector
    feature_names : list
        List of feature names
    n_repeats : int, default=10
        Number of times to permute each feature
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with feature names and importance scores
    """
    # Get baseline score
    baseline_score = accuracy_score(y, model.predict(X))
    
    # Initialize importance scores
    importances = np.zeros((n_repeats, len(feature_names)))
    
    # Set random seed
    np.random.seed(random_state)
    
    # Calculate importance for each feature
    for i, feature in enumerate(range(X.shape[1])):
        for j in range(n_repeats):
            # Create a copy of the feature matrix
            X_permuted = X.copy()
            
            # Permute the feature
            X_permuted[:, feature] = np.random.permutation(X_permuted[:, feature])
            
            # Calculate score with permuted feature
            permuted_score = accuracy_score(y, model.predict(X_permuted))
            
            # Calculate importance
            importances[j, i] = baseline_score - permuted_score
    
    # Calculate mean importance
    mean_importances = np.mean(importances, axis=0)
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': mean_importances
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    return importance_df

def plot_permutation_importance(importance_df, title="Permutation Importance", figsize=(10, 6)):
    """
    Plot permutation importance.
    
    Parameters:
    -----------
    importance_df : pandas.DataFrame
        DataFrame with feature names and importance scores
    title : str, default="Permutation Importance"
        Plot title
    figsize : tuple, default=(10, 6)
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot importance
    sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
    
    # Set title and labels
    ax.set_title(title)
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    
    # Add grid
    ax.grid(True, axis='x')
    
    return fig

def calculate_permutation_importance_for_sequence_model(model, X, y, feature_names, n_repeats=10, random_state=42):
    """
    Calculate permutation importance for a sequence model.
    
    Parameters:
    -----------
    model : object
        Model with a predict method
    X : numpy.ndarray
        Feature matrix with shape (samples, timesteps, features)
    y : numpy.ndarray
        Target vector
    feature_names : list
        List of feature names
    n_repeats : int, default=10
        Number of times to permute each feature
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with feature names and importance scores
    """
    # Get baseline score
    baseline_preds = model.predict(X)
    baseline_score = accuracy_score(y, (baseline_preds > 0.5).astype(int))
    
    # Initialize importance scores
    importances = np.zeros((n_repeats, len(feature_names)))
    
    # Set random seed
    np.random.seed(random_state)
    
    # Calculate importance for each feature
    for i, feature in enumerate(range(X.shape[2])):
        for j in range(n_repeats):
            # Create a copy of the feature matrix
            X_permuted = X.copy()
            
            # Permute the feature across all timesteps
            for t in range(X.shape[1]):
                X_permuted[:, t, feature] = np.random.permutation(X_permuted[:, t, feature])
            
            # Calculate score with permuted feature
            permuted_preds = model.predict(X_permuted)
            permuted_score = accuracy_score(y, (permuted_preds > 0.5).astype(int))
            
            # Calculate importance
            importances[j, i] = baseline_score - permuted_score
    
    # Calculate mean importance
    mean_importances = np.mean(importances, axis=0)
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': mean_importances
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    return importance_df

print("✅ Fixed permutation importance module loaded")
