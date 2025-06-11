"""
Feature Set Selection Fix

This script provides a fix for the feature set selection and saving process.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from feature_set_utils import save_feature_set_results, save_best_feature_set

def evaluate_and_save_feature_sets(feature_matrix, target, selected_feature_sets, lookback_window, symbol):
    """
    Evaluate feature sets and save the results and best feature set.
    
    Parameters:
    -----------
    feature_matrix : pandas.DataFrame
        DataFrame containing the features
    target : pandas.Series
        Series containing the target values
    selected_feature_sets : dict
        Dictionary mapping feature set names to lists of feature names
    lookback_window : int
        Number of time steps to use for sequence data
    symbol : str
        Trading symbol (e.g., 'EURUSD')
        
    Returns:
    --------
    tuple
        (results_df, best_feature_set)
    """
    # Ensure target is aligned with feature_matrix
    common_index = feature_matrix.index.intersection(target.index)
    feature_matrix = feature_matrix.loc[common_index]
    target = target.loc[common_index]
    
    # Convert target to numpy array
    y = target.values
    
    # Evaluate each feature set
    results = []
    for set_name, feature_list in selected_feature_sets.items():
        # Filter features to only include those available in feature_matrix
        feature_list = [f for f in feature_list if f in feature_matrix.columns]
        
        if not feature_list:
            print(f"Skipping feature set '{set_name}': no valid features present in feature_matrix.")
            continue
        
        print(f"Evaluating feature set: {set_name} with features: {feature_list}")
        
        # Extract and scale features
        X_set = feature_matrix[feature_list].values
        X_scaled = StandardScaler().fit_transform(X_set)
        
        # Create sequences for LSTM
        X_seq = np.array([X_scaled[i-lookback_window:i] for i in range(lookback_window, len(X_scaled))])
        y_seq = y[lookback_window:]
        
        # Train/test split
        split = int(len(X_seq) * 0.8)
        X_train, X_test = X_seq[:split], X_seq[split:]
        y_train, y_test = y_seq[:split], y_seq[split:]
        
        # Flatten sequences for RandomForest
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        
        # Train model
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train_flat, y_train)
        
        # Evaluate model
        y_pred = rf.predict(X_test_flat)
        
        # Ensure y_test and y_pred are the same length
        min_len = min(len(y_test), len(y_pred))
        y_test_aligned = y_test[:min_len]
        y_pred_aligned = y_pred[:min_len]
        
        # Calculate accuracy
        acc = accuracy_score(y_test_aligned, y_pred_aligned)
        
        # Store results
        results.append({
            "Feature Set": set_name,
            "Accuracy": acc,
            "Features": feature_list
        })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    save_feature_set_results(results_df, symbol, filename=f"feature_set_results_{symbol}.csv")
    
    # Find best feature set
    if not results_df.empty:
        best_row = results_df.loc[results_df['Accuracy'].idxmax()]
        best_feature_set = best_row['Features']
        
        # Save best feature set
        save_best_feature_set(best_row, symbol, filename=f"best_feature_set_{symbol}.csv")
        
        print(f"✅ Best feature set: {best_row['Feature Set']} with accuracy {best_row['Accuracy']:.4f}")
        return results_df, best_feature_set
    else:
        print("⚠️ No feature sets evaluated")
        return results_df, []

def get_multiple_feature_sets(feature_matrix, target=None, feature_selection_strategy="all"):
    """
    Get multiple feature sets for evaluation.
    
    Parameters:
    -----------
    feature_matrix : pandas.DataFrame
        DataFrame containing the features
    target : pandas.Series, optional
        Series containing the target values
    feature_selection_strategy : str, default="all"
        Strategy for selecting feature sets. Options: "all", "manual", "permutation", "shap"
        
    Returns:
    --------
    dict
        Dictionary mapping feature set names to lists of feature names
    """
    # Define manual feature sets
    manual_feature_sets = {}
    
    # Basic technical indicators
    manual_feature_sets["Basic Indicators"] = [
        'rsi', 'macd', 'momentum', 'cci'
    ]
    
    # Volatility indicators
    volatility_indicators = [
        'atr', 'bbw', 'rolling_std_5'
    ]
    volatility_cols = [col for col in volatility_indicators if col in feature_matrix.columns]
    if volatility_cols:
        manual_feature_sets["Volatility Indicators"] = volatility_cols
    
    # Trend indicators
    trend_indicators = [
        'adx', 'rolling_mean_5', 'macd'
    ]
    trend_cols = [col for col in trend_indicators if col in feature_matrix.columns]
    if trend_cols:
        manual_feature_sets["Trend Indicators"] = trend_cols
    
    # Momentum indicators
    momentum_indicators = [
        'rsi', 'momentum', 'stoch_k', 'stoch_d', 'roc'
    ]
    momentum_cols = [col for col in momentum_indicators if col in feature_matrix.columns]
    if momentum_cols:
        manual_feature_sets["Momentum Indicators"] = momentum_cols
    
    # Return-based indicators
    return_indicators = [
        'return_1d', 'return_3d'
    ]
    return_cols = [col for col in return_indicators if col in feature_matrix.columns]
    if return_cols:
        manual_feature_sets["Return Indicators"] = return_cols
    
    # All available indicators
    manual_feature_sets["All Indicators"] = feature_matrix.columns.tolist()
    
    # Top 5 indicators (based on common usage)
    top5_indicators = [
        'rsi', 'macd', 'adx', 'atr', 'cci'
    ]
    top5_cols = [col for col in top5_indicators if col in feature_matrix.columns]
    if len(top5_cols) >= 3:  # Only include if at least 3 indicators are available
        manual_feature_sets["Top 5 Common"] = top5_cols
    
    # If feature_selection_strategy is "all" or "manual", return manual feature sets
    if feature_selection_strategy in ["all", "manual"]:
        return manual_feature_sets
    
    # If feature_selection_strategy is "permutation", add permutation-based feature sets
    if feature_selection_strategy in ["all", "permutation"] and target is not None:
        from sklearn.inspection import permutation_importance
        
        print("🔁 Running permutation-based feature importance...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(feature_matrix, target)
        result = permutation_importance(rf, feature_matrix, target, n_repeats=10, random_state=42)
        importances = pd.Series(result.importances_mean, index=feature_matrix.columns).sort_values(ascending=False)
        
        # Add top 5 and top 10 permutation-based feature sets
        manual_feature_sets["Top 5 Permutation"] = importances.head(5).index.tolist()
        manual_feature_sets["Top 10 Permutation"] = importances.head(10).index.tolist()
    
    # If feature_selection_strategy is "shap", add SHAP-based feature sets
    if feature_selection_strategy in ["all", "shap"] and target is not None:
        print("📊 Running SHAP-based feature importance...")
        
        # Train a RandomForest model first
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(feature_matrix, target)
        
        try:
            # Try to use SHAP
            from shap_utils import compute_shap_feature_importance
            shap_importance = compute_shap_feature_importance(rf, feature_matrix)
            
            # Add top 5 and top 10 SHAP-based feature sets
            manual_feature_sets["Top 5 SHAP"] = shap_importance.head(5).index.tolist()
            manual_feature_sets["Top 10 SHAP"] = shap_importance.head(10).index.tolist()
        except Exception as e:
            print(f"⚠️ Error computing SHAP values: {str(e)}")
            print("Falling back to RandomForest feature importance")
            
            # Use RandomForest feature importance instead
            importances = pd.Series(rf.feature_importances_, index=feature_matrix.columns).sort_values(ascending=False)
            
            # Add top 5 and top 10 feature importance-based feature sets
            manual_feature_sets["Top 5 RF"] = importances.head(5).index.tolist()
            manual_feature_sets["Top 10 RF"] = importances.head(10).index.tolist()
    
    return manual_feature_sets

def fix_notebook_code():
    """
    Print the fixed code to use in the notebook.
    """
    fixed_code = """
# --- Evaluate feature sets and save results ---
from feature_set_selection_fix import evaluate_and_save_feature_sets, get_multiple_feature_sets

# Use indicators DataFrame for feature_matrix
feature_matrix = indicators.copy()

# Get multiple feature sets for evaluation
selected_feature_sets = get_multiple_feature_sets(
    feature_matrix=feature_matrix,
    target=target,
    feature_selection_strategy="all"  # Options: "all", "manual", "permutation", "shap"
)

print(f"✅ Generated {len(selected_feature_sets)} feature sets for evaluation")
for set_name, features in selected_feature_sets.items():
    print(f"  - {set_name}: {len(features)} features")

# Evaluate feature sets and save results
results_df, best_features = evaluate_and_save_feature_sets(
    feature_matrix=feature_matrix,
    target=target,
    selected_feature_sets=selected_feature_sets,
    lookback_window=lookback_window,
    symbol=symbol_to_predict
)

print("✅ Feature set evaluation complete")
print(f"Best feature set: {best_features}")
"""
    print("Copy and paste this code into your notebook:")
    print(fixed_code)

if __name__ == "__main__":
    fix_notebook_code()
