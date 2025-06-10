# --- Use only aligned feature_matrix and target for modeling and feature selection ---

import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

feature_selection_strategy = "shap"  # Options: "manual", "permutation", "shap"
lookback_window = 20
symbol_to_predict = "EURUSD"

# Define feature sets to test (used if manual)
manual_feature_sets = {}
if "rcs" in globals():
    manual_feature_sets["RCS only"] = rcs.columns.tolist()
    manual_feature_sets["RCS + RSI + MACD"] = rcs.columns.tolist() + ['rsi', 'macd']
manual_feature_sets["Indicators only"] = ['rsi', 'macd', 'momentum', 'cci']
manual_feature_sets["All features"] = indicators.columns.tolist()

# Align feature_matrix and target
common_index = indicators.index.intersection(target.index)
feature_matrix = indicators.loc[common_index]
target = target.loc[common_index]

print("feature_matrix shape:", feature_matrix.shape)
print("target shape:", target.shape)

selected_feature_sets = {}

if feature_selection_strategy == "manual":
    selected_feature_sets = manual_feature_sets

elif feature_selection_strategy == "permutation":
    print("🔁 Running permutation-based feature importance...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(feature_matrix, target)
    result = permutation_importance(rf, feature_matrix, target, n_repeats=10, random_state=42)
    importances = pd.Series(result.importances_mean, index=feature_matrix.columns).sort_values(ascending=False)
    top_feats = importances.head(10).index.tolist()
    selected_feature_sets = {"Top 10 Permutation": top_feats}

elif feature_selection_strategy == "shap":
    print("📊 Running SHAP-based feature importance...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(feature_matrix, target)
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(feature_matrix)
    shap_sum = np.abs(shap_values[1]).mean(axis=0)
    shap_importance = pd.Series(shap_sum, index=feature_matrix.columns).sort_values(ascending=False)
    top_feats = shap_importance.head(10).index.tolist()
    selected_feature_sets = {"Top 10 SHAP": top_feats}

print("✅ Selected feature sets:")
print(selected_feature_sets)
