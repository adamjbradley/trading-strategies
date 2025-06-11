import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

def detect_feature_drift(df, time_col="time", window=100, threshold=0.1):
    # Compare distribution between early and recent window for each feature
    recent = df.sort_values(by=time_col).tail(window)
    early = df.sort_values(by=time_col).head(window)

    drift_scores = {}
    for col in df.columns:
        if col == time_col or df[col].nunique() <= 1:
            continue
        stat, pval = ks_2samp(early[col].dropna(), recent[col].dropna())
        drift_scores[col] = pval

    # Return features likely to have drifted (low p-value)
    drifted = [k for k, v in drift_scores.items() if v < threshold]
    return drifted

def remove_correlated_features(df, threshold=0.95):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    drop_cols = [col for col in upper.columns if any(upper[col] > threshold)]
    return df.drop(columns=drop_cols), drop_cols
