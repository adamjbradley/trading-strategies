import numpy as np
import pandas as pd
import copy
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

def compute_permutation_importance(model, X_val, y_val, feature_names, threshold=0.5):
    base_preds = (model.predict(X_val) > threshold).astype(int).flatten()
    min_len = min(len(y_val), len(base_preds))
    y_val_aligned = y_val[:min_len]
    base_preds_aligned = base_preds[:min_len]
    base_acc = accuracy_score(y_val_aligned, base_preds_aligned)
    importances = []
    for i in range(X_val.shape[2]):
        X_permuted = copy.deepcopy(X_val)
        np.random.shuffle(X_permuted[:, :, i])
        perm_preds = (model.predict(X_permuted) > threshold).astype(int).flatten()
        min_len_perm = min(len(y_val), len(perm_preds))
        y_val_perm = y_val[:min_len_perm]
        perm_preds_aligned = perm_preds[:min_len_perm]
        perm_acc = accuracy_score(y_val_perm, perm_preds_aligned)
        importance = base_acc - perm_acc
        importances.append((feature_names[i], importance))
    return sorted(importances, key=lambda x: x[1], reverse=True)

def compare_feature_sets(feature_matrix, y, feature_sets, lookback_window=20, model_fn=None, epochs=10, batch_size=32):
    results = []
    for set_name, feature_list in feature_sets.items():
        feature_list = [f for f in feature_list if f in feature_matrix.columns]
        if not feature_list:
            print(f"Skipping feature set '{set_name}': no valid features present in feature_matrix.")
            continue
        print(f"Evaluating feature set: {set_name} with features: {feature_list}")
        X_set = feature_matrix[feature_list].dropna().values
        X_scaled = StandardScaler().fit_transform(X_set)
        X_seq = np.array([X_scaled[i-lookback_window:i] for i in range(lookback_window, len(X_scaled))])
        y_seq = y[-len(X_seq):]
        split = int(len(X_seq) * 0.8)
        X_train, X_test = X_seq[:split], X_seq[split:]
        y_train, y_test = y_seq[:split], y_seq[split:]
        if model_fn is not None:
            model, _ = model_fn(X_train, y_train, X_test, y_test, X_train.shape[1:], epochs=epochs, batch_size=batch_size)
            acc = accuracy_score(y_test, (model.predict(X_test) > 0.5).astype(int).flatten()[:len(y_test)])
        else:
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            X_test_flat = X_test.reshape(X_test.shape[0], -1)
            rf.fit(X_train_flat, y_train)
            y_pred = rf.predict(X_test_flat)
            min_len = min(len(y_test), len(y_pred))
            y_test_aligned = y_test[:min_len]
            y_pred_aligned = y_pred[:min_len]
            acc = accuracy_score(y_test_aligned, y_pred_aligned)
        results.append({"Feature Set": set_name, "Accuracy": acc, "Features": feature_list})
    results_df = pd.DataFrame(results)
    return results_df

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
