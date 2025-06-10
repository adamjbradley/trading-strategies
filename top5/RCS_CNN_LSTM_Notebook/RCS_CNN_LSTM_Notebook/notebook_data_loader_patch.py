# --- Use best combination of parameters from previous executions ---

import pandas as pd

symbol = "EURUSD"
best_set_path = f"best_feature_set_{symbol}.csv"
best_row = pd.read_csv(best_set_path)
best_features = eval(best_row["Features"].iloc[0])  # Assumes features are stored as a list in CSV

print("Best features:", best_features)

# Align features and target
features_sel = features[best_features]
common_index = features_sel.index.intersection(target.index)
features_sel_aligned = features_sel.loc[common_index]
target_aligned = target.loc[common_index]
aligned = features_sel_aligned.join(target_aligned.rename("target")).dropna()
features_final = aligned.drop(columns=["target"])
target_final = aligned["target"]

# Standardize and create rolling windows
from sklearn.preprocessing import StandardScaler
features_scaled = StandardScaler().fit_transform(features_final)
lookback = 20
X = create_rolling_windows(features_scaled, lookback)
y = target_final.values[lookback:]

# Train/test split, model training, and evaluation as before
X_train, X_test, y_train, y_test = train_test_split_rolling(X, y, test_size=0.2)
model, history = train_cnn_lstm_model(X_train, y_train, X_test, y_test, X_train.shape[1:], epochs=10, batch_size=32)
acc, report = evaluate_cnn_lstm_model(model, X_test, y_test)
print("Test accuracy:", acc)
print(report)
