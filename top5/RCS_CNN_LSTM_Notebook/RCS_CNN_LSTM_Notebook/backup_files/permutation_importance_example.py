"""
Permutation Importance Example

This script demonstrates how to import and use the compute_permutation_importance function
directly in the notebook.
"""

# --- Copy and paste this into your notebook ---

# Import the compute_permutation_importance function
from feature_importance_utils import compute_permutation_importance

# Verify feature alignment
print("feature_names:", feature_names)
print("X_test shape:", X_test.shape)
assert X_test.shape[2] == len(feature_names), "Mismatch between X_test features and feature_names"

# Compute feature importance
importances = compute_permutation_importance(model, X_test, y_test, feature_names)
importance_df = pd.DataFrame(importances, columns=["Feature", "Importance"])

# Plot feature importance
plt.figure(figsize=(10,5))
sns.barplot(data=importance_df, x="Importance", y="Feature")
plt.title(f"Feature Importance via Permutation: {symbol}")
plt.grid(True)
plt.show()

print("✅ Permutation importance example is ready to use")
print("Copy and paste this into your notebook")
