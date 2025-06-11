"""
Shape Mismatch Example

This script demonstrates how to handle mismatches between model input shapes and data shapes
using the input_shape_handler.py module.
"""

# --- Copy and paste this into your notebook ---

# Import the input shape handler functions
from input_shape_handler import compute_permutation_importance_with_shape_handling

# Verify feature alignment
print("feature_names:", feature_names)
print("X_test shape:", X_test.shape)
print("Model input shape:", model.input_shape)

# Compute feature importance with shape handling
importances = compute_permutation_importance_with_shape_handling(
    model=model,
    X_val=X_test,
    y_val=y_test,
    feature_names=feature_names,
    n_repeats=3,
    verbose=True
)

# Convert to DataFrame and plot
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

importance_df = pd.DataFrame(importances, columns=["Feature", "Importance"])

plt.figure(figsize=(10,5))
sns.barplot(data=importance_df, x="Importance", y="Feature")
plt.title(f"Feature Importance via Permutation: {symbol}")
plt.grid(True)
plt.show()

print("✅ Shape mismatch example is ready to use")
print("Copy and paste this into your notebook")
