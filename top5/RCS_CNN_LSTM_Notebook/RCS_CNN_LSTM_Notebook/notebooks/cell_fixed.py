"""
Notebook Cell Fixed

This script provides a fixed notebook cell that can be copied into the notebook.
"""

def get_fixed_notebook_cell():
    """
    Get a fixed notebook cell that can be copied into the notebook.
    
    Returns:
    --------
    str
        Fixed notebook cell code
    """
    code = get_fixed_notebook_cell_code()
    return code

def get_fixed_notebook_cell_code():
    """
    Get a fixed notebook cell that can be copied into the notebook.
    
    Returns:
    --------
    str
        Fixed notebook cell code
    """
    code = """
# --- Fix the model_utils module and reload modules ---
from fix_notebook import fix_model_utils, fix_model_builder_func
from src.utils.reload_modules import reload_modules

# Reload modules to ensure we have the latest versions
reload_results = reload_modules()
print("Module reload results:", reload_results)

# Patch the model_utils module
fix_model_utils()

# Define a fixed model builder function
def model_builder_func(X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    # Print diagnostic information about input shapes
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    if X_val is not None:
        print(f"X_val shape: {X_val.shape}")
    if y_val is not None:
        print(f"y_val shape: {y_val.shape}")
    
    return fix_model_builder_func(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=epochs,
        batch_size=batch_size
    )

# Import necessary modules
from src.models.training import train_model_with_best_features, evaluate_model, train_model_with_random_features, generate_feature_sets_from_random

# Ensure we have the right data format
if 'data' not in locals() or data is None:
    try:
        # Try to use indicators if available
        data = indicators
        print("Using 'indicators' DataFrame for model training")
    except NameError:
        # If indicators is not defined, try to use features
        try:
            data = features
            print("Using 'features' DataFrame for model training")
        except NameError:
            # If neither is defined, raise an error
            raise ValueError("No data found for model training. Please define 'data', 'indicators', or 'features'")

# Train the model with the best features
model, feature_names, X_train, y_train, X_val, y_val, X_test, y_test = train_model_with_best_features(
    symbol=symbol_to_predict,
    data=data,
    model_builder_func=model_builder_func,
    use_saved_features=True,  # Use the features we just saved
    epochs=50,
    batch_size=32
)

# Evaluate the model
metrics = evaluate_model(model, X_test, y_test)

# Print metrics
print("\\nModel evaluation metrics:")
for metric, value in metrics.items():
    print(f"  - {metric}: {value:.4f}")

# Save the model
model_path = f"{symbol_to_predict}_CNN_LSTM.h5"
model.save(model_path)
print(f"\\nModel saved to {model_path}")

# --- Train with random features and generate feature sets ---
# This code trains models with random features and generates feature sets

print("\n\n--- Training model with random features for comparison ---")
from src.models.training import train_model_with_random_features, generate_feature_sets_from_random

# Option 1: Train a single model with random features
random_model, random_features, X_train_r, y_train_r, X_val_r, y_val_r, X_test_r, y_test_r = train_model_with_random_features(
    symbol=symbol_to_predict,
    data=data,
    model_builder_func=model_builder_func,
    n_features=15,  # Number of random features to select
    random_seed=42,  # For reproducibility
    epochs=50,
    batch_size=32
)

# Evaluate the random model
random_metrics = evaluate_model(random_model, X_test_r, y_test_r)

# Print metrics
print("\nRandom model evaluation metrics:")
for metric, value in random_metrics.items():
    print(f"  - {metric}: {value:.4f}")

# Compare with the best features model
print("\nComparison between best features and random features:")
for metric in metrics.keys():
    best_value = metrics[metric]
    random_value = random_metrics[metric]
    diff = best_value - random_value
    print(f"  - {metric}: Best={best_value:.4f}, Random={random_value:.4f}, Diff={diff:.4f}")

# Save the random model
random_model_path = f"{symbol_to_predict}_CNN_LSTM_Random.h5"
random_model.save(random_model_path)
print(f"\nRandom model saved to {random_model_path}")

# Option 2: Generate feature sets from multiple random trials
print("\n\n--- Generating feature sets from random trials ---")
best_random_model, best_random_features, feature_sets, trial_results = generate_feature_sets_from_random(
    symbol=symbol_to_predict,
    data=data,
    model_builder_func=model_builder_func,
    n_random_trials=5,           # Number of random trials to run
    features_per_trial=15,       # Number of features per trial
    n_top_features=10,           # Number of top features to select
    n_feature_sets=3,            # Number of feature sets to generate
    random_seed=42,              # Base random seed
    epochs=20,                   # Fewer epochs for faster training
    batch_size=32
)

# Print the generated feature sets
print("\nGenerated feature sets:")
for i, feature_set in enumerate(feature_sets):
    print(f"Feature Set {i+1}: {feature_set}")

# Save the best random model
best_random_model_path = f"{symbol_to_predict}_CNN_LSTM_Best_Random.h5"
best_random_model.save(best_random_model_path)
print(f"\nBest random model saved to {best_random_model_path}")

# Print summary of trials
print("\nRandom trials summary:")
for trial in trial_results:
    print(f"Trial {trial['trial']}: Accuracy = {trial['metrics']['accuracy']:.4f}, F1 = {trial['metrics']['f1_score']:.4f}")
"""
    return code

if __name__ == "__main__":
    # Get the fixed notebook cell
    fixed_cell = get_fixed_notebook_cell()
    
    # Print the fixed notebook cell
    print("Copy and paste this code into your notebook:")
    print(fixed_cell)
