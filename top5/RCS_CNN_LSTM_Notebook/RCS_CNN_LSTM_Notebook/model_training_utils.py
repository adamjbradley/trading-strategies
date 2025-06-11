"""
Model Training Utilities

This module provides utilities for training models using the best feature sets.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
# Import the functions directly from the module
from src.features.selection import load_best_feature_set, get_feature_importance_ranking, get_default_features, select_top_n_features, filter_available_features, append_to_best_feature_set

def train_model_with_best_features(symbol, data, model_builder_func, n_features=15, use_saved_features=True, **kwargs):
    """
    Train a model using the best feature set.
    
    Parameters:
    -----------
    symbol : str
        Trading symbol (e.g., 'EURUSD')
    data : pandas.DataFrame
        DataFrame containing the data for training
    model_builder_func : function
        Function that builds and trains the model. Should accept X_train, y_train, X_val, y_val as arguments.
    n_features : int, default=15
        Number of top features to use if no saved feature set is found
    use_saved_features : bool, default=True
        If True, use the saved best feature set. If False, use feature importance to select features.
    **kwargs : dict
        Additional keyword arguments to pass to the model_builder_func
        
    Returns:
    --------
    tuple
        (model, feature_names, X_train, y_train, X_val, y_val, X_test, y_test)
    """
    print(f"🔍 Training model for {symbol} using best features")
    
    # Print available columns for debugging
    print("Available columns in DataFrame:")
    print(data.columns.tolist())
    
    # Check if target column exists, if not create it
    if 'target' not in data.columns:
        print("⚠️ Target column 'target' not found in data, creating it")
        
        # Handle different symbol formats (with or without =X suffix)
        base_symbol = symbol.replace('=X', '')
        
        # Try different possible column formats
        if (symbol, 'close') in data.columns:
            data['target'] = (data[(symbol, 'close')].shift(-1) > data[(symbol, 'close')]).astype(int)
            print(f"✅ Created target using ({symbol}, 'close')")
        elif (base_symbol, 'close') in data.columns:
            data['target'] = (data[(base_symbol, 'close')].shift(-1) > data[(base_symbol, 'close')]).astype(int)
            print(f"✅ Created target using ({base_symbol}, 'close')")
        elif symbol in data.columns:
            data['target'] = (data[symbol].shift(-1) > data[symbol]).astype(int)
            print(f"✅ Created target using {symbol}")
        elif base_symbol in data.columns:
            data['target'] = (data[base_symbol].shift(-1) > data[base_symbol]).astype(int)
            print(f"✅ Created target using {base_symbol}")
        else:
            # Try to find a price column
            price_cols = [col for col in data.columns if 'close' in str(col).lower() or 'price' in str(col).lower()]
            
            # Print available columns for debugging
            print(f"Looking for price columns among: {data.columns.tolist()}")
            print(f"Found potential price columns: {price_cols}")
            
            if price_cols:
                # Use the first price column found
                price_col = price_cols[0]
                data['target'] = (data[price_col].shift(-1) > data[price_col]).astype(int)
                print(f"✅ Created target using {price_col}")
            else:
                # If no price column is found, create a random target (50/50 split)
                print("⚠️ No price column found, creating random target")
                np.random.seed(42)  # For reproducibility
                data['target'] = np.random.randint(0, 2, size=len(data))
                print("⚠️ Created random target column as fallback")
    
    # Get the best feature set
    best_features = []
    
    if use_saved_features:
        # Try to load the best feature set from the saved file
        best_features = load_best_feature_set(symbol)
    
    # If no saved feature set is found or use_saved_features is False, use feature importance
    if not best_features:
        print(f"⚠️ No saved feature set found for {symbol}, using feature importance ranking")
        
        # Get the feature importance ranking
        importance_df = get_feature_importance_ranking(symbol)
        
        # If no feature importance ranking is found, use default features
        if importance_df.empty:
            print(f"⚠️ No feature importance ranking found for {symbol}, using default features")
            best_features = get_default_features()
        else:
            # Select the top N features
            best_features = select_top_n_features(importance_df, n=n_features)
    
    # Filter the features to only include those available in the data
    available_features = filter_available_features(data, best_features)
    
    if not available_features:
        # If no features are available, use all numeric columns except target
        print("⚠️ No valid features found, using all numeric columns")
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        available_features = [col for col in numeric_cols if col != 'target']
        
        if not available_features:
            raise ValueError(f"No numeric features found in data")
    
    print(f"✅ Using {len(available_features)} features for model training: {available_features}")
    
    # Check for NaN values in features and target
    feature_data = data[available_features]
    nan_counts = feature_data.isna().sum()
    
    if nan_counts.sum() > 0:
        print("⚠️ NaN values found in features:")
        for col, count in nan_counts[nan_counts > 0].items():
            print(f"  - {col}: {count} NaN values")
        
        print("Filling NaN values with appropriate methods...")
        # Fill NaN values with appropriate methods
        for col in feature_data.columns:
            if nan_counts[col] > 0:
                # Use forward fill first
                feature_data[col] = feature_data[col].ffill()
                # Then use backward fill for any remaining NaNs
                feature_data[col] = feature_data[col].bfill()
                # If still NaN (e.g., all NaN column), fill with 0
                feature_data[col] = feature_data[col].fillna(0)
    
    # Check for NaN values in target
    target_nan_count = data['target'].isna().sum()
    if target_nan_count > 0:
        print(f"⚠️ {target_nan_count} NaN values found in target, filling with forward fill")
        data['target'] = data['target'].ffill().bfill().fillna(0)
    
    # Drop any remaining rows with NaN values
    valid_rows = ~(feature_data.isna().any(axis=1) | data['target'].isna())
    if valid_rows.sum() < len(data):
        print(f"⚠️ Dropping {len(data) - valid_rows.sum()} rows with NaN values")
        feature_data = feature_data[valid_rows]
        target_data = data['target'][valid_rows]
    else:
        target_data = data['target']
    
    # Prepare the data for training
    X = feature_data.values
    y = target_data.values
    
    # Verify data shapes
    print(f"Feature data shape: {X.shape}")
    print(f"Target data shape: {y.shape}")
    
    # Split the data into train, validation, and test sets
    # Assuming a 70/15/15 split
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]
    
    # Reshape the data for CNN-LSTM model if needed
    # Assuming the model expects 3D input (samples, timesteps, features)
    if len(X_train.shape) == 2:
        # Reshape to (samples, 1, features)
        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
        X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    
    # Build and train the model
    model = model_builder_func(X_train, y_train, X_val, y_val, **kwargs)
    
    # Evaluate the model to get accuracy
    metrics = evaluate_model(model, X_test, y_test)
    accuracy = metrics['accuracy']
    
    # Create a descriptive name for the feature set
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    feature_set_name = f"Best Features {timestamp}"
    
    # Add the result to the best_feature_set CSV file
    append_to_best_feature_set(
        feature_set_name=feature_set_name,
        accuracy=accuracy,
        features=available_features,
        symbol=symbol
    )
    
    print(f"✅ Added best feature set with accuracy {accuracy:.4f} to best_feature_set_{symbol}.csv")
    
    return model, available_features, X_train, y_train, X_val, y_val, X_test, y_test

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test set.
    
    Parameters:
    -----------
    model : keras.Model
        Trained model
    X_test : numpy.ndarray
        Test features
    y_test : numpy.ndarray
        Test targets
        
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    """
    # Make predictions
    y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Return metrics
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    print(f"✅ Model evaluation metrics:")
    for metric, value in metrics.items():
        print(f"  - {metric}: {value:.4f}")
    
    return metrics

def train_model_with_random_features(symbol, data, model_builder_func, n_features=15, random_seed=42, **kwargs):
    """
    Train a model using randomly selected features and add the best result to the best_feature_set CSV file.
    
    Parameters:
    -----------
    symbol : str
        Trading symbol (e.g., 'EURUSD')
    data : pandas.DataFrame
        DataFrame containing the data for training
    model_builder_func : function
        Function that builds and trains the model. Should accept X_train, y_train, X_val, y_val as arguments.
    n_features : int, default=15
        Number of random features to select
    random_seed : int, default=42
        Random seed for reproducibility
    **kwargs : dict
        Additional keyword arguments to pass to the model_builder_func
        
    Returns:
    --------
    tuple
        (model, feature_names, X_train, y_train, X_val, y_val, X_test, y_test)
    """
    print(f"🔍 Training model for {symbol} using {n_features} random features")
    
    # Print available columns for debugging
    print("Available columns in DataFrame:")
    print(data.columns.tolist())
    
    # Check if target column exists, if not create it
    if 'target' not in data.columns:
        print("⚠️ Target column 'target' not found in data, creating it")
        
        # Handle different symbol formats (with or without =X suffix)
        base_symbol = symbol.replace('=X', '')
        
        # Try different possible column formats
        if (symbol, 'close') in data.columns:
            data['target'] = (data[(symbol, 'close')].shift(-1) > data[(symbol, 'close')]).astype(int)
            print(f"✅ Created target using ({symbol}, 'close')")
        elif (base_symbol, 'close') in data.columns:
            data['target'] = (data[(base_symbol, 'close')].shift(-1) > data[(base_symbol, 'close')]).astype(int)
            print(f"✅ Created target using ({base_symbol}, 'close')")
        elif symbol in data.columns:
            data['target'] = (data[symbol].shift(-1) > data[symbol]).astype(int)
            print(f"✅ Created target using {symbol}")
        elif base_symbol in data.columns:
            data['target'] = (data[base_symbol].shift(-1) > data[base_symbol]).astype(int)
            print(f"✅ Created target using {base_symbol}")
        else:
            # Try to find a price column
            price_cols = [col for col in data.columns if 'close' in str(col).lower() or 'price' in str(col).lower()]
            
            # Print available columns for debugging
            print(f"Looking for price columns among: {data.columns.tolist()}")
            print(f"Found potential price columns: {price_cols}")
            
            if price_cols:
                # Use the first price column found
                price_col = price_cols[0]
                data['target'] = (data[price_col].shift(-1) > data[price_col]).astype(int)
                print(f"✅ Created target using {price_col}")
            else:
                # If no price column is found, create a random target (50/50 split)
                print("⚠️ No price column found, creating random target")
                np.random.seed(random_seed)  # For reproducibility
                data['target'] = np.random.randint(0, 2, size=len(data))
                print("⚠️ Created random target column as fallback")
    
    # Get all numeric columns except target
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    available_features = [col for col in numeric_cols if col != 'target']
    
    if not available_features:
        raise ValueError(f"No numeric features found in data")
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Select random features
    if len(available_features) <= n_features:
        # If we have fewer features than requested, use all of them
        selected_features = available_features
        print(f"⚠️ Only {len(selected_features)} features available, using all of them")
    else:
        # Randomly select n_features
        selected_features = np.random.choice(available_features, size=n_features, replace=False).tolist()
        print(f"✅ Randomly selected {len(selected_features)} features")
    
    print(f"Selected features: {selected_features}")
    
    # Check for NaN values in features and target
    feature_data = data[selected_features]
    nan_counts = feature_data.isna().sum()
    
    if nan_counts.sum() > 0:
        print("⚠️ NaN values found in features:")
        for col, count in nan_counts[nan_counts > 0].items():
            print(f"  - {col}: {count} NaN values")
        
        print("Filling NaN values with appropriate methods...")
        # Fill NaN values with appropriate methods
        for col in feature_data.columns:
            if nan_counts[col] > 0:
                # Use forward fill first
                feature_data[col] = feature_data[col].ffill()
                # Then use backward fill for any remaining NaNs
                feature_data[col] = feature_data[col].bfill()
                # If still NaN (e.g., all NaN column), fill with 0
                feature_data[col] = feature_data[col].fillna(0)
    
    # Check for NaN values in target
    target_nan_count = data['target'].isna().sum()
    if target_nan_count > 0:
        print(f"⚠️ {target_nan_count} NaN values found in target, filling with forward fill")
        data['target'] = data['target'].ffill().bfill().fillna(0)
    
    # Drop any remaining rows with NaN values
    valid_rows = ~(feature_data.isna().any(axis=1) | data['target'].isna())
    if valid_rows.sum() < len(data):
        print(f"⚠️ Dropping {len(data) - valid_rows.sum()} rows with NaN values")
        feature_data = feature_data[valid_rows]
        target_data = data['target'][valid_rows]
    else:
        target_data = data['target']
    
    # Prepare the data for training
    X = feature_data.values
    y = target_data.values
    
    # Verify data shapes
    print(f"Feature data shape: {X.shape}")
    print(f"Target data shape: {y.shape}")
    
    # Split the data into train, validation, and test sets
    # Assuming a 70/15/15 split
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]
    
    # Reshape the data for CNN-LSTM model if needed
    # Assuming the model expects 3D input (samples, timesteps, features)
    if len(X_train.shape) == 2:
        # Reshape to (samples, 1, features)
        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
        X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    
    # Build and train the model
    model = model_builder_func(X_train, y_train, X_val, y_val, **kwargs)
    
    # Evaluate the model to get accuracy
    metrics = evaluate_model(model, X_test, y_test)
    accuracy = metrics['accuracy']
    
    # Create a descriptive name for the feature set
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    feature_set_name = f"Random Features {timestamp}"
    
    # Add the result to the best_feature_set CSV file
    # Ensure we always add the result, regardless of accuracy
    append_to_best_feature_set(
        feature_set_name=feature_set_name,
        accuracy=accuracy,
        features=selected_features,
        symbol=symbol,
        max_entries=100  # Increase max entries to ensure it's always added
    )
    
    print(f"✅ Added random feature set with accuracy {accuracy:.4f} to best_feature_set_{symbol}.csv")
    
    return model, selected_features, X_train, y_train, X_val, y_val, X_test, y_test

def generate_feature_sets_from_random(symbol, data, model_builder_func, n_random_trials=5, features_per_trial=15, 
                                      n_top_features=10, n_feature_sets=3, random_seed=42, **kwargs):
    """
    Generate feature sets by training multiple models with random features and selecting the best performing features.
    
    Parameters:
    -----------
    symbol : str
        Trading symbol (e.g., 'EURUSD')
    data : pandas.DataFrame
        DataFrame containing the data for training
    model_builder_func : function
        Function that builds and trains the model. Should accept X_train, y_train, X_val, y_val as arguments.
    n_random_trials : int, default=5
        Number of random feature trials to run
    features_per_trial : int, default=15
        Number of features to use in each random trial
    n_top_features : int, default=10
        Number of top features to select from all trials
    n_feature_sets : int, default=3
        Number of feature sets to generate from the top features
    random_seed : int, default=42
        Base random seed for reproducibility
    **kwargs : dict
        Additional keyword arguments to pass to the model_builder_func
        
    Returns:
    --------
    tuple
        (best_model, best_features, feature_sets, trial_results)
    """
    print(f"🔍 Generating feature sets from random trials for {symbol}")
    
    # Store results from each trial
    trial_results = []
    all_features = []
    feature_metrics = {}
    
    # Run multiple trials with random features
    for trial in range(n_random_trials):
        print(f"\n--- Random Trial {trial+1}/{n_random_trials} ---")
        
        # Use a different random seed for each trial
        trial_seed = random_seed + trial
        
        # Train model with random features
        model, features, X_train, y_train, X_val, y_val, X_test, y_test = train_model_with_random_features(
            symbol=symbol,
            data=data,
            model_builder_func=model_builder_func,
            n_features=features_per_trial,
            random_seed=trial_seed,
            **kwargs
        )
        
        # Evaluate the model
        metrics = evaluate_model(model, X_test, y_test)
        
        # Store results
        trial_result = {
            'trial': trial + 1,
            'features': features,
            'metrics': metrics,
            'model': model
        }
        trial_results.append(trial_result)
        
        # Add features to the overall list
        all_features.extend(features)
        
        # Update feature metrics
        for feature in features:
            if feature not in feature_metrics:
                feature_metrics[feature] = []
            
            # Store metrics for this feature
            feature_metrics[feature].append({
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score']
            })
    
    # Get unique features
    unique_features = list(set(all_features))
    print(f"\n✅ Found {len(unique_features)} unique features across all trials")
    
    # Calculate average metrics for each feature
    feature_scores = {}
    for feature, metrics_list in feature_metrics.items():
        # Calculate average metrics
        avg_accuracy = sum(m['accuracy'] for m in metrics_list) / len(metrics_list)
        avg_f1 = sum(m['f1_score'] for m in metrics_list) / len(metrics_list)
        
        # Calculate a combined score (you can adjust the weights)
        combined_score = 0.7 * avg_accuracy + 0.3 * avg_f1
        
        # Store the score
        feature_scores[feature] = combined_score
    
    # Sort features by score
    sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Select top N features
    top_features = [f[0] for f in sorted_features[:n_top_features]]
    print(f"Top {len(top_features)} features: {top_features}")
    
    # Generate feature sets from top features
    feature_sets = []
    np.random.seed(random_seed)
    
    # Find the best trial
    best_trial = max(trial_results, key=lambda x: x['metrics']['accuracy'])
    best_model = best_trial['model']
    best_features = best_trial['features']
    
    # Generate feature sets
    for i in range(n_feature_sets):
        # Determine set size (between 50% and 100% of top features)
        set_size = np.random.randint(max(3, n_top_features // 2), n_top_features + 1)
        
        # Select random features from top features
        feature_set = np.random.choice(top_features, size=set_size, replace=False).tolist()
        
        # Add to feature sets
        feature_sets.append(feature_set)
        
        # Save this feature set
        save_best_feature_set(
            feature_set,
            symbol=f"{symbol}_random_set_{i+1}",
            filename=f"best_feature_set_{symbol}_random_set_{i+1}.csv"
        )
        
        print(f"Feature Set {i+1}: {feature_set}")
    
    # Save the overall best feature set
    save_best_feature_set(
        top_features,
        symbol=f"{symbol}_top_random",
        filename=f"best_feature_set_{symbol}_top_random.csv"
    )
    
    # Add the best overall result to the main best_feature_set CSV file
    best_accuracy = best_trial['metrics']['accuracy']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    feature_set_name = f"Best Random Trial {timestamp}"
    
    # Add the best result to the best_feature_set CSV file
    append_to_best_feature_set(
        feature_set_name=feature_set_name,
        accuracy=best_accuracy,
        features=best_features,
        symbol=symbol
    )
    
    print(f"✅ Added best random trial with accuracy {best_accuracy:.4f} to best_feature_set_{symbol}.csv")
    
    return best_model, best_features, feature_sets, trial_results

print("✅ Model training utilities loaded")
