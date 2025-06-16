"""
Feature Set Utilities

This module provides utilities for loading and selecting feature sets.
"""

import os
import pandas as pd
import numpy as np

def load_best_feature_set(symbol):
    """
    Load the best feature set for a given symbol.
    
    Parameters:
    -----------
    symbol : str
        Trading symbol (e.g., 'EURUSD')
        
    Returns:
    --------
    list
        List of feature names
    """
    # Check if the best feature set file exists
    file_path = f"best_feature_set_{symbol}.csv"
    if not os.path.exists(file_path):
        print(f"⚠️ No best feature set file found for {symbol}")
        return []
    
    try:
        # Load the best feature set
        best_set_df = pd.read_csv(file_path)
        
        # Check if the Features column exists
        if 'Features' not in best_set_df.columns:
            print(f"⚠️ No Features column found in {file_path}")
            return []
        
        # Get the features
        features_str = best_set_df['Features'].iloc[0]
        
        # Convert string representation of list to actual list
        try:
            features = eval(features_str)
            
            # Ensure features is a list of strings
            if not isinstance(features, list):
                print(f"⚠️ Features in {file_path} is not a list")
                return []
            
            # Filter out any non-string or NaN features
            features = [f for f in features if isinstance(f, str) and not pd.isna(f)]
            
            print(f"✅ Loaded {len(features)} features from {file_path}")
            return features
        except:
            print(f"⚠️ Error parsing features in {file_path}")
            return []
    except Exception as e:
        print(f"⚠️ Error loading best feature set: {e}")
        return []

def get_feature_importance_ranking(symbol):
    """
    Get the feature importance ranking for a given symbol.
    
    Parameters:
    -----------
    symbol : str
        Trading symbol (e.g., 'EURUSD')
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with feature names and importance scores
    """
    # Check if the feature importance file exists
    file_path = f"feature_importance_{symbol}.csv"
    if not os.path.exists(file_path):
        print(f"⚠️ No feature importance file found for {symbol}")
        return pd.DataFrame()
    
    try:
        # Load the feature importance
        importance_df = pd.read_csv(file_path)
        
        # Check if the required columns exist
        if 'Feature' not in importance_df.columns or 'Importance' not in importance_df.columns:
            print(f"⚠️ Required columns not found in {file_path}")
            return pd.DataFrame()
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        print(f"✅ Loaded feature importance from {file_path}")
        return importance_df
    except Exception as e:
        print(f"⚠️ Error loading feature importance: {e}")
        return pd.DataFrame()

def select_top_n_features(importance_df, n=15):
    """
    Select the top N features from an importance DataFrame.
    
    Parameters:
    -----------
    importance_df : pandas.DataFrame
        DataFrame with feature names and importance scores
    n : int, default=15
        Number of top features to select
        
    Returns:
    --------
    list
        List of top feature names
    """
    if importance_df.empty:
        return []
    
    # Get the top N features
    top_n = importance_df.head(n)
    
    # Get the feature names
    features = top_n['Feature'].tolist()
    
    print(f"✅ Selected top {len(features)} features")
    return features

def get_default_features():
    """
    Get a default set of features with forex-specific enhancements.
    
    Returns:
    --------
    list
        List of default feature names including forex-specific features
    """
    return [
        # Technical Indicators
        'rsi', 'macd', 'momentum', 'cci', 'atr', 'adx', 'stoch_k', 'stoch_d',
        'roc', 'bbw',
        
        # Price Features
        'return_1d', 'return_3d', 'rolling_mean_5', 'rolling_std_5', 'momentum_slope',
        
        # Forex-Specific Trading Session Features
        'asian_session', 'london_session', 'ny_session', 'session_overlap', 
        'session_volatility_ratio',
        
        # Currency Strength & Correlation Features
        'eur_strength_proxy', 'eur_strength_trend', 'usd_strength_impact',
        
        # Advanced Volatility Features
        'volatility_regime', 'volatility_persistence',
        
        # Market Structure Features  
        'range_ratio', 'close_position', 'risk_sentiment',
        
        # Time-based Features
        'hour', 'is_friday', 'is_monday'
    ]

def filter_available_features(data, feature_list):
    """
    Filter a list of features to only include those available in the data.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing the data
    feature_list : list
        List of feature names to filter
        
    Returns:
    --------
    list
        List of available feature names
    """
    # Get the available columns
    available_columns = data.columns.tolist()
    
    # Filter the feature list
    available_features = [f for f in feature_list if f in available_columns]
    
    if not available_features:
        print("⚠️ None of the specified features are available in the data")
        print("Available columns:", available_columns)
        
        # Try to find some numeric columns to use as features
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclude common non-feature columns
        exclude_cols = ['timestamp', 'date', 'time', 'target', 'symbol']
        available_features = [col for col in numeric_cols if col not in exclude_cols]
        
        if available_features:
            print(f"✅ Using {len(available_features)} numeric columns as features")
        else:
            print("⚠️ No numeric columns found to use as features")
    else:
        print(f"✅ Found {len(available_features)} available features")
    
    return available_features

def save_feature_set_results(results_df, symbol, filename=None):
    """
    Save feature set results to a CSV file.
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame containing the results
    symbol : str
        Trading symbol (e.g., 'EURUSD')
    filename : str, optional
        Filename to save the results to. If None, a default name will be used.
        
    Returns:
    --------
    str
        Path to the saved file
    """
    if filename is None:
        filename = f"feature_set_results_{symbol}.csv"
    
    # Save the results
    results_df.to_csv(filename, index=False)
    
    print(f"✅ Saved feature set results to {filename}")
    
    return filename

def save_best_feature_set(best_row, symbol, filename=None):
    """
    Save the best feature set to a CSV file.
    
    Parameters:
    -----------
    best_row : pandas.Series, pandas.DataFrame, list, or any
        Series, DataFrame, list, or any object containing the best feature set
    symbol : str
        Trading symbol (e.g., 'EURUSD')
    filename : str, optional
        Filename to save the best feature set to. If None, a default name will be used.
        
    Returns:
    --------
    str
        Path to the saved file
    """
    if filename is None:
        filename = f"best_feature_set_{symbol}.csv"
    
    # Convert to DataFrame based on input type
    if isinstance(best_row, pd.Series):
        df = best_row.to_frame().T
    elif isinstance(best_row, pd.DataFrame):
        df = best_row
    elif isinstance(best_row, list):
        # Create a DataFrame with a Features column containing the list as a string
        df = pd.DataFrame({'Features': [str(best_row)]})
    else:
        # Try to convert to string and save
        try:
            df = pd.DataFrame({'Features': [str(best_row)]})
        except:
            raise ValueError(f"Cannot convert {type(best_row)} to DataFrame")
    
    # Save the best feature set
    df.to_csv(filename, index=False)
    
    print(f"✅ Saved best feature set to {filename}")
    
    return filename

def append_to_best_feature_set(feature_set_name, accuracy, features, symbol, filename=None, max_entries=50):
    """
    Append a new feature set to the best feature set CSV file if it's in the top N entries based on accuracy.
    
    Parameters:
    -----------
    feature_set_name : str
        Descriptive name for the feature set
    accuracy : float
        Accuracy value for the feature set
    features : list
        List of feature names
    symbol : str
        Trading symbol (e.g., 'EURUSD')
    filename : str, optional
        Filename to append to. If None, a default name will be used.
    max_entries : int, default=50
        Maximum number of entries to keep in the file, sorted by accuracy
        
    Returns:
    --------
    str
        Path to the updated file
    """
    if filename is None:
        filename = f"best_feature_set_{symbol}.csv"
    
    # Create a new row to append
    new_row = pd.DataFrame({
        'Feature Set': [feature_set_name],
        'Accuracy': [accuracy],
        'Features': [str(features)]
    })
    
    # Check if the file exists
    if os.path.exists(filename):
        # Read the existing file
        try:
            existing_df = pd.read_csv(filename)
            
            # Check if the file has the expected columns
            expected_columns = ['Feature Set', 'Accuracy', 'Features']
            if not all(col in existing_df.columns for col in expected_columns):
                # If the file doesn't have the expected columns, create a new file
                print(f"⚠️ Existing file {filename} doesn't have the expected columns. Creating a new file.")
                new_row.to_csv(filename, index=False)
                print(f"✅ Added new feature set to {filename}")
                return filename
            
            # Combine existing data with new row
            combined_df = pd.concat([existing_df, new_row], ignore_index=True)
            
            # Sort by accuracy in descending order
            combined_df = combined_df.sort_values('Accuracy', ascending=False)
            
            # Keep only the top max_entries
            if len(combined_df) > max_entries:
                combined_df = combined_df.head(max_entries)
                
                # Always add the entry, regardless of whether it makes it into the top max_entries
                if feature_set_name in combined_df['Feature Set'].values:
                    print(f"✅ New feature set with accuracy {accuracy:.4f} added to top {max_entries} in {filename}")
                else:
                    print(f"ℹ️ New feature set with accuracy {accuracy:.4f} did not make it into top {max_entries} in {filename}")
                    # But we'll still add it to a separate file to ensure it's always recorded
                    all_entries_filename = f"all_{filename}"
                    if os.path.exists(all_entries_filename):
                        all_entries_df = pd.read_csv(all_entries_filename)
                        all_entries_df = pd.concat([all_entries_df, new_row], ignore_index=True)
                    else:
                        all_entries_df = new_row
                    all_entries_df.to_csv(all_entries_filename, index=False)
                    print(f"✅ Added new feature set to {all_entries_filename} for complete record keeping")
            
            # Save the updated file
            combined_df.to_csv(filename, index=False)
            print(f"✅ Updated feature sets in {filename}")
            
        except Exception as e:
            print(f"⚠️ Error reading existing file {filename}: {e}. Creating a new file.")
            new_row.to_csv(filename, index=False)
            print(f"✅ Added new feature set to {filename}")
    else:
        # Create a new file
        new_row.to_csv(filename, index=False)
        print(f"✅ Created new file {filename} with feature set")
    
    return filename

print("✅ Feature set utilities loaded")
