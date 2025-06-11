"""
Notebook Target Fix

This script provides a fix for the target variable issue in the notebook.
"""

def get_target_fix_cell():
    """
    Get a notebook cell that fixes the target variable issue.
    
    Returns:
    --------
    str
        Notebook cell code that fixes the target variable issue
    """
    code = """
# --- Fix for target variable ---
# Create target variable from price data
if 'target' not in locals() or target is None:
    print("Creating target variable...")
    
    # Check if we have a symbol_to_predict variable
    if 'symbol_to_predict' in locals():
        symbol = symbol_to_predict
    else:
        # Default to EURUSD if no symbol is defined
        symbol = "EURUSD"
        print(f"No symbol_to_predict found, using default: {symbol}")
    
    # Try different approaches to create the target
    if 'data' in locals() and data is not None:
        if 'target' in data.columns:
            # Use existing target column if available
            target = data['target']
            print("Using existing target column from data DataFrame")
        elif symbol in data.columns:
            # Create target from symbol column
            target = (data[symbol].shift(-1) > data[symbol]).astype(int)
            print(f"Created target using {symbol} column")
        elif 'close' in data.columns:
            # Create target from close column
            target = (data['close'].shift(-1) > data['close']).astype(int)
            print("Created target using close column")
        else:
            # Try to find a price column
            price_cols = [col for col in data.columns if 'close' in str(col).lower() or 'price' in str(col).lower()]
            
            if price_cols:
                # Use the first price column found
                price_col = price_cols[0]
                target = (data[price_col].shift(-1) > data[price_col]).astype(int)
                print(f"Created target using {price_col} column")
            else:
                # If no price column is found, create a random target (50/50 split)
                print("⚠️ No price column found, creating random target")
                import numpy as np
                np.random.seed(42)  # For reproducibility
                target = pd.Series(np.random.randint(0, 2, size=len(data)), index=data.index)
                print("⚠️ Created random target as fallback")
    elif 'features' in locals() and features is not None:
        # Try to create target from features DataFrame
        if 'close' in features.columns:
            target = (features['close'].shift(-1) > features['close']).astype(int)
            print("Created target using close column from features DataFrame")
        else:
            # Try to find a price column
            price_cols = [col for col in features.columns if 'close' in str(col).lower() or 'price' in str(col).lower()]
            
            if price_cols:
                # Use the first price column found
                price_col = price_cols[0]
                target = (features[price_col].shift(-1) > features[price_col]).astype(int)
                print(f"Created target using {price_col} column from features DataFrame")
            else:
                # If no price column is found, create a random target (50/50 split)
                print("⚠️ No price column found in features, creating random target")
                import numpy as np
                np.random.seed(42)  # For reproducibility
                target = pd.Series(np.random.randint(0, 2, size=len(features)), index=features.index)
                print("⚠️ Created random target as fallback")
    else:
        # If no data or features are available, create a dummy target
        print("⚠️ No data or features found, creating dummy target")
        import numpy as np
        import pandas as pd
        target = pd.Series(np.ones(100))  # Dummy target
        print("⚠️ Created dummy target as fallback")
    
    # Drop NaN values
    target = target.dropna()
    print(f"Target shape: {target.shape}")
    
    # Add target to data if it exists
    if 'data' in locals() and data is not None and 'target' not in data.columns:
        data['target'] = target
        print("Added target column to data DataFrame")

# Now you can use the target variable in your code
lookback = 20
X = np.array([features_scaled[i-lookback:i] for i in range(lookback, len(features_scaled))])

# Make sure target is aligned with features
if hasattr(target, 'loc') and hasattr(features, 'index'):
    aligned_target = target.loc[features.index].dropna()
    y = aligned_target.values[lookback:]
else:
    # Fallback if target is not a Series or features doesn't have an index
    y = target.values[lookback:] if hasattr(target, 'values') else target[lookback:]

print("X shape:", X.shape, "y shape:", y.shape)
"""
    return code

if __name__ == "__main__":
    # Get the target fix cell
    target_fix_cell = get_target_fix_cell()
    
    # Print the target fix cell
    print("Copy and paste this code into your notebook:")
    print(target_fix_cell)
