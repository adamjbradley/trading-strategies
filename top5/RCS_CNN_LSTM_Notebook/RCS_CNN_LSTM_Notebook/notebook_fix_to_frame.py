"""
Notebook Fix for .to_frame() AttributeError

This script provides a fix for the common AttributeError that occurs when
calling .to_frame() on a list object instead of a pandas Series.

Usage in notebook cells:
1. Import this fix at the beginning of your notebook
2. Use safe_to_dataframe() instead of .to_frame()
"""

import pandas as pd


def safe_to_dataframe(data, default_column_name='Features'):
    """
    Safely convert various data types to pandas DataFrame.
    
    This function handles the common issue where `.to_frame()` is called on
    objects that don't have this method (like lists).
    
    Parameters:
    -----------
    data : pandas.Series, pandas.DataFrame, list, or any
        Data to convert to DataFrame
    default_column_name : str, default='Features'
        Column name to use when creating DataFrame from non-pandas objects
        
    Returns:
    --------
    pandas.DataFrame
        Converted DataFrame
    """
    if isinstance(data, pd.Series):
        return data.to_frame().T
    elif isinstance(data, pd.DataFrame):
        return data
    elif isinstance(data, list):
        # Create a DataFrame with the specified column containing the list as a string
        return pd.DataFrame({default_column_name: [str(data)]})
    else:
        # Try to convert to string and save
        try:
            return pd.DataFrame({default_column_name: [str(data)]})
        except Exception as e:
            raise ValueError(f"Cannot convert {type(data)} to DataFrame: {e}")


# For use in notebook cells - replace the problematic pattern:
# OLD (problematic): best_row.to_frame().T.to_csv(f'best_feature_set_{symbol}.csv', index=False)
# NEW (safe): safe_to_dataframe(best_row).to_csv(f'best_feature_set_{symbol}.csv', index=False)

print("✅ Notebook .to_frame() fix loaded. Use safe_to_dataframe() instead of .to_frame()")