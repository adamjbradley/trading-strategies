"""
DataFrame Utilities

Helper functions for handling data type conversions and DataFrame operations.
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
        
    Examples:
    ---------
    >>> import pandas as pd
    >>> # Works with pandas Series
    >>> series = pd.Series([1, 2, 3], name='test')
    >>> df = safe_to_dataframe(series)
    
    >>> # Works with lists
    >>> list_data = ['feature1', 'feature2', 'feature3']
    >>> df = safe_to_dataframe(list_data)
    
    >>> # Works with DataFrames (returns as-is)
    >>> existing_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    >>> df = safe_to_dataframe(existing_df)
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


def safe_save_to_csv(data, filepath, default_column_name='Features', **kwargs):
    """
    Safely convert data to DataFrame and save to CSV.
    
    Parameters:
    -----------
    data : pandas.Series, pandas.DataFrame, list, or any
        Data to convert and save
    filepath : str
        Path to save the CSV file
    default_column_name : str, default='Features'
        Column name to use when creating DataFrame from non-pandas objects
    **kwargs : additional arguments to pass to pandas.DataFrame.to_csv()
        
    Returns:
    --------
    str
        Path to the saved file
    """
    df = safe_to_dataframe(data, default_column_name)
    df.to_csv(filepath, **kwargs)
    return filepath


# Example usage that replaces the problematic pattern:
def example_usage():
    """
    Example showing how to replace the problematic .to_frame() pattern
    """
    # Instead of this (which fails if best_row is a list):
    # best_row.to_frame().T.to_csv('output.csv', index=False)
    
    # Use this pattern:
    # safe_save_to_csv(best_row, 'output.csv', index=False)
    
    # Or this pattern:
    # df = safe_to_dataframe(best_row)
    # df.to_csv('output.csv', index=False)
    
    pass


if __name__ == "__main__":
    # Test the functions
    import pandas as pd
    
    # Test with Series
    series = pd.Series([1, 2, 3], name='test')
    df1 = safe_to_dataframe(series)
    print("Series test:", df1.shape)
    
    # Test with list
    list_data = ['feature1', 'feature2', 'feature3']
    df2 = safe_to_dataframe(list_data)
    print("List test:", df2.shape)
    
    # Test with DataFrame
    existing_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    df3 = safe_to_dataframe(existing_df)
    print("DataFrame test:", df3.shape)
    
    print("All tests passed!")