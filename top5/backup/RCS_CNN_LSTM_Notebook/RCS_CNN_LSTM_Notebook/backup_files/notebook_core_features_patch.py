"""
Notebook Core Features Patch

This module provides a patch for the core_features variable used in the RCS_CNN_LSTM notebook.
"""

from core_features import get_core_features

# Define the core_features variable for use in the notebook
core_features = get_core_features()

def get_filtered_features(data_columns):
    """
    Returns a filtered list of core features that exist in the provided data columns.
    
    Parameters:
    -----------
    data_columns : list or pandas.Index
        The columns available in the data DataFrame
        
    Returns:
    --------
    list
        Filtered list of core feature names that exist in data_columns
    """
    return [f for f in core_features if f in data_columns]
