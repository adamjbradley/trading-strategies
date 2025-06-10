"""
Notebook Combined Fix

This script provides a combined fix for the notebook, including both the fixed notebook cell
and the target variable fix.
"""

from notebook_cell_fixed import get_fixed_notebook_cell_code
from notebook_target_fix import get_target_fix_cell

def get_combined_fix():
    """
    Get a combined fix for the notebook, including both the fixed notebook cell
    and the target variable fix.
    
    Returns:
    --------
    dict
        Dictionary containing the fixed notebook cell and the target fix
    """
    return {
        "target_fix": get_target_fix_cell(),
        "fixed_notebook_cell": get_fixed_notebook_cell_code()
    }

def print_combined_fix():
    """
    Print the combined fix for the notebook.
    """
    combined_fix = get_combined_fix()
    
    print("=" * 80)
    print("STEP 1: First, add this cell to fix the target variable issue:")
    print("=" * 80)
    print(combined_fix["target_fix"])
    
    print("\n\n")
    print("=" * 80)
    print("STEP 2: Then, add this cell to run the fixed notebook code:")
    print("=" * 80)
    print(combined_fix["fixed_notebook_cell"])

if __name__ == "__main__":
    print_combined_fix()
