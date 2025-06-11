#!/usr/bin/env python3
"""
Script to execute the RCS_CNN_LSTM notebook
"""
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import sys
import os

def run_notebook(notebook_path, output_path=None):
    """Execute a Jupyter notebook and optionally save the result"""
    
    # Set working directory to notebook directory
    notebook_dir = os.path.dirname(os.path.abspath(notebook_path))
    os.chdir(notebook_dir)
    print(f"Working directory: {os.getcwd()}")
    
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    # Create executor
    ep = ExecutePreprocessor(timeout=1800, kernel_name='python3')  # 30 minute timeout
    
    try:
        # Execute the notebook
        print(f"Executing notebook: {notebook_path}")
        ep.preprocess(nb, {'metadata': {'path': notebook_dir}})
        print("✅ Notebook executed successfully!")
        
        # Save the executed notebook if output path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                nbformat.write(nb, f)
            print(f"📁 Executed notebook saved to: {output_path}")
            
    except Exception as e:
        print(f"❌ Error executing notebook: {e}")
        sys.exit(1)

if __name__ == "__main__":
    notebook_path = "./RCS_CNN_LSTM_Notebook/RCS_CNN_LSTM_Notebook/RCS_CNN_LSTM.ipynb"
    output_path = "./RCS_CNN_LSTM_Notebook/RCS_CNN_LSTM_Notebook/RCS_CNN_LSTM_executed.ipynb"
    
    run_notebook(notebook_path, output_path)