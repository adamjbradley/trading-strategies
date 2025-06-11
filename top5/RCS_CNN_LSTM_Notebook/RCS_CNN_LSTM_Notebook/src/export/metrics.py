"""
Migrate Metrics

This script migrates metrics from older individual CSV files to the new consolidated format.
"""

import os
import glob
import pandas as pd
import re

def migrate_metrics_to_consolidated_file(models_dir="models", output_file=None):
    """
    Migrate metrics from older individual CSV files to a new consolidated CSV file.
    
    Parameters:
    -----------
    models_dir : str, default="models"
        Directory containing the model files and metrics CSV files
    output_file : str, optional
        Path to the output consolidated CSV file. If None, defaults to 'all_models_metrics.csv' in the models_dir
        
    Returns:
    --------
    str
        Path to the consolidated CSV file
    """
    if not os.path.exists(models_dir):
        print(f"⚠️ Models directory '{models_dir}' does not exist.")
        return None
    
    # Set default output file if not provided
    if output_file is None:
        output_file = os.path.join(models_dir, "all_models_metrics.csv")
    
    # Find all CSV files in the models directory that match the pattern *_metrics.csv
    csv_files = glob.glob(os.path.join(models_dir, "*_metrics.csv"))
    
    if not csv_files:
        print(f"⚠️ No metrics CSV files found in '{models_dir}'.")
        return None
    
    print(f"Found {len(csv_files)} metrics CSV files.")
    
    # Initialize an empty list to store all metrics
    all_metrics = []
    
    # Process each CSV file
    for csv_file in csv_files:
        try:
            # Extract model name and symbol from the filename
            filename = os.path.basename(csv_file)
            match = re.match(r"(.+)_CNN_LSTM_(\d{8}_\d{6})_metrics\.csv", filename)
            
            if match:
                symbol = match.group(1)
                timestamp = match.group(2)
                model_name = f"{symbol}_CNN_LSTM_{timestamp}"
            else:
                # Try another pattern
                match = re.match(r"(.+)_(\d{8}_\d{6})_metrics\.csv", filename)
                if match:
                    symbol = match.group(1)
                    timestamp = match.group(2)
                    model_name = f"{symbol}_{timestamp}"
                else:
                    # If we can't extract from filename, use the filename without extension
                    model_name = os.path.splitext(filename)[0].replace("_metrics", "")
                    symbol = "unknown"
                    timestamp = "unknown"
            
            # Read the CSV file
            df = pd.read_csv(csv_file)
            
            # If the DataFrame is empty, skip this file
            if df.empty:
                print(f"⚠️ Skipping empty file: {csv_file}")
                continue
            
            # If model_name and symbol are not in the DataFrame, add them
            if "model_name" not in df.columns:
                df["model_name"] = model_name
            if "symbol" not in df.columns:
                df["symbol"] = symbol
            if "timestamp" not in df.columns:
                df["timestamp"] = timestamp
            
            # Add to the list of all metrics
            all_metrics.append(df)
            
            print(f"✅ Processed: {csv_file}")
            
        except Exception as e:
            print(f"⚠️ Error processing {csv_file}: {e}")
    
    if not all_metrics:
        print("⚠️ No metrics data found in the CSV files.")
        return None
    
    # Concatenate all metrics
    consolidated_df = pd.concat(all_metrics, ignore_index=True)
    
    # Check if the output file already exists
    if os.path.exists(output_file):
        # Read the existing file
        existing_df = pd.read_csv(output_file)
        
        # Concatenate with the new data
        consolidated_df = pd.concat([existing_df, consolidated_df], ignore_index=True)
        
        # Remove duplicates based on model_name
        consolidated_df = consolidated_df.drop_duplicates(subset=["model_name"], keep="first")
        
        print(f"✅ Merged with existing file: {output_file}")
    
    # Save the consolidated DataFrame to the output file
    consolidated_df.to_csv(output_file, index=False)
    
    print(f"✅ Saved consolidated metrics to: {output_file}")
    print(f"Total models: {len(consolidated_df)}")
    
    return output_file

if __name__ == "__main__":
    # Example usage
    migrate_metrics_to_consolidated_file(models_dir="models")
