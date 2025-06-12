#!/usr/bin/env python3
"""
Comprehensive Hyperparameter Analysis for Trading Strategy Optimization
Analyzes all optimization results to identify best performing parameter ranges and patterns.
"""

import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

def load_all_results(results_dir):
    """Load all optimization results from JSON files"""
    results = []
    results_path = Path(results_dir)
    
    # Load best_params files
    for json_file in results_path.glob("best_params_*.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                data['source_file'] = json_file.name
                results.append(data)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    # Load all_trials files for more comprehensive data
    trial_results = []
    for json_file in results_path.glob("all_trials_*.json"):
        try:
            with open(json_file, 'r') as f:
                trials = json.load(f)
                for trial in trials:
                    if 'value' in trial and trial['value'] > -1.0:  # Filter out failed trials
                        trial_data = {
                            'objective_value': trial['value'],
                            'source_file': json_file.name,
                            **trial['params']
                        }
                        if 'user_attrs' in trial:
                            trial_data.update(trial['user_attrs'])
                        trial_results.append(trial_data)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    return results, trial_results

def analyze_parameter_performance(results, trial_results):
    """Analyze parameter performance across all results"""
    
    # Combine best results and trial results
    all_data = []
    
    # Add best results
    for result in results:
        if result['objective_value'] > 0:  # Only successful runs
            data_point = {
                'objective_value': result['objective_value'],
                'mean_accuracy': result.get('mean_accuracy', 0),
                'mean_sharpe': result.get('mean_sharpe', 0),
                'symbol': result.get('symbol', 'UNKNOWN'),
                'source_type': 'best_params',
                **result['best_params']
            }
            all_data.append(data_point)
    
    # Add trial results
    for trial in trial_results:
        if trial['objective_value'] > 0:  # Only successful runs
            trial['source_type'] = 'trial'
            all_data.append(trial)
    
    df = pd.DataFrame(all_data)
    
    if df.empty:
        print("No valid data found!")
        return None
    
    return df

def categorize_performance(df):
    """Categorize results by performance levels"""
    df['performance_category'] = pd.cut(
        df['objective_value'], 
        bins=[0, 0.5, 0.75, 0.85, 1.0], 
        labels=['Poor', 'Good', 'Very Good', 'Excellent'],
        include_lowest=True
    )
    return df

def analyze_parameter_ranges(df, param_name):
    """Analyze optimal ranges for a specific parameter"""
    if param_name not in df.columns:
        return None
    
    analysis = {}
    
    # Overall statistics
    analysis['overall'] = {
        'min': df[param_name].min(),
        'max': df[param_name].max(),
        'mean': df[param_name].mean(),
        'median': df[param_name].median(),
        'std': df[param_name].std()
    }
    
    # Performance-based analysis
    for category in ['Very Good', 'Excellent']:
        if category in df['performance_category'].values:
            subset = df[df['performance_category'] == category]
            if not subset.empty:
                analysis[category.lower().replace(' ', '_')] = {
                    'min': subset[param_name].min(),
                    'max': subset[param_name].max(),
                    'mean': subset[param_name].mean(),
                    'median': subset[param_name].median(),
                    'std': subset[param_name].std(),
                    'count': len(subset)
                }
    
    # Top 10% analysis
    top_10_pct = df.nlargest(max(1, len(df) // 10), 'objective_value')
    analysis['top_10_percent'] = {
        'min': top_10_pct[param_name].min(),
        'max': top_10_pct[param_name].max(),
        'mean': top_10_pct[param_name].mean(),
        'median': top_10_pct[param_name].median(),
        'std': top_10_pct[param_name].std(),
        'count': len(top_10_pct),
        'values': top_10_pct[param_name].tolist()
    }
    
    return analysis

def main():
    """Main analysis function"""
    results_dir = "/mnt/c/Users/user/Projects/Finance/Strategies/trading-strategies/top5/optimization_results"
    
    print("Loading optimization results...")
    results, trial_results = load_all_results(results_dir)
    
    print(f"Loaded {len(results)} best parameter files and {len(trial_results)} trial results")
    
    # Analyze performance
    df = analyze_parameter_performance(results, trial_results)
    if df is None:
        return
    
    df = categorize_performance(df)
    
    print(f"\nTotal data points: {len(df)}")
    print(f"Performance distribution:")
    print(df['performance_category'].value_counts())
    
    # Define parameters to analyze
    key_parameters = [
        'lookback_window', 'max_features', 'conv1d_filters_1', 'conv1d_filters_2',
        'conv1d_kernel_size', 'lstm_units', 'dense_units', 'dropout_rate',
        'l1_reg', 'l2_reg', 'learning_rate', 'batch_size', 'epochs', 'patience'
    ]
    
    print("\n" + "="*80)
    print("COMPREHENSIVE HYPERPARAMETER ANALYSIS")
    print("="*80)
    
    # Analyze each parameter
    parameter_analysis = {}
    for param in key_parameters:
        if param in df.columns:
            analysis = analyze_parameter_ranges(df, param)
            if analysis:
                parameter_analysis[param] = analysis
    
    # Print detailed analysis
    for param, analysis in parameter_analysis.items():
        print(f"\n{param.upper().replace('_', ' ')} ANALYSIS:")
        print("-" * 50)
        
        print(f"Overall Range: {analysis['overall']['min']:.6f} - {analysis['overall']['max']:.6f}")
        print(f"Overall Mean: {analysis['overall']['mean']:.6f}")
        print(f"Overall Median: {analysis['overall']['median']:.6f}")
        
        if 'top_10_percent' in analysis:
            top_data = analysis['top_10_percent']
            print(f"\nTOP 10% PERFORMERS ({top_data['count']} models):")
            print(f"  Range: {top_data['min']:.6f} - {top_data['max']:.6f}")
            print(f"  Mean: {top_data['mean']:.6f}")
            print(f"  Median: {top_data['median']:.6f}")
            print(f"  Values: {[round(v, 6) for v in top_data['values']]}")
        
        if 'excellent' in analysis:
            exc_data = analysis['excellent']
            print(f"\nEXCELLENT PERFORMERS (obj > 0.85, {exc_data['count']} models):")
            print(f"  Range: {exc_data['min']:.6f} - {exc_data['max']:.6f}")
            print(f"  Mean: {exc_data['mean']:.6f}")
            print(f"  Median: {exc_data['median']:.6f}")
    
    # Identify best performing models
    print(f"\n{'='*80}")
    print("TOP PERFORMING MODELS (Objective > 0.85)")
    print("="*80)
    
    top_models = df[df['objective_value'] > 0.85].sort_values('objective_value', ascending=False)
    
    for idx, (_, model) in enumerate(top_models.iterrows()):
        print(f"\nRank {idx+1}: Objective = {model['objective_value']:.4f}")
        print(f"Symbol: {model.get('symbol', 'Unknown')}")
        print(f"Accuracy: {model.get('mean_accuracy', 0):.4f}, Sharpe: {model.get('mean_sharpe', 0):.4f}")
        
        param_summary = []
        for param in key_parameters:
            if param in model and pd.notna(model[param]):
                if isinstance(model[param], float):
                    param_summary.append(f"{param}={model[param]:.4f}")
                else:
                    param_summary.append(f"{param}={model[param]}")
        
        print("Parameters:")
        for i in range(0, len(param_summary), 3):
            print("  " + ", ".join(param_summary[i:i+3]))
    
    # Parameter correlation analysis
    print(f"\n{'='*80}")
    print("PARAMETER RECOMMENDATIONS BASED ON TOP PERFORMERS")
    print("="*80)
    
    top_performers = df[df['objective_value'] > 0.75]  # Use more data for better statistics
    
    recommendations = {}
    for param in key_parameters:
        if param in top_performers.columns and not top_performers[param].isna().all():
            param_data = top_performers[param].dropna()
            if len(param_data) > 0:
                recommendations[param] = {
                    'recommended_min': param_data.quantile(0.25),
                    'recommended_max': param_data.quantile(0.75),
                    'optimal_value': param_data.median(),
                    'sample_size': len(param_data)
                }
    
    for param, rec in recommendations.items():
        print(f"\n{param.upper().replace('_', ' ')}:")
        print(f"  Recommended Range: {rec['recommended_min']:.6f} - {rec['recommended_max']:.6f}")
        print(f"  Optimal Value: {rec['optimal_value']:.6f}")
        print(f"  Based on {rec['sample_size']} top-performing models")
    
    print(f"\n{'='*80}")
    print("SUMMARY AND INSIGHTS")
    print("="*80)
    
    # Generate insights
    insights = []
    
    # Best objective values
    best_obj = df['objective_value'].max()
    best_models_count = len(df[df['objective_value'] > 0.85])
    insights.append(f"Best objective value achieved: {best_obj:.4f}")
    insights.append(f"Number of models with objective > 0.85: {best_models_count}")
    
    # Parameter insights
    if 'dropout_rate' in recommendations:
        dropout_rec = recommendations['dropout_rate']
        insights.append(f"Optimal dropout rate range: {dropout_rec['recommended_min']:.3f} - {dropout_rec['recommended_max']:.3f}")
    
    if 'learning_rate' in recommendations:
        lr_rec = recommendations['learning_rate']
        insights.append(f"Optimal learning rate range: {lr_rec['recommended_min']:.6f} - {lr_rec['recommended_max']:.6f}")
    
    if 'lookback_window' in recommendations:
        lw_rec = recommendations['lookback_window']
        insights.append(f"Optimal lookback window range: {int(lw_rec['recommended_min'])} - {int(lw_rec['recommended_max'])} periods")
    
    for insight in insights:
        print(f"• {insight}")
    
    print(f"\nAnalysis complete! Data saved to DataFrame with {len(df)} total experiments.")
    return df, parameter_analysis, recommendations

if __name__ == "__main__":
    df, param_analysis, recommendations = main()