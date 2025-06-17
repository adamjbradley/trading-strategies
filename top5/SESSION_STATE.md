# Session State File
*Generated: 2025-06-17*

## Project Overview
This is a comprehensive trading strategy project focused on CNN-LSTM models for forex trading. The project contains:

### Core Components
- **Main Directory**: `/mnt/c/Users/user/Projects/Finance/Strategies/trading-strategies/top5`
- **Primary Models**: CNN-LSTM hybrid models for forex pair prediction
- **Supported Pairs**: EURUSD, GBPUSD, AUDUSD, EURJPY, GBPJPY, USDCAD, USDJPY
- **Data Format**: HDF5 and Parquet files from MetaTrader

### Key Directories
- `data/`: Contains forex data files (metatrader_*.h5, metatrader_*.parquet)
- `exported_models/`: Trained models and metadata (.onnx, .json files)
- `optimization_results/`: Hyperparameter optimization results and checkpoints
- `RCS_CNN_LSTM_Notebook/`: Main development environment with structured codebase
- `backup/`: Backup files and previous versions

### Current Git Status
- **Branch**: main
- **Modified Files**: 2 Jupyter notebooks
- **Deleted Files**: 9 exported model files
- **Untracked Files**: 37+ metadata and optimization result files

### Active Components
1. **Training Infrastructure**: Hyperparameter optimization with Optuna
2. **Model Export**: ONNX format for deployment
3. **Data Processing**: Feature engineering and selection
4. **Backtesting**: Performance validation system

### Recent Activity
- Heavy optimization work with multiple currency pairs
- Model training with enhanced correlation features
- ONNX export pipeline development
- Comprehensive testing and validation

### Key Files to Remember
- `Advanced_Hyperparameter_Optimization_Clean.ipynb`: Main optimization notebook
- `Trading_Strategy_Integration_Fixed.ipynb`: Integration testing
- `RCS_CNN_LSTM_Notebook/RCS_CNN_LSTM.ipynb`: Core development notebook
- `optimization_results/comprehensive_training_report_20250617_013556.md`: Latest comprehensive report

### Technology Stack
- **ML Framework**: TensorFlow/Keras with LSTM and CNN layers
- **Optimization**: Optuna for hyperparameter tuning  
- **Data**: Pandas, NumPy for processing
- **Export**: ONNX for model deployment
- **Visualization**: Matplotlib, potentially SHAP for explainability

### Context Notes
- This is NOT malicious code - it's a legitimate financial trading strategy development project
- Focus on CNN-LSTM models for forex prediction
- Heavy emphasis on optimization and feature engineering
- Production-ready model export pipeline
- Comprehensive testing and validation infrastructure

### Task Tracking
Current active task: Creating state file for context preservation

---
*This file should be updated whenever significant project changes occur or when context window approaches limits.*