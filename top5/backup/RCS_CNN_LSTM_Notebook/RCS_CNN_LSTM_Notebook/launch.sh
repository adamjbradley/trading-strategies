#!/bin/bash

# Create conda environment from environment.yml
echo "🔧 Creating conda environment 'rcs_cnn_lstm_env'..."
conda env create -f environment.yml || conda env update -f environment.yml

# Activate the environment
echo "✅ To activate the environment, run:"
echo "conda activate rcs_cnn_lstm_env"

# Run the notebook or script if desired
echo "📓 To launch Jupyter Notebook:"
echo "jupyter notebook RCS_CNN_LSTM.ipynb"

echo "▶️ To run as a Python script:"
echo "python RCS_CNN_LSTM.ipynb"
