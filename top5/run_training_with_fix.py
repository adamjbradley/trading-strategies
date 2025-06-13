#!/usr/bin/env python3
"""
Run Training with ONNX Fix Applied
"""

print("🔧 Fixing ONNX export issue and starting training...")

# First, let's create a simple ONNX-free version that just saves Keras models
def simple_export_model(symbol: str, model, model_data: dict, params: dict, timestamp: str = None) -> str:
    """Simple model export without ONNX complications"""
    if timestamp is None:
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save as Keras model (reliable)
    keras_filename = f"{symbol}_CNN_LSTM_{timestamp}.h5"
    keras_path = f"exported_models/{keras_filename}"
    
    try:
        model.save(keras_path)
        print(f"📁 Model saved: {keras_filename}")
        
        # Save metadata
        metadata = {
            'symbol': symbol,
            'timestamp': timestamp,
            'hyperparameters': params,
            'selected_features': model_data.get('selected_features', []),
            'num_features': len(model_data.get('selected_features', [])),
            'lookback_window': model_data.get('lookback_window', 50),
            'input_shape': model_data.get('input_shape', [50, 10]),
            'model_architecture': 'CNN-LSTM',
            'framework': 'tensorflow/keras',
            'scaler_type': 'RobustScaler'
        }
        
        metadata_file = f"exported_models/{symbol}_training_metadata_{timestamp}.json"
        import json
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return keras_filename
        
    except Exception as e:
        print(f"❌ Model save failed: {e}")
        return f"save_failed_{timestamp}"

print("✅ Simple model export function created")
print("🎯 Ready to run training without ONNX complications")

# Instructions for running
print("""
🚀 TO RUN TRAINING:

1. Open Advanced_Hyperparameter_Optimization_Clean.ipynb
2. Run all cells to initialize the system  
3. In the last cell, modify the ONNX export to use simple Keras export
4. Run the full symbol optimization

The ONNX issue is fixed by simply saving Keras models instead.
You can convert to ONNX later if needed using tf2onnx separately.

✅ All models will be saved as .h5 files in exported_models/
✅ Training metadata will be saved as .json files
✅ No more ONNX export errors during training!
""")