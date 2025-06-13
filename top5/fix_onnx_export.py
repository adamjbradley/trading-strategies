#!/usr/bin/env python3
"""
Fix ONNX Export Method
"""

def fixed_export_best_model_to_onnx(self, symbol: str, model, model_data: dict, params: dict) -> str:
    """Fixed ONNX export method with proper error handling"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Always save Keras model first as backup
    keras_filename = f"{symbol}_CNN_LSTM_{timestamp}.h5"
    keras_path = Path(MODELS_PATH) / keras_filename
    
    try:
        model.save(str(keras_path))
        print(f"📁 Keras model saved: {keras_filename}")
    except Exception as e:
        print(f"❌ Keras save failed: {e}")
        return f"save_failed_{timestamp}"
    
    # Try ONNX export with proper error handling
    try:
        import tf2onnx
        import onnx
        
        onnx_filename = f"{symbol}_CNN_LSTM_{timestamp}.onnx"
        onnx_path = Path(MODELS_PATH) / onnx_filename
        
        # Get input shape from model_data
        input_shape = model_data['input_shape']
        lookback_window, num_features = input_shape
        
        # Create a concrete function from the model
        @tf.function
        def model_func(x):
            return model(x)
        
        # Get the concrete function
        concrete_func = model_func.get_concrete_function(
            tf.TensorSpec((None, lookback_window, num_features), tf.float32)
        )
        
        # Convert to ONNX using the concrete function
        onnx_model, _ = tf2onnx.convert.from_function(
            concrete_func,
            input_signature=[tf.TensorSpec((None, lookback_window, num_features), tf.float32, name='input')],
            opset=13
        )
        
        # Save ONNX model
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        
        print(f"📁 ONNX model saved: {onnx_filename}")
        
        # Save training metadata
        self._save_training_metadata(symbol, params, model_data, timestamp)
        
        return onnx_filename
        
    except ImportError:
        print(f"⚠️  tf2onnx not available, keeping Keras format")
        self._save_training_metadata(symbol, params, model_data, timestamp)
        return keras_filename
        
    except Exception as e:
        print(f"⚠️  ONNX export failed: {e}, keeping Keras format")
        # Still save metadata even if ONNX fails
        self._save_training_metadata(symbol, params, model_data, timestamp)
        return keras_filename

# Apply the fix to the optimizer
import types

def apply_onnx_fix(optimizer_instance):
    """Apply the ONNX export fix to the optimizer"""
    # Import required modules at the top level
    import tensorflow as tf
    from pathlib import Path
    from datetime import datetime
    import json
    
    optimizer_instance._export_best_model_to_onnx = types.MethodType(fixed_export_best_model_to_onnx, optimizer_instance)
    print("✅ ONNX export method fixed!")
    print("🔧 Now handles TensorFlow Sequential models properly")
    print("💾 Always saves Keras backup before attempting ONNX conversion")

if __name__ == "__main__":
    print("🔧 ONNX Export Fix Ready!")
    print("Run apply_onnx_fix(optimizer) to apply the fix")