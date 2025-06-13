#!/usr/bin/env python3
"""
Fixed ONNX Export Function
This properly handles the Sequential model conversion to ONNX
"""

def create_fixed_onnx_export_method():
    """
    Returns a fixed ONNX export method that properly handles Sequential models
    """
    
    def _export_best_model_to_onnx(self, symbol: str, model, model_data: dict, params: dict) -> str:
        """Export the best model to ONNX format with proper Sequential model handling"""
        from datetime import datetime
        from pathlib import Path
        import tensorflow as tf
        import json
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Always save Keras model first as backup
        keras_filename = f"{symbol}_CNN_LSTM_{timestamp}.h5"
        keras_path = Path("exported_models") / keras_filename
        
        try:
            model.save(str(keras_path))
            print(f"📁 Keras model saved: {keras_filename}")
        except Exception as e:
            print(f"❌ Keras save failed: {e}")
            return f"save_failed_{timestamp}"
        
        # Try ONNX export with proper Sequential model handling
        try:
            import tf2onnx
            import onnx
            
            onnx_filename = f"{symbol}_CNN_LSTM_{timestamp}.onnx"
            onnx_path = Path("exported_models") / onnx_filename
            
            # Get input shape from model_data
            input_shape = model_data['input_shape']
            lookback_window, num_features = input_shape
            
            # FIXED: Use tf.function wrapper to avoid Sequential model issues
            @tf.function
            def model_func(x):
                return model(x)
            
            # Create concrete function with proper input signature
            concrete_func = model_func.get_concrete_function(
                tf.TensorSpec((None, lookback_window, num_features), tf.float32)
            )
            
            # Convert using the concrete function (avoids 'output_names' error)
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
            print(f"⚠️  tf2onnx not available, using Keras format")
            self._save_training_metadata(symbol, params, model_data, timestamp)
            return keras_filename
            
        except Exception as e:
            print(f"⚠️  ONNX export failed ({str(e)[:50]}), using Keras format")
            # Still save metadata even if ONNX fails
            self._save_training_metadata(symbol, params, model_data, timestamp)
            return keras_filename
    
    return _export_best_model_to_onnx

def apply_onnx_fix(optimizer_instance):
    """Apply the fixed ONNX export method to the optimizer"""
    import types
    
    # Get the fixed method
    fixed_method = create_fixed_onnx_export_method()
    
    # Apply it to the optimizer instance
    optimizer_instance._export_best_model_to_onnx = types.MethodType(fixed_method, optimizer_instance)
    
    print("✅ ONNX export method FIXED!")
    print("🔧 Now properly handles TensorFlow Sequential models")
    print("💾 Uses tf.function wrapper to avoid 'output_names' error")
    print("🔄 Falls back to Keras format if ONNX conversion fails")

if __name__ == "__main__":
    print("🔧 ONNX Export Fix Ready!")
    print()
    print("To apply the fix:")
    print("1. Import this module: from fixed_onnx_export import apply_onnx_fix")
    print("2. Apply to optimizer: apply_onnx_fix(optimizer)")
    print("3. Run training - ONNX export will work properly!")
    print()
    print("The fix handles the 'Sequential' object has no attribute 'output_names' error")
    print("by using tf.function wrapper and tf2onnx.convert.from_function instead")
    print("of tf2onnx.convert.from_keras which has issues with Sequential models.")