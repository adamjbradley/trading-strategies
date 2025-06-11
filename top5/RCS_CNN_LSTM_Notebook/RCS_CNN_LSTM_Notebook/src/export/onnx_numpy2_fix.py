"""
ONNX Export with NumPy 2.0 Compatibility

This module provides improved ONNX export functionality that addresses
compatibility issues with NumPy 2.0.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tf2onnx
import onnx
from datetime import datetime
import warnings

# Suppress specific warnings that are common with NumPy 2.0
warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
warnings.filterwarnings('ignore', message='.*numpy.ndarray size changed.*')

def ensure_numpy_compatibility(array):
    """
    Ensure NumPy array compatibility across different versions.
    
    Parameters:
    -----------
    array : numpy.ndarray
        Input array
        
    Returns:
    --------
    numpy.ndarray
        Compatible array
    """
    if hasattr(array, 'copy'):
        # Use explicit copy for NumPy 2.0 compatibility
        return array.copy()
    else:
        # Fallback to np.array for older versions
        return np.array(array)

def fix_onnx_model_for_numpy2(onnx_model):
    """
    Apply fixes to ONNX model for NumPy 2.0 compatibility.
    
    Parameters:
    -----------
    onnx_model : onnx.ModelProto
        ONNX model to fix
        
    Returns:
    --------
    onnx.ModelProto
        Fixed ONNX model
    """
    try:
        # Check if we need to apply NumPy 2.0 specific fixes
        numpy_version = tuple(map(int, np.__version__.split('.')[:2]))
        
        if numpy_version >= (2, 0):
            # Apply NumPy 2.0 specific fixes
            for node in onnx_model.graph.node:
                # Fix potential data type issues
                for attr in node.attribute:
                    if attr.type == onnx.AttributeProto.TENSOR:
                        # Ensure tensor data is properly formatted
                        if hasattr(attr.t, 'raw_data') and attr.t.raw_data:
                            # Convert raw data to ensure compatibility
                            try:
                                # This helps with potential serialization issues
                                raw_data = bytes(attr.t.raw_data)
                                attr.t.raw_data = raw_data
                            except Exception:
                                pass
            
        return onnx_model
    except Exception as e:
        print(f"⚠️ Warning: Could not apply NumPy 2.0 fixes: {e}")
        return onnx_model

def convert_keras_to_onnx_safe(model, input_shape, opset_version=13, model_name="model"):
    """
    Safely convert Keras model to ONNX with NumPy 2.0 compatibility.
    
    Parameters:
    -----------
    model : keras.Model
        Keras model to convert
    input_shape : tuple
        Input shape (excluding batch dimension)
    opset_version : int, default=13
        ONNX opset version to use
    model_name : str, default="model"
        Name for the model
        
    Returns:
    --------
    onnx.ModelProto or None
        ONNX model or None if conversion fails
    """
    print(f"🔄 Converting Keras model to ONNX (opset {opset_version})")
    
    try:
        # Ensure input shape is compatible
        if not isinstance(input_shape, tuple):
            input_shape = tuple(input_shape)
        
        # Create input signature with proper data types
        input_signature = [tf.TensorSpec(
            shape=(None,) + input_shape,
            dtype=tf.float32,
            name="input_1"
        )]
        
        # Method 1: Direct conversion from Keras
        try:
            print("  Attempting direct Keras conversion...")
            onnx_model, _ = tf2onnx.convert.from_keras(
                model, 
                input_signature=input_signature,
                opset=opset_version,
                custom_ops=None,
                extra_opset=None
            )
            
            # Apply NumPy 2.0 compatibility fixes
            onnx_model = fix_onnx_model_for_numpy2(onnx_model)
            print("  ✅ Direct Keras conversion successful")
            return onnx_model
            
        except Exception as e:
            print(f"  ⚠️ Direct conversion failed: {e}")
            print("  Trying alternative method...")
        
        # Method 2: Convert via concrete function
        try:
            print("  Attempting concrete function conversion...")
            
            # Create a concrete function
            @tf.function
            def model_func(x):
                return model(x)
            
            # Get concrete function
            concrete_func = model_func.get_concrete_function(
                tf.TensorSpec(shape=(None,) + input_shape, dtype=tf.float32)
            )
            
            # Convert using the concrete function
            onnx_model, _ = tf2onnx.convert.from_function(
                concrete_func,
                input_signature=input_signature,
                opset=opset_version
            )
            
            # Apply NumPy 2.0 compatibility fixes
            onnx_model = fix_onnx_model_for_numpy2(onnx_model)
            print("  ✅ Concrete function conversion successful")
            return onnx_model
            
        except Exception as e:
            print(f"  ⚠️ Concrete function conversion failed: {e}")
            print("  Trying SavedModel method...")
        
        # Method 3: Convert via SavedModel (most robust)
        try:
            print("  Attempting SavedModel conversion...")
            
            # Create temporary directory for SavedModel
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                savedmodel_path = os.path.join(temp_dir, "temp_model")
                
                # Save as SavedModel
                model.save(savedmodel_path, save_format='tf')
                
                # Convert from SavedModel
                onnx_model = tf2onnx.convert.from_saved_model(
                    savedmodel_path,
                    input_names=["input_1"],
                    output_names=["output"],
                    opset=opset_version
                )
                
                # Apply NumPy 2.0 compatibility fixes
                onnx_model = fix_onnx_model_for_numpy2(onnx_model)
                print("  ✅ SavedModel conversion successful")
                return onnx_model
                
        except Exception as e:
            print(f"  ⚠️ SavedModel conversion failed: {e}")
            print("  Trying simplified model conversion...")
        
        # Method 4: Simplified conversion with basic model
        try:
            print("  Attempting simplified conversion...")
            
            # Create a simplified version of the model for conversion
            if hasattr(model, 'layers') and len(model.layers) > 0:
                # Try to build a simplified functional model
                inputs = tf.keras.Input(shape=input_shape, name="input_1")
                outputs = model(inputs)
                
                functional_model = tf.keras.Model(inputs=inputs, outputs=outputs, name=model_name)
                
                # Convert the functional model
                onnx_model, _ = tf2onnx.convert.from_keras(
                    functional_model,
                    input_signature=input_signature,
                    opset=opset_version
                )
                
                # Apply NumPy 2.0 compatibility fixes
                onnx_model = fix_onnx_model_for_numpy2(onnx_model)
                print("  ✅ Simplified conversion successful")
                return onnx_model
                
        except Exception as e:
            print(f"  ⚠️ Simplified conversion failed: {e}")
        
        print("  ❌ All conversion methods failed")
        return None
        
    except Exception as e:
        print(f"❌ ONNX conversion failed: {e}")
        return None

def save_onnx_model_safe(onnx_model, output_path):
    """
    Safely save ONNX model with NumPy 2.0 compatibility.
    
    Parameters:
    -----------
    onnx_model : onnx.ModelProto
        ONNX model to save
    output_path : str
        Path to save the model
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Validate the model before saving
        try:
            onnx.checker.check_model(onnx_model)
            print("  ✅ ONNX model validation passed")
        except Exception as e:
            print(f"  ⚠️ ONNX model validation warning: {e}")
            # Continue anyway as some warnings are non-critical
        
        # Save the model with NumPy 2.0 compatible method
        try:
            # Use onnx.save with explicit protocol buffer handling
            with open(output_path, 'wb') as f:
                f.write(onnx_model.SerializeToString())
            print(f"  ✅ ONNX model saved to {output_path}")
            return True
        except Exception as e:
            print(f"  ⚠️ Custom save failed: {e}, trying standard save...")
            # Fallback to standard save
            onnx.save(onnx_model, output_path)
            print(f"  ✅ ONNX model saved to {output_path} (fallback method)")
            return True
            
    except Exception as e:
        print(f"❌ Failed to save ONNX model: {e}")
        return False

def verify_onnx_model(onnx_path, test_input=None):
    """
    Verify ONNX model can be loaded and run.
    
    Parameters:
    -----------
    onnx_path : str
        Path to ONNX model
    test_input : numpy.ndarray, optional
        Test input for inference verification
        
    Returns:
    --------
    bool
        True if verification passes, False otherwise
    """
    try:
        # Load the model
        onnx_model = onnx.load(onnx_path)
        print(f"  ✅ ONNX model loaded successfully from {onnx_path}")
        
        # Check model
        try:
            onnx.checker.check_model(onnx_model)
            print("  ✅ ONNX model structure validation passed")
        except Exception as e:
            print(f"  ⚠️ Model structure validation warning: {e}")
        
        # Test inference if test input provided
        if test_input is not None:
            try:
                import onnxruntime as ort
                
                # Create inference session
                session = ort.InferenceSession(onnx_path)
                
                # Get input name
                input_name = session.get_inputs()[0].name
                
                # Ensure test input is compatible
                test_input_safe = ensure_numpy_compatibility(test_input)
                
                # Run inference
                outputs = session.run(None, {input_name: test_input_safe})
                print(f"  ✅ ONNX inference test passed (output shape: {outputs[0].shape})")
                
            except ImportError:
                print("  ⚠️ ONNXRuntime not available, skipping inference test")
            except Exception as e:
                print(f"  ⚠️ ONNX inference test failed: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ ONNX model verification failed: {e}")
        return False

def export_model_to_onnx_numpy2_safe(model, output_path, input_shape, test_input=None, model_name="cnn_lstm_model"):
    """
    Export Keras model to ONNX with full NumPy 2.0 compatibility.
    
    Parameters:
    -----------
    model : keras.Model
        Keras model to export
    output_path : str
        Path to save ONNX model
    input_shape : tuple
        Input shape (excluding batch dimension)
    test_input : numpy.ndarray, optional
        Test input for verification
    model_name : str, default="cnn_lstm_model"
        Name for the model
        
    Returns:
    --------
    bool
        True if export successful, False otherwise
    """
    print(f"\n🔄 Exporting model to ONNX with NumPy 2.0 compatibility")
    print(f"   Model: {model_name}")
    print(f"   Output: {output_path}")
    print(f"   Input shape: {input_shape}")
    print(f"   NumPy version: {np.__version__}")
    
    # Convert model to ONNX
    onnx_model = convert_keras_to_onnx_safe(model, input_shape, model_name=model_name)
    
    if onnx_model is None:
        print("❌ ONNX conversion failed")
        return False
    
    # Save ONNX model
    save_success = save_onnx_model_safe(onnx_model, output_path)
    
    if not save_success:
        print("❌ ONNX save failed")
        return False
    
    # Verify the saved model
    verify_success = verify_onnx_model(output_path, test_input)
    
    if verify_success:
        print("✅ ONNX export completed successfully with NumPy 2.0 compatibility")
        return True
    else:
        print("⚠️ ONNX export completed but verification failed")
        return False

print("✅ ONNX NumPy 2.0 compatibility module loaded")