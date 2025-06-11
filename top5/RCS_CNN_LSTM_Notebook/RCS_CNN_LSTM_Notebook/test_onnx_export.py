#!/usr/bin/env python3
"""
Test ONNX Export with NumPy 2.0 Compatibility

This script tests the improved ONNX export functionality.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, BatchNormalization
import os
import sys

# Add src to path
sys.path.append('src')

def create_test_model(input_shape):
    """Create a test CNN-LSTM model."""
    model = Sequential([
        tf.keras.layers.Input(shape=input_shape),
        Conv1D(filters=32, kernel_size=1, padding='same', activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        LSTM(25, return_sequences=False),
        BatchNormalization(),
        Dropout(0.2),
        Dense(10, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def test_onnx_export():
    """Test the ONNX export functionality."""
    print("🧪 Testing ONNX export with NumPy 2.0 compatibility")
    print(f"NumPy version: {np.__version__}")
    print(f"TensorFlow version: {tf.__version__}")
    
    # Create test data
    input_shape = (1, 10)  # (timesteps, features)
    X_test = np.random.random((20, 1, 10)).astype(np.float32)
    
    print(f"Test input shape: {X_test.shape}")
    print(f"Model input shape: {input_shape}")
    
    # Create test model
    print("\n🔧 Creating test model...")
    model = create_test_model(input_shape)
    print("Model created successfully")
    
    # Test ONNX export
    print("\n📤 Testing ONNX export...")
    
    try:
        from src.export.onnx_numpy2_fix import export_model_to_onnx_numpy2_safe
        
        # Create output directory
        os.makedirs("test_exports", exist_ok=True)
        output_path = "test_exports/test_model.onnx"
        
        # Export model
        success = export_model_to_onnx_numpy2_safe(
            model=model,
            output_path=output_path,
            input_shape=input_shape,
            test_input=X_test[:5],  # Use small sample for testing
            model_name="test_cnn_lstm"
        )
        
        if success:
            print("✅ ONNX export test PASSED")
            
            # Test file size
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                print(f"ONNX file size: {file_size} bytes")
                
                # Clean up
                print("🧹 Cleaning up test files...")
                os.remove(output_path)
                os.rmdir("test_exports")
                print("Test files cleaned up")
            
            return True
        else:
            print("❌ ONNX export test FAILED")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Export test failed: {e}")
        return False

def test_numpy_compatibility():
    """Test NumPy compatibility functions."""
    print("\n🧪 Testing NumPy compatibility functions")
    
    try:
        from src.export.onnx_numpy2_fix import ensure_numpy_compatibility, fix_onnx_model_for_numpy2
        
        # Test array compatibility
        test_array = np.random.random((10, 5))
        compatible_array = ensure_numpy_compatibility(test_array)
        
        print(f"Original array shape: {test_array.shape}")
        print(f"Compatible array shape: {compatible_array.shape}")
        print("✅ NumPy compatibility test PASSED")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ NumPy compatibility test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting ONNX export compatibility tests\n")
    
    # Test NumPy compatibility
    numpy_test_passed = test_numpy_compatibility()
    
    # Test ONNX export
    onnx_test_passed = test_onnx_export()
    
    # Summary
    print(f"\n📊 Test Results:")
    print(f"NumPy compatibility: {'✅ PASSED' if numpy_test_passed else '❌ FAILED'}")
    print(f"ONNX export: {'✅ PASSED' if onnx_test_passed else '❌ FAILED'}")
    
    if numpy_test_passed and onnx_test_passed:
        print("\n🎉 All tests PASSED! ONNX export is ready for NumPy 2.0")
        sys.exit(0)
    else:
        print("\n⚠️ Some tests FAILED. Check the output above for details.")
        sys.exit(1)