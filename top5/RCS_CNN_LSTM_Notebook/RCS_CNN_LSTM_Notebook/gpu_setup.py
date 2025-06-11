#!/usr/bin/env python3
"""
GPU Setup and Performance Testing

This script sets up GPU acceleration and tests performance.
"""

import sys
import os

# Add src to path
sys.path.append('src')

def main():
    """Main GPU setup and testing function."""
    print("🚀 GPU Setup and Performance Testing")
    print("=" * 50)
    
    # Import GPU configuration utilities
    try:
        from src.utils.gpu_config import print_gpu_summary, configure_tensorflow_gpu
        
        # Print comprehensive GPU summary
        gpu_config, cuda_info = print_gpu_summary()
        
        # Configure TensorFlow for GPU
        print("\n🔧 Configuring TensorFlow for GPU...")
        gpu_success = configure_tensorflow_gpu()
        
        if gpu_success:
            print("\n✅ GPU configuration successful!")
            
            # Test GPU performance with a simple model
            print("\n🧪 Testing GPU performance...")
            test_gpu_performance()
        else:
            print("\n⚠️ GPU configuration failed. Check your CUDA installation.")
            
    except ImportError as e:
        print(f"❌ Could not import GPU utilities: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ GPU setup failed: {e}")
        sys.exit(1)

def test_gpu_performance():
    """Test GPU performance with a simple CNN-LSTM model."""
    try:
        import tensorflow as tf
        import numpy as np
        import time
        
        # Check available devices
        print("📊 Available devices:")
        for device in tf.config.list_physical_devices():
            print(f"  - {device}")
        
        # Create test data
        X_test = np.random.random((1000, 10, 20)).astype(np.float32)
        y_test = np.random.randint(0, 2, (1000, 1)).astype(np.float32)
        
        print(f"📈 Test data shape: X={X_test.shape}, y={y_test.shape}")
        
        # Create simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(64, 3, activation='relu', input_shape=(10, 20)),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Test training performance
        print("\n⏱️ Testing training performance...")
        
        # GPU test
        if tf.config.list_physical_devices('GPU'):
            with tf.device('/GPU:0'):
                start_time = time.time()
                model.fit(X_test, y_test, epochs=3, batch_size=64, verbose=0)
                gpu_time = time.time() - start_time
                print(f"🎮 GPU training time (3 epochs): {gpu_time:.2f} seconds")
        
        # CPU test for comparison
        with tf.device('/CPU:0'):
            # Create new model for fair comparison
            cpu_model = tf.keras.Sequential([
                tf.keras.layers.Conv1D(64, 3, activation='relu', input_shape=(10, 20)),
                tf.keras.layers.LSTM(32),
                tf.keras.layers.Dense(16, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            cpu_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            
            start_time = time.time()
            cpu_model.fit(X_test, y_test, epochs=3, batch_size=64, verbose=0)
            cpu_time = time.time() - start_time
            print(f"🖥️ CPU training time (3 epochs): {cpu_time:.2f} seconds")
        
        # Calculate speedup
        if tf.config.list_physical_devices('GPU') and 'gpu_time' in locals():
            speedup = cpu_time / gpu_time
            print(f"🚀 GPU speedup: {speedup:.2f}x faster than CPU")
            
            if speedup > 1.5:
                print("✅ GPU acceleration is working well!")
            else:
                print("⚠️ GPU speedup is lower than expected. Check your configuration.")
        else:
            print("ℹ️ GPU not available for comparison")
            
    except Exception as e:
        print(f"❌ Performance test failed: {e}")

def check_cuda_installation():
    """Check CUDA installation and provide recommendations."""
    print("\n🔍 Checking CUDA Installation")
    print("-" * 30)
    
    try:
        import subprocess
        
        # Check nvidia-smi
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ nvidia-smi is working")
            
            # Extract CUDA version
            for line in result.stdout.split('\n'):
                if 'CUDA Version:' in line:
                    cuda_version = line.split('CUDA Version:')[1].strip().split()[0]
                    print(f"🔧 CUDA Version: {cuda_version}")
                    break
        else:
            print("❌ nvidia-smi failed")
            
    except FileNotFoundError:
        print("❌ nvidia-smi not found. Please install NVIDIA drivers.")
    except Exception as e:
        print(f"⚠️ Error checking CUDA: {e}")

def print_recommendations():
    """Print GPU optimization recommendations."""
    print("\n💡 GPU Optimization Recommendations")
    print("-" * 40)
    print("For RTX 3060 Ti (8GB VRAM):")
    print("  • Use batch sizes: 64-128 for training")
    print("  • Enable mixed precision (float16)")
    print("  • Set memory growth to avoid OOM")
    print("  • Use TensorFlow-GPU >= 2.8.0")
    print("  • Install CUDA 11.8 and cuDNN 8.6")
    print("  • Monitor GPU memory usage with nvidia-smi")
    print("\nOptimal settings:")
    print("  • Batch size: 64-128")
    print("  • Learning rate: 0.001-0.01")
    print("  • Memory limit: 6GB (leaves 2GB for system)")

if __name__ == "__main__":
    main()
    check_cuda_installation()
    print_recommendations()