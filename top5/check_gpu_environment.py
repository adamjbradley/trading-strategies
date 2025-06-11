#!/usr/bin/env python3
"""
GPU Environment Checker for Trading Strategy Notebooks
This script helps diagnose and fix GPU configuration issues.
"""

import subprocess
import sys
import os
import importlib.util

def check_conda_environment():
    """Check current conda environment and available environments."""
    print("🔍 Checking Conda Environment...")
    
    try:
        # Check current environment
        current_env = os.environ.get('CONDA_DEFAULT_ENV', 'Unknown')
        print(f"Current environment: {current_env}")
        
        # List all environments
        result = subprocess.run(['conda', 'info', '--envs'], capture_output=True, text=True)
        print("\nAvailable environments:")
        trading_env_found = False
        
        for line in result.stdout.split('\n'):
            if 'trading-env' in line:
                print(f"✅ {line}")
                trading_env_found = True
            elif line.strip() and not line.startswith('#'):
                print(f"   {line}")
        
        if not trading_env_found:
            print("❌ trading-env not found!")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking conda: {e}")
        return False

def check_nvidia_gpu():
    """Check NVIDIA GPU availability."""
    print("\n🎮 Checking NVIDIA GPU...")
    
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA GPU detected:")
            # Extract GPU name
            for line in result.stdout.split('\n'):
                if 'GeForce' in line or 'RTX' in line or 'GTX' in line:
                    gpu_info = line.split('|')[1].strip()
                    print(f"   {gpu_info}")
            return True
        else:
            print("❌ nvidia-smi failed")
            return False
            
    except Exception as e:
        print(f"❌ Error checking GPU: {e}")
        return False

def check_tensorflow_gpu():
    """Check TensorFlow GPU support."""
    print("\n🧠 Checking TensorFlow GPU Support...")
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow version: {tf.__version__}")
        
        # Check CUDA support
        cuda_available = tf.test.is_built_with_cuda()
        print(f"CUDA support: {'✅' if cuda_available else '❌'} {cuda_available}")
        
        # Check GPU devices
        gpu_devices = tf.config.list_physical_devices('GPU')
        if gpu_devices:
            print(f"✅ GPU devices found: {len(gpu_devices)}")
            for i, gpu in enumerate(gpu_devices):
                print(f"   GPU {i}: {gpu}")
                
                # Get GPU details
                try:
                    details = tf.config.experimental.get_device_details(gpu)
                    print(f"      {details}")
                except:
                    pass
            
            # Test GPU computation
            try:
                with tf.device('/GPU:0'):
                    a = tf.random.normal([100, 100])
                    b = tf.random.normal([100, 100])
                    c = tf.matmul(a, b)
                    print(f"✅ GPU computation test successful")
                    print(f"   Device used: {c.device}")
                
                return True
                
            except Exception as e:
                print(f"❌ GPU computation test failed: {e}")
                return False
                
        else:
            print("❌ No GPU devices found")
            return False
            
    except ImportError:
        print("❌ TensorFlow not installed")
        return False
    except Exception as e:
        print(f"❌ Error checking TensorFlow: {e}")
        return False

def provide_solutions():
    """Provide solutions for common issues."""
    print("\n💡 Solutions for Common Issues:")
    
    print("\n1. If GPU not detected in notebook:")
    print("   - Restart Jupyter kernel")
    print("   - Ensure you're running: conda activate trading-env")
    print("   - Run notebook from activated environment")
    
    print("\n2. If TensorFlow doesn't see GPU:")
    print("   - Install GPU TensorFlow: conda install tensorflow-gpu")
    print("   - Update CUDA drivers")
    print("   - Restart Python session")
    
    print("\n3. If conda environment issues:")
    print("   - Create new environment:")
    print("     conda create -n trading-env python=3.10")
    print("     conda activate trading-env")
    print("     conda install tensorflow-gpu")
    
    print("\n4. To test in notebook, run:")
    print("   import tensorflow as tf")
    print("   print(tf.config.list_physical_devices('GPU'))")

def main():
    """Main diagnostic function."""
    print("🔧 GPU Environment Diagnostic Tool")
    print("=" * 50)
    
    # Run all checks
    conda_ok = check_conda_environment()
    gpu_ok = check_nvidia_gpu()
    tf_ok = check_tensorflow_gpu()
    
    print("\n" + "=" * 50)
    print("📊 Summary:")
    print(f"Conda Environment: {'✅' if conda_ok else '❌'}")
    print(f"NVIDIA GPU: {'✅' if gpu_ok else '❌'}")
    print(f"TensorFlow GPU: {'✅' if tf_ok else '❌'}")
    
    if conda_ok and gpu_ok and tf_ok:
        print("\n🎉 All checks passed! GPU should work in notebooks.")
    else:
        print("\n⚠️ Issues detected. See solutions below.")
        provide_solutions()

if __name__ == "__main__":
    main()