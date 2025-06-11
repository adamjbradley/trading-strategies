"""
GPU Environment Fix for Jupyter Notebooks
Run this in your notebook to ensure proper environment activation
"""

import os
import sys
import subprocess

def fix_environment_path():
    """Fix Python path to use trading-env conda environment."""
    
    # Get the trading-env Python path
    try:
        result = subprocess.run([
            'conda', 'run', '-n', 'trading-env', 'python', '-c', 
            'import sys; print(sys.executable)'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            trading_env_python = result.stdout.strip()
            current_python = sys.executable
            
            print(f"Current Python: {current_python}")
            print(f"Trading-env Python: {trading_env_python}")
            
            if trading_env_python not in current_python:
                print("⚠️ Not using trading-env Python!")
                print("Solutions:")
                print("1. Change Jupyter kernel to 'Python (trading-env)'")
                print("2. Restart Jupyter from activated trading-env")
                print("3. Run: conda activate trading-env && jupyter notebook")
                return False
            else:
                print("✅ Using correct trading-env Python")
                return True
                
    except Exception as e:
        print(f"Error checking environment: {e}")
        return False

def verify_gpu_in_current_session():
    """Verify GPU works in current Python session."""
    try:
        import tensorflow as tf
        
        print(f"\nTensorFlow version: {tf.__version__}")
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            print(f"✅ {len(gpus)} GPU(s) detected in current session:")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu}")
            
            # Test computation
            with tf.device('/GPU:0'):
                a = tf.random.normal([100, 100])
                b = tf.random.normal([100, 100])
                c = tf.matmul(a, b)
                print(f"✅ GPU computation successful: {c.device}")
            
            return True
        else:
            print("❌ No GPUs detected in current session")
            return False
            
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        return False

# Run the fixes
if __name__ == "__main__":
    print("🔧 Environment Fix Tool")
    print("=" * 40)
    
    env_ok = fix_environment_path()
    gpu_ok = verify_gpu_in_current_session()
    
    if env_ok and gpu_ok:
        print("\n🎉 Environment is correctly configured!")
    else:
        print("\n⚠️ Issues detected. Please follow the solutions above.")