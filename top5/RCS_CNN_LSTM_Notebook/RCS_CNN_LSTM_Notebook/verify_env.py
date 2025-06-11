#!/usr/bin/env python3
"""
Verify that the trading-env environment is properly configured
"""

import sys
import os

def main():
    print("🔍 Environment Verification")
    print("=" * 40)
    
    # Check Python path
    print(f"Python executable: {sys.executable}")
    print(f"Environment path: {sys.prefix}")
    
    # Check if we're in the trading-env
    if "trading-env" in sys.executable:
        print("✅ Running in trading-env")
    else:
        print("⚠️ NOT running in trading-env")
        return False
    
    # Check TensorFlow GPU
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow version: {tf.__version__}")
        
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ GPU available: {len(gpus)} GPU(s)")
            print(f"✅ GPU details: {gpus[0]}")
        else:
            print("⚠️ No GPU detected")
            
        # Test GPU computation
        with tf.device('/GPU:0' if gpus else '/CPU:0'):
            a = tf.constant([1.0, 2.0])
            b = tf.constant([3.0, 4.0])
            c = tf.add(a, b)
            device = c.device
            print(f"✅ Computation device: {device}")
            
    except ImportError:
        print("❌ TensorFlow not available")
        return False
    
    # Check other key packages
    packages = ['pandas', 'numpy', 'scikit-learn', 'matplotlib']
    for pkg in packages:
        try:
            __import__(pkg)
            print(f"✅ {pkg} available")
        except ImportError:
            print(f"❌ {pkg} not available")
    
    print("\n🚀 Environment verification complete!")
    return True

if __name__ == "__main__":
    main()