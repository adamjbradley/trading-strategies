# GPU Configuration for Hyperparameter Optimization
# Add this to the beginning of cell 2 in the optimization notebook

import tensorflow as tf

def configure_gpu():
    """
    Configure TensorFlow for optimal GPU usage.
    """
    print("🔧 Configuring GPU settings...")
    
    # List available GPUs
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            print(f"🎮 Found {len(gpus)} GPU(s):")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu}")
            
            # Configure memory growth for all GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"  ✅ Memory growth enabled for {gpu}")
            
            # Set GPU memory limit (optional - useful for multiple processes)
            # Uncomment and adjust if you need to limit GPU memory
            # tf.config.experimental.set_memory_limit(gpus[0], 8192)  # 8GB limit
            
            # Enable mixed precision (faster training on modern GPUs)
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("  ✅ Mixed precision enabled (float16)")
            
            # Verify GPU is available for TensorFlow
            print(f"  ✅ GPU acceleration: {tf.test.is_gpu_available()}")
            print(f"  ✅ GPU device name: {tf.test.gpu_device_name()}")
            
            return True
            
        except RuntimeError as e:
            print(f"  ❌ GPU setup failed: {e}")
            return False
    else:
        print("  ⚠️ No GPUs found, using CPU")
        return False

def verify_gpu_usage():
    """
    Verify that TensorFlow is actually using GPU.
    """
    print("\n🔍 GPU Usage Verification:")
    
    # Check if GPU is being used
    with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
        # Create a simple computation
        a = tf.random.normal([1000, 1000])
        b = tf.random.normal([1000, 1000])
        c = tf.matmul(a, b)
        
        print(f"  Test computation device: {c.device}")
        print(f"  GPU available: {tf.config.list_physical_devices('GPU')}")
        
    # Memory info
    if tf.config.list_physical_devices('GPU'):
        gpu_details = tf.config.experimental.get_device_details(tf.config.list_physical_devices('GPU')[0])
        print(f"  GPU details: {gpu_details}")

# Configure GPU
gpu_available = configure_gpu()
verify_gpu_usage()

# Additional optimization settings for GPU
if gpu_available:
    print("\n⚡ GPU Optimization Settings Applied:")
    print("  - Memory growth enabled")
    print("  - Mixed precision training (float16)")
    print("  - GPU device verification completed")
    
    # Set additional TensorFlow GPU options
    tf.config.optimizer.set_jit(True)  # Enable XLA compilation
    print("  - XLA compilation enabled")
else:
    print("\n🖥️ CPU Optimization Settings:")
    # Optimize for CPU if no GPU available
    tf.config.threading.set_intra_op_parallelism_threads(0)  # Use all cores
    tf.config.threading.set_inter_op_parallelism_threads(0)  # Use all cores
    print("  - Multi-threading enabled for CPU")