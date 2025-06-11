"""
GPU Configuration and CUDA Setup

This module provides utilities for configuring GPU usage and CUDA acceleration.
"""

import os
import sys

def setup_gpu_config():
    """
    Configure GPU settings for optimal performance.
    
    Returns:
    --------
    dict
        GPU configuration information
    """
    config_info = {
        'gpu_available': False,
        'gpu_count': 0,
        'cuda_version': None,
        'tensorflow_gpu': False,
        'memory_growth': False,
        'devices': []
    }
    
    try:
        import tensorflow as tf
        
        # Get TensorFlow version
        tf_version = tf.__version__
        print(f"🔧 TensorFlow version: {tf_version}")
        
        # Check if TensorFlow was built with CUDA
        config_info['tensorflow_gpu'] = tf.test.is_built_with_cuda()
        print(f"🔧 TensorFlow built with CUDA: {config_info['tensorflow_gpu']}")
        
        # Get physical GPU devices
        physical_devices = tf.config.list_physical_devices('GPU')
        config_info['gpu_count'] = len(physical_devices)
        config_info['gpu_available'] = config_info['gpu_count'] > 0
        
        print(f"🎮 GPU devices found: {config_info['gpu_count']}")
        
        if config_info['gpu_available']:
            # Configure memory growth to avoid allocating all GPU memory at once
            for gpu in physical_devices:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    config_info['memory_growth'] = True
                    print(f"✅ Memory growth enabled for GPU: {gpu.name}")
                    
                    # Get GPU details
                    device_details = tf.config.experimental.get_device_details(gpu)
                    config_info['devices'].append({
                        'name': gpu.name,
                        'details': device_details
                    })
                    
                except Exception as e:
                    print(f"⚠️ Could not configure memory growth for {gpu.name}: {e}")
            
            # Set GPU as preferred device
            try:
                tf.config.set_soft_device_placement(True)
                print("✅ Soft device placement enabled")
            except Exception as e:
                print(f"⚠️ Could not enable soft device placement: {e}")
                
            # Enable mixed precision for better performance
            try:
                from tensorflow.keras.mixed_precision import set_global_policy
                set_global_policy('mixed_float16')
                print("✅ Mixed precision (float16) enabled for better performance")
                config_info['mixed_precision'] = True
            except Exception as e:
                print(f"⚠️ Could not enable mixed precision: {e}")
                config_info['mixed_precision'] = False
                
        else:
            print("⚠️ No GPU devices found, using CPU")
            
    except ImportError:
        print("❌ TensorFlow not available")
        config_info['error'] = 'TensorFlow not installed'
    except Exception as e:
        print(f"❌ GPU configuration failed: {e}")
        config_info['error'] = str(e)
    
    return config_info

def check_cuda_environment():
    """
    Check CUDA environment and drivers.
    
    Returns:
    --------
    dict
        CUDA environment information
    """
    cuda_info = {
        'nvidia_smi_available': False,
        'cuda_version': None,
        'driver_version': None,
        'gpu_name': None,
        'gpu_memory': None,
        'gpu_utilization': None
    }
    
    try:
        import subprocess
        
        # Run nvidia-smi to get GPU info
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,utilization.gpu,driver_version', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            cuda_info['nvidia_smi_available'] = True
            lines = result.stdout.strip().split('\n')
            if lines and lines[0]:
                parts = [p.strip() for p in lines[0].split(',')]
                if len(parts) >= 4:
                    cuda_info['gpu_name'] = parts[0]
                    cuda_info['gpu_memory'] = f"{parts[1]} MB"
                    cuda_info['gpu_utilization'] = f"{parts[2]}%"
                    cuda_info['driver_version'] = parts[3]
        
        # Get CUDA version
        cuda_result = subprocess.run(['nvidia-smi', '--query-gpu=cuda_version', '--format=csv,noheader,nounits'], 
                                   capture_output=True, text=True, timeout=10)
        if cuda_result.returncode == 0:
            cuda_version = cuda_result.stdout.strip()
            if cuda_version:
                cuda_info['cuda_version'] = cuda_version
                
    except Exception as e:
        print(f"⚠️ Could not check CUDA environment: {e}")
        cuda_info['error'] = str(e)
    
    return cuda_info

def print_gpu_summary():
    """Print a comprehensive GPU and CUDA summary."""
    print("🎮 GPU Configuration Summary")
    print("=" * 50)
    
    # Check CUDA environment
    cuda_info = check_cuda_environment()
    
    if cuda_info['nvidia_smi_available']:
        print(f"🔧 GPU Name: {cuda_info.get('gpu_name', 'Unknown')}")
        print(f"🔧 GPU Memory: {cuda_info.get('gpu_memory', 'Unknown')}")
        print(f"🔧 GPU Utilization: {cuda_info.get('gpu_utilization', 'Unknown')}")
        print(f"🔧 Driver Version: {cuda_info.get('driver_version', 'Unknown')}")
        print(f"🔧 CUDA Version: {cuda_info.get('cuda_version', 'Unknown')}")
    else:
        print("❌ NVIDIA GPU not detected or nvidia-smi not available")
    
    print("\n🤖 TensorFlow GPU Configuration")
    print("-" * 30)
    
    # Setup GPU configuration
    gpu_config = setup_gpu_config()
    
    if gpu_config.get('error'):
        print(f"❌ Configuration Error: {gpu_config['error']}")
    else:
        print(f"✅ GPU Available: {gpu_config['gpu_available']}")
        print(f"✅ GPU Count: {gpu_config['gpu_count']}")
        print(f"✅ TensorFlow GPU Support: {gpu_config['tensorflow_gpu']}")
        print(f"✅ Memory Growth: {gpu_config['memory_growth']}")
        print(f"✅ Mixed Precision: {gpu_config.get('mixed_precision', False)}")
    
    return gpu_config, cuda_info

def configure_tensorflow_gpu():
    """
    Configure TensorFlow for optimal GPU usage.
    
    Returns:
    --------
    bool
        True if GPU configuration successful, False otherwise
    """
    try:
        import tensorflow as tf
        
        # Set environment variables for better GPU performance
        os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        
        # Configure GPU devices
        physical_devices = tf.config.list_physical_devices('GPU')
        
        if physical_devices:
            # Enable memory growth
            for gpu in physical_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Configure logical devices with memory limit if needed
            # This prevents OOM errors with large models
            try:
                tf.config.experimental.set_virtual_device_configuration(
                    physical_devices[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6144)]  # 6GB limit for RTX 3060 Ti
                )
                print("✅ GPU memory limit set to 6GB")
            except Exception as e:
                print(f"⚠️ Could not set memory limit: {e}")
            
            # Enable mixed precision
            try:
                from tensorflow.keras.mixed_precision import set_global_policy
                set_global_policy('mixed_float16')
                print("✅ Mixed precision enabled")
            except Exception as e:
                print(f"⚠️ Could not enable mixed precision: {e}")
            
            # Test GPU computation
            try:
                with tf.device('/GPU:0'):
                    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                    b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
                    c = tf.matmul(a, b)
                print("✅ GPU computation test successful")
                return True
            except Exception as e:
                print(f"⚠️ GPU computation test failed: {e}")
                return False
        else:
            print("⚠️ No GPU devices found")
            return False
            
    except ImportError:
        print("❌ TensorFlow not available")
        return False
    except Exception as e:
        print(f"❌ GPU configuration failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 GPU Configuration Check")
    gpu_config, cuda_info = print_gpu_summary()
    
    print("\n🔧 Configuring TensorFlow GPU...")
    gpu_success = configure_tensorflow_gpu()
    
    if gpu_success:
        print("\n🎉 GPU configuration completed successfully!")
        print("Your models will train on GPU with CUDA acceleration.")
    else:
        print("\n⚠️ GPU configuration failed. Models will train on CPU.")
        
print("✅ GPU configuration utilities loaded")