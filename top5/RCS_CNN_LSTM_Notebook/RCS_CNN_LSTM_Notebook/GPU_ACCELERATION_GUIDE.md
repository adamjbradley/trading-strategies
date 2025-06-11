# GPU Acceleration Guide

This guide explains how to leverage your **NVIDIA GeForce RTX 3060 Ti** with **CUDA 12.9** for accelerated model training.

## Current GPU Setup

✅ **Detected Hardware:**
- GPU: NVIDIA GeForce RTX 3060 Ti
- VRAM: 8GB GDDR6
- CUDA Version: 12.9
- Driver Version: 576.52

## Installation Requirements

### Option 1: Conda Environment (Recommended)

```bash
# Create environment with GPU support
conda env create -f environment.yml
conda activate rcs_cnn_lstm_env
```

### Option 2: Pip Installation

```bash
# Install GPU-specific packages
pip install tensorflow-gpu>=2.8.0
pip install onnxruntime-gpu>=1.12.0
pip install -r requirements.txt
```

## GPU Configuration

The system automatically configures GPU acceleration:

### Automatic Setup
```python
# GPU configuration is automatically applied when importing models
from src.models.cnn_lstm import build_cnn_lstm_model
# GPU will be configured automatically
```

### Manual Setup
```python
# Run GPU setup script
python gpu_setup.py
```

### Check GPU Status
```python
from src.utils.gpu_config import print_gpu_summary
gpu_config, cuda_info = print_gpu_summary()
```

## Performance Optimizations

### Batch Size Optimization
- **GPU Training**: Automatically uses batch sizes 64-128
- **Memory Efficient**: 6GB memory limit (leaves 2GB for system)
- **Mixed Precision**: Float16 for better performance

### Training Configuration
```python
# GPU-optimized training parameters
model = build_cnn_lstm_model(
    X_train, y_train, X_val, y_val,
    epochs=50,
    batch_size=64,  # Automatically increased to 64+ on GPU
    dropout_rate=0.3,
    l1_reg=0.01,
    l2_reg=0.01
)
```

## Performance Comparison

### Expected Speedup
- **CNN-LSTM Training**: 3-5x faster than CPU
- **Large Datasets**: Up to 10x faster for complex models
- **ONNX Export**: 2-3x faster conversion

### Memory Usage
- **Model Training**: ~2-4GB VRAM for typical CNN-LSTM
- **Batch Processing**: ~1-2GB additional for large batches
- **Buffer**: 2GB reserved for system stability

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```
   Solution: Reduce batch size or enable memory growth
   - Set batch_size=32 instead of 64
   - Restart Python kernel
   ```

2. **GPU Not Detected**
   ```bash
   # Check NVIDIA drivers
   nvidia-smi
   
   # Reinstall CUDA toolkit
   conda install cudatoolkit=11.8 cudnn=8.6.0
   ```

3. **TensorFlow GPU Not Working**
   ```python
   # Check TensorFlow GPU support
   import tensorflow as tf
   print(tf.config.list_physical_devices('GPU'))
   print(tf.test.is_built_with_cuda())
   ```

### Performance Issues

1. **Low GPU Utilization**
   - Increase batch size (64-128)
   - Enable mixed precision
   - Check data loading bottlenecks

2. **Slow Training**
   - Verify GPU is being used: `nvidia-smi`
   - Check memory growth settings
   - Monitor CPU-GPU data transfer

## Monitoring GPU Usage

### During Training
```bash
# Monitor GPU usage in real-time
watch -n 1 nvidia-smi

# Or use the integrated monitoring
python -c "from src.utils.gpu_config import print_gpu_summary; print_gpu_summary()"
```

### Memory Management
```python
# Check memory usage
import tensorflow as tf
print(tf.config.experimental.get_memory_info('GPU:0'))
```

## Advanced Configuration

### Custom Memory Limits
```python
# Set specific memory limit (in MB)
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6144)]
    )
```

### Mixed Precision Training
```python
# Enable mixed precision for RTX 30 series
from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy('mixed_float16')
```

## Optimal Settings for RTX 3060 Ti

### Training Parameters
```python
optimal_config = {
    'batch_size': 64,          # Good balance for 8GB VRAM
    'epochs': 50,              # Standard training length
    'learning_rate': 0.001,    # Works well with mixed precision
    'dropout_rate': 0.3,       # Regularization
    'memory_limit': 6144,      # 6GB limit (MB)
    'mixed_precision': True,   # Enable for RTX 30 series
    'memory_growth': True      # Prevent OOM errors
}
```

### Model Architecture
- **CNN Layers**: 32-128 filters work well
- **LSTM Units**: 25-100 units optimal for GPU
- **Dense Layers**: 10-50 units for output layers
- **Regularization**: L1/L2 + Dropout for generalization

## Performance Metrics

### Expected Training Times (RTX 3060 Ti)
- **Small Model** (1000 samples): ~30 seconds/epoch
- **Medium Model** (5000 samples): ~2 minutes/epoch  
- **Large Model** (10000+ samples): ~5 minutes/epoch

### Memory Usage Guidelines
- **Batch 32**: ~1.5GB VRAM
- **Batch 64**: ~2.5GB VRAM
- **Batch 128**: ~4GB VRAM
- **Reserve**: 2GB for system/other processes

## Verification

Run the verification script to ensure everything is working:

```bash
python gpu_setup.py
```

Expected output:
```
🎮 GPU acceleration enabled for CNN-LSTM training
✅ GPU Available: True
✅ Mixed Precision: True
✅ Memory Growth: True
🚀 GPU speedup: 4.2x faster than CPU
```

## Additional Resources

- [TensorFlow GPU Guide](https://www.tensorflow.org/guide/gpu)
- [CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
- [Mixed Precision Training](https://www.tensorflow.org/guide/mixed_precision)

Your RTX 3060 Ti is excellent for deep learning and should provide significant speedup for CNN-LSTM model training!