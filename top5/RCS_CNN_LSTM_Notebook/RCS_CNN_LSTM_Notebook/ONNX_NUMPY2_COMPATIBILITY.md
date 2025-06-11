# ONNX Export NumPy 2.0 Compatibility

This document explains the improvements made to ensure ONNX export functionality works with NumPy 2.0 and provides guidance for troubleshooting.

## Overview

NumPy 2.0 introduced several breaking changes that can affect ONNX model export and inference. This project includes comprehensive fixes to ensure compatibility.

## Key Improvements

### 1. Enhanced ONNX Export (`src/export/onnx_numpy2_fix.py`)

- **Multiple Conversion Methods**: Implements 4 different conversion strategies with automatic fallback
- **NumPy Compatibility Layer**: Handles array operations safely across NumPy versions
- **Model Validation**: Comprehensive validation and verification of exported models
- **Error Handling**: Graceful degradation when ONNX export fails

### 2. Conversion Strategies

1. **Direct Keras Conversion**: Standard tf2onnx conversion from Keras model
2. **Concrete Function**: Conversion via TensorFlow concrete function
3. **SavedModel**: Most robust method using temporary SavedModel
4. **Simplified Model**: Fallback using functional model reconstruction

### 3. NumPy 2.0 Specific Fixes

- **Array Copying**: Explicit array copying for NumPy 2.0 compatibility
- **Data Type Handling**: Proper tensor data serialization
- **Warning Suppression**: Filters NumPy 2.0 deprecation warnings
- **Protocol Buffer Safety**: Safe model serialization

## Usage

### Basic Export

```python
from src.export.onnx_numpy2_fix import export_model_to_onnx_numpy2_safe

# Export with automatic compatibility handling
success = export_model_to_onnx_numpy2_safe(
    model=your_keras_model,
    output_path="model.onnx",
    input_shape=(timesteps, features),
    test_input=sample_data,
    model_name="your_model"
)
```

### Integration with Training Pipeline

The main ONNX export (`src/export/onnx.py`) automatically uses the NumPy 2.0 compatible functions with fallback to standard methods.

## Testing

Run the compatibility test:

```bash
python test_onnx_export.py
```

This tests:
- NumPy compatibility functions
- ONNX export with different conversion methods
- Model validation and verification

## Troubleshooting

### Common Issues

1. **NumPy Version Conflicts**
   - Ensure NumPy version is within supported range (>=1.21.0,<3.0.0)
   - Update tf2onnx to version >=1.13.0

2. **ONNX Runtime Issues**
   - Update onnxruntime to >=1.12.0
   - Some models may require specific opset versions

3. **TensorFlow Compatibility**
   - Ensure TensorFlow version is >=2.8.0,<3.0.0
   - Some layer types may not be fully supported in ONNX

### Fallback Behavior

If ONNX export fails:
1. The system will try multiple conversion methods
2. If all methods fail, the model is still saved in HDF5 format
3. The system continues operation without ONNX models
4. Error messages provide specific guidance

## Version Requirements

```
numpy>=1.21.0,<3.0.0
onnxruntime>=1.12.0
onnx>=1.12.0
tf2onnx>=1.13.0
tensorflow>=2.8.0,<3.0.0
```

## Architecture Considerations

### Model Types Supported

- Sequential models ✅
- Functional models ✅
- Models with custom layers ⚠️ (may require special handling)
- Subclassed models ⚠️ (limited support)

### Layer Compatibility

Fully supported:
- Conv1D, LSTM, Dense
- Dropout, BatchNormalization
- Activation functions
- Pooling layers

Limited support:
- Custom layers
- Lambda layers
- Complex attention mechanisms

## Performance Notes

- SavedModel conversion is most reliable but slower
- Direct conversion is fastest but may fail with complex models
- Model validation adds overhead but ensures reliability
- Test input verification prevents runtime issues

## Future Improvements

1. **Enhanced Custom Layer Support**: Better handling of custom layers
2. **Automatic Opset Selection**: Dynamic opset version selection
3. **Model Optimization**: Post-export model optimization
4. **Quantization Support**: INT8 and FP16 model variants

## Error Codes

- **Import Error**: NumPy 2.0 compatibility module unavailable
- **Conversion Error**: All conversion methods failed
- **Validation Error**: Model structure validation failed
- **Runtime Error**: Model inference test failed

For detailed error information, check the console output during export.