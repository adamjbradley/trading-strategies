"""
Fix Notebook

This script provides a function to fix the CNN-LSTM model in the notebook.
"""

import sys
import importlib
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, BatchNormalization

def fix_model_builder_func(X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    """
    Fixed model builder function that adapts kernel size based on input shape.
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training features
    y_train : numpy.ndarray
        Training targets
    X_val : numpy.ndarray
        Validation features
    y_val : numpy.ndarray
        Validation targets
    epochs : int, default=50
        Number of epochs to train for
    batch_size : int, default=32
        Batch size for training
        
    Returns:
    --------
    keras.Model
        Trained model
    """
    # Get input shape
    input_shape = X_train.shape[1:]
    
    # Adapt kernel size based on sequence length (first dimension of input_shape)
    sequence_length = input_shape[0]
    kernel_size = min(3, sequence_length)
    
    # Print diagnostic information
    print(f"Input shape: {input_shape}")
    print(f"Sequence length: {sequence_length}")
    print(f"Using kernel size: {kernel_size}")
    
    # Build model with adaptive kernel size
    model = Sequential([
        # CNN layers with adaptive kernel size
        Conv1D(filters=64, kernel_size=kernel_size, padding='same', activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.2),
        
        # Add second Conv1D layer only if sequence length allows
        Conv1D(filters=32, kernel_size=kernel_size if sequence_length > 1 else 1, padding='same', activation='relu') if sequence_length > 0 else None,
        BatchNormalization() if sequence_length > 0 else None,
        Dropout(0.2) if sequence_length > 0 else None,
        
        # LSTM layers
        LSTM(units=50, return_sequences=False),
        BatchNormalization(),
        Dropout(0.2),
        
        # Output layer
        Dense(1, activation='sigmoid')
    ])
    
    # Remove None layers (if any)
    model = Sequential([layer for layer in model.layers if layer is not None])
    
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Print model summary
    print("Model summary:")
    model.summary()
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val) if X_val is not None and y_val is not None else None,
        verbose=1
    )
    
    return model

def train_cnn_lstm_model_fixed(X_train, y_train, X_val=None, y_val=None, input_shape=None, epochs=10, batch_size=32):
    """
    Fixed version of train_cnn_lstm_model that adapts kernel size based on input shape.
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training features
    y_train : numpy.ndarray
        Training targets
    X_val : numpy.ndarray, optional
        Validation features
    y_val : numpy.ndarray, optional
        Validation targets
    input_shape : tuple, optional
        Input shape (timesteps, features)
    epochs : int, default=10
        Number of epochs to train for
    batch_size : int, default=32
        Batch size for training
        
    Returns:
    --------
    tuple
        (model, history)
    """
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Dropout, BatchNormalization
    
    if input_shape is None:
        input_shape = X_train.shape[1:]
    
    # Adapt kernel size based on sequence length (first dimension of input_shape)
    sequence_length = input_shape[0]
    kernel_size = min(3, sequence_length)
    
    # Print diagnostic information
    print(f"Input shape: {input_shape}")
    print(f"Sequence length: {sequence_length}")
    print(f"Using kernel size: {kernel_size}")
    
    # Build model with adaptive kernel size
    input_layer = Input(shape=input_shape)
    
    # First Conv1D layer with adaptive kernel size
    x = Conv1D(filters=64, kernel_size=kernel_size, padding='same', activation='relu')(input_layer)
    x = BatchNormalization()(x)
    
    # Second Conv1D layer with adaptive kernel size (only if sequence length allows)
    if sequence_length > 1:
        x = Conv1D(filters=32, kernel_size=kernel_size, padding='same', activation='relu')(x)
    else:
        # For sequence length of 1, use kernel_size=1
        x = Conv1D(filters=32, kernel_size=1, padding='same', activation='relu')(x)
    
    x = LSTM(64, return_sequences=False)(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Print model summary
    print("Model summary:")
    model.summary()
    
    # Train the model
    if X_val is not None and y_val is not None:
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=2
        )
    else:
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=2
        )
    
    return model, history

def fix_model_utils():
    """
    Fix the model_utils module by monkey patching the build_cnn_lstm_model function.
    """
    try:
        import model_utils
        
        # Store the original function
        original_build_cnn_lstm_model = model_utils.build_cnn_lstm_model
        original_train_cnn_lstm_model = model_utils.train_cnn_lstm_model
        
        # Define the fixed function
        def fixed_build_cnn_lstm_model(input_shape):
            from tensorflow.keras.models import Model
            from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Dropout, BatchNormalization
            
            # Adapt kernel size based on sequence length (first dimension of input_shape)
            sequence_length = input_shape[0]
            kernel_size = min(3, sequence_length)
            
            # Print diagnostic information
            print(f"Input shape: {input_shape}")
            print(f"Sequence length: {sequence_length}")
            print(f"Using kernel size: {kernel_size}")
            
            # Build model with adaptive kernel size
            input_layer = Input(shape=input_shape)
            
            # First Conv1D layer with adaptive kernel size
            x = Conv1D(filters=64, kernel_size=kernel_size, padding='same', activation='relu')(input_layer)
            x = BatchNormalization()(x)
            
            # Second Conv1D layer with adaptive kernel size (only if sequence length allows)
            if sequence_length > 1:
                x = Conv1D(filters=32, kernel_size=kernel_size, padding='same', activation='relu')(x)
            else:
                # For sequence length of 1, use kernel_size=1
                x = Conv1D(filters=32, kernel_size=1, padding='same', activation='relu')(x)
            
            x = LSTM(64, return_sequences=False)(x)
            x = Dropout(0.3)(x)
            output = Dense(1, activation='sigmoid')(x)
            model = Model(inputs=input_layer, outputs=output)
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            return model
        
        # Replace the original function with the fixed one
        model_utils.build_cnn_lstm_model = fixed_build_cnn_lstm_model
        model_utils.train_cnn_lstm_model = train_cnn_lstm_model_fixed
        
        print("✅ Successfully patched model_utils.build_cnn_lstm_model")
        print("✅ Successfully patched model_utils.train_cnn_lstm_model")
        
        return True
    except ImportError:
        print("❌ Could not import model_utils module")
        return False
    except Exception as e:
        print(f"❌ Error patching model_utils: {e}")
        return False

def notebook_code_snippet():
    """
    Print a code snippet to use in the notebook.
    """
    code = """
# --- Fix the model_utils module ---
from fix_notebook import fix_model_utils, fix_model_builder_func

# Patch the model_utils module
fix_model_utils()

# Define a fixed model builder function
def model_builder_func(X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    return fix_model_builder_func(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=epochs,
        batch_size=batch_size
    )

# Train the model with the best features
model, feature_names, X_train, y_train, X_val, y_val, X_test, y_test = train_model_with_best_features(
    symbol=symbol_to_predict,
    data=data,
    model_builder_func=model_builder_func,
    use_saved_features=True,  # Use the features we just saved
    epochs=50,
    batch_size=32
)

# Evaluate the model
metrics = evaluate_model(model, X_test, y_test)
"""
    print("Copy and paste this code into your notebook:")
    print(code)

def reload_modules():
    """
    Reload all relevant modules to ensure the fixes take effect.
    """
    modules_to_reload = [
        'model_utils',
        'model_training_utils',
        'build_cnn_lstm_model',
        'direct_train_with_best_features',
        'train_with_best_features'
    ]
    
    for module_name in modules_to_reload:
        try:
            if module_name in sys.modules:
                importlib.reload(sys.modules[module_name])
                print(f"✅ Reloaded {module_name}")
            else:
                print(f"⚠️ Module {module_name} not loaded yet")
        except Exception as e:
            print(f"❌ Error reloading {module_name}: {e}")

if __name__ == "__main__":
    # Fix the model_utils module
    fix_model_utils()
    
    # Reload modules
    reload_modules()
    
    # Print notebook code snippet
    notebook_code_snippet()
