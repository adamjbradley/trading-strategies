import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization

def build_cnn_lstm_model(X_train, y_train, X_val, y_val, epochs=50, batch_size=32, **kwargs):
    """
    Build and train a CNN-LSTM model.
    
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
        Number of epochs to train
    batch_size : int, default=32
        Batch size for training
    **kwargs : dict
        Additional keyword arguments
        
    Returns:
    --------
    keras.Model
        Trained model
    """
    # Get input shape
    input_shape = X_train.shape[1:]
    
    # Build model based on input shape
    model = Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape))
    
    # Add Conv1D layer
    model.add(Conv1D(filters=64, kernel_size=1, padding='same', activation='relu'))
    
    # Add MaxPooling1D layer only if the sequence length is sufficient
    if input_shape[0] >= 2:
        model.add(MaxPooling1D(pool_size=2))
    
    # Add LSTM layer
    model.add(LSTM(50, return_sequences=False))
    
    # Add Dense layers
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    print("Model summary:")
    model.summary()
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        verbose=1
    )
    
    return model
