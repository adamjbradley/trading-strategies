import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Dropout, BatchNormalization

def create_rolling_windows(features_scaled, lookback):
    X = np.array([features_scaled[i-lookback:i] for i in range(lookback, len(features_scaled))])
    return X

def train_test_split_rolling(X, y, test_size=0.2):
    split = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    return X_train, X_test, y_train, y_test

def build_cnn_lstm_model(input_shape):
    input_layer = Input(shape=input_shape)
    
    # Adapt kernel size based on sequence length (first dimension of input_shape)
    sequence_length = input_shape[0]
    kernel_size = min(3, sequence_length)
    
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

def train_cnn_lstm_model(X_train, y_train, X_val=None, y_val=None, input_shape=None, epochs=10, batch_size=32):
    if input_shape is None:
        input_shape = X_train.shape[1:]
    model = build_cnn_lstm_model(input_shape)
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

def evaluate_cnn_lstm_model(model, X_test, y_test, threshold=0.5):
    """
    Evaluate a trained CNN+LSTM model, return accuracy and classification report.
    """
    y_pred = (model.predict(X_test) > threshold).astype(int).flatten()
    min_len = min(len(y_test), len(y_pred))
    y_test_aligned = y_test[:min_len]
    y_pred_aligned = y_pred[:min_len]
    acc = accuracy_score(y_test_aligned, y_pred_aligned)
    report = classification_report(y_test_aligned, y_pred_aligned, output_dict=True)
    return acc, report
