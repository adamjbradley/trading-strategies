import argparse
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import onnxruntime as ort
import joblib

def load_data(input_file, lookback=20):
    df = pd.read_csv(input_file, index_col=0)
    scaled = joblib.load("scaler.pkl").transform(df)
    X = np.array([scaled[i-lookback:i] for i in range(lookback, len(scaled))])
    return X

def predict_with_keras(model_path, X):
    model = load_model(model_path)
    preds = model.predict(X)
    return preds

def predict_with_onnx(model_path, X):
    sess = ort.InferenceSession(model_path)
    input_name = sess.get_inputs()[0].name
    preds = sess.run(None, {input_name: X.astype(np.float32)})[0]
    return preds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="CSV input feature file", required=True)
    parser.add_argument("--model", help="Path to .h5 or .onnx model", required=True)
    parser.add_argument("--onnx", action="store_true", help="Use ONNX model for inference")
    args = parser.parse_args()

    X = load_data(args.input)
    preds = predict_with_onnx(args.model, X) if args.onnx else predict_with_keras(args.model, X)
    print("Predictions:", preds.flatten())

if __name__ == "__main__":
    main()
