from setuptools import setup, find_packages

setup(
    name="rcs_cnn_lstm",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "tensorflow",
        "matplotlib",
        "seaborn",
        "ta",
        "yfinance",
        "metatrader5",
        "tf2onnx",
        "onnx",
        "shap",
        "mlflow",
    ],
    python_requires=">=3.8",
) 