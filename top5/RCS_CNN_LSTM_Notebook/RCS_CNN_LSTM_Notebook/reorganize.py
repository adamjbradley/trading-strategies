import os
import shutil
from pathlib import Path

# 1. Backup
def backup():
    if not os.path.exists("backup"):
        os.makedirs("backup")
    if not os.path.exists("backup/RCS_CNN_LSTM_Notebook"):
        shutil.copytree("RCS_CNN_LSTM_Notebook", "backup/RCS_CNN_LSTM_Notebook")
    print("✅ Backup complete.")

# 2. Create new structure
def create_structure(base):
    dirs = [
        "src/data", "src/features", "src/models", "src/utils", "src/export",
        "notebooks", "tests", "config", "data/raw", "models/saved", "logs"
    ]
    for d in dirs:
        Path(os.path.join(base, d)).mkdir(parents=True, exist_ok=True)
        if d.startswith("src/"):
            Path(os.path.join(base, d, "__init__.py")).touch()
    print("✅ Directory structure created.")

# 3. Move files
def move_files(base):
    moves = {
        # Data
        "mt5_downloader.py": "src/data/downloader.py",
        "download_data.py": "src/data/downloader.py",
        "async_data_loader.py": "src/data/loader.py",
        "feature_engineering.py": "src/data/preprocessing.py",
        # Features
        "feature_set_utils.py": "src/features/selection.py",
        "feature_importance_utils.py": "src/features/importance.py",
        "shap_utils.py": "src/features/explainability.py",
        # Models
        "build_cnn_lstm_model.py": "src/models/cnn_lstm.py",
        "model_training_utils.py": "src/models/training.py",
        "model_export_utils.py": "src/models/evaluation.py",
        # Utils
        "input_shape_handler.py": "src/utils/shape_handler.py",
        "signal_logger.py": "src/utils/logging.py",
        "backtest.py": "src/utils/backtesting.py",
        # Export
        "train_and_export_onnx.py": "src/export/onnx.py",
        "migrate_metrics.py": "src/export/metrics.py",
        # Notebooks and configs
        "RCS_CNN_LSTM.ipynb": "notebooks/RCS_CNN_LSTM.ipynb",
        "RCS_CNN_LSTM_Pipeline.ipynb": "notebooks/RCS_CNN_LSTM_Pipeline.ipynb",
        "config.yaml": "config/config.yaml",
        "batch_config.yaml": "config/batch_config.yaml",
        # Root docs
        "README.md": "../../README.md",
        "requirements.txt": "../../requirements.txt",
    }
    for src, dst in moves.items():
        src_path = os.path.join(base, src)
        dst_path = os.path.join(base, dst)
        if os.path.exists(src_path):
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.move(src_path, dst_path)
    print("✅ Files moved.")

# 4. Update imports in all .py files under src/
def update_imports(base):
    import_map = {
        "async_data_loader": "src.data.loader",
        "feature_engineering": "src.data.preprocessing",
        "feature_set_utils": "src.features.selection",
        "feature_importance_utils": "src.features.importance",
        "shap_utils": "src.features.explainability",
        "build_cnn_lstm_model": "src.models.cnn_lstm",
        "model_training_utils": "src.models.training",
        "model_export_utils": "src.models.evaluation",
        "input_shape_handler": "src.utils.shape_handler",
        "signal_logger": "src.utils.logging",
        "backtest": "src.utils.backtesting",
        "train_and_export_onnx": "src.export.onnx",
        "migrate_metrics": "src.export.metrics",
        # Add more as needed
    }
    for root, _, files in os.walk(os.path.join(base, "src")):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                for old, new in import_map.items():
                    content = content.replace(f"import {old}", f"import {new}")
                    content = content.replace(f"from {old} import", f"from {new} import")
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
    print("✅ Imports updated.")

def main():
    base = "RCS_CNN_LSTM_Notebook/RCS_CNN_LSTM_Notebook"
    print("Starting reorganization...")
    backup()
    create_structure(base)
    move_files(base)
    update_imports(base)
    print("🎉 Reorganization complete! Please test your new structure.")

if __name__ == "__main__":
    main()