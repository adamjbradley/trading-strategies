import os
import shutil
from pathlib import Path

def create_structure(base):
    """Create the new directory structure"""
    dirs = [
        "src/data", "src/features", "src/models", "src/utils", "src/export",
        "notebooks", "tests", "config", "data/raw", "models/saved", "logs"
    ]
    for d in dirs:
        Path(os.path.join(base, d)).mkdir(parents=True, exist_ok=True)
        if d.startswith("src/"):
            Path(os.path.join(base, d, "__init__.py")).touch()
    print("✅ Directory structure created.")

def move_files(base):
    """Move files to their new locations"""
    moves = {
        # Data-related files
        "compare_broker_data.py": "src/data/broker_comparison.py",
        "test_async_data_loader.py": "src/data/test_loader.py",
        "notebook_data_loader_patch.py": "src/data/loader_patch.py",
        "async_data_loader.py": "src/data/loader.py",
        
        # Feature-related files
        "feature_filtering.py": "src/features/filtering.py",
        "feature_set_analysis.py": "src/features/analysis.py",
        "define_core_features.py": "src/features/core.py",
        "autofeatures.py": "src/features/auto.py",
        "fixed_permutation_importance.py": "src/features/permutation.py",
        "feature_engineering.py": "src/features/engineering.py",
        "feature_set_selection_fix.py": "src/features/selection_fix.py",
        
        # Model-related files
        "model_utils.py": "src/models/utils.py",
        "train_with_best_features.py": "src/models/training.py",
        "direct_train_with_best_features.py": "src/models/direct_training.py",
        "inference.py": "src/models/inference.py",
        "rcs_cnn_lstm_pipeline.py": "src/models/pipeline.py",
        "optuna_tuner.py": "src/models/tuner.py",
        
        # Utility files
        "mlflow_utils.py": "src/utils/mlflow.py",
        "manifest_utils.py": "src/utils/manifest.py",
        "rolling_labeling.py": "src/utils/labeling.py",
        "backtester.py": "src/utils/backtesting.py",
        "explainability.py": "src/utils/explainability.py",
        "reload_modules.py": "src/utils/reload.py",
        
        # Export-related files
        "onnx_benchmark.py": "src/export/benchmark.py",
        "train_and_export_onnx.py": "src/export/onnx.py",
        "model_export_utils.py": "src/export/utils.py",
        
        # Application files
        "app.py": "src/app.py",
        
        # Configuration files
        "environment.yml": "environment.yml",
        "requirements_colab.txt": "requirements_colab.txt",
        "config.yaml": "config/config.yaml",
        "batch_config.yaml": "config/batch_config.yaml",
        
        # Notebook-related files
        "notebook_combined_fix.py": "notebooks/combined_fix.py",
        "notebook_cell_fixed.py": "notebooks/cell_fixed.py",
        "notebook_target_fix.py": "notebooks/target_fix.py",
        "notebook_cell_with_best_features.py": "notebooks/cell_with_features.py",
        "notebook_cell_simple.py": "notebooks/cell_simple.py",
        "notebook_cell_final.py": "notebooks/cell_final.py",
        "notebook_cell.py": "notebooks/cell.py",
        "RCS_CNN_LSTM.ipynb": "notebooks/RCS_CNN_LSTM.ipynb",
        "fix_notebook.py": "notebooks/fix.py",
    }
    
    # Move Python files
    for src, dst in moves.items():
        src_path = os.path.join(base, src)
        dst_path = os.path.join(base, dst)
        if os.path.exists(src_path):
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.move(src_path, dst_path)
            print(f"✅ Moved {src} to {dst}")
    
    # Move model files
    model_files = [f for f in os.listdir(base) if f.endswith(('.h5', '.onnx'))]
    for model_file in model_files:
        src_path = os.path.join(base, model_file)
        dst_path = os.path.join(base, "models/saved", model_file)
        shutil.move(src_path, dst_path)
        print(f"✅ Moved {model_file} to models/saved/")
    
    # Move feature set files
    feature_files = [f for f in os.listdir(base) if f.startswith(('best_feature_set_', 'feature_set_results_'))]
    for feature_file in feature_files:
        src_path = os.path.join(base, feature_file)
        dst_path = os.path.join(base, "data/raw", feature_file)
        shutil.move(src_path, dst_path)
        print(f"✅ Moved {feature_file} to data/raw/")
    
    # Move image files
    image_files = [f for f in os.listdir(base) if f.endswith(('.png', '.jpg', '.jpeg'))]
    for image_file in image_files:
        src_path = os.path.join(base, image_file)
        dst_path = os.path.join(base, "data/raw", image_file)
        shutil.move(src_path, dst_path)
        print(f"✅ Moved {image_file} to data/raw/")

def cleanup(base):
    """Remove temporary files and directories"""
    # Remove __pycache__ directories
    for root, dirs, files in os.walk(base):
        if "__pycache__" in dirs:
            shutil.rmtree(os.path.join(root, "__pycache__"))
            print(f"✅ Removed __pycache__ from {root}")
    
    # Remove backup_files directory if it exists
    backup_dir = os.path.join(base, "backup_files")
    if os.path.exists(backup_dir):
        shutil.rmtree(backup_dir)
        print("✅ Removed backup_files directory")

def update_imports(base):
    """Update import statements in Python files"""
    import_map = {
        "async_data_loader": "src.data.loader",
        "feature_engineering": "src.features.engineering",
        "feature_set_utils": "src.features.selection",
        "feature_importance_utils": "src.features.importance",
        "shap_utils": "src.features.explainability",
        "build_cnn_lstm_model": "src.models.cnn_lstm",
        "model_training_utils": "src.models.training",
        "model_export_utils": "src.export.utils",
        "input_shape_handler": "src.utils.shape_handler",
        "signal_logger": "src.utils.logging",
        "backtest": "src.utils.backtesting",
        "train_and_export_onnx": "src.export.onnx",
        "migrate_metrics": "src.export.metrics",
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
    print("✅ Updated imports in Python files")

def main():
    base = "RCS_CNN_LSTM_Notebook/RCS_CNN_LSTM_Notebook"
    print("Starting cleanup and reorganization...")
    
    # Create new structure
    create_structure(base)
    
    # Move files
    move_files(base)
    
    # Clean up temporary files
    cleanup(base)
    
    # Update imports
    update_imports(base)
    
    print("🎉 Cleanup and reorganization complete!")

if __name__ == "__main__":
    main() 