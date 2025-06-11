import json
import os

def fix_notebook_imports(notebook_path):
    """Fix import paths in the notebook."""
    # Check if file is empty
    if os.path.getsize(notebook_path) == 0:
        print(f"⚠️ Skipping empty file: {notebook_path}")
        return
        
    with open(notebook_path, 'r', encoding='utf-8') as f:
        try:
            notebook = json.load(f)
        except json.JSONDecodeError as e:
            print(f"⚠️ Error reading {notebook_path}: {str(e)}")
            return
    
    # Import path mappings
    import_mappings = {
        'from async_data_loader import': 'from src.data.loader import',
        'from feature_engineering import': 'from src.features.engineering import',
        'from feature_set_utils import': 'from src.features.selection import',
        'from feature_importance_utils import': 'from src.features.importance import',
        'from shap_utils import': 'from src.features.explainability import',
        'from build_cnn_lstm_model import': 'from src.models.cnn_lstm import',
        'from model_training_utils import': 'from src.models.training import',
        'from model_export_utils import': 'from src.export.utils import',
        'from input_shape_handler import': 'from src.utils.shape_handler import',
        'from signal_logger import': 'from src.utils.logging import',
        'from backtest import': 'from src.utils.backtesting import',
        'from train_and_export_onnx import': 'from src.export.onnx import',
        'from migrate_metrics import': 'from src.export.metrics import',
        'from model_utils import': 'from src.models.utils import',
        'from feature_set_analysis import': 'from src.features.analysis import',
        'from feature_engineering import': 'from src.features.engineering import',
        'from data_loader import': 'from src.data.loader import',
        'from feature_selection import': 'from src.features.selection import',
        'from model_training import': 'from src.models.training import',
        'from model_evaluation import': 'from src.models.evaluation import',
        'from feature_analysis import': 'from src.features.analysis import',
        'from data_preprocessing import': 'from src.data.preprocessing import',
        'from model_export import': 'from src.export.utils import',
        'from utils import': 'from src.utils.core import',
        'from feature_importance import': 'from src.features.importance import',
        'from model_metrics import': 'from src.models.metrics import',
        'from data_utils import': 'from src.data.utils import',
        'from feature_utils import': 'from src.features.utils import',
        'from model_utils import': 'from src.models.utils import',
        'from export_utils import': 'from src.export.utils import',
        'from utils_common import': 'from src.utils.common import',
        'from utils_data import': 'from src.utils.data import',
        'from utils_features import': 'from src.utils.features import',
        'from utils_models import': 'from src.utils.models import',
        'from utils_export import': 'from src.utils.export import',
        'from utils_analysis import': 'from src.utils.analysis import',
        'from utils_visualization import': 'from src.utils.visualization import',
        'from utils_evaluation import': 'from src.utils.evaluation import',
        'from utils_metrics import': 'from src.utils.metrics import',
        'from utils_preprocessing import': 'from src.utils.preprocessing import',
        'from utils_training import': 'from src.utils.training import',
        'from utils_testing import': 'from src.utils.testing import',
        'from utils_validation import': 'from src.utils.validation import',
        'from utils_optimization import': 'from src.utils.optimization import',
        'from utils_hyperparameters import': 'from src.utils.hyperparameters import',
        'from utils_configuration import': 'from src.utils.configuration import',
        'from utils_logging import': 'from src.utils.logging import',
        'from utils_monitoring import': 'from src.utils.monitoring import',
        'from utils_debugging import': 'from src.utils.debugging import',
        'from utils_profiling import': 'from src.utils.profiling import',
    }
    
    # Update each cell
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            for old, new in import_mappings.items():
                if old in source:
                    cell['source'] = [line.replace(old, new) for line in cell['source']]
    
    # Save the updated notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"✅ Updated imports in {notebook_path}")

if __name__ == "__main__":
    # Fix all notebooks in the current directory
    notebook_files = [
        "RCS_CNN_LSTM.ipynb",
        "RCS_CNN_LSTM_Pipeline.ipynb",
        "RCS_CNN_LSTM_Fixed.ipynb"
    ]
    
    for notebook_file in notebook_files:
        if os.path.exists(notebook_file):
            fix_notebook_imports(notebook_file)
        else:
            print(f"⚠️ Notebook not found: {notebook_file}") 