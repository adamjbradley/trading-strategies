"""
Reload Modules

This script provides a function to reload all relevant modules to ensure changes take effect.
"""

import sys
import importlib
from src.utils.reload_modules import reload_modules

def reload_modules():
    """
    Reload all relevant modules to ensure changes take effect.
    
    Returns:
    --------
    dict
        Dictionary of module names and their reload status
    """
    modules_to_reload = [
        'model_utils',
        'model_training_utils',
        'build_cnn_lstm_model',
        'direct_train_with_best_features',
        'train_with_best_features',
        'feature_set_utils',
        'feature_importance_utils',
        'shap_utils',
        'input_shape_handler'
    ]
    
    results = {}
    
    for module_name in modules_to_reload:
        try:
            if module_name in sys.modules:
                importlib.reload(sys.modules[module_name])
                print(f"✅ Reloaded {module_name}")
                results[module_name] = "reloaded"
            else:
                try:
                    # Try to import the module
                    importlib.import_module(module_name)
                    print(f"✅ Imported {module_name}")
                    results[module_name] = "imported"
                except ImportError:
                    print(f"⚠️ Module {module_name} not found")
                    results[module_name] = "not found"
        except Exception as e:
            print(f"❌ Error reloading {module_name}: {e}")
            results[module_name] = f"error: {str(e)}"
    
    return results

def notebook_code_snippet():
    """
    Print a code snippet to use in the notebook.
    """
    code = """
# --- Reload all modules to ensure changes take effect ---
from src.utils.reload_modules import reload_modules

# Reload modules
reload_results = reload_modules()
print("Module reload results:", reload_results)
"""
    print("Copy and paste this code into your notebook:")
    print(code)

if __name__ == "__main__":
    # Reload modules
    results = reload_modules()
    
    # Print results
    print("\nModule reload results:")
    for module_name, status in results.items():
        print(f"  - {module_name}: {status}")
    
    # Print notebook code snippet
    print("\n")
    notebook_code_snippet()
