#!/usr/bin/env python3
"""
Apply ONNX-Only Fix Script

This script applies the ONNX-only fix to replace the problematic H5 fallback
implementation in the Advanced_Hyperparameter_Optimization_Clean.ipynb notebook.

Execute this after running the notebook to apply the SCRATCHPAD requirements:
1. Remove ALL H5 fallback functionality
2. Make LSTM layers ONNX-compatible (solve CudnnRNNV3 issue) 
3. System should FAIL if ONNX export fails (no fallback)
"""

import sys
import os
from pathlib import Path

# Add current directory to path to import the fix
sys.path.append(str(Path(__file__).parent))

try:
    from onnx_only_fix import apply_onnx_only_fix
    print("✅ ONNX-only fix module imported successfully")
except ImportError as e:
    print(f"❌ Failed to import ONNX-only fix: {e}")
    sys.exit(1)


def main():
    """Main function to apply the ONNX-only fix"""
    print("🚀 APPLYING ONNX-ONLY FIX")
    print("="*50)
    print("This will replace the H5 fallback implementation with ONNX-only export")
    print("as requested in SCRATCHPAD.md")
    print("")
    
    # Instructions for manual application
    print("📋 MANUAL APPLICATION INSTRUCTIONS:")
    print("")
    print("1. In your notebook, run this code in a new cell:")
    print("   ```python")
    print("   exec(open('apply_onnx_fix.py').read())")
    print("   ```")
    print("")
    print("2. Or import and apply directly:")
    print("   ```python")
    print("   from onnx_only_fix import apply_onnx_only_fix")
    print("   apply_onnx_only_fix(optimizer)")
    print("   ```")
    print("")
    print("3. This will replace the problematic Cell 8 functionality with:")
    print("   ✅ ONNX-only export (no H5 fallback)")
    print("   ✅ ONNX-compatible LSTM layers (implementation=1)")
    print("   ✅ System fails if ONNX export impossible")
    print("   ✅ GPU performance maintained")
    print("")
    
    # Check if we're being run from a notebook context
    try:
        # Try to detect if we're in a Jupyter environment
        get_ipython
        print("🔬 Jupyter environment detected!")
        
        # Check if optimizer exists in the global namespace
        if 'optimizer' in globals():
            print("🎯 Optimizer instance found, applying fix...")
            apply_onnx_only_fix(globals()['optimizer'])
            print("✅ ONNX-only fix applied to optimizer instance!")
        else:
            print("⚠️  Optimizer instance not found in global namespace")
            print("   Make sure to run this after initializing the optimizer")
            
    except NameError:
        print("💻 Running as standalone script")
        print("   Execute the manual instructions above in your notebook")
    
    print("\n🎯 EXPECTED RESULTS AFTER APPLYING FIX:")
    print("   • Only .onnx files will be created in exported_models/")
    print("   • No .h5 files will be created under any circumstances")
    print("   • System will fail with clear error if ONNX conversion impossible")
    print("   • LSTM layers will work with tf2onnx (no more CudnnRNNV3 error)")
    print("   • GPU performance maintained during training and inference")
    print("")
    print("✅ ONNX-only fix ready for application!")


if __name__ == "__main__":
    main()


# Auto-apply if optimizer is available
try:
    # This will work if the script is exec()'d from the notebook
    if 'optimizer' in locals() or 'optimizer' in globals():
        optimizer_instance = locals().get('optimizer') or globals().get('optimizer')
        if optimizer_instance is not None:
            print("\n🔧 Auto-applying ONNX-only fix...")
            apply_onnx_only_fix(optimizer_instance)
            print("✅ ONNX-only fix auto-applied successfully!")
except Exception as e:
    print(f"ℹ️  Auto-apply not possible: {e}")
    print("   Use manual application instructions above")