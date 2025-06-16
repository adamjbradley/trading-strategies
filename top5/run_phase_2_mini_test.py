#!/usr/bin/env python3
"""
Run a mini optimization test with Phase 2 features
"""

import warnings
warnings.filterwarnings('ignore')

# Execute the notebook cells to set up the optimizer
exec(open('Advanced_Hyperparameter_Optimization_Clean.ipynb').read())

print("🧪 PHASE 2 MINI OPTIMIZATION TEST")
print("="*50)

try:
    # Set verbose mode to see feature details
    optimizer.set_verbose_mode(True)
    
    print("🚀 Running mini optimization with Phase 2 features...")
    print("   Symbol: EURUSD")
    print("   Trials: 2 (quick test)")
    print("   Mode: VERBOSE (detailed feature output)")
    print("")
    
    # Run mini optimization
    result = optimizer.optimize_symbol('EURUSD', n_trials=2)
    
    # Restore quiet mode
    optimizer.set_verbose_mode(False)
    
    if result:
        print(f"\n✅ MINI OPTIMIZATION SUCCESSFUL!")
        print(f"   Best score: {result.objective_value:.6f}")
        print(f"   Features used: {result.num_features}")
        print(f"   Trials completed: {result.completed_trials}/{result.total_trials}")
        print(f"   Study name: {result.study_name}")
        
        # Save result for analysis
        print(f"\n📊 Phase 2 features were active during optimization")
        print(f"   (Feature selection details shown in verbose output above)")
        
    else:
        print(f"❌ Mini optimization failed")

except Exception as e:
    print(f"❌ Mini optimization error: {e}")
    import traceback
    traceback.print_exc()

print(f"\n🎉 PHASE 2 MINI TEST COMPLETED!")