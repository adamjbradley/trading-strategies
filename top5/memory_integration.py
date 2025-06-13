#!/usr/bin/env python3
"""
Memory Monitoring Integration for Trading Strategy Optimization

Simple integration script to add memory monitoring to the existing optimization system.

Usage:
  from memory_integration import enable_memory_monitoring, show_memory_status
  
  # Enable monitoring
  monitored_optimizer = enable_memory_monitoring(optimizer)
  
  # Check memory status
  show_memory_status()

Created: 2025-06-13
Purpose: Optional memory tracking for production resource optimization
"""

from memory_monitor import OptimizationMemoryWrapper, MemoryMonitor


def enable_memory_monitoring(optimizer):
    """
    Enable memory monitoring for the optimizer
    
    Args:
        optimizer: The AdvancedHyperparameterOptimizer instance
        
    Returns:
        OptimizationMemoryWrapper: Wrapped optimizer with monitoring
    """
    print("🔧 Enabling memory monitoring...")
    
    wrapper = OptimizationMemoryWrapper(optimizer, enable_monitoring=True)
    
    print("✅ Memory monitoring enabled!")
    print("📊 Usage will be tracked during optimization runs")
    print("💾 Memory data will be saved to optimization_results/")
    print("")
    print("💡 Memory monitoring functions:")
    print("  • monitored_optimizer.show_current_usage()")
    print("  • monitored_optimizer.get_memory_estimates()")
    print("")
    
    return wrapper


def show_memory_status():
    """Show current memory status and estimates"""
    monitor = MemoryMonitor()
    
    print("📊 MEMORY STATUS REPORT")
    print("="*50)
    
    # Current usage
    usage = monitor.get_current_usage()
    print("🔍 Current System Status:")
    print(f"  • Process Memory: {usage['process_memory_mb']:.1f} MB ({usage['process_memory_percent']:.1f}%)")
    print(f"  • System Memory: {usage['system_memory_used_gb']:.1f} / {monitor.system_memory:.1f} GB ({usage['system_memory_percent']:.1f}%)")
    print(f"  • Available Memory: {usage['system_memory_available_gb']:.1f} GB")
    print(f"  • CPU Usage: {usage['cpu_percent']:.1f}%")
    
    if usage.get('gpu_available'):
        print(f"  • GPU: ✅ Available for monitoring")
    else:
        print(f"  • GPU: ❌ Not available for monitoring")
    
    print("")
    
    # Memory estimates
    estimates = monitor.estimate_memory_requirements()
    print("📈 Optimization Memory Estimates:")
    print(f"  • Current Usage: {estimates['current_usage_mb']:.1f} MB")
    print(f"  • Estimated Peak per Trial: {estimates['estimated_peak_per_trial_mb']:.1f} MB")
    print(f"  • Estimated Total (7 symbols): {estimates['estimated_total_for_optimization_gb']:.1f} GB")
    print(f"  • Expected Utilization: {estimates['estimated_memory_utilization_percent']:.1f}%")
    print(f"  • Recommendation: {estimates['recommendation']}")
    
    print("")
    print("💡 Tips for Memory Optimization:")
    if estimates['estimated_memory_utilization_percent'] > 80:
        print("  🚨 High memory usage expected:")
        print("    - Consider reducing trials per symbol (100 → 50)")
        print("    - Run symbols sequentially instead of batching")
        print("    - Monitor system during optimization")
    elif estimates['estimated_memory_utilization_percent'] > 60:
        print("  ⚠️  Moderate memory usage expected:")
        print("    - Monitor memory usage during optimization")
        print("    - Consider running during off-peak hours")
    else:
        print("  ✅ Memory usage should be manageable")
        print("    - Sufficient memory for full optimization")
    
    return usage, estimates


def quick_memory_check():
    """Quick memory check before starting optimization"""
    monitor = MemoryMonitor()
    usage = monitor.get_current_usage()
    
    # Simple check
    available_gb = usage['system_memory_available_gb']
    
    if available_gb < 2:
        print("❌ LOW MEMORY WARNING!")
        print(f"   Only {available_gb:.1f} GB available")
        print("   Consider freeing memory before optimization")
        return False
    elif available_gb < 4:
        print("⚠️  Memory Warning:")
        print(f"   Only {available_gb:.1f} GB available")
        print("   Monitor usage during optimization")
        return True
    else:
        print(f"✅ Memory OK: {available_gb:.1f} GB available")
        return True


def create_memory_optimized_functions(optimizer):
    """Create memory-optimized versions of common functions"""
    
    def run_quality_test_with_monitoring():
        """Quality test with memory monitoring"""
        print("🧪 QUALITY TEST WITH MEMORY MONITORING")
        print("="*45)
        
        # Check memory before starting
        if not quick_memory_check():
            print("Skipping test due to low memory")
            return None
        
        # Enable monitoring
        monitored_optimizer = enable_memory_monitoring(optimizer)
        
        # Run test
        result = monitored_optimizer.optimize_symbol('EURUSD', n_trials=20)  # Reduced trials for testing
        
        if result:
            print(f"\n✅ Quality test completed with monitoring!")
            print(f"Objective: {result.objective_value:.6f}")
        
        return result
    
    def run_memory_efficient_optimization():
        """Run optimization with memory monitoring and efficiency settings"""
        print("🚀 MEMORY-EFFICIENT OPTIMIZATION")
        print("="*40)
        
        # Check system first
        usage, estimates = show_memory_status()
        
        if estimates['estimated_memory_utilization_percent'] > 85:
            print("\n🚨 High memory usage expected - using conservative settings:")
            trials_per_symbol = 50  # Reduced trials
            print(f"   • Reduced trials: {trials_per_symbol}")
        else:
            trials_per_symbol = 100  # Normal trials
            print(f"\n✅ Normal memory usage expected - using standard settings:")
            print(f"   • Standard trials: {trials_per_symbol}")
        
        # Enable monitoring
        monitored_optimizer = enable_memory_monitoring(optimizer)
        
        # Test symbols (adjust based on memory)
        test_symbols = ['EURUSD', 'GBPUSD'] if estimates['estimated_memory_utilization_percent'] > 85 else ['EURUSD', 'GBPUSD', 'USDJPY']
        
        results = {}
        for symbol in test_symbols:
            print(f"\n🎯 Optimizing {symbol} with memory monitoring...")
            result = monitored_optimizer.optimize_symbol(symbol, n_trials=trials_per_symbol)
            if result:
                results[symbol] = result
                print(f"✅ {symbol}: {result.objective_value:.6f}")
        
        print(f"\n📊 Optimization completed with memory monitoring!")
        print(f"Successful: {len(results)}/{len(test_symbols)} symbols")
        
        return results
    
    return run_quality_test_with_monitoring, run_memory_efficient_optimization


# Convenience functions for notebook integration
def setup_memory_monitoring(optimizer):
    """Complete setup for memory monitoring"""
    print("🔧 SETTING UP MEMORY MONITORING")
    print("="*40)
    
    # Show initial status
    show_memory_status()
    
    # Create memory-optimized functions
    test_func, optimize_func = create_memory_optimized_functions(optimizer)
    
    print("\n✅ Memory monitoring setup complete!")
    print("\n💡 Available functions:")
    print("  • show_memory_status()                    # Check current memory")
    print("  • quick_memory_check()                    # Quick pre-optimization check")
    print("  • test_func()                            # Quality test with monitoring")
    print("  • optimize_func()                        # Memory-efficient optimization")
    
    return {
        'monitor': MemoryMonitor(),
        'test_with_monitoring': test_func,
        'optimize_with_monitoring': optimize_func,
        'show_status': show_memory_status,
        'quick_check': quick_memory_check
    }


if __name__ == "__main__":
    # Demo the integration
    print("🧪 Memory Integration Demo")
    show_memory_status()