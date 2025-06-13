#!/usr/bin/env python3
"""
Memory Monitor for Trading Strategy Optimization

Simple memory usage tracking during hyperparameter optimization to help with:
1. Production resource planning
2. Memory leak detection
3. GPU memory management
4. System stability monitoring

Created: 2025-06-13
Purpose: Optional production resource tracking
"""

import psutil
import time
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import threading
import os


class MemoryMonitor:
    """Simple memory monitoring for optimization runs"""
    
    def __init__(self, results_path: str = "optimization_results"):
        self.results_path = Path(results_path)
        self.results_path.mkdir(exist_ok=True)
        
        self.monitoring = False
        self.monitor_thread = None
        self.memory_data = []
        self.start_time = None
        self.symbol = None
        
        # Get system info
        self.system_memory = psutil.virtual_memory().total / (1024**3)  # GB
        
        # Check for GPU
        self.gpu_available = self._check_gpu()
        
    def _check_gpu(self) -> bool:
        """Check if GPU monitoring is available"""
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            return len(gpus) > 0
        except:
            return False
    
    def _get_memory_usage(self) -> Dict:
        """Get current memory usage statistics"""
        # System memory
        vm = psutil.virtual_memory()
        process = psutil.Process()
        
        memory_stats = {
            'timestamp': datetime.now().isoformat(),
            'system_memory_percent': vm.percent,
            'system_memory_used_gb': vm.used / (1024**3),
            'system_memory_available_gb': vm.available / (1024**3),
            'process_memory_mb': process.memory_info().rss / (1024**2),
            'process_memory_percent': process.memory_percent(),
            'cpu_percent': process.cpu_percent()
        }
        
        # GPU memory (if available)
        if self.gpu_available:
            try:
                import tensorflow as tf
                gpu_devices = tf.config.experimental.list_physical_devices('GPU')
                
                if gpu_devices:
                    # Get GPU memory info
                    try:
                        # This is a simple approximation - more detailed monitoring would need nvidia-ml-py
                        memory_stats['gpu_available'] = True
                        memory_stats['gpu_device_count'] = len(gpu_devices)
                    except:
                        memory_stats['gpu_available'] = False
                else:
                    memory_stats['gpu_available'] = False
            except:
                memory_stats['gpu_available'] = False
        else:
            memory_stats['gpu_available'] = False
        
        return memory_stats
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                memory_stats = self._get_memory_usage()
                self.memory_data.append(memory_stats)
                
                # Sleep for monitoring interval (5 seconds)
                time.sleep(5)
            except Exception as e:
                print(f"Memory monitoring error: {e}")
                break
    
    def start_monitoring(self, symbol: str):
        """Start memory monitoring for a symbol optimization"""
        if self.monitoring:
            self.stop_monitoring()
        
        self.symbol = symbol
        self.start_time = datetime.now()
        self.memory_data = []
        self.monitoring = True
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        print(f"📊 Memory monitoring started for {symbol}")
        print(f"   System RAM: {self.system_memory:.1f} GB")
        if self.gpu_available:
            print(f"   GPU monitoring: ✅ Available")
        else:
            print(f"   GPU monitoring: ❌ Not available")
    
    def stop_monitoring(self) -> Optional[Dict]:
        """Stop monitoring and return summary statistics"""
        if not self.monitoring:
            return None
        
        self.monitoring = False
        
        # Wait for thread to finish
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        
        if not self.memory_data:
            return None
        
        # Calculate summary statistics
        summary = self._calculate_summary()
        
        # Save detailed data
        self._save_monitoring_data(summary)
        
        print(f"📊 Memory monitoring stopped for {self.symbol}")
        print(f"   Duration: {summary['duration_minutes']:.1f} minutes")
        print(f"   Peak process memory: {summary['peak_process_memory_mb']:.1f} MB")
        print(f"   Average system memory: {summary['avg_system_memory_percent']:.1f}%")
        
        return summary
    
    def _calculate_summary(self) -> Dict:
        """Calculate summary statistics from memory data"""
        if not self.memory_data:
            return {}
        
        df = pd.DataFrame(self.memory_data)
        
        # Calculate duration
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds() / 60  # minutes
        
        summary = {
            'symbol': self.symbol,
            'start_time': self.start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_minutes': duration,
            'samples_collected': len(df),
            
            # System memory statistics
            'peak_system_memory_percent': df['system_memory_percent'].max(),
            'avg_system_memory_percent': df['system_memory_percent'].mean(),
            'peak_system_memory_used_gb': df['system_memory_used_gb'].max(),
            'avg_system_memory_used_gb': df['system_memory_used_gb'].mean(),
            
            # Process memory statistics
            'peak_process_memory_mb': df['process_memory_mb'].max(),
            'avg_process_memory_mb': df['process_memory_mb'].mean(),
            'final_process_memory_mb': df['process_memory_mb'].iloc[-1],
            'peak_process_memory_percent': df['process_memory_percent'].max(),
            'avg_process_memory_percent': df['process_memory_percent'].mean(),
            
            # CPU usage
            'peak_cpu_percent': df['cpu_percent'].max(),
            'avg_cpu_percent': df['cpu_percent'].mean(),
            
            # GPU info
            'gpu_available': df['gpu_available'].iloc[0] if 'gpu_available' in df.columns else False,
            
            # System info
            'system_total_memory_gb': self.system_memory
        }
        
        return summary
    
    def _save_monitoring_data(self, summary: Dict):
        """Save monitoring data to files"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save summary
        summary_file = self.results_path / f"memory_summary_{self.symbol}_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save detailed data
        detailed_file = self.results_path / f"memory_detailed_{self.symbol}_{timestamp}.json"
        with open(detailed_file, 'w') as f:
            json.dump(self.memory_data, f, indent=2)
        
        print(f"📁 Memory data saved: {summary_file.name}")
    
    def get_current_usage(self) -> Dict:
        """Get current memory usage without starting monitoring"""
        return self._get_memory_usage()
    
    def estimate_memory_requirements(self, symbol_count: int = 7, trials_per_symbol: int = 100) -> Dict:
        """Estimate memory requirements for full optimization"""
        current = self._get_memory_usage()
        
        # Simple estimation based on current usage
        base_memory_mb = current['process_memory_mb']
        
        # Rough estimates (these would be refined with actual monitoring data)
        estimated_peak_per_trial = base_memory_mb * 1.5  # 50% increase per trial
        estimated_total_gb = (estimated_peak_per_trial * symbol_count) / 1024
        
        estimates = {
            'current_usage_mb': base_memory_mb,
            'estimated_peak_per_trial_mb': estimated_peak_per_trial,
            'estimated_total_for_optimization_gb': estimated_total_gb,
            'system_memory_gb': self.system_memory,
            'estimated_memory_utilization_percent': (estimated_total_gb / self.system_memory) * 100,
            'recommendation': self._get_memory_recommendation(estimated_total_gb)
        }
        
        return estimates
    
    def _get_memory_recommendation(self, estimated_gb: float) -> str:
        """Get memory recommendation based on estimates"""
        utilization = (estimated_gb / self.system_memory) * 100
        
        if utilization < 50:
            return "✅ Sufficient memory available"
        elif utilization < 75:
            return "⚠️  Moderate memory usage expected - monitor during optimization"
        elif utilization < 90:
            return "🚨 High memory usage expected - consider reducing trials or running sequentially"
        else:
            return "❌ Insufficient memory - reduce trials or upgrade system"


class OptimizationMemoryWrapper:
    """Wrapper to integrate memory monitoring with optimization"""
    
    def __init__(self, optimizer, enable_monitoring: bool = True):
        self.optimizer = optimizer
        self.monitor = MemoryMonitor() if enable_monitoring else None
        self.enable_monitoring = enable_monitoring
        
        # Store original optimize_symbol method
        self.original_optimize_symbol = optimizer.optimize_symbol
        
        # Replace with monitored version
        optimizer.optimize_symbol = self._monitored_optimize_symbol
    
    def _monitored_optimize_symbol(self, symbol: str, n_trials: int = 50):
        """Wrapped optimize_symbol with memory monitoring"""
        if self.enable_monitoring and self.monitor:
            # Start monitoring
            self.monitor.start_monitoring(symbol)
            
            try:
                # Run original optimization
                result = self.original_optimize_symbol(symbol, n_trials)
                
                # Stop monitoring and get summary
                memory_summary = self.monitor.stop_monitoring()
                
                # Add memory info to result if available
                if result and memory_summary:
                    # Could extend result object with memory info if needed
                    print(f"💾 Memory usage for {symbol}: {memory_summary['peak_process_memory_mb']:.1f} MB peak")
                
                return result
                
            except Exception as e:
                # Make sure to stop monitoring even if optimization fails
                if self.monitor.monitoring:
                    self.monitor.stop_monitoring()
                raise e
        else:
            # Run without monitoring
            return self.original_optimize_symbol(symbol, n_trials)
    
    def get_memory_estimates(self):
        """Get memory requirement estimates"""
        if self.monitor:
            return self.monitor.estimate_memory_requirements()
        else:
            return {"error": "Memory monitoring not enabled"}
    
    def show_current_usage(self):
        """Show current memory usage"""
        if self.monitor:
            usage = self.monitor.get_current_usage()
            print(f"📊 Current Memory Usage:")
            print(f"   Process: {usage['process_memory_mb']:.1f} MB ({usage['process_memory_percent']:.1f}%)")
            print(f"   System: {usage['system_memory_used_gb']:.1f} GB ({usage['system_memory_percent']:.1f}%)")
            print(f"   CPU: {usage['cpu_percent']:.1f}%")
            return usage
        else:
            print("Memory monitoring not enabled")
            return None


# Example usage functions
def enable_memory_monitoring(optimizer):
    """Enable memory monitoring for an optimizer"""
    wrapper = OptimizationMemoryWrapper(optimizer, enable_monitoring=True)
    print("✅ Memory monitoring enabled")
    print("📊 Memory usage will be tracked during optimization")
    return wrapper

def show_memory_requirements_estimate():
    """Show estimated memory requirements for full optimization"""
    monitor = MemoryMonitor()
    estimates = monitor.estimate_memory_requirements()
    
    print("📊 MEMORY REQUIREMENTS ESTIMATE")
    print("="*40)
    print(f"Current usage: {estimates['current_usage_mb']:.1f} MB")
    print(f"Estimated peak per trial: {estimates['estimated_peak_per_trial_mb']:.1f} MB")
    print(f"Estimated total for 7 symbols: {estimates['estimated_total_for_optimization_gb']:.1f} GB")
    print(f"System memory: {estimates['system_memory_gb']:.1f} GB")
    print(f"Estimated utilization: {estimates['estimated_memory_utilization_percent']:.1f}%")
    print(f"Recommendation: {estimates['recommendation']}")
    
    return estimates

def show_current_memory_usage():
    """Show current system memory usage"""
    monitor = MemoryMonitor()
    usage = monitor.get_current_usage()
    
    print("📊 CURRENT MEMORY USAGE")
    print("="*30)
    print(f"Process Memory: {usage['process_memory_mb']:.1f} MB ({usage['process_memory_percent']:.1f}%)")
    print(f"System Memory: {usage['system_memory_used_gb']:.1f} GB ({usage['system_memory_percent']:.1f}%)")
    print(f"Available Memory: {usage['system_memory_available_gb']:.1f} GB")
    print(f"CPU Usage: {usage['cpu_percent']:.1f}%")
    if usage.get('gpu_available'):
        print(f"GPU: ✅ Available")
    else:
        print(f"GPU: ❌ Not available for monitoring")
    
    return usage


if __name__ == "__main__":
    # Demo the memory monitoring
    print("🧪 Memory Monitor Demo")
    print("="*30)
    
    # Show current usage
    show_current_memory_usage()
    print()
    
    # Show estimates
    show_memory_requirements_estimate()