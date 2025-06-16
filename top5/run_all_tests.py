#!/usr/bin/env python3
"""
Master Test Runner for Enhanced Feature Engineering
Runs all test suites and provides comprehensive validation
"""

import os
import sys
import subprocess
import time
from pathlib import Path
import json

def run_test_suite(test_file, description):
    """Run a specific test suite and return results"""
    print(f"\n{'='*60}")
    print(f"🧪 RUNNING: {description}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Run the test file
        result = subprocess.run([
            sys.executable, test_file
        ], capture_output=True, text=True, timeout=300)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Parse results
        success = result.returncode == 0
        output = result.stdout
        error_output = result.stderr
        
        print(output)
        if error_output:
            print("STDERR:")
            print(error_output)
        
        print(f"\n⏱️ Duration: {duration:.2f} seconds")
        
        return {
            'name': description,
            'success': success,
            'duration': duration,
            'output': output,
            'errors': error_output
        }
        
    except subprocess.TimeoutExpired:
        print("❌ Test suite timed out after 5 minutes")
        return {
            'name': description,
            'success': False,
            'duration': 300,
            'output': '',
            'errors': 'Test timed out'
        }
    except Exception as e:
        print(f"❌ Failed to run test suite: {e}")
        return {
            'name': description,
            'success': False,
            'duration': 0,
            'output': '',
            'errors': str(e)
        }

def validate_notebook_file():
    """Validate that the notebook file exists and has been updated"""
    notebook_path = Path("Advanced_Hyperparameter_Optimization_Clean.ipynb")
    
    if not notebook_path.exists():
        print("❌ Notebook file not found!")
        return False
    
    # Check file modification time
    mod_time = notebook_path.stat().st_mtime
    current_time = time.time()
    
    # File should have been modified within the last day
    if current_time - mod_time > 86400:  # 24 hours
        print("⚠️ Notebook file appears to be old - may not have latest enhancements")
    
    print(f"✅ Notebook file found: {notebook_path}")
    return True

def check_dependencies():
    """Check that required dependencies are available"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'pandas', 'numpy', 'sklearn', 'tensorflow', 'optuna'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"  ❌ {package} - MISSING")
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {missing_packages}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ All dependencies available")
    return True

def generate_test_report(results):
    """Generate a comprehensive test report"""
    print(f"\n{'='*80}")
    print("📊 COMPREHENSIVE TEST REPORT")
    print(f"{'='*80}")
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r['success'])
    failed_tests = total_tests - passed_tests
    total_duration = sum(r['duration'] for r in results)
    
    print(f"📈 SUMMARY:")
    print(f"  Total Test Suites: {total_tests}")
    print(f"  Passed: {passed_tests}")
    print(f"  Failed: {failed_tests}")
    print(f"  Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    print(f"  Total Duration: {total_duration:.2f} seconds")
    
    print(f"\n📋 DETAILED RESULTS:")
    for i, result in enumerate(results, 1):
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        print(f"  {i}. {result['name']}")
        print(f"     Status: {status}")
        print(f"     Duration: {result['duration']:.2f}s")
        if not result['success'] and result['errors']:
            print(f"     Error: {result['errors'][:100]}...")
    
    # Overall assessment
    if passed_tests == total_tests:
        print(f"\n🎉 ALL TESTS PASSED!")
        print("✅ Enhanced feature engineering is fully validated")
        print("✅ Notebook integration is working correctly")
        print("✅ Production ready for deployment")
        
        print(f"\n🚀 NEXT STEPS:")
        print("1. Run the notebook with enhanced features")
        print("2. Execute hyperparameter optimization")
        print("3. Monitor performance improvements")
        print("4. Export optimized models to ONNX")
        
    else:
        print(f"\n⚠️ {failed_tests} TEST SUITE(S) FAILED")
        print("🔧 Review failed tests before proceeding:")
        
        for result in results:
            if not result['success']:
                print(f"  - {result['name']}: {result['errors'][:50]}...")
        
        print("\n💡 Troubleshooting tips:")
        print("1. Check that all dependencies are installed")
        print("2. Verify notebook file has been properly updated")
        print("3. Ensure test data can be created successfully")
        print("4. Review error messages for specific issues")
    
    return passed_tests == total_tests

def create_test_summary_file(results):
    """Create a JSON summary file of test results"""
    summary = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_suites': len(results),
        'passed_suites': sum(1 for r in results if r['success']),
        'failed_suites': sum(1 for r in results if not r['success']),
        'total_duration': sum(r['duration'] for r in results),
        'results': results
    }
    
    summary_file = Path("test_results_summary.json")
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📄 Test summary saved to: {summary_file}")

def main():
    """Main test execution function"""
    print("🚀 ENHANCED FEATURE ENGINEERING - MASTER TEST SUITE")
    print("=" * 80)
    print("Testing all enhanced features and notebook integration")
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Pre-flight checks
    print("\n🔍 PRE-FLIGHT CHECKS")
    print("-" * 40)
    
    if not check_dependencies():
        print("❌ Dependency check failed - aborting tests")
        return False
    
    if not validate_notebook_file():
        print("❌ Notebook validation failed - aborting tests")
        return False
    
    # Define test suites
    test_suites = [
        {
            'file': 'test_enhanced_features.py',
            'description': 'Enhanced Feature Engineering Tests'
        },
        {
            'file': 'test_notebook_integration.py', 
            'description': 'Notebook Integration Tests'
        }
    ]
    
    # Run all test suites
    results = []
    
    for suite in test_suites:
        if not Path(suite['file']).exists():
            print(f"⚠️ Test file {suite['file']} not found - skipping")
            continue
            
        result = run_test_suite(suite['file'], suite['description'])
        results.append(result)
    
    # Generate comprehensive report
    success = generate_test_report(results)
    
    # Save test summary
    create_test_summary_file(results)
    
    print(f"\n{'='*80}")
    print(f"🏁 Testing completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    return success

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n🎉 All tests passed - system is ready for production!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed - please review and fix issues")
        sys.exit(1)