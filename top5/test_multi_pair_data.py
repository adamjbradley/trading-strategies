#!/usr/bin/env python3
"""
Step 2: Multi-Pair Data Availability Test
Test and validate availability of multi-pair forex data for enhanced correlations
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os

def test_multi_pair_data_availability():
    """Test availability and quality of multi-pair forex data"""
    
    print("🔄 Step 2: Testing Multi-Pair Data Availability")
    print("=" * 60)
    
    data_dir = Path("data")
    
    # Define major currency pairs for correlation analysis
    major_pairs = [
        'EURUSD', 'GBPUSD', 'USDJPY', 'USDCAD',
        'AUDUSD', 'EURJPY', 'GBPJPY'
    ]
    
    available_pairs = []
    pair_stats = {}
    
    print("📊 Checking data availability for major currency pairs...")
    
    for pair in major_pairs:
        parquet_file = data_dir / f"metatrader_{pair}.parquet"
        h5_file = data_dir / f"metatrader_{pair}.h5"
        
        if parquet_file.exists():
            try:
                # Load and analyze the data
                df = pd.read_parquet(parquet_file)
                
                # Basic statistics
                stats = {
                    'format': 'parquet',
                    'rows': len(df),
                    'start_date': df.index.min() if hasattr(df.index, 'min') else 'N/A',
                    'end_date': df.index.max() if hasattr(df.index, 'max') else 'N/A',
                    'columns': list(df.columns),
                    'missing_data': df.isnull().sum().sum(),
                    'file_size_mb': parquet_file.stat().st_size / (1024 * 1024)
                }
                
                available_pairs.append(pair)
                pair_stats[pair] = stats
                
                print(f"✅ {pair}: {stats['rows']:,} rows, {stats['file_size_mb']:.1f}MB")
                
            except Exception as e:
                print(f"❌ {pair}: Error loading parquet - {e}")
                
        elif h5_file.exists():
            try:
                # Try H5 format as fallback
                df = pd.read_hdf(h5_file)
                
                stats = {
                    'format': 'h5',
                    'rows': len(df),
                    'start_date': df.index.min() if hasattr(df.index, 'min') else 'N/A',
                    'end_date': df.index.max() if hasattr(df.index, 'max') else 'N/A',
                    'columns': list(df.columns),
                    'missing_data': df.isnull().sum().sum(),
                    'file_size_mb': h5_file.stat().st_size / (1024 * 1024)
                }
                
                available_pairs.append(pair)
                pair_stats[pair] = stats
                
                print(f"✅ {pair}: {stats['rows']:,} rows, {stats['file_size_mb']:.1f}MB (H5)")
                
            except Exception as e:
                print(f"❌ {pair}: Error loading H5 - {e}")
        else:
            print(f"❌ {pair}: No data file found")
    
    print(f"\n📈 Data Availability Summary:")
    print(f"   Available pairs: {len(available_pairs)}/{len(major_pairs)}")
    print(f"   Coverage: {len(available_pairs)/len(major_pairs)*100:.1f}%")
    
    if len(available_pairs) >= 4:
        print("✅ Sufficient data for enhanced correlation features!")
    else:
        print("⚠️ Limited data - basic correlations only")
    
    # Test data quality and synchronization
    print(f"\n🔍 Testing Data Quality & Synchronization:")
    
    if len(available_pairs) >= 2:
        # Load two pairs to test synchronization
        pair1, pair2 = available_pairs[0], available_pairs[1]
        
        try:
            df1 = pd.read_parquet(data_dir / f"metatrader_{pair1}.parquet")
            df2 = pd.read_parquet(data_dir / f"metatrader_{pair2}.parquet")
            
            # Check time alignment
            common_times = df1.index.intersection(df2.index)
            sync_ratio = len(common_times) / max(len(df1), len(df2))
            
            print(f"   {pair1} vs {pair2} synchronization: {sync_ratio:.1%}")
            
            if sync_ratio > 0.8:
                print("✅ Good time synchronization for correlation analysis")
            else:
                print("⚠️ Poor synchronization - may affect correlation quality")
                
        except Exception as e:
            print(f"❌ Synchronization test failed: {e}")
    
    # Currency coverage analysis
    currencies = set()
    for pair in available_pairs:
        currencies.add(pair[:3])  # Base currency
        currencies.add(pair[3:])  # Quote currency
    
    print(f"\n💰 Currency Coverage:")
    print(f"   Unique currencies: {sorted(currencies)}")
    print(f"   USD pairs: {len([p for p in available_pairs if 'USD' in p])}")
    print(f"   EUR pairs: {len([p for p in available_pairs if 'EUR' in p])}")
    print(f"   GBP pairs: {len([p for p in available_pairs if 'GBP' in p])}")
    print(f"   JPY pairs: {len([p for p in available_pairs if 'JPY' in p])}")
    
    # Correlation feature readiness assessment
    print(f"\n🎯 Enhanced Correlation Features Readiness:")
    
    readiness_score = 0
    max_score = 7
    
    # 1. Currency Strength Index (CSI)
    if 'USD' in currencies and len([p for p in available_pairs if 'USD' in p]) >= 3:
        print("✅ Currency Strength Index: Ready (3+ USD pairs)")
        readiness_score += 1
    else:
        print("⚠️ Currency Strength Index: Limited (need 3+ USD pairs)")
    
    # 2. Cross-pair correlation matrix
    if len(available_pairs) >= 4:
        print("✅ Cross-pair correlation matrix: Ready (4+ pairs)")
        readiness_score += 1
    else:
        print("⚠️ Cross-pair correlation matrix: Limited (need 4+ pairs)")
    
    # 3. Risk-on/risk-off sentiment
    risk_pairs = [p for p in available_pairs if any(curr in p for curr in ['AUD', 'GBP', 'JPY'])]
    if len(risk_pairs) >= 2:
        print("✅ Risk sentiment detection: Ready (risk currencies available)")
        readiness_score += 1
    else:
        print("⚠️ Risk sentiment detection: Limited (need risk currencies)")
    
    # 4. Carry trade indicators
    carry_pairs = [p for p in available_pairs if any(curr in p for curr in ['AUD', 'JPY', 'USD'])]
    if len(carry_pairs) >= 2:
        print("✅ Carry trade indicators: Ready")
        readiness_score += 1
    else:
        print("⚠️ Carry trade indicators: Limited")
    
    # 5. USD Index proxy
    usd_pairs = [p for p in available_pairs if 'USD' in p]
    if len(usd_pairs) >= 4:
        print("✅ USD Index proxy: Ready (4+ USD pairs)")
        readiness_score += 1
    else:
        print("⚠️ USD Index proxy: Limited (need 4+ USD pairs)")
    
    # 6. Correlation regime detection
    if len(available_pairs) >= 3:
        print("✅ Correlation regime detection: Ready")
        readiness_score += 1
    else:
        print("⚠️ Correlation regime detection: Limited")
    
    # 7. Divergence indicators
    if len(available_pairs) >= 2:
        print("✅ Divergence indicators: Ready")
        readiness_score += 1
    else:
        print("⚠️ Divergence indicators: Limited")
    
    print(f"\n📊 Overall Readiness: {readiness_score}/{max_score} ({readiness_score/max_score*100:.0f}%)")
    
    if readiness_score >= 5:
        print("🎉 Excellent! Ready for full enhanced correlation implementation")
        recommendation = "full_implementation"
    elif readiness_score >= 3:
        print("👍 Good! Ready for partial enhanced correlation implementation")
        recommendation = "partial_implementation"
    else:
        print("⚠️ Limited data - basic correlation features only")
        recommendation = "basic_implementation"
    
    # Create multi-pair data loader configuration
    config = {
        'available_pairs': available_pairs,
        'total_pairs': len(available_pairs),
        'currencies': sorted(currencies),
        'readiness_score': readiness_score,
        'recommendation': recommendation,
        'usd_pairs': usd_pairs,
        'data_quality': 'good' if sync_ratio > 0.8 else 'moderate',
        'implementation_features': {
            'currency_strength_index': readiness_score >= 1,
            'cross_pair_correlation': readiness_score >= 2,
            'risk_sentiment': readiness_score >= 3,
            'carry_trade': readiness_score >= 4,
            'usd_index_proxy': readiness_score >= 5,
            'correlation_regimes': readiness_score >= 6,
            'divergence_indicators': readiness_score >= 7
        }
    }
    
    return config

def create_multi_pair_data_loader_demo():
    """Demo the multi-pair data loader functionality"""
    
    print("\n🔧 Creating Multi-Pair Data Loader Demo...")
    
    demo_code = '''
def create_multi_pair_data_loader(target_symbol, correlation_pairs=None):
    """Load multiple currency pairs for enhanced correlation analysis"""
    
    data_dir = Path("data")
    all_data = {}
    
    # Load target symbol
    target_file = data_dir / f"metatrader_{target_symbol}.parquet"
    if target_file.exists():
        all_data[target_symbol] = pd.read_parquet(target_file)
    
    # Load correlation pairs
    if correlation_pairs is None:
        correlation_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCAD', 'AUDUSD']
    
    for pair in correlation_pairs:
        if pair != target_symbol:
            pair_file = data_dir / f"metatrader_{pair}.parquet"
            if pair_file.exists():
                all_data[pair] = pd.read_parquet(pair_file)
    
    # Synchronize all data to common timeframe
    if len(all_data) > 1:
        common_index = None
        for symbol, df in all_data.items():
            if common_index is None:
                common_index = df.index
            else:
                common_index = common_index.intersection(df.index)
        
        # Align all data to common timeframe
        for symbol in all_data:
            all_data[symbol] = all_data[symbol].loc[common_index]
    
    return all_data

# This function is ready for integration!
'''
    
    print("✅ Multi-pair data loader function created!")
    print("🔧 Ready for integration into correlation system")
    
    return demo_code

if __name__ == "__main__":
    # Run the multi-pair data availability test
    config = test_multi_pair_data_availability()
    
    # Create the data loader demo
    loader_code = create_multi_pair_data_loader_demo()
    
    print("\n" + "=" * 60)
    print("🎯 Step 2 Results Summary:")
    print("=" * 60)
    print(f"✅ Available pairs: {config['total_pairs']}")
    print(f"✅ Readiness score: {config['readiness_score']}/7")
    print(f"✅ Recommendation: {config['recommendation']}")
    print(f"✅ Data quality: {config['data_quality']}")
    
    print("🔄 Ready to proceed to Step 3: Run optimization with enhanced features")