#!/usr/bin/env python3
"""
Test script for load_or_fetch function
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.loader import load_or_fetch, load_metatrader_data

def test_load_or_fetch():
    """Test the load_or_fetch function"""
    print("🧪 Testing load_or_fetch function...")
    
    try:
        # Try to load EURUSD data (should work since we have it cached)
        symbol = "EURUSD"
        provider = "metatrader"
        
        print(f"Testing load_or_fetch for {symbol} using {provider}...")
        
        # This should load from cache if available
        df = load_or_fetch(
            symbol=symbol,
            provider=provider,
            loader_func=load_metatrader_data,
            api_key="",  # Not needed for cached data
            force_refresh=False
        )
        
        if df is not None and not df.empty:
            print(f"✅ Successfully loaded {len(df)} rows of {symbol} data")
            print(f"Columns: {df.columns.tolist()}")
            print(f"Date range: {df.index[0] if hasattr(df.index, '__getitem__') else 'N/A'} to {df.index[-1] if hasattr(df.index, '__getitem__') else 'N/A'}")
            return True
        else:
            print("⚠️ No data returned")
            return False
            
    except Exception as e:
        print(f"❌ Error testing load_or_fetch: {e}")
        return False

if __name__ == "__main__":
    success = test_load_or_fetch()
    if success:
        print("\n🎉 load_or_fetch test passed!")
    else:
        print("\n❌ load_or_fetch test failed!")
        sys.exit(1)