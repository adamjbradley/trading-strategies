import MetaTrader5 as mt5
import pandas as pd
import sys
import os
from datetime import datetime, timedelta

def download_2year_data():
    """Download 2 years of historical data for major currency pairs"""
    print("MetaTrader5 - 2 Year Data Download")
    print("=" * 40)
    
    # Initialize MT5
    if not mt5.initialize():
        print("✗ MetaTrader5 initialization failed")
        print(f"Error: {mt5.last_error()}")
        return False
    
    print("✓ MetaTrader5 initialized successfully")
    
    # Define symbols and timeframes
    symbols = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]
    timeframes = {
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1
    }
    
    # Calculate date range (2 years from now)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # 2 years
    
    print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Create output directory
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    
    total_downloads = 0
    successful_downloads = 0
    
    for symbol in symbols:
        print(f"\n--- Processing {symbol} ---")
        
        # Check if symbol is available
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            print(f"✗ {symbol}: Not available")
            # Try to enable the symbol
            if mt5.symbol_select(symbol, True):
                print(f"  → Enabled {symbol} in Market Watch")
                symbol_info = mt5.symbol_info(symbol)
            else:
                print(f"  → Failed to enable {symbol}")
                continue
        
        if symbol_info is not None:
            print(f"✓ {symbol}: Available (Spread: {symbol_info.spread})")
        
        for tf_name, tf_value in timeframes.items():
            total_downloads += 1
            print(f"  Downloading {symbol} {tf_name}...")
            
            try:
                # Get rates from start_date to end_date
                rates = mt5.copy_rates_range(symbol, tf_value, start_date, end_date)
                
                if rates is not None and len(rates) > 0:
                    df = pd.DataFrame(rates)
                    df['time'] = pd.to_datetime(df['time'], unit='s')
                    
                    # Add additional columns for analysis
                    df['symbol'] = symbol
                    df['timeframe'] = tf_name
                    df['hl_range'] = df['high'] - df['low']
                    df['oc_range'] = abs(df['close'] - df['open'])
                    
                    filename = os.path.join(output_dir, f"{symbol}_{tf_name}_2year.csv")
                    df.to_csv(filename, index=False)
                    
                    print(f"    ✓ {len(df)} bars saved to {filename}")
                    print(f"    Date range: {df['time'].min()} to {df['time'].max()}")
                    successful_downloads += 1
                    
                else:
                    print(f"    ✗ No data available for {symbol} {tf_name}")
                    last_error = mt5.last_error()
                    if last_error[0] != 1:  # 1 means success
                        print(f"    Error: {last_error}")
                        
            except Exception as e:
                print(f"    ✗ Exception: {e}")
    
    # Summary
    print(f"\n=== Download Summary ===")
    print(f"Total attempts: {total_downloads}")
    print(f"Successful downloads: {successful_downloads}")
    print(f"Success rate: {(successful_downloads/total_downloads)*100:.1f}%")
    
    # List downloaded files
    print(f"\nDownloaded files in '{output_dir}' directory:")
    try:
        files = [f for f in os.listdir(output_dir) if f.endswith('_2year.csv')]
        for file in sorted(files):
            file_path = os.path.join(output_dir, file)
            file_size = os.path.getsize(file_path)
            print(f"  {file} ({file_size:,} bytes)")
    except Exception as e:
        print(f"  Error listing files: {e}")
    
    mt5.shutdown()
    return successful_downloads > 0

def get_data_summary():
    """Get summary of downloaded data"""
    print("\n=== Data Summary ===")
    
    data_dir = "data"
    if not os.path.exists(data_dir):
        print("No data directory found")
        return
    
    files = [f for f in os.listdir(data_dir) if f.endswith('_2year.csv')]
    
    if not files:
        print("No 2-year data files found")
        return
    
    total_size = 0
    for file in sorted(files):
        file_path = os.path.join(data_dir, file)
        file_size = os.path.getsize(file_path)
        total_size += file_size
        
        # Read first few rows to get info
        try:
            df = pd.read_csv(file_path, nrows=5)
            print(f"{file}:")
            print(f"  Size: {file_size:,} bytes")
            if len(df) > 0:
                print(f"  Columns: {', '.join(df.columns.tolist())}")
        except Exception as e:
            print(f"  Error reading {file}: {e}")
    
    print(f"\nTotal data size: {total_size:,} bytes ({total_size/1024/1024:.1f} MB)")

if __name__ == "__main__":
    success = download_2year_data()
    
    if success:
        get_data_summary()
        print("\n✓ 2-year data download completed successfully!")
    else:
        print("\n✗ Data download failed. Please check the output above.")
        sys.exit(1)
