import MetaTrader5 as mt5
import pandas as pd
import sys
import os
from datetime import datetime, timedelta

def connect_to_icmarkets():
    """Connect specifically to ICMarkets MT5"""
    print("=== Connecting to ICMarkets MT5 ===")
    
    # Try to initialize MT5 with ICMarkets path
    icmarkets_paths = [
        "C:\\Program Files\\ICMarkets - MetaTrader 5\\terminal64.exe",
        "C:\\Program Files (x86)\\ICMarkets - MetaTrader 5\\terminal64.exe",
        "C:\\Program Files\\ICMarkets - MetaTrader 5",
        "C:\\Program Files (x86)\\ICMarkets - MetaTrader 5"
    ]
    
    # First try default initialization
    if mt5.initialize():
        terminal_info = mt5.terminal_info()
        if terminal_info and "ICMarkets" in terminal_info.name:
            print(f"✓ Connected to {terminal_info.name}")
            return True
        else:
            print(f"Connected to {terminal_info.name if terminal_info else 'Unknown'}, but not ICMarkets")
            mt5.shutdown()
    
    # Try specific ICMarkets paths
    for path in icmarkets_paths:
        print(f"Trying path: {path}")
        if mt5.initialize(path):
            terminal_info = mt5.terminal_info()
            if terminal_info and "ICMarkets" in terminal_info.name:
                print(f"✓ Connected to {terminal_info.name}")
                return True
            else:
                mt5.shutdown()
    
    print("✗ Could not connect to ICMarkets MT5")
    print("\nTroubleshooting:")
    print("1. Make sure ICMarkets MT5 is installed")
    print("2. Open ICMarkets MT5 terminal manually")
    print("3. Log in to your ICMarkets account")
    print("4. Make sure the terminal is running")
    
    return False

def download_icmarkets_2year_data():
    """Download 2 years of data from ICMarkets"""
    print("ICMarkets MT5 - 2 Year Data Download")
    print("=" * 40)
    
    # Connect to ICMarkets
    if not connect_to_icmarkets():
        return False
    
    # Get terminal and account info
    terminal_info = mt5.terminal_info()
    account_info = mt5.account_info()
    
    print(f"\nTerminal: {terminal_info.name}")
    print(f"Connected: {terminal_info.connected}")
    print(f"Path: {terminal_info.path}")
    
    if account_info:
        print(f"Account: {account_info.login}")
        print(f"Server: {account_info.server}")
        print(f"Currency: {account_info.currency}")
        print(f"Company: {account_info.company}")
    
    # Define symbols - ICMarkets standard naming
    symbols = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]
    
    # Test symbol availability first
    print(f"\n=== Testing Symbol Availability ===")
    available_symbols = []
    
    for symbol in symbols:
        # Try to select the symbol
        if mt5.symbol_select(symbol, True):
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is not None:
                print(f"✓ {symbol}: Available (Bid: {symbol_info.bid:.5f}, Ask: {symbol_info.ask:.5f})")
                available_symbols.append(symbol)
            else:
                print(f"✗ {symbol}: Info not available")
        else:
            print(f"✗ {symbol}: Cannot select")
    
    if not available_symbols:
        print("No symbols available for download")
        mt5.shutdown()
        return False
    
    print(f"\nWill download data for: {available_symbols}")
    
    # Define timeframes
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
    
    for symbol in available_symbols:
        print(f"\n--- Processing {symbol} ---")
        
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
                    df['broker'] = 'ICMarkets'
                    df['hl_range'] = df['high'] - df['low']
                    df['oc_range'] = abs(df['close'] - df['open'])
                    
                    filename = os.path.join(output_dir, f"ICM_{symbol}_{tf_name}_2year.csv")
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
    if total_downloads > 0:
        print(f"Success rate: {(successful_downloads/total_downloads)*100:.1f}%")
    else:
        print("No downloads attempted")
    
    # List downloaded files
    print(f"\nDownloaded files in '{output_dir}' directory:")
    try:
        files = [f for f in os.listdir(output_dir) if f.startswith('ICM_') and f.endswith('_2year.csv')]
        total_size = 0
        for file in sorted(files):
            file_path = os.path.join(output_dir, file)
            file_size = os.path.getsize(file_path)
            total_size += file_size
            print(f"  {file} ({file_size:,} bytes)")
        
        if total_size > 0:
            print(f"\nTotal ICMarkets data size: {total_size:,} bytes ({total_size/1024/1024:.1f} MB)")
            
    except Exception as e:
        print(f"  Error listing files: {e}")
    
    mt5.shutdown()
    return successful_downloads > 0

if __name__ == "__main__":
    success = download_icmarkets_2year_data()
    
    if success:
        print("\n✓ ICMarkets 2-year data download completed successfully!")
    else:
        print("\n✗ ICMarkets data download failed. Please check the output above.")
        print("\nMake sure:")
        print("1. ICMarkets MT5 is installed and running")
        print("2. You are logged into your ICMarkets account")
        print("3. The symbols are available in Market Watch")
        sys.exit(1)
