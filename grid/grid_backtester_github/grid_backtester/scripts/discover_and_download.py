import MetaTrader5 as mt5
import pandas as pd
import sys
import os
from datetime import datetime, timedelta

def discover_available_symbols():
    """Discover available symbols in the MT5 terminal"""
    print("=== Discovering Available Symbols ===")
    
    # Get all symbols
    symbols = mt5.symbols_get()
    if symbols is None:
        print("No symbols found")
        return []
    
    print(f"Total symbols available: {len(symbols)}")
    
    # Filter for major forex pairs
    forex_symbols = []
    major_pairs = ["EUR", "GBP", "USD", "JPY", "CHF", "AUD", "CAD", "NZD"]
    
    for symbol in symbols:
        symbol_name = symbol.name
        # Check if it's a forex pair (contains major currencies)
        if any(curr in symbol_name for curr in major_pairs) and len(symbol_name) <= 8:
            # Check if symbol is visible and can be selected
            if mt5.symbol_select(symbol_name, True):
                symbol_info = mt5.symbol_info(symbol_name)
                if symbol_info and symbol_info.visible:
                    forex_symbols.append(symbol_name)
    
    print(f"Available forex symbols: {len(forex_symbols)}")
    for symbol in sorted(forex_symbols)[:20]:  # Show first 20
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info:
            print(f"  {symbol}: Bid={symbol_info.bid:.5f}, Ask={symbol_info.ask:.5f}")
    
    if len(forex_symbols) > 20:
        print(f"  ... and {len(forex_symbols) - 20} more")
    
    return forex_symbols

def download_2year_data_available_symbols():
    """Download 2 years of data for available symbols"""
    print("MetaTrader5 - 2 Year Data Download (Available Symbols)")
    print("=" * 55)
    
    # Initialize MT5
    if not mt5.initialize():
        print("✗ MetaTrader5 initialization failed")
        print(f"Error: {mt5.last_error()}")
        return False
    
    print("✓ MetaTrader5 initialized successfully")
    
    # Get terminal and account info
    terminal_info = mt5.terminal_info()
    account_info = mt5.account_info()
    
    if terminal_info:
        print(f"Terminal: {terminal_info.name}")
        print(f"Connected: {terminal_info.connected}")
    
    if account_info:
        print(f"Account: {account_info.login}")
        print(f"Server: {account_info.server}")
        print(f"Currency: {account_info.currency}")
    
    # Discover available symbols
    available_symbols = discover_available_symbols()
    
    if not available_symbols:
        print("No forex symbols available")
        mt5.shutdown()
        return False
    
    # Select top symbols for download (limit to avoid too much data)
    priority_symbols = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD", "EURGBP", "EURJPY", "GBPJPY"]
    symbols_to_download = []
    
    # First, try to get priority symbols
    for symbol in priority_symbols:
        if symbol in available_symbols:
            symbols_to_download.append(symbol)
    
    # If we don't have enough, add more available symbols
    if len(symbols_to_download) < 5:
        for symbol in available_symbols:
            if symbol not in symbols_to_download and len(symbols_to_download) < 10:
                symbols_to_download.append(symbol)
    
    print(f"\nSelected symbols for download: {symbols_to_download}")
    
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
    
    for symbol in symbols_to_download:
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
    if total_downloads > 0:
        print(f"Success rate: {(successful_downloads/total_downloads)*100:.1f}%")
    else:
        print("No downloads attempted")
    
    # List downloaded files
    print(f"\nDownloaded files in '{output_dir}' directory:")
    try:
        files = [f for f in os.listdir(output_dir) if f.endswith('_2year.csv')]
        total_size = 0
        for file in sorted(files):
            file_path = os.path.join(output_dir, file)
            file_size = os.path.getsize(file_path)
            total_size += file_size
            print(f"  {file} ({file_size:,} bytes)")
        
        if total_size > 0:
            print(f"\nTotal data size: {total_size:,} bytes ({total_size/1024/1024:.1f} MB)")
            
    except Exception as e:
        print(f"  Error listing files: {e}")
    
    mt5.shutdown()
    return successful_downloads > 0

if __name__ == "__main__":
    success = download_2year_data_available_symbols()
    
    if success:
        print("\n✓ 2-year data download completed successfully!")
    else:
        print("\n✗ Data download failed. Please check the output above.")
        sys.exit(1)
