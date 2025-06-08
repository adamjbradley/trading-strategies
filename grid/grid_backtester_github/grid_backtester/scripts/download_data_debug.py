import MetaTrader5 as mt5
import pandas as pd
import sys
import os

def check_mt5_connection():
    """Check MT5 connection and provide detailed diagnostics"""
    print("=== MetaTrader5 Connection Diagnostics ===")
    
    # Check if MT5 package is available
    try:
        import MetaTrader5 as mt5
        print("✓ MetaTrader5 package imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import MetaTrader5: {e}")
        return False
    
    # Try to initialize MT5
    print("\nAttempting to initialize MetaTrader5...")
    if not mt5.initialize():
        print("✗ MetaTrader5 initialization failed")
        
        # Get last error
        last_error = mt5.last_error()
        print(f"Last error: {last_error}")
        
        # Common troubleshooting steps
        print("\nTroubleshooting steps:")
        print("1. Make sure MetaTrader5 terminal is installed")
        print("2. Open MetaTrader5 terminal manually")
        print("3. Log in to your trading account")
        print("4. Ensure 'Allow DLL imports' is enabled in Tools > Options > Expert Advisors")
        print("5. Check if your antivirus is blocking MT5")
        
        mt5.shutdown()
        return False
    
    print("✓ MetaTrader5 initialized successfully")
    
    # Get terminal info
    terminal_info = mt5.terminal_info()
    if terminal_info is not None:
        print(f"\nTerminal Info:")
        print(f"  Company: {terminal_info.company}")
        print(f"  Name: {terminal_info.name}")
        print(f"  Path: {terminal_info.path}")
        print(f"  Data Path: {terminal_info.data_path}")
        print(f"  Connected: {terminal_info.connected}")
        print(f"  Trade Allowed: {terminal_info.trade_allowed}")
    else:
        print("✗ Could not get terminal info")
    
    # Get account info
    account_info = mt5.account_info()
    if account_info is not None:
        print(f"\nAccount Info:")
        print(f"  Login: {account_info.login}")
        print(f"  Server: {account_info.server}")
        print(f"  Currency: {account_info.currency}")
        print(f"  Company: {account_info.company}")
        print(f"  Connected: {account_info.connected if hasattr(account_info, 'connected') else 'N/A'}")
    else:
        print("✗ Could not get account info - you may not be logged in")
    
    return True

def test_symbol_availability(symbols):
    """Test if symbols are available"""
    print(f"\n=== Testing Symbol Availability ===")
    
    available_symbols = []
    for symbol in symbols:
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is not None:
            print(f"✓ {symbol}: Available")
            print(f"    Bid: {symbol_info.bid}")
            print(f"    Ask: {symbol_info.ask}")
            print(f"    Spread: {symbol_info.spread}")
            available_symbols.append(symbol)
        else:
            print(f"✗ {symbol}: Not available")
            
            # Try to enable the symbol
            if mt5.symbol_select(symbol, True):
                print(f"  → Enabled {symbol} in Market Watch")
                available_symbols.append(symbol)
            else:
                print(f"  → Failed to enable {symbol}")
    
    return available_symbols

def download_data_with_diagnostics():
    """Download data with comprehensive error handling"""
    symbols = ["EURUSD", "GBPUSD", "USDJPY"]
    timeframe = mt5.TIMEFRAME_H1
    bars = 5000
    
    # Check connection first
    if not check_mt5_connection():
        return False
    
    # Test symbol availability
    available_symbols = test_symbol_availability(symbols)
    
    if not available_symbols:
        print("\n✗ No symbols available for download")
        mt5.shutdown()
        return False
    
    print(f"\n=== Downloading Data ===")
    success_count = 0
    
    for symbol in available_symbols:
        print(f"\nFetching data for {symbol}...")
        
        try:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
            
            if rates is not None and len(rates) > 0:
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                
                # Create output directory if it doesn't exist
                output_dir = "data"
                os.makedirs(output_dir, exist_ok=True)
                
                filename = os.path.join(output_dir, f"{symbol}_H1.csv")
                df.to_csv(filename, index=False)
                
                print(f"✓ {symbol}: {len(df)} rows saved to {filename}")
                print(f"  Date range: {df['time'].min()} to {df['time'].max()}")
                success_count += 1
                
            else:
                print(f"✗ Failed to fetch {symbol}")
                last_error = mt5.last_error()
                print(f"  Error: {last_error}")
                
        except Exception as e:
            print(f"✗ Exception while fetching {symbol}: {e}")
    
    print(f"\n=== Summary ===")
    print(f"Successfully downloaded: {success_count}/{len(available_symbols)} symbols")
    
    mt5.shutdown()
    return success_count > 0

if __name__ == "__main__":
    print("MetaTrader5 Data Download with Diagnostics")
    print("=" * 50)
    
    success = download_data_with_diagnostics()
    
    if success:
        print("\n✓ Data download completed successfully!")
    else:
        print("\n✗ Data download failed. Please check the diagnostics above.")
        sys.exit(1)
