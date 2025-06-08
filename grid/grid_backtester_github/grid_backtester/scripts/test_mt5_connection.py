import MetaTrader5 as mt5

def quick_mt5_test():
    """Quick MT5 connection test"""
    print("Quick MT5 Connection Test")
    print("=" * 30)
    
    # Initialize MT5
    if not mt5.initialize():
        print("❌ MT5 initialization failed")
        print(f"Error: {mt5.last_error()}")
        return False
    
    print("✅ MT5 connected successfully")
    
    # Get basic info
    terminal_info = mt5.terminal_info()
    account_info = mt5.account_info()
    
    if terminal_info:
        print(f"Terminal: {terminal_info.name}")
        print(f"Connected: {terminal_info.connected}")
    
    if account_info:
        print(f"Account: {account_info.login}")
        print(f"Server: {account_info.server}")
        print(f"Currency: {account_info.currency}")
    
    # Test a simple symbol
    symbol_info = mt5.symbol_info("EURUSD")
    if symbol_info:
        print(f"EURUSD Bid: {symbol_info.bid}")
        print(f"EURUSD Ask: {symbol_info.ask}")
    
    mt5.shutdown()
    return True

if __name__ == "__main__":
    quick_mt5_test()
