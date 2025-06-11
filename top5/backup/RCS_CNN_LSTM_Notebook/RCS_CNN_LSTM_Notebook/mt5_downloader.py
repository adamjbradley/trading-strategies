import MetaTrader5 as mt5
import pandas as pd
import os
from datetime import datetime, timedelta
import glob

def list_mt5_terminals():
    """List all MetaTrader5 terminal.exe files in common install locations and assign broker names."""
    possible_dirs = [
        r"C:\Program Files\MetaTrader 5*",
        r"C:\Program Files (x86)\MetaTrader 5*",
        r"C:\Program Files\ICMarkets*",
        r"C:\Program Files (x86)\ICMarkets*",
        r"C:\Program Files\AMP Global*",
        r"C:\Program Files (x86)\AMP Global*",
        r"C:\Program Files\*MT5*",
        r"C:\Program Files (x86)\*MT5*",
    ]
    found = []
    for d in possible_dirs:
        for path in glob.glob(os.path.join(d, "terminal64.exe")) + glob.glob(os.path.join(d, "terminal.exe")):
            found.append(path)
    broker_map = {}
    print("Detected MetaTrader5 terminals:")
    if found:
        for i, path in enumerate(found):
            # Auto-generate broker name from parent directory
            broker_name = os.path.basename(os.path.dirname(path)).replace(" ", "_").lower()
            broker_map[broker_name] = path
            print(f"  [{i+1}] {broker_name}: {path}")
    else:
        print("  No MT5 terminals found in common locations.")
    return broker_map

class MT5Downloader:
    def __init__(self, output_dir="data", broker="default", mt5_path=None):
        self.output_dir = output_dir
        self.broker = broker
        # If broker is specified and mt5_path is not, try to map broker to path
        if mt5_path is None and broker != "default":
            broker_map = list_mt5_terminals()
            self.mt5_path = broker_map.get(broker)
            if self.mt5_path:
                print(f"Using terminal for broker '{broker}': {self.mt5_path}")
            else:
                print(f"Warning: No terminal found for broker '{broker}', using default.")
                self.mt5_path = None
        else:
            self.mt5_path = mt5_path
        os.makedirs(self.output_dir, exist_ok=True)

    def initialize(self, path=None):
        # Priority: explicit path > self.mt5_path > default
        use_path = path if path is not None else self.mt5_path
        if use_path:
            initialized = mt5.initialize(use_path)
        else:
            initialized = mt5.initialize()
        if not initialized:
            print("✗ MetaTrader5 initialization failed")
            print(f"Error: {mt5.last_error()}")
            return False
        print(f"✓ MetaTrader5 initialized successfully (path: {use_path if use_path else 'default'})")
        # Detailed connection summary
        try:
            term_info = mt5.terminal_info()
            acct_info = mt5.account_info()
            print("---- MetaTrader5 Connection Summary ----")
            if term_info:
                print(f"Terminal Path: {getattr(term_info, 'path', '(not available)')}")
                print(f"Terminal Version: {getattr(term_info, 'version', '(not available)')}")
                print(f"Data Path: {getattr(term_info, 'data_path', '(not available)')}")
                print(f"Community Account: {getattr(term_info, 'community_account', '(not available)')}")
                print(f"Community Connection: {getattr(term_info, 'community_connection', '(not available)')}")
                print(f"Connected: {getattr(term_info, 'connected', '(not available)')}")
            else:
                print("Terminal info: Not available")
            if acct_info:
                print(f"Account Number: {acct_info.login}")
                print(f"Account Name: {acct_info.name}")
                print(f"Server: {acct_info.server}")
                print(f"Company: {acct_info.company}")
                print(f"Leverage: {acct_info.leverage}")
                print(f"Trade Allowed: {acct_info.trade_allowed}")
                print(f"Balance: {acct_info.balance}")
                print(f"Currency: {acct_info.currency}")
            else:
                print("Account info: Not available")
            print("----------------------------------------")
        except Exception as e:
            print(f"Warning: Could not retrieve full MT5 connection info: {e}")
        return True

    def shutdown(self):
        mt5.shutdown()
        print("MetaTrader5 shutdown.")

    def ensure_symbol(self, symbol):
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            print(f"✗ {symbol}: Not available")
            if mt5.symbol_select(symbol, True):
                print(f"  → Enabled {symbol} in Market Watch")
                symbol_info = mt5.symbol_info(symbol)
            else:
                print(f"  → Failed to enable {symbol}")
                return None
        if symbol_info is not None:
            print(f"✓ {symbol}: Available (Spread: {symbol_info.spread})")
        return symbol_info

    def download_symbol_timeframe(self, symbol, timeframe, start_date, end_date, extra_columns=True):
        tf_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
        }
        tf_value = tf_map.get(timeframe.upper())
        if tf_value is None:
            print(f"Unsupported timeframe: {timeframe}")
            return None

        # Ensure start_date and end_date are datetime objects
        if isinstance(start_date, str):
            try:
                start_date = datetime.fromisoformat(start_date)
            except Exception:
                start_date = datetime.strptime(start_date, "%Y-%m-%d")
        if isinstance(end_date, str):
            try:
                end_date = datetime.fromisoformat(end_date)
            except Exception:
                end_date = datetime.strptime(end_date, "%Y-%m-%d")

        base_filename = os.path.join(self.output_dir, f"{symbol}_{timeframe}_2year_{self.broker}")
        parquet_filename = base_filename + ".parquet"
        # Check if file exists and is non-empty
        if os.path.exists(parquet_filename):
            try:
                df_existing = pd.read_parquet(parquet_filename)
                if not df_existing.empty:
                    print(f"    ⏩ Skipping {symbol} {timeframe} ({self.broker}): {parquet_filename} already exists and is non-empty.")
                    return parquet_filename
            except Exception as e:
                print(f"    Warning: Could not read {parquet_filename} ({e}), will re-download and overwrite.")
                try:
                    os.remove(parquet_filename)
                except Exception as del_e:
                    print(f"    Error deleting corrupt file {parquet_filename}: {del_e}")

        rates = mt5.copy_rates_range(symbol, tf_value, start_date, end_date)
        if rates is not None and len(rates) > 0:
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            if extra_columns:
                df['symbol'] = symbol
                df['timeframe'] = timeframe
                df['broker'] = self.broker
                df['hl_range'] = df['high'] - df['low']
                df['oc_range'] = abs(df['close'] - df['open'])
            csv_filename = base_filename + ".csv"
            h5_filename = base_filename + ".h5"
            df.to_csv(csv_filename, index=False)
            df.to_parquet(parquet_filename, index=False)
            df.to_hdf(h5_filename, key="df", mode="w")
            print(f"    ✓ {len(df)} bars saved to {csv_filename}, {parquet_filename}, and {h5_filename}")
            print(f"    Date range: {df['time'].min()} to {df['time'].max()}")
            return csv_filename
        else:
            print(f"    ✗ No data available for {symbol} {timeframe} ({self.broker})")
            last_error = mt5.last_error()
            if last_error[0] != 1:
                print(f"    Error: {last_error}")
            return None

    def batch_download(self, symbols, timeframes, start_date, end_date):
        total_downloads = 0
        successful_downloads = 0
        files = []
        for symbol in symbols:
            print(f"\n--- Processing {symbol} ---")
            symbol_info = self.ensure_symbol(symbol)
            if symbol_info is None:
                continue
            for tf in timeframes:
                total_downloads += 1
                print(f"  Downloading {symbol} {tf}...")
                filename = self.download_symbol_timeframe(symbol, tf, start_date, end_date)
                if filename:
                    successful_downloads += 1
                    files.append(filename)
        print(f"\n=== Download Summary ===")
        print(f"Total attempts: {total_downloads}")
        print(f"Successful downloads: {successful_downloads}")
        print(f"Success rate: {(successful_downloads/total_downloads)*100:.1f}%")
        return files

    def list_downloaded_files(self):
        print(f"\nDownloaded files in '{self.output_dir}' directory:")
        try:
            files = [f for f in os.listdir(self.output_dir) if f.endswith('_2year.csv')]
            for file in sorted(files):
                file_path = os.path.join(self.output_dir, file)
                file_size = os.path.getsize(file_path)
                print(f"  {file} ({file_size:,} bytes)")
        except Exception as e:
            print(f"  Error listing files: {e}")

    def get_data_summary(self):
        print("\n=== Data Summary ===")
        if not os.path.exists(self.output_dir):
            print("No data directory found")
            return
        files = [f for f in os.listdir(self.output_dir) if f.endswith('_2year.csv')]
        if not files:
            print("No 2-year data files found")
            return
        total_size = 0
        for file in sorted(files):
            file_path = os.path.join(self.output_dir, file)
            file_size = os.path.getsize(file_path)
            total_size += file_size
            try:
                df = pd.read_csv(file_path, nrows=5)
                print(f"{file}:")
                print(f"  Size: {file_size:,} bytes")
                if len(df) > 0:
                    print(f"  Columns: {', '.join(df.columns.tolist())}")
            except Exception as e:
                print(f"  Error reading {file}: {e}")
        print(f"\nTotal data size: {total_size:,} bytes ({total_size/1024/1024:.1f} MB)")
