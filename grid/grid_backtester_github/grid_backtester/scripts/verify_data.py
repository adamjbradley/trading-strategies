import pandas as pd
import os
import glob

def verify_downloaded_data():
    """Verify the quality and completeness of downloaded data"""
    print("Data Verification Report")
    print("=" * 30)
    
    data_dir = "data"
    if not os.path.exists(data_dir):
        print("No data directory found")
        return
    
    # Get all ICMarkets 2-year files
    icm_files = glob.glob(os.path.join(data_dir, "ICM_*_2year.csv"))
    
    if not icm_files:
        print("No ICMarkets 2-year data files found")
        return
    
    print(f"Found {len(icm_files)} ICMarkets data files")
    
    total_size = 0
    total_rows = 0
    
    # Group files by symbol and timeframe
    symbols = {}
    
    for file_path in sorted(icm_files):
        filename = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        total_size += file_size
        
        # Parse filename: ICM_SYMBOL_TIMEFRAME_2year.csv
        parts = filename.replace('.csv', '').split('_')
        if len(parts) >= 3:
            symbol = parts[1]
            timeframe = parts[2]
            
            if symbol not in symbols:
                symbols[symbol] = {}
            
            try:
                df = pd.read_csv(file_path)
                row_count = len(df)
                total_rows += row_count
                
                # Basic data validation
                df['time'] = pd.to_datetime(df['time'])
                date_range = f"{df['time'].min().strftime('%Y-%m-%d')} to {df['time'].max().strftime('%Y-%m-%d')}"
                
                # Check for missing values
                missing_values = df[['open', 'high', 'low', 'close', 'tick_volume']].isnull().sum().sum()
                
                # Check data integrity
                invalid_ohlc = ((df['high'] < df['low']) | 
                               (df['high'] < df['open']) | 
                               (df['high'] < df['close']) |
                               (df['low'] > df['open']) |
                               (df['low'] > df['close'])).sum()
                
                symbols[symbol][timeframe] = {
                    'rows': row_count,
                    'size': file_size,
                    'date_range': date_range,
                    'missing_values': missing_values,
                    'invalid_ohlc': invalid_ohlc,
                    'filename': filename
                }
                
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    
    # Display summary by symbol
    print(f"\n=== Data Summary by Symbol ===")
    for symbol in sorted(symbols.keys()):
        print(f"\n{symbol}:")
        for timeframe in ['H1', 'H4', 'D1']:
            if timeframe in symbols[symbol]:
                data = symbols[symbol][timeframe]
                print(f"  {timeframe}: {data['rows']:,} bars, {data['size']:,} bytes")
                print(f"       Range: {data['date_range']}")
                if data['missing_values'] > 0:
                    print(f"       ⚠️  Missing values: {data['missing_values']}")
                if data['invalid_ohlc'] > 0:
                    print(f"       ⚠️  Invalid OHLC: {data['invalid_ohlc']}")
                else:
                    print(f"       ✓ Data integrity: OK")
    
    print(f"\n=== Overall Summary ===")
    print(f"Total files: {len(icm_files)}")
    print(f"Total data size: {total_size:,} bytes ({total_size/1024/1024:.1f} MB)")
    print(f"Total rows: {total_rows:,}")
    print(f"Average file size: {total_size/len(icm_files):,.0f} bytes")
    
    # Check timeframe coverage
    expected_timeframes = ['H1', 'H4', 'D1']
    expected_symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD']
    
    print(f"\n=== Coverage Check ===")
    missing_data = []
    for symbol in expected_symbols:
        for tf in expected_timeframes:
            if symbol not in symbols or tf not in symbols[symbol]:
                missing_data.append(f"{symbol}_{tf}")
    
    if missing_data:
        print(f"⚠️  Missing data: {', '.join(missing_data)}")
    else:
        print("✓ All expected symbol/timeframe combinations present")
    
    # Sample data preview
    print(f"\n=== Sample Data Preview (EURUSD H1) ===")
    sample_file = os.path.join(data_dir, "ICM_EURUSD_H1_2year.csv")
    if os.path.exists(sample_file):
        df_sample = pd.read_csv(sample_file, nrows=5)
        print(df_sample.to_string(index=False))
    
    print(f"\n✓ Data verification completed!")

if __name__ == "__main__":
    verify_downloaded_data()
