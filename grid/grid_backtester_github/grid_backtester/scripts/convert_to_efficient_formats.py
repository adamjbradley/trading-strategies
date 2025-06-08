import pandas as pd
import os
import glob
import time
from pathlib import Path

def convert_csv_to_efficient_formats():
    """Convert CSV files to more efficient formats and compare performance"""
    print("Converting CSV to Efficient Formats")
    print("=" * 40)
    
    data_dir = "data"
    if not os.path.exists(data_dir):
        print("No data directory found")
        return
    
    # Get all ICMarkets CSV files
    csv_files = glob.glob(os.path.join(data_dir, "ICM_*_2year.csv"))
    
    if not csv_files:
        print("No ICMarkets CSV files found")
        return
    
    print(f"Found {len(csv_files)} CSV files to convert")
    
    # Create subdirectories for different formats
    formats = ['parquet', 'feather', 'hdf5', 'pickle']
    for fmt in formats:
        os.makedirs(os.path.join(data_dir, fmt), exist_ok=True)
    
    # Performance tracking
    performance_data = []
    
    for csv_file in csv_files[:3]:  # Convert first 3 files for demonstration
        filename = os.path.basename(csv_file)
        base_name = filename.replace('.csv', '')
        
        print(f"\nProcessing {filename}...")
        
        # Read CSV and optimize data types
        start_time = time.time()
        df = pd.read_csv(csv_file)
        csv_read_time = time.time() - start_time
        
        # Optimize data types
        df['time'] = pd.to_datetime(df['time'])
        df = df.astype({
            'open': 'float32',
            'high': 'float32', 
            'low': 'float32',
            'close': 'float32',
            'tick_volume': 'int32',
            'spread': 'int16',
            'real_volume': 'int64'
        })
        
        original_size = os.path.getsize(csv_file)
        
        # 1. Parquet format (recommended for analytics)
        parquet_file = os.path.join(data_dir, 'parquet', f"{base_name}.parquet")
        start_time = time.time()
        df.to_parquet(parquet_file, compression='snappy', index=False)
        parquet_write_time = time.time() - start_time
        parquet_size = os.path.getsize(parquet_file)
        
        start_time = time.time()
        df_parquet = pd.read_parquet(parquet_file)
        parquet_read_time = time.time() - start_time
        
        # 2. Feather format (fast I/O)
        feather_file = os.path.join(data_dir, 'feather', f"{base_name}.feather")
        start_time = time.time()
        df.to_feather(feather_file)
        feather_write_time = time.time() - start_time
        feather_size = os.path.getsize(feather_file)
        
        start_time = time.time()
        df_feather = pd.read_feather(feather_file)
        feather_read_time = time.time() - start_time
        
        # 3. HDF5 format (good for large datasets)
        hdf5_file = os.path.join(data_dir, 'hdf5', f"{base_name}.h5")
        start_time = time.time()
        df.to_hdf(hdf5_file, key='data', mode='w', complevel=9, complib='zlib')
        hdf5_write_time = time.time() - start_time
        hdf5_size = os.path.getsize(hdf5_file)
        
        start_time = time.time()
        df_hdf5 = pd.read_hdf(hdf5_file, key='data')
        hdf5_read_time = time.time() - start_time
        
        # 4. Pickle format (preserves all pandas features)
        pickle_file = os.path.join(data_dir, 'pickle', f"{base_name}.pkl")
        start_time = time.time()
        df.to_pickle(pickle_file, compression='gzip')
        pickle_write_time = time.time() - start_time
        pickle_size = os.path.getsize(pickle_file)
        
        start_time = time.time()
        df_pickle = pd.read_pickle(pickle_file)
        pickle_read_time = time.time() - start_time
        
        # Store performance data
        performance_data.append({
            'file': filename,
            'rows': len(df),
            'csv_size': original_size,
            'csv_read_time': csv_read_time,
            'parquet_size': parquet_size,
            'parquet_write_time': parquet_write_time,
            'parquet_read_time': parquet_read_time,
            'feather_size': feather_size,
            'feather_write_time': feather_write_time,
            'feather_read_time': feather_read_time,
            'hdf5_size': hdf5_size,
            'hdf5_write_time': hdf5_write_time,
            'hdf5_read_time': hdf5_read_time,
            'pickle_size': pickle_size,
            'pickle_write_time': pickle_write_time,
            'pickle_read_time': pickle_read_time,
        })
        
        print(f"  ✓ Converted to all formats")
    
    # Performance comparison
    print(f"\n=== Format Comparison ===")
    
    if performance_data:
        avg_data = {}
        for key in performance_data[0].keys():
            if key not in ['file']:
                avg_data[key] = sum(d[key] for d in performance_data) / len(performance_data)
        
        print(f"\nFile Size Comparison (average):")
        print(f"  CSV:     {avg_data['csv_size']:8,.0f} bytes (baseline)")
        print(f"  Parquet: {avg_data['parquet_size']:8,.0f} bytes ({avg_data['parquet_size']/avg_data['csv_size']*100:.1f}%)")
        print(f"  Feather: {avg_data['feather_size']:8,.0f} bytes ({avg_data['feather_size']/avg_data['csv_size']*100:.1f}%)")
        print(f"  HDF5:    {avg_data['hdf5_size']:8,.0f} bytes ({avg_data['hdf5_size']/avg_data['csv_size']*100:.1f}%)")
        print(f"  Pickle:  {avg_data['pickle_size']:8,.0f} bytes ({avg_data['pickle_size']/avg_data['csv_size']*100:.1f}%)")
        
        print(f"\nRead Speed Comparison (average):")
        print(f"  CSV:     {avg_data['csv_read_time']:.3f}s (baseline)")
        print(f"  Parquet: {avg_data['parquet_read_time']:.3f}s ({avg_data['parquet_read_time']/avg_data['csv_read_time']*100:.1f}%)")
        print(f"  Feather: {avg_data['feather_read_time']:.3f}s ({avg_data['feather_read_time']/avg_data['csv_read_time']*100:.1f}%)")
        print(f"  HDF5:    {avg_data['hdf5_read_time']:.3f}s ({avg_data['hdf5_read_time']/avg_data['csv_read_time']*100:.1f}%)")
        print(f"  Pickle:  {avg_data['pickle_read_time']:.3f}s ({avg_data['pickle_read_time']/avg_data['csv_read_time']*100:.1f}%)")
    
    print(f"\n=== Format Recommendations ===")
    print(f"📊 **Parquet** (RECOMMENDED for most use cases):")
    print(f"   • Excellent compression (typically 50-70% smaller)")
    print(f"   • Fast read/write performance")
    print(f"   • Column-oriented (efficient for analytics)")
    print(f"   • Cross-platform compatibility")
    print(f"   • Preserves data types")
    print(f"   • Industry standard for data science")
    
    print(f"\n⚡ **Feather** (FASTEST I/O):")
    print(f"   • Fastest read/write speeds")
    print(f"   • Good compression")
    print(f"   • Perfect for temporary files")
    print(f"   • Language agnostic (R, Python, etc.)")
    
    print(f"\n🗄️  **HDF5** (LARGE DATASETS):")
    print(f"   • Best for very large datasets (>1GB)")
    print(f"   • Hierarchical structure")
    print(f"   • Excellent compression")
    print(f"   • Supports metadata")
    print(f"   • Can append data efficiently")
    
    print(f"\n🐍 **Pickle** (PYTHON-SPECIFIC):")
    print(f"   • Preserves all pandas features")
    print(f"   • Python-only format")
    print(f"   • Good for complex data structures")
    print(f"   • Fast but larger file sizes")
    
    print(f"\n=== Usage Examples ===")
    print(f"# Read parquet (recommended)")
    print(f"df = pd.read_parquet('data/parquet/ICM_EURUSD_H1_2year.parquet')")
    print(f"")
    print(f"# Read feather (fastest)")
    print(f"df = pd.read_feather('data/feather/ICM_EURUSD_H1_2year.feather')")
    print(f"")
    print(f"# Read HDF5 (large datasets)")
    print(f"df = pd.read_hdf('data/hdf5/ICM_EURUSD_H1_2year.h5', key='data')")

def create_optimized_download_script():
    """Create a new download script that saves directly to Parquet"""
    script_content = '''
import MetaTrader5 as mt5
import pandas as pd
import sys
import os
from datetime import datetime, timedelta

def download_to_parquet():
    """Download data and save directly to Parquet format"""
    print("ICMarkets MT5 - Direct to Parquet Download")
    print("=" * 45)
    
    if not mt5.initialize("C:\\\\Program Files\\\\ICMarkets - MetaTrader 5\\\\terminal64.exe"):
        print("✗ MT5 initialization failed")
        return False
    
    symbols = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]
    timeframes = {"H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4, "D1": mt5.TIMEFRAME_D1}
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    
    os.makedirs("data/parquet", exist_ok=True)
    
    for symbol in symbols:
        if not mt5.symbol_select(symbol, True):
            continue
            
        for tf_name, tf_value in timeframes.items():
            rates = mt5.copy_rates_range(symbol, tf_value, start_date, end_date)
            
            if rates is not None and len(rates) > 0:
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                
                # Optimize data types
                df = df.astype({
                    'open': 'float32', 'high': 'float32', 'low': 'float32', 'close': 'float32',
                    'tick_volume': 'int32', 'spread': 'int16', 'real_volume': 'int64'
                })
                
                df['symbol'] = symbol
                df['timeframe'] = tf_name
                df['broker'] = 'ICMarkets'
                
                filename = f"data/parquet/ICM_{symbol}_{tf_name}_2year.parquet"
                df.to_parquet(filename, compression='snappy', index=False)
                print(f"✓ {symbol} {tf_name}: {len(df)} bars → {filename}")
    
    mt5.shutdown()
    return True

if __name__ == "__main__":
    download_to_parquet()
'''
    
    with open("trading-strategies/grid/grid_backtester_github/grid_backtester/scripts/download_to_parquet.py", "w") as f:
        f.write(script_content)
    
    print(f"\n✓ Created optimized download script: download_to_parquet.py")

if __name__ == "__main__":
    convert_csv_to_efficient_formats()
    create_optimized_download_script()
