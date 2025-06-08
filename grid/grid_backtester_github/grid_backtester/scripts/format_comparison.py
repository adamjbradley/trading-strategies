import pandas as pd
import os
import glob
import time

def compare_file_formats():
    """Compare CSV vs Parquet vs Feather formats"""
    print("File Format Comparison")
    print("=" * 30)
    
    data_dir = "data"
    csv_files = glob.glob(os.path.join(data_dir, "ICM_*_H1_2year.csv"))
    
    if not csv_files:
        print("No H1 CSV files found")
        return
    
    # Test with one H1 file (largest)
    test_file = csv_files[0]
    filename = os.path.basename(test_file)
    base_name = filename.replace('.csv', '')
    
    print(f"Testing with: {filename}")
    
    # Create format directories
    os.makedirs(os.path.join(data_dir, 'parquet'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'feather'), exist_ok=True)
    
    # Read CSV
    print("\n1. Reading CSV...")
    start_time = time.time()
    df = pd.read_csv(test_file)
    csv_read_time = time.time() - start_time
    csv_size = os.path.getsize(test_file)
    
    print(f"   Rows: {len(df):,}")
    print(f"   Size: {csv_size:,} bytes")
    print(f"   Read time: {csv_read_time:.3f}s")
    
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
    
    # Test Parquet
    print("\n2. Testing Parquet...")
    parquet_file = os.path.join(data_dir, 'parquet', f"{base_name}.parquet")
    
    start_time = time.time()
    df.to_parquet(parquet_file, compression='snappy', index=False)
    parquet_write_time = time.time() - start_time
    parquet_size = os.path.getsize(parquet_file)
    
    start_time = time.time()
    df_parquet = pd.read_parquet(parquet_file)
    parquet_read_time = time.time() - start_time
    
    print(f"   Size: {parquet_size:,} bytes ({parquet_size/csv_size*100:.1f}% of CSV)")
    print(f"   Write time: {parquet_write_time:.3f}s")
    print(f"   Read time: {parquet_read_time:.3f}s ({parquet_read_time/csv_read_time*100:.1f}% of CSV)")
    
    # Test Feather
    print("\n3. Testing Feather...")
    feather_file = os.path.join(data_dir, 'feather', f"{base_name}.feather")
    
    start_time = time.time()
    df.to_feather(feather_file)
    feather_write_time = time.time() - start_time
    feather_size = os.path.getsize(feather_file)
    
    start_time = time.time()
    df_feather = pd.read_feather(feather_file)
    feather_read_time = time.time() - start_time
    
    print(f"   Size: {feather_size:,} bytes ({feather_size/csv_size*100:.1f}% of CSV)")
    print(f"   Write time: {feather_write_time:.3f}s")
    print(f"   Read time: {feather_read_time:.3f}s ({feather_read_time/csv_read_time*100:.1f}% of CSV)")
    
    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"Format    | Size Reduction | Read Speed | Write Speed | Recommendation")
    print(f"----------|----------------|------------|-------------|---------------")
    print(f"CSV       | 0% (baseline) | 100%       | N/A         | Human readable")
    print(f"Parquet   | {100-parquet_size/csv_size*100:.0f}%            | {parquet_read_time/csv_read_time*100:.0f}%         | Fast        | ⭐ RECOMMENDED")
    print(f"Feather   | {100-feather_size/csv_size*100:.0f}%            | {feather_read_time/csv_read_time*100:.0f}%         | Fastest     | Speed critical")
    
    print(f"\n=== RECOMMENDATIONS ===")
    print(f"🏆 **Use Parquet for production**: Best balance of size and speed")
    print(f"⚡ **Use Feather for speed**: When you need fastest I/O")
    print(f"📝 **Keep CSV for debugging**: Human readable, universal compatibility")
    
    print(f"\n=== USAGE ===")
    print(f"# Load parquet (recommended)")
    print(f"df = pd.read_parquet('{parquet_file}')")
    print(f"")
    print(f"# Load feather (fastest)")
    print(f"df = pd.read_feather('{feather_file}')")

def convert_all_to_parquet():
    """Convert all CSV files to Parquet format"""
    print(f"\n" + "="*50)
    print("Converting All Files to Parquet")
    print("="*50)
    
    data_dir = "data"
    csv_files = glob.glob(os.path.join(data_dir, "ICM_*_2year.csv"))
    
    if not csv_files:
        print("No CSV files to convert")
        return
    
    os.makedirs(os.path.join(data_dir, 'parquet'), exist_ok=True)
    
    total_csv_size = 0
    total_parquet_size = 0
    converted_count = 0
    
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        base_name = filename.replace('.csv', '')
        parquet_file = os.path.join(data_dir, 'parquet', f"{base_name}.parquet")
        
        # Skip if already exists
        if os.path.exists(parquet_file):
            continue
        
        try:
            df = pd.read_csv(csv_file)
            
            # Optimize data types
            df['time'] = pd.to_datetime(df['time'])
            df = df.astype({
                'open': 'float32', 'high': 'float32', 'low': 'float32', 'close': 'float32',
                'tick_volume': 'int32', 'spread': 'int16', 'real_volume': 'int64'
            })
            
            df.to_parquet(parquet_file, compression='snappy', index=False)
            
            csv_size = os.path.getsize(csv_file)
            parquet_size = os.path.getsize(parquet_file)
            
            total_csv_size += csv_size
            total_parquet_size += parquet_size
            converted_count += 1
            
            print(f"✓ {filename} → {base_name}.parquet ({parquet_size/csv_size*100:.1f}% of original)")
            
        except Exception as e:
            print(f"✗ Failed to convert {filename}: {e}")
    
    if converted_count > 0:
        print(f"\n=== Conversion Summary ===")
        print(f"Files converted: {converted_count}")
        print(f"Total CSV size: {total_csv_size:,} bytes ({total_csv_size/1024/1024:.1f} MB)")
        print(f"Total Parquet size: {total_parquet_size:,} bytes ({total_parquet_size/1024/1024:.1f} MB)")
        print(f"Space saved: {total_csv_size - total_parquet_size:,} bytes ({(1-total_parquet_size/total_csv_size)*100:.1f}%)")
        
        print(f"\n✓ All files converted to Parquet format!")
        print(f"📁 Parquet files location: {os.path.join(data_dir, 'parquet')}")

if __name__ == "__main__":
    compare_file_formats()
    convert_all_to_parquet()
