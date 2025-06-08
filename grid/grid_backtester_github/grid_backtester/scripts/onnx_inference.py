import pandas as pd
import numpy as np
import onnxruntime as ort
import joblib
import os
from datetime import datetime
import MetaTrader5 as mt5

class GridTradingONNXPredictor:
    def __init__(self, model_path="models/grid_trading_model.onnx"):
        """Initialize the ONNX predictor"""
        self.model_path = model_path
        self.scaler_path = model_path.replace('.onnx', '_scaler.joblib')
        self.features_path = model_path.replace('.onnx', '_features.joblib')
        
        # Load model components
        self.session = None
        self.scaler = None
        self.feature_names = None
        self.input_name = None
        
        self.load_model()
    
    def load_model(self):
        """Load ONNX model and preprocessing components"""
        try:
            # Load ONNX model
            self.session = ort.InferenceSession(self.model_path)
            self.input_name = self.session.get_inputs()[0].name
            
            # Load scaler and feature names
            self.scaler = joblib.load(self.scaler_path)
            self.feature_names = joblib.load(self.features_path)
            
            print(f"✓ ONNX model loaded: {self.model_path}")
            print(f"✓ Features: {len(self.feature_names)}")
            print(f"✓ Input shape: {self.session.get_inputs()[0].shape}")
            print(f"✓ Output shape: {self.session.get_outputs()[0].shape}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def engineer_features_single(self, df):
        """Engineer features for a single symbol's data"""
        if len(df) < 50:  # Need enough data for rolling calculations
            raise ValueError("Need at least 50 data points for feature engineering")
        
        features_df = df.copy()
        
        # Basic price features
        features_df['hl_range'] = features_df['high'] - features_df['low']
        features_df['oc_range'] = abs(features_df['close'] - features_df['open'])
        features_df['price_change'] = features_df['close'] - features_df['open']
        features_df['price_change_pct'] = features_df['price_change'] / features_df['open']
        
        # Moving averages
        features_df['sma_5'] = features_df['close'].rolling(5).mean()
        features_df['sma_20'] = features_df['close'].rolling(20).mean()
        features_df['sma_50'] = features_df['close'].rolling(50).mean()
        
        # Price relative to moving averages
        features_df['price_vs_sma5'] = features_df['close'] / features_df['sma_5'] - 1
        features_df['price_vs_sma20'] = features_df['close'] / features_df['sma_20'] - 1
        features_df['price_vs_sma50'] = features_df['close'] / features_df['sma_50'] - 1
        
        # Volatility measures
        features_df['volatility_5'] = features_df['hl_range'].rolling(5).mean()
        features_df['volatility_20'] = features_df['hl_range'].rolling(20).mean()
        features_df['price_volatility'] = features_df['close'].rolling(20).std()
        
        # Momentum indicators
        features_df['roc_5'] = features_df['close'].pct_change(5)
        features_df['roc_20'] = features_df['close'].pct_change(20)
        
        # Support/Resistance levels
        features_df['recent_high'] = features_df['high'].rolling(20).max()
        features_df['recent_low'] = features_df['low'].rolling(20).min()
        features_df['price_position'] = (features_df['close'] - features_df['recent_low']) / (features_df['recent_high'] - features_df['recent_low'])
        
        # Volume indicators
        features_df['volume_sma'] = features_df['tick_volume'].rolling(20).mean()
        features_df['volume_ratio'] = features_df['tick_volume'] / features_df['volume_sma']
        
        # Trend strength
        features_df['trend_strength'] = abs(features_df['sma_5'] - features_df['sma_20']) / features_df['close']
        
        # Time features
        features_df['hour'] = features_df['time'].dt.hour
        features_df['day_of_week'] = features_df['time'].dt.dayofweek
        
        # Symbol encoding (for single symbol, use a default value)
        features_df['symbol_encoded'] = 0  # Will be updated if needed
        
        return features_df
    
    def predict(self, df, symbol="EURUSD"):
        """Make predictions on new data"""
        # Engineer features
        features_df = self.engineer_features_single(df)
        
        # Select only the features used in training
        try:
            X = features_df[self.feature_names].copy()
        except KeyError as e:
            missing_features = set(self.feature_names) - set(features_df.columns)
            raise ValueError(f"Missing features: {missing_features}")
        
        # Remove rows with NaN values
        valid_mask = ~X.isnull().any(axis=1)
        X_clean = X[valid_mask]
        
        if len(X_clean) == 0:
            raise ValueError("No valid data points after feature engineering")
        
        # Scale features
        X_scaled = self.scaler.transform(X_clean).astype(np.float32)
        
        # Make predictions
        predictions = self.session.run(None, {self.input_name: X_scaled})[0]
        
        # Create results dataframe
        results = pd.DataFrame({
            'time': features_df.loc[valid_mask, 'time'].values,
            'close': features_df.loc[valid_mask, 'close'].values,
            'grid_opportunity': predictions,
            'grid_probability': predictions  # For binary classification, this is 0 or 1
        })
        
        return results
    
    def predict_live_mt5(self, symbol="EURUSD", timeframe=mt5.TIMEFRAME_H1, bars=100):
        """Make predictions on live MT5 data"""
        try:
            import MetaTrader5 as mt5
        except ImportError:
            print("MetaTrader5 package not found. Skipping live prediction.")
            return None

        if not mt5.initialize():
            print("Failed to initialize MT5. Skipping live prediction.")
            print(f"Error: {mt5.last_error()}")
            print("Please ensure the MetaTrader 5 terminal is running and you are logged in.")
            return None
        
        try:
            # Get recent data
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
            if rates is None:
                raise ValueError(f"Failed to get data for {symbol}")
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Make predictions
            results = self.predict(df, symbol)
            
            # Get latest prediction
            latest = results.iloc[-1]
            
            print(f"\n=== Live Prediction for {symbol} ===")
            print(f"Time: {latest['time']}")
            print(f"Price: {latest['close']:.5f}")
            print(f"Grid Opportunity: {'YES' if latest['grid_opportunity'] == 1 else 'NO'}")
            print(f"Confidence: {latest['grid_probability']:.3f}")

            return results
            
        finally:
            mt5.shutdown()
    
    def predict_from_file(self, file_path, symbol_filter=None):
        """Make predictions on data from file"""
        # Load data
        if file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        else:
            df = pd.read_csv(file_path)
            df['time'] = pd.to_datetime(df['time'])
        
        # Filter by symbol if specified
        if symbol_filter and 'symbol' in df.columns:
            df = df[df['symbol'] == symbol_filter]
        
        if len(df) == 0:
            raise ValueError("No data found after filtering")
        
        # Make predictions
        results = self.predict(df)
        
        # Summary statistics
        total_predictions = len(results)
        grid_opportunities = (results['grid_opportunity'] == 1).sum()
        opportunity_rate = grid_opportunities / total_predictions * 100
        
        print(f"\n=== Prediction Summary ===")
        print(f"Total predictions: {total_predictions:,}")
        print(f"Grid opportunities: {grid_opportunities:,} ({opportunity_rate:.1f}%)")
        print(f"Date range: {results['time'].min()} to {results['time'].max()}")
        
        return results

def demo_predictions():
    """Demonstrate the ONNX predictor"""
    print("Grid Trading ONNX Model Inference Demo")
    print("=" * 45)
    
    try:
        # Initialize predictor
        predictor = GridTradingONNXPredictor()
        
        # Test with saved data
        print("\n1. Testing with saved Parquet data...")
        parquet_file = "data/parquet/ICM_EURUSD_H1_2year.parquet"
        if os.path.exists(parquet_file):
            results = predictor.predict_from_file(parquet_file)
            
            # Show recent predictions
            print("\nRecent predictions:")
            print(results.tail(10)[['time', 'close', 'grid_opportunity']].to_string(index=False))
        
        # Test with live MT5 data
        print("\n2. Testing with live MT5 data...")
        try:
            live_results = predictor.predict_live_mt5("EURUSD", mt5.TIMEFRAME_H1, 100)
            if live_results is not None:
                print("✓ Live prediction successful")
        except Exception as e:
            print(f"Live prediction failed: {e}")
        
        print("\n✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")

def main():
    """Main function for command line usage"""
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "demo":
            demo_predictions()
        elif command == "live":
            symbol = sys.argv[2] if len(sys.argv) > 2 else "EURUSD"
            predictor = GridTradingONNXPredictor()
            results = predictor.predict_live_mt5(symbol)
            if results is None:
                print("Could not retrieve live data.")
        elif command == "file":
            if len(sys.argv) < 3:
                print("Usage: python onnx_inference.py file <file_path> [symbol]")
                return
            file_path = sys.argv[2]
            symbol = sys.argv[3] if len(sys.argv) > 3 else None
            predictor = GridTradingONNXPredictor()
            predictor.predict_from_file(file_path, symbol)
        else:
            print("Usage: python onnx_inference.py [demo|live|file]")
    else:
        demo_predictions()

if __name__ == "__main__":
    main()
