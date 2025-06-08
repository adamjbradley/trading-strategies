import pandas as pd
import numpy as np
import os
import glob
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnx
import onnxruntime as ort
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class GridTradingModelTrainer:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = []
        
    def load_data(self, use_parquet=True):
        """Load and combine all available data"""
        print("Loading data...")
        
        if use_parquet and os.path.exists(os.path.join(self.data_dir, 'parquet')):
            # Load from optimized Parquet files
            parquet_files = glob.glob(os.path.join(self.data_dir, 'parquet', 'ICM_*_H1_2year.parquet'))
            if parquet_files:
                print(f"Loading {len(parquet_files)} Parquet files...")
                dfs = []
                for file in parquet_files:
                    df = pd.read_parquet(file)
                    dfs.append(df)
                combined_df = pd.concat(dfs, ignore_index=True)
                print(f"✓ Loaded {len(combined_df):,} rows from Parquet files")
                return combined_df
        
        # Fallback to CSV files
        csv_files = glob.glob(os.path.join(self.data_dir, 'ICM_*_H1_2year.csv'))
        if not csv_files:
            raise FileNotFoundError("No data files found. Please run data download first.")
        
        print(f"Loading {len(csv_files)} CSV files...")
        dfs = []
        for file in csv_files:
            df = pd.read_csv(file)
            df['time'] = pd.to_datetime(df['time'])
            dfs.append(df)
        
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"✓ Loaded {len(combined_df):,} rows from CSV files")
        return combined_df
    
    def engineer_features(self, df):
        """Create features for grid trading strategy"""
        print("Engineering features...")
        
        # Sort by symbol and time
        df = df.sort_values(['symbol', 'time']).reset_index(drop=True)
        
        features_df = df.copy()
        
        # Basic price features
        features_df['hl_range'] = features_df['high'] - features_df['low']
        features_df['oc_range'] = abs(features_df['close'] - features_df['open'])
        features_df['price_change'] = features_df['close'] - features_df['open']
        features_df['price_change_pct'] = features_df['price_change'] / features_df['open']
        
        # Technical indicators by symbol
        symbol_features = []
        
        for symbol in features_df['symbol'].unique():
            symbol_data = features_df[features_df['symbol'] == symbol].copy()
            
            # Moving averages
            symbol_data['sma_5'] = symbol_data['close'].rolling(5).mean()
            symbol_data['sma_20'] = symbol_data['close'].rolling(20).mean()
            symbol_data['sma_50'] = symbol_data['close'].rolling(50).mean()
            
            # Price relative to moving averages
            symbol_data['price_vs_sma5'] = symbol_data['close'] / symbol_data['sma_5'] - 1
            symbol_data['price_vs_sma20'] = symbol_data['close'] / symbol_data['sma_20'] - 1
            symbol_data['price_vs_sma50'] = symbol_data['close'] / symbol_data['sma_50'] - 1
            
            # Volatility measures
            symbol_data['volatility_5'] = symbol_data['hl_range'].rolling(5).mean()
            symbol_data['volatility_20'] = symbol_data['hl_range'].rolling(20).mean()
            symbol_data['price_volatility'] = symbol_data['close'].rolling(20).std()
            
            # Momentum indicators
            symbol_data['roc_5'] = symbol_data['close'].pct_change(5)
            symbol_data['roc_20'] = symbol_data['close'].pct_change(20)
            
            # Support/Resistance levels (simplified)
            symbol_data['recent_high'] = symbol_data['high'].rolling(20).max()
            symbol_data['recent_low'] = symbol_data['low'].rolling(20).min()
            symbol_data['price_position'] = (symbol_data['close'] - symbol_data['recent_low']) / (symbol_data['recent_high'] - symbol_data['recent_low'])
            
            # Volume indicators
            symbol_data['volume_sma'] = symbol_data['tick_volume'].rolling(20).mean()
            symbol_data['volume_ratio'] = symbol_data['tick_volume'] / symbol_data['volume_sma']
            
            # Trend strength
            symbol_data['trend_strength'] = abs(symbol_data['sma_5'] - symbol_data['sma_20']) / symbol_data['close']
            
            symbol_features.append(symbol_data)
        
        # Combine all symbols
        features_df = pd.concat(symbol_features, ignore_index=True)
        
        # Hour of day (important for forex)
        features_df['hour'] = features_df['time'].dt.hour
        features_df['day_of_week'] = features_df['time'].dt.dayofweek
        
        # Encode symbol as numeric
        le = LabelEncoder()
        features_df['symbol_encoded'] = le.fit_transform(features_df['symbol'])
        
        print(f"✓ Created {len(features_df.columns)} features")
        return features_df
    
    def create_targets(self, df):
        """Create target variables for grid trading"""
        print("Creating target variables...")
        
        targets_df = df.copy()
        
        # Target 1: High volatility in next period (good for grid trading)
        volatility_threshold = df['hl_range'].quantile(0.75)
        
        # Create targets by symbol to avoid index alignment issues
        target_list = []
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol].copy()
            
            # High volatility next period
            symbol_data['high_volatility_next'] = (symbol_data['hl_range'].shift(-1) > volatility_threshold).astype(int)
            
            # Range expansion
            rolling_mean = symbol_data['hl_range'].rolling(5).mean()
            symbol_data['range_expansion'] = (symbol_data['hl_range'].shift(-1) > rolling_mean).astype(int)
            
            # Mean reversion opportunity (if price_vs_sma20 exists)
            if 'price_vs_sma20' in symbol_data.columns:
                rolling_std = symbol_data['price_vs_sma20'].rolling(20).std()
                symbol_data['mean_reversion'] = (abs(symbol_data['price_vs_sma20'].shift(-1)) > rolling_std).astype(int)
            else:
                symbol_data['mean_reversion'] = 0
            
            target_list.append(symbol_data)
        
        # Combine all symbols
        targets_df = pd.concat(target_list, ignore_index=True)
        
        # Combined target: Good grid trading conditions
        targets_df['grid_opportunity'] = (
            (targets_df['high_volatility_next'] == 1) | 
            (targets_df['range_expansion'] == 1)
        ).astype(int)
        
        print("✓ Created target variables")
        return targets_df
    
    def prepare_training_data(self, df):
        """Prepare features and targets for training"""
        print("Preparing training data...")
        
        # Select feature columns (exclude metadata and targets)
        feature_cols = [col for col in df.columns if col not in [
            'time', 'symbol', 'timeframe', 'broker', 'open', 'high', 'low', 'close',
            'tick_volume', 'spread', 'real_volume', 'high_volatility_next',
            'range_expansion', 'mean_reversion', 'grid_opportunity'
        ]]
        
        X = df[feature_cols].copy()
        y = df['grid_opportunity'].copy()
        
        # Remove rows with NaN values
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        self.feature_names = feature_cols
        
        print(f"✓ Training data shape: {X.shape}")
        print(f"✓ Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def train_model(self, X, y, model_type='random_forest'):
        """Train the machine learning model"""
        print(f"Training {model_type} model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Choose model
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        print(f"✓ Training accuracy: {train_score:.3f}")
        print(f"✓ Test accuracy: {test_score:.3f}")
        
        # Detailed evaluation
        y_pred = self.model.predict(X_test_scaled)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nTop 10 Most Important Features:")
            print(feature_importance.head(10).to_string(index=False))
        
        return X_test_scaled, y_test, y_pred
    
    def export_to_onnx(self, output_path="models/grid_trading_model.onnx"):
        """Export trained model to ONNX format"""
        print("Exporting model to ONNX...")
        
        if self.model is None:
            raise ValueError("No trained model found. Train a model first.")
        
        # Create models directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Define input type
        initial_type = [('float_input', FloatTensorType([None, len(self.feature_names)]))]
        
        # Convert to ONNX
        onnx_model = convert_sklearn(
            self.model, 
            initial_types=initial_type,
            target_opset=11
        )
        
        # Save ONNX model
        with open(output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        
        # Save scaler and feature names
        scaler_path = output_path.replace('.onnx', '_scaler.joblib')
        features_path = output_path.replace('.onnx', '_features.joblib')
        
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.feature_names, features_path)
        
        print(f"✓ ONNX model saved: {output_path}")
        print(f"✓ Scaler saved: {scaler_path}")
        print(f"✓ Features saved: {features_path}")
        
        # Test ONNX model
        self.test_onnx_model(output_path, scaler_path)
        
        return output_path
    
    def test_onnx_model(self, onnx_path, scaler_path):
        """Test the exported ONNX model"""
        print("Testing ONNX model...")
        
        try:
            # Load ONNX model
            ort_session = ort.InferenceSession(onnx_path)
            
            # Load scaler
            scaler = joblib.load(scaler_path)
            
            # Create test data
            test_data = np.random.randn(5, len(self.feature_names)).astype(np.float32)
            test_data_scaled = scaler.transform(test_data).astype(np.float32)
            
            # Run inference
            input_name = ort_session.get_inputs()[0].name
            predictions = ort_session.run(None, {input_name: test_data_scaled})
            
            print(f"✓ ONNX model test successful")
            print(f"✓ Input shape: {test_data_scaled.shape}")
            print(f"✓ Output shape: {predictions[0].shape}")
            
        except Exception as e:
            print(f"✗ ONNX model test failed: {e}")

def main():
    """Main training pipeline"""
    print("Grid Trading ONNX Model Training Pipeline")
    print("=" * 50)
    
    # Initialize trainer
    trainer = GridTradingModelTrainer()
    
    try:
        # Load data
        df = trainer.load_data(use_parquet=True)
        
        # Engineer features
        df = trainer.engineer_features(df)
        
        # Create targets
        df = trainer.create_targets(df)
        
        # Prepare training data
        X, y = trainer.prepare_training_data(df)
        
        # Train model
        X_test, y_test, y_pred = trainer.train_model(X, y, model_type='random_forest')
        
        # Export to ONNX
        onnx_path = trainer.export_to_onnx("models/grid_trading_model.onnx")
        
        print(f"\n" + "=" * 50)
        print("✅ Model training completed successfully!")
        print(f"📁 Model files saved in: models/")
        print(f"🤖 ONNX model: {onnx_path}")
        print(f"📊 Features: {len(trainer.feature_names)}")
        print(f"📈 Training data: {len(X):,} samples")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
