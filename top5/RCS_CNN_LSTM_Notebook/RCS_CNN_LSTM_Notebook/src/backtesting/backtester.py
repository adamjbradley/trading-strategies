"""
Backtesting Module

This module provides functionality for backtesting trading strategies using ONNX models.

Example:
--------
from src.backtesting import run_backtest

# Run a backtest for EURUSD
results = run_backtest(
    model_path="exported_models/EURUSD_CNN_LSTM_20250611_083000.onnx",
    symbol="EURUSD",
    data_dir="data",
    scaler_path="models/scaler.pkl",  # Optional
    lookback=20,
    initial_capital=10000,
    commission=0.0001,
    plot=True
)
"""

import os
import numpy as np
import pandas as pd
import onnxruntime
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from ..features.selection import load_best_feature_set, filter_available_features
from ..utils.shape_handler import ensure_compatible_input_shape
import ta

class ONNXBacktester:
    def __init__(self, model_path, symbol, data_dir, lookback=20, initial_capital=10000, commission=0.0001):
        """
        Initialize the backtester.
        
        Parameters:
        -----------
        model_path : str
            Path to the ONNX model file
        symbol : str
            Trading symbol (e.g., 'EURUSD')
        data_dir : str
            Directory containing the data files
        lookback : int, default=20
            Number of time steps to use for sequence data
        initial_capital : float, default=10000
            Initial capital for backtesting
        commission : float, default=0.0001
            Commission rate per trade
        """
        print("\n🔧 Initializing backtester...")
        self.model_path = model_path
        self.symbol = symbol
        self.data_dir = data_dir
        self.lookback = lookback
        self.initial_capital = initial_capital
        self.commission = commission
        
        # Load the ONNX model
        print(f"\n📥 Loading ONNX model from {model_path}")
        self.session = onnxruntime.InferenceSession(model_path)
        
        # Get model metadata
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.expected_features = self.input_shape[-1]
        
        print("\n📊 Model Information:")
        print(f"  • Input shape: {self.input_shape}")
        print(f"  • Input name: {self.input_name}")
        print(f"  • Output name: {self.output_name}")
        print(f"  • Expected features: {self.expected_features}")
        print(f"  • Lookback period: {lookback}")
        print(f"  • Initial capital: ${initial_capital:,.2f}")
        print(f"  • Commission rate: {commission*100:.4f}%")
        
        # Define the features the model was trained with
        self.default_features = [
            'rsi', 'macd', 'momentum', 'cci', 'atr', 'adx', 'stoch_k', 'stoch_d', 
            'roc', 'bbw', 'return_1d', 'return_3d', 'rolling_mean_5', 'rolling_std_5', 
            'momentum_slope', 'dxy', '^vix', '^gspc', 'gc=f', 'cl=f', 
            'gold_oil_ratio', 'day_of_week', 'month'
        ]
    
    def load_data(self):
        """
        Load and prepare the data for backtesting.
        
        Returns:
        --------
        pandas.DataFrame
            Prepared data for backtesting
        """
        print(f"\n📂 Loading data for {self.symbol}")
        
        # Try to load data from different file formats
        data_files = [
            os.path.join(self.data_dir, f"metatrader_{self.symbol}.parquet"),
            os.path.join(self.data_dir, f"metatrader_{self.symbol}.h5"),
            os.path.join(self.data_dir, f"{self.symbol}.parquet"),
            os.path.join(self.data_dir, f"{self.symbol}.h5")
        ]
        
        df = None
        for file_path in data_files:
            if os.path.exists(file_path):
                print(f"  • Found data file: {file_path}")
                if file_path.endswith('.parquet'):
                    df = pd.read_parquet(file_path)
                else:  # .h5 file
                    df = pd.read_hdf(file_path)
                break
        
        if df is None:
            raise FileNotFoundError(f"No data file found for {self.symbol}")
        
        print(f"\n📊 Data Information:")
        print(f"  • Total rows: {len(df):,}")
        print(f"  • Date range: {df.index[0]} to {df.index[-1]}")
        print(f"  • Available columns: {', '.join(df.columns.tolist())}")
        return df
    
    def prepare_features(self, data):
        """
        Prepare features for the model.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Raw data
            
        Returns:
        --------
        pandas.DataFrame
            Data with prepared features
        """
        print("\n🔧 Preparing features...")
        features = []
        # Calculate technical indicators and only add if possible
        # RSI
        try:
            data['rsi'] = ta.momentum.RSIIndicator(data['close'], window=14).rsi()
            features.append('rsi')
        except Exception as e:
            print(f"  • Skipping rsi: {e}")
        # MACD
        try:
            macd = ta.trend.MACD(data['close'])
            data['macd'] = macd.macd()
            features.append('macd')
        except Exception as e:
            print(f"  • Skipping macd: {e}")
        # Momentum
        try:
            data['momentum'] = ta.momentum.ROCIndicator(data['close'], window=10).roc()
            features.append('momentum')
        except Exception as e:
            print(f"  • Skipping momentum: {e}")
        # CCI
        try:
            data['cci'] = ta.trend.CCIIndicator(data['high'], data['low'], data['close']).cci()
            features.append('cci')
        except Exception as e:
            print(f"  • Skipping cci: {e}")
        # ATR
        try:
            data['atr'] = ta.volatility.AverageTrueRange(data['high'], data['low'], data['close']).average_true_range()
            features.append('atr')
        except Exception as e:
            print(f"  • Skipping atr: {e}")
        # ADX
        try:
            data['adx'] = ta.trend.ADXIndicator(data['high'], data['low'], data['close']).adx()
            features.append('adx')
        except Exception as e:
            print(f"  • Skipping adx: {e}")
        # Stochastic
        try:
            stoch = ta.momentum.StochasticOscillator(data['high'], data['low'], data['close'])
            data['stoch_k'] = stoch.stoch()
            data['stoch_d'] = stoch.stoch_signal()
            features.extend(['stoch_k', 'stoch_d'])
        except Exception as e:
            print(f"  • Skipping stoch_k/stoch_d: {e}")
        # Rate of Change
        try:
            data['roc'] = ta.momentum.ROCIndicator(data['close']).roc()
            features.append('roc')
        except Exception as e:
            print(f"  • Skipping roc: {e}")
        # Bollinger Band Width
        try:
            bb = ta.volatility.BollingerBands(data['close'])
            data['bbw'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
            features.append('bbw')
        except Exception as e:
            print(f"  • Skipping bbw: {e}")
        # Returns
        try:
            data['return_1d'] = data['close'].pct_change()
            features.append('return_1d')
        except Exception as e:
            print(f"  • Skipping return_1d: {e}")
        try:
            data['return_3d'] = data['close'].pct_change(periods=3)
            features.append('return_3d')
        except Exception as e:
            print(f"  • Skipping return_3d: {e}")
        # Rolling statistics
        try:
            data['rolling_mean_5'] = data['close'].rolling(window=5).mean()
            features.append('rolling_mean_5')
        except Exception as e:
            print(f"  • Skipping rolling_mean_5: {e}")
        try:
            data['rolling_std_5'] = data['close'].rolling(window=5).std()
            features.append('rolling_std_5')
        except Exception as e:
            print(f"  • Skipping rolling_std_5: {e}")
        # Momentum slope
        try:
            data['momentum_slope'] = data['momentum'].diff()
            features.append('momentum_slope')
        except Exception as e:
            print(f"  • Skipping momentum_slope: {e}")
        # Market indicators (do not use proxies)
        for col in ['dxy', '^vix', '^gspc', 'gc=f', 'cl=f', 'gold_oil_ratio']:
            if col in data.columns:
                features.append(col)
        # Time features
        try:
            data['day_of_week'] = pd.to_datetime(data['time']).dt.dayofweek
            features.append('day_of_week')
        except Exception as e:
            print(f"  • Skipping day_of_week: {e}")
        try:
            data['month'] = pd.to_datetime(data['time']).dt.month
            features.append('month')
        except Exception as e:
            print(f"  • Skipping month: {e}")
        # Fill missing values
        print("  • Filling missing values...")
        data = data.fillna(method='ffill').fillna(method='bfill').fillna(0)
        # Only use features that are present in the data
        available_features = [f for f in features if f in data.columns]
        data = data[available_features]
        print("\n📊 Feature Information:")
        print(f"  • Total features: {len(available_features)}")
        print(f"  • Features used: {', '.join(available_features)}")
        if len(available_features) != self.expected_features:
            print(f"\n⚠️ Warning: Number of features ({len(available_features)}) does not match model's expected number ({self.expected_features})")
        return data
    
    def predict(self, X):
        """
        Make predictions using the ONNX model.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input features
            
        Returns:
        --------
        numpy.ndarray
            Model predictions
        """
        # Ensure input shape is compatible
        expected_shape = self.input_shape[1:]  # Remove batch dimension
        X_compatible = ensure_compatible_input_shape(X, expected_shape)
        
        # Make prediction
        pred = self.session.run([self.output_name], {self.input_name: X_compatible.astype(np.float32)})[0]
        
        return pred
    
    def run_backtest(self, plot=True):
        """
        Run the backtest.
        
        Parameters:
        -----------
        plot : bool, default=True
            Whether to plot the results
            
        Returns:
        --------
        dict
            Backtest results
        """
        print(f"\n🚀 Starting backtest for {self.symbol}")
        start_time = datetime.now()
        
        # Load and prepare data
        df = self.load_data()
        data = self.prepare_features(df)
        
        print("\n📊 Creating sequences...")
        # Create sequences
        X = np.array([data.iloc[i-self.lookback:i].values for i in tqdm(range(self.lookback, len(data)), desc="Creating sequences")])
        print(f"  • Input shape: {X.shape}")
        
        print("\n🤖 Making predictions...")
        # Make predictions
        predictions = self.predict(X)
        
        print("\n📈 Calculating returns...")
        # Calculate returns
        returns = df['close'].pct_change().fillna(0)
        strategy_returns = (predictions > 0.5).astype(int).flatten() * returns[self.lookback:]
        
        # Calculate equity curve
        equity = (1 + strategy_returns).cumprod() * self.initial_capital
        
        # Calculate metrics
        total_return = (equity.iloc[-1] / self.initial_capital) - 1
        sharpe_ratio = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std()
        
        # Store results
        results = {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'equity': equity,
            'returns': strategy_returns,
            'predictions': predictions
        }
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n📊 Backtest Results:")
        print(f"  • Total Return: {total_return*100:.2f}%")
        print(f"  • Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"  • Final Equity: ${equity.iloc[-1]:,.2f}")
        print(f"  • Number of Trades: {len(strategy_returns[strategy_returns != 0])}")
        print(f"  • Backtest Duration: {duration}")
        
        if plot:
            print("\n📊 Plotting results...")
            self.plot_results(results)
        
        return results
    
    def plot_results(self, results):
        """
        Plot the backtest results.
        
        Parameters:
        -----------
        results : dict
            Backtest results
        """
        plt.figure(figsize=(12, 8))
        
        # Plot equity curve
        plt.subplot(2, 1, 1)
        plt.plot(results['equity'])
        plt.title(f'{self.symbol} Strategy Equity Curve')
        plt.xlabel('Time')
        plt.ylabel('Equity')
        plt.grid(True)
        
        # Plot returns distribution
        plt.subplot(2, 1, 2)
        plt.hist(results['returns'], bins=50)
        plt.title('Strategy Returns Distribution')
        plt.xlabel('Returns')
        plt.ylabel('Frequency')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

    def run_optimization(self, lookback_range, commission_range):
        """
        Run a grid search over lookback and commission parameters.
        
        Parameters:
        -----------
        lookback_range : list
            List of lookback periods to test
        commission_range : list
            List of commission rates to test
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with results for each parameter combination
        """
        results = []
        for lookback in lookback_range:
            for commission in commission_range:
                print(f"\n🔍 Testing lookback={lookback}, commission={commission}")
                self.lookback = lookback
                self.commission = commission
                self.run()
                results.append({
                    'lookback': lookback,
                    'commission': commission,
                    'total_return': self.total_return,
                    'sharpe_ratio': self.sharpe_ratio,
                    'final_equity': self.equity_curve[-1],
                    'num_trades': len(self.trades)
                })
        return pd.DataFrame(results)

def run_backtest(model_path, symbol, data_dir="data", scaler_path=None, lookback=20, initial_capital=10000, commission=0.0001, plot=True):
    """
    Run a backtest using an ONNX model.
    
    Parameters:
    -----------
    model_path : str
        Path to the ONNX model file
    symbol : str
        Trading symbol (e.g., 'EURUSD')
    data_dir : str, default="data"
        Directory containing the data files
    scaler_path : str, optional
        Path to the scaler file
    lookback : int, default=20
        Number of time steps to use for sequence data
    initial_capital : float, default=10000
        Initial capital for backtesting
    commission : float, default=0.0001
        Commission rate per trade
    plot : bool, default=True
        Whether to plot the results
        
    Returns:
    --------
    dict
        Backtest results
    """
    backtester = ONNXBacktester(
        model_path=model_path,
        symbol=symbol,
        data_dir=data_dir,
        lookback=lookback,
        initial_capital=initial_capital,
        commission=commission
    )
    
    return backtester.run_backtest(plot=plot)

def run_optimization_backtest(model_path, symbol, data_dir, lookback_range, commission_range, initial_capital=10000, plot=False):
    """
    Run a grid search over lookback and commission parameters.
    
    Parameters:
    -----------
    model_path : str
        Path to the ONNX model file
    symbol : str
        Trading symbol (e.g., 'EURUSD')
    data_dir : str
        Directory containing the data files
    lookback_range : list
        List of lookback periods to test
    commission_range : list
        List of commission rates to test
    initial_capital : float, default=10000
        Initial capital for the backtest
    plot : bool, default=False
        Whether to plot the results
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with results for each parameter combination
    """
    backtester = ONNXBacktester(model_path, symbol, data_dir, lookback=lookback_range[0], initial_capital=initial_capital, commission=commission_range[0], plot=plot)
    results = backtester.run_optimization(lookback_range, commission_range)
    return results

if __name__ == "__main__":
    # Example usage
    results = run_backtest(
        model_path="exported_models/EURUSD_CNN_LSTM_20250611_083000.onnx",
        symbol="EURUSD",
        data_dir="data",
        scaler_path="models/scaler.pkl",  # Optional
        lookback=20,
        initial_capital=10000,
        commission=0.0001,
        plot=True
    )

print("✅ Backtester module loaded") 