"""
ONNX Strategy for Backtesting

This module implements a trading strategy using the backtesting library
that incorporates predictions from our ONNX model.
"""

import numpy as np
import pandas as pd
import onnxruntime as ort
from backtesting import Strategy, Backtest
from backtesting.lib import crossover
import ta

class ONNXStrategy(Strategy):
    """
    A strategy that uses ONNX model predictions to make trading decisions.
    """
    
    # Define parameters that can be optimized
    lookback = 20
    commission = 0.0001
    
    def init(self):
        """Initialize the strategy."""
        # Load the ONNX model
        self.model = ort.InferenceSession('RCS_CNN_LSTM_Notebook/RCS_CNN_LSTM_Notebook/exported_models/EURUSD_CNN_LSTM_20250610_211730.onnx')
        
        # Convert data to pandas Series for calculations
        self.close = pd.Series(self.data.Close)
        self.high = pd.Series(self.data.High)
        self.low = pd.Series(self.data.Low)
        
        # Calculate technical indicators
        self.rsi = self.I(self.calculate_rsi)
        self.macd = self.I(self.calculate_macd)
        self.momentum = self.I(self.calculate_momentum)
        self.cci = self.I(self.calculate_cci)
        self.atr = self.I(self.calculate_atr)
        self.adx = self.I(self.calculate_adx)
        self.stoch_k = self.I(self.calculate_stoch_k)
        self.stoch_d = self.I(self.calculate_stoch_d)
        self.roc = self.I(self.calculate_roc)
        self.bbw = self.I(self.calculate_bbw)
        self.return_1d = self.I(self.calculate_return_1d)
        self.return_3d = self.I(self.calculate_return_3d)
        self.rolling_mean_5 = self.I(self.calculate_rolling_mean_5)
        self.rolling_std_5 = self.I(self.calculate_rolling_std_5)
        
        # Calculate momentum slope using pandas Series
        momentum_values = self.calculate_momentum()
        self.momentum_slope = self.I(lambda: pd.Series(momentum_values).diff())
        
        # Initialize prediction array
        self.predictions = np.zeros(len(self.data))
        
    def calculate_rsi(self):
        """Calculate RSI indicator."""
        return ta.momentum.RSIIndicator(self.close, window=14).rsi()
    
    def calculate_macd(self):
        """Calculate MACD indicator."""
        return ta.trend.MACD(self.close).macd()
    
    def calculate_momentum(self):
        """Calculate momentum indicator."""
        return ta.momentum.ROCIndicator(self.close, window=10).roc()
    
    def calculate_cci(self):
        """Calculate CCI indicator."""
        return ta.trend.CCIIndicator(self.high, self.low, self.close).cci()
    
    def calculate_atr(self):
        """Calculate ATR indicator."""
        return ta.volatility.AverageTrueRange(self.high, self.low, self.close).average_true_range()
    
    def calculate_adx(self):
        """Calculate ADX indicator."""
        return ta.trend.ADXIndicator(self.high, self.low, self.close).adx()
    
    def calculate_stoch_k(self):
        """Calculate Stochastic K indicator."""
        return ta.momentum.StochasticOscillator(self.high, self.low, self.close).stoch()
    
    def calculate_stoch_d(self):
        """Calculate Stochastic D indicator."""
        return ta.momentum.StochasticOscillator(self.high, self.low, self.close).stoch_signal()
    
    def calculate_roc(self):
        """Calculate Rate of Change indicator."""
        return ta.momentum.ROCIndicator(self.close).roc()
    
    def calculate_bbw(self):
        """Calculate Bollinger Band Width indicator."""
        bb = ta.volatility.BollingerBands(self.close)
        return (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
    
    def calculate_return_1d(self):
        """Calculate 1-day return."""
        return self.close.pct_change()
    
    def calculate_return_3d(self):
        """Calculate 3-day return."""
        return self.close.pct_change(periods=3)
    
    def calculate_rolling_mean_5(self):
        """Calculate 5-day rolling mean."""
        return self.close.rolling(window=5).mean()
    
    def calculate_rolling_std_5(self):
        """Calculate 5-day rolling standard deviation."""
        return self.close.rolling(window=5).std()
    
    def next(self):
        """Execute the strategy for each bar."""
        # Skip if we don't have enough data
        if len(self.data) < self.lookback:
            return
        
        # Prepare features for prediction
        features = np.column_stack([
            self.rsi[-self.lookback:],
            self.macd[-self.lookback:],
            self.momentum[-self.lookback:],
            self.cci[-self.lookback:],
            self.atr[-self.lookback:],
            self.adx[-self.lookback:],
            self.stoch_k[-self.lookback:],
            self.stoch_d[-self.lookback:],
            self.roc[-self.lookback:],
            self.bbw[-self.lookback:],
            self.return_1d[-self.lookback:],
            self.return_3d[-self.lookback:],
            self.rolling_mean_5[-self.lookback:],
            self.rolling_std_5[-self.lookback:],
            self.momentum_slope[-self.lookback:],
            # Add market indicators if available
            np.zeros(self.lookback),  # dxy
            np.zeros(self.lookback),  # ^vix
            np.zeros(self.lookback),  # ^gspc
            np.zeros(self.lookback),  # gc=f
            np.zeros(self.lookback),  # cl=f
            np.zeros(self.lookback),  # gold_oil_ratio
            # Add time features
            np.array([pd.Timestamp(self.data.index[-i]).dayofweek for i in range(self.lookback)]),  # day_of_week
            np.array([pd.Timestamp(self.data.index[-i]).month for i in range(self.lookback)])  # month
        ])
        
        # Reshape for ONNX model
        features = features.reshape(1, self.lookback, -1).astype(np.float32)
        
        # Make prediction
        pred = self.model.run(None, {'args_0': features})[0][0]
        self.predictions[-1] = pred
        
        # Trading logic based on prediction
        if pred > 0.5 and not self.position:
            self.buy()
        elif pred < 0.5 and self.position:
            self.position.close()

def run_backtest(data, lookback=20, commission=0.0001, cash=10000):
    """
    Run a backtest using the ONNX strategy.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame with OHLCV data
    lookback : int, default=20
        Lookback period for the strategy
    commission : float, default=0.0001
        Commission rate
    cash : float, default=10000
        Initial cash
        
    Returns:
    --------
    backtesting.backtesting.Backtest
        Backtest results
    """
    # Create and run the backtest
    bt = Backtest(
        data,
        ONNXStrategy,
        cash=cash,
        commission=commission,
        trade_on_close=True,
        exclusive_orders=True
    )
    
    # Run the backtest
    results = bt.run(lookback=lookback, commission=commission)
    
    # Plot the results
    bt.plot()
    
    return results

if __name__ == "__main__":
    # Example usage
    import pandas as pd
    
    # Load data
    data = pd.read_parquet('RCS_CNN_LSTM_Notebook/RCS_CNN_LSTM_Notebook/data/metatrader_EURUSD.parquet')
    
    # Rename columns to match backtesting library requirements
    data = data.rename(columns={
        'time': 'Date',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'tick_volume': 'Volume'
    })
    
    # Set index
    data.set_index('Date', inplace=True)
    
    # Run backtest
    results = run_backtest(data)
    
    # Print results
    print("\nBacktest Results:")
    print(f"Total Return: {results['Return [%]']:.2f}%")
    print(f"Sharpe Ratio: {results['Sharpe Ratio']:.2f}")
    print(f"Max Drawdown: {results['Max. Drawdown [%]']:.2f}%")
    print(f"# Trades: {results['# Trades']}") 