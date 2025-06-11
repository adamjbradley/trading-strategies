"""
ONNX Strategy for Backtrader

This module implements a trading strategy using Backtrader
that incorporates predictions from our ONNX model.
"""

import backtrader as bt
import numpy as np
import pandas as pd
import onnxruntime as ort
import ta
from datetime import datetime
import itertools
import sys
import os

class ONNXStrategy(bt.Strategy):
    """
    A strategy that uses ONNX model predictions to make trading decisions.
    """
    
    params = (
        ('threshold', 0.3),
        ('stop_loss_pct', 0.05),
        ('take_profit_pct', 0.09),
        ('lookback', 20),
        ('commission', 0.0001),
        ('atr_period', 14),
        ('atr_multiplier', 2.0),
        ('min_atr', 0.001),
    )
    
    def __init__(self):
        """Initialize the strategy."""
        # Load the ONNX model
        model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'exported_models', 'EURUSD_CNN_LSTM_20250610_211730.onnx')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        try:
            self.model = ort.InferenceSession(model_path)
            # Get model input shape and name
            self.input_shape = self.model.get_inputs()[0].shape
            self.input_name = self.model.get_inputs()[0].name
            print(f"Model loaded successfully. Input shape: {self.input_shape}, Input name: {self.input_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model: {str(e)}")
        
        # Initialize indicators
        self.rsi = bt.indicators.RSI(period=14)
        self.macd = bt.indicators.MACD()
        self.momentum = bt.indicators.MomentumOscillator(period=10)
        self.cci = bt.indicators.CCI(period=20)
        self.atr = bt.indicators.ATR(period=self.params.atr_period)
        self.adx = bt.indicators.DirectionalMovement(period=14)
        self.stoch = bt.indicators.Stochastic(period=14)
        self.roc = bt.indicators.ROC(period=10)
        self.bb = bt.indicators.BollingerBands(period=20)
        
        # Calculate additional indicators
        self.return_1d = bt.indicators.PercentChange(period=1)
        self.return_3d = bt.indicators.PercentChange(period=3)
        self.rolling_mean_5 = bt.indicators.SMA(period=5)
        self.rolling_std_5 = bt.indicators.StandardDeviation(period=5)
        
        # Initialize prediction array
        self.predictions = []
        
        # Track orders
        self.order = None
        self.stop_loss = None
        self.take_profit = None
        
        # Track trades
        self.total_trades = 0
        self.wins = 0
        self.losses = 0
        self.last_trade_price = None
        self.last_trade_type = None
        self.trade_pnls = []
        
    def notify_order(self, order):
        """Handle order notifications."""
        if order.status in [order.Submitted, order.Accepted]:
            return
            
        if order.status in [order.Completed]:
            if order.isbuy():
                self.last_trade_price = order.executed.price
                self.last_trade_type = 'buy'
            else:
                if self.last_trade_type == 'buy' and self.last_trade_price is not None:
                    pnl = order.executed.price - self.last_trade_price
                    self.trade_pnls.append(pnl)
                    if pnl > 0:
                        self.wins += 1
                    else:
                        self.losses += 1
                self.last_trade_price = order.executed.price
                self.last_trade_type = 'sell'
            self.total_trades += 1
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
            
        self.order = None
        
    def log(self, txt, dt=None):
        """Log messages."""
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}, {txt}')
        
    def next(self):
        """Execute the strategy for each bar."""
        # Skip if we don't have enough data
        if len(self.data) < self.params.lookback + 1:
            return
            
        # Check if indicators are ready
        if not all([
            len(self.rsi) >= self.params.lookback,
            len(self.macd.macd) >= self.params.lookback,
            len(self.momentum) >= self.params.lookback,
            len(self.cci) >= self.params.lookback,
            len(self.atr) >= self.params.lookback,
            len(self.adx) >= self.params.lookback,
            len(self.stoch.percK) >= self.params.lookback,
            len(self.stoch.percD) >= self.params.lookback,
            len(self.roc) >= self.params.lookback,
            len(self.bb.lines.top) >= self.params.lookback,
            len(self.return_1d) >= self.params.lookback,
            len(self.return_3d) >= self.params.lookback,
            len(self.rolling_mean_5) >= self.params.lookback,
            len(self.rolling_std_5) >= self.params.lookback
        ]):
            return
            
        # Calculate momentum slope using numpy diff
        momentum_vals = np.array(self.momentum.get(size=self.params.lookback + 1))
        momentum_slope = np.diff(momentum_vals)
        # Use only the last lookback values
        momentum_slope = momentum_slope[-self.params.lookback:]
        
        try:
            # Prepare features for prediction
            features = np.column_stack([
                self.rsi.get(size=self.params.lookback),
                self.macd.macd.get(size=self.params.lookback),
                self.momentum.get(size=self.params.lookback),
                self.cci.get(size=self.params.lookback),
                self.atr.get(size=self.params.lookback),
                self.adx.get(size=self.params.lookback),
                self.stoch.percK.get(size=self.params.lookback),
                self.stoch.percD.get(size=self.params.lookback),
                self.roc.get(size=self.params.lookback),
                [(self.bb.lines.top[i] - self.bb.lines.bot[i]) / self.bb.lines.mid[i] for i in range(-self.params.lookback, 0)],
                self.return_1d.get(size=self.params.lookback),
                self.return_3d.get(size=self.params.lookback),
                self.rolling_mean_5.get(size=self.params.lookback),
                self.rolling_std_5.get(size=self.params.lookback),
                momentum_slope,
                # Add market indicators if available
                np.zeros(self.params.lookback),  # dxy
                np.zeros(self.params.lookback),  # ^vix
                np.zeros(self.params.lookback),  # ^gspc
                np.zeros(self.params.lookback),  # gc=f
                np.zeros(self.params.lookback),  # cl=f
                np.zeros(self.params.lookback),  # gold_oil_ratio
                # Add time features
                np.array([self.data.datetime.date(i).weekday() for i in range(-self.params.lookback, 0)]),  # day_of_week
                np.array([self.data.datetime.date(i).month for i in range(-self.params.lookback, 0)])  # month
            ])
            
            # Check for NaN values
            if np.isnan(features).any():
                return
                
            # Reshape for ONNX model
            features = features.reshape(1, self.params.lookback, -1).astype(np.float32)
            
            # Print shapes for debugging
            print(f"Feature shape: {features.shape}")
            print(f"Model input shape: {self.input_shape}")
            
            # Ensure input shape matches model expectations (ignore batch dimension)
            expected_shape = tuple(self.input_shape[-2:])  # (time_steps, features)
            if features.shape[1:] != expected_shape:
                print(f"Warning: Feature shape {features.shape} does not match expected model input shape (batch, {expected_shape})")
                return
            
            try:
                # Make prediction
                pred = self.model.run(None, {self.input_name: features.astype(np.float32)})[0]
                pred_value = float(pred[0])  # Extract the scalar value from the prediction array
                print(f"Prediction: {pred_value:.4f}, Threshold: {self.params.threshold}")
                
                # Check if ATR is above minimum threshold
                if self.atr[0] < self.params.min_atr:
                    return
                
                # Check for buy signal (prediction > threshold)
                if pred_value > self.params.threshold:
                    if not self.position:
                        self.log(f'BUY CREATE, {self.data.close[0]:.2f}, pred: {pred_value:.4f}, threshold: {self.params.threshold:.4f}')
                        self.order = self.buy()
                        # Set stop loss and take profit levels
                        self.stop_loss = self.data.close[0] * (1 - self.params.stop_loss_pct)
                        self.take_profit = self.data.close[0] * (1 + self.params.take_profit_pct)
                
                # Check for sell signal (prediction < (1 - threshold))
                elif pred_value < (1 - self.params.threshold):
                    if self.position:
                        self.log(f'SELL CREATE, {self.data.close[0]:.2f}, pred: {pred_value:.4f}, threshold: {self.params.threshold:.4f}')
                        self.order = self.sell()
                
                # Check stop loss and take profit for existing positions
                if self.position:
                    current_price = self.data.close[0]
                    if current_price <= self.stop_loss:
                        self.log(f'STOP LOSS, {current_price:.2f}, pred: {pred_value:.4f}, threshold: {self.params.threshold:.4f}')
                        self.order = self.sell()
                    elif current_price >= self.take_profit:
                        self.log(f'TAKE PROFIT, {current_price:.2f}, pred: {pred_value:.4f}, threshold: {self.params.threshold:.4f}')
                        self.order = self.sell()
                
                # Log key metrics
                print(f'Total Trades: {self.total_trades}, Wins: {self.wins}, Losses: {self.losses}')
                
            except Exception as e:
                print(f"Error making prediction: {str(e)}")
                return
            
        except Exception as e:
            print(f"Error in next(): {str(e)}")
            return

def run_backtest(data, lookback=20, commission=0.0001, cash=10000):
    """
    Run a backtest using the ONNX strategy with Backtrader.
    
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
    backtrader.Cerebro
        Backtrader cerebro instance with results
    """
    # Create a cerebro entity
    cerebro = bt.Cerebro()
    
    # Add a strategy
    cerebro.addstrategy(ONNXStrategy, lookback=lookback, commission=commission)
    
    # Ensure datetime index is properly formatted
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    
    # Create a Data Feed
    data_feed = bt.feeds.PandasData(
        dataname=data,
        datetime=None,  # Use index as datetime
        open='open',
        high='high',
        low='low',
        close='close',
        volume='tick_volume',
        openinterest=-1
    )
    
    # Add the Data Feed to Cerebro
    cerebro.adddata(data_feed)
    
    # Set our desired cash start
    cerebro.broker.setcash(cash)
    
    # Set the commission
    cerebro.broker.setcommission(commission=commission)
    
    # Print out the starting conditions
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    
    # Add analyzers for drawdown and Sharpe ratio
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    
    # Run over everything
    results = cerebro.run()
    strat = results[0]
    
    # Print out the final result
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
    
    # Print detailed summary from results
    pnl = cerebro.broker.getvalue() - cash
    total_trades = strat.total_trades
    win_rate = (strat.wins / total_trades) * 100 if total_trades > 0 else 0
    max_drawdown = strat.analyzers.drawdown.get_analysis()['max']['drawdown'] if hasattr(strat.analyzers, 'drawdown') else None
    sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio', None) if hasattr(strat.analyzers, 'sharpe') else None
    print("\n===== Backtest Summary =====")
    print(f"Final Portfolio Value: {cerebro.broker.getvalue():.2f}")
    print(f"Net Profit/Loss: {pnl:.2f}")
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.2f}%")
    if max_drawdown is not None:
        print(f"Max Drawdown: {max_drawdown:.2f}%")
    if sharpe is not None:
        print(f"Sharpe Ratio: {sharpe:.2f}")
    print("===========================\n")
    
    # Plot the result
    cerebro.plot()
    
    return cerebro

def optimize_parameters(data):
    """
    Optimize strategy parameters using two distinct sets for comparison.
    """
    param_sets = [
        {'threshold': 0.3, 'stop_loss_pct': 0.05, 'take_profit_pct': 0.09, 'lookback': 20},
        {'threshold': 0.7, 'stop_loss_pct': 0.02, 'take_profit_pct': 0.15, 'lookback': 20}
    ]
    results = []
    for params in param_sets:
        print(f"\nRunning backtest with params: {params}")
        cerebro = bt.Cerebro()
        cerebro.broker.set_cash(10000)
        cerebro.broker.setcommission(commission=0.0001)
        data_feed = bt.feeds.PandasData(dataname=data)
        cerebro.adddata(data_feed)
        cerebro.addstrategy(ONNXStrategy,
                            threshold=params['threshold'],
                            stop_loss_pct=params['stop_loss_pct'],
                            take_profit_pct=params['take_profit_pct'],
                            lookback=params['lookback'])
        try:
            result = cerebro.run()
            strat = result[0]
            final_value = cerebro.broker.getvalue()
            print(f"Final Portfolio Value: {final_value:.2f}")
            print(f"Total Trades: {strat.total_trades}, Wins: {strat.wins}, Losses: {strat.losses}")
            results.append({
                'params': params,
                'final_value': final_value,
                'total_trades': strat.total_trades,
                'wins': strat.wins,
                'losses': strat.losses
            })
        except Exception as e:
            print(f"Error during backtest: {e}")
    print("\n===== Summary of All Parameter Sets =====")
    for res in results:
        print(f"Params: {res['params']}, Final Value: {res['final_value']:.2f}, Trades: {res['total_trades']}, Wins: {res['wins']}, Losses: {res['losses']}")

if __name__ == '__main__':
    # Load data
    print("Loading data...")
    data = pd.read_parquet('RCS_CNN_LSTM_Notebook/RCS_CNN_LSTM_Notebook/data/metatrader_EURUSD.parquet')
    # Ensure datetime index for Backtrader
    if 'time' in data.columns:
        data['time'] = pd.to_datetime(data['time'])
        data = data.set_index('time')
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    print("Data loaded successfully")
    print(f"Data shape: {data.shape}")
    print(f"Data index type: {type(data.index)}")
    print(f"First 5 index values: {data.index[:5]}")
    
    # Run a single backtest first
    print("\nRunning single backtest...")
    try:
        cerebro = run_backtest(data)
        print("Single backtest completed successfully")
    except Exception as e:
        print(f"Error during single backtest: {str(e)}")
        sys.exit(1)
    
    # Run optimization
    print("\nCalling optimization function...")
    try:
        optimize_parameters(data)
        print("Optimization completed successfully")
    except Exception as e:
        print(f"Error during optimization: {str(e)}") 