#!/usr/bin/env python3
"""
Simple backtest runner for ONNX strategy
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.backtesting.backtester import run_backtest

def main():
    """Run a simple backtest"""
    print("🚀 Starting ONNX Strategy Backtest")
    
    # Configuration
    model_path = "exported_models/EURUSD_CNN_LSTM_20250610_211730.onnx"
    symbol = "EURUSD"
    data_dir = "data"
    lookback = 20
    initial_capital = 100000
    commission = 0.0001
    
    print(f"Model: {model_path}")
    print(f"Symbol: {symbol}")
    print(f"Initial Capital: ${initial_capital:,}")
    print(f"Commission: {commission*100:.4f}%")
    print(f"Lookback: {lookback}")
    
    try:
        # Run backtest
        results = run_backtest(
            model_path=model_path,
            symbol=symbol,
            data_dir=data_dir,
            lookback=lookback,
            initial_capital=initial_capital,
            commission=commission,
            plot=False  # Set to False to avoid display issues
        )
        
        # Print detailed results
        print("\n" + "="*50)
        print("📊 BACKTEST RESULTS")
        print("="*50)
        print(f"Total Return: {results['total_return']*100:.2f}%")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Final Equity: ${results['equity'].iloc[-1]:,.2f}")
        print(f"Initial Capital: ${initial_capital:,.2f}")
        print(f"Net Profit: ${results['equity'].iloc[-1] - initial_capital:,.2f}")
        print(f"Number of Predictions: {len(results['predictions'])}")
        print(f"Number of Active Trading Days: {len(results['returns'][results['returns'] != 0])}")
        
        # Calculate additional metrics
        max_equity = results['equity'].max()
        min_equity = results['equity'].min()
        max_drawdown = (max_equity - min_equity) / max_equity
        
        print(f"Max Equity: ${max_equity:,.2f}")
        print(f"Max Drawdown: {max_drawdown*100:.2f}%")
        
        # Win rate calculation
        winning_trades = len(results['returns'][results['returns'] > 0])
        total_trades = len(results['returns'][results['returns'] != 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        print(f"Winning Trades: {winning_trades}")
        print(f"Total Trades: {total_trades}")
        print(f"Win Rate: {win_rate*100:.2f}%")
        
        print("="*50)
        print("✅ Backtest completed successfully!")
        
        return results
        
    except Exception as e:
        print(f"❌ Error running backtest: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()