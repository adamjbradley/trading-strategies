"""
Backtesting Package

This package provides functionality for backtesting trading strategies.
"""

from .backtester import ONNXBacktester, run_backtest

__all__ = ['ONNXBacktester', 'run_backtest'] 