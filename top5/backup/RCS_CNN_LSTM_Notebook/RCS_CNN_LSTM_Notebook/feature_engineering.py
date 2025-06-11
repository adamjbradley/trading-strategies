import pandas as pd
import numpy as np
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import ADXIndicator, MACD, CCIIndicator
from ta.momentum import StochasticOscillator, ROCIndicator, RSIIndicator

def engineer_features(prices, symbol="EURUSD"):
    """
    Given a MultiIndex prices DataFrame (symbol, field), return a DataFrame of engineered features for the given symbol.
    """
    ohlc = prices.copy()

    # Synthesize missing columns if needed
    for col in ["open", "high", "low", "close"]:
        if (symbol, col) not in ohlc.columns:
            ohlc[(symbol, col)] = ohlc[(symbol, "close")]

    data = pd.DataFrame(index=ohlc.index)
    data['rsi'] = RSIIndicator(close=ohlc[(symbol, "close")]).rsi()
    data['macd'] = MACD(ohlc[(symbol, "close")]).macd()
    data['momentum'] = ROCIndicator(close=ohlc[(symbol, "close")]).roc()
    data['cci'] = CCIIndicator(
        high=ohlc[(symbol, "high")],
        low=ohlc[(symbol, "low")],
        close=ohlc[(symbol, "close")]
    ).cci()
    data['atr'] = AverageTrueRange(
        high=ohlc[(symbol, "high")],
        low=ohlc[(symbol, "low")],
        close=ohlc[(symbol, "close")]
    ).average_true_range()
    data['adx'] = ADXIndicator(
        high=ohlc[(symbol, "high")],
        low=ohlc[(symbol, "low")],
        close=ohlc[(symbol, "close")]
    ).adx()
    data['stoch_k'] = StochasticOscillator(
        high=ohlc[(symbol, "high")],
        low=ohlc[(symbol, "low")],
        close=ohlc[(symbol, "close")]
    ).stoch()
    data['stoch_d'] = StochasticOscillator(
        high=ohlc[(symbol, "high")],
        low=ohlc[(symbol, "low")],
        close=ohlc[(symbol, "close")]
    ).stoch_signal()
    data['roc'] = ROCIndicator(close=ohlc[(symbol, "close")]).roc()
    data['bbw'] = BollingerBands(close=ohlc[(symbol, "close")]).bollinger_wband()

    # Lagged/Engineered Features
    data['return_1d'] = ohlc[(symbol, "close")].pct_change(1)
    data['return_3d'] = ohlc[(symbol, "close")].pct_change(3)
    data['rolling_mean_5'] = ohlc[(symbol, "close")].rolling(window=5).mean()
    data['rolling_std_5'] = ohlc[(symbol, "close")].rolling(window=5).std()
    data['momentum_slope'] = ohlc[(symbol, "close")].diff(1)

    # Macro & Cross-Asset Features (fill with zeros if not present)
    for macro in ['DXY', '^VIX', '^GSPC', 'GC=F', 'CL=F']:
        if (macro, "close") in prices.columns:
            data[macro.lower()] = prices[(macro, "close")].reindex(data.index).ffill()
        else:
            data[macro.lower()] = 0.0

    if ('GC=F', "close") in prices.columns and ('CL=F', "close") in prices.columns:
        data['gold_oil_ratio'] = prices[('GC=F', "close")].reindex(data.index).ffill() / prices[('CL=F', "close")].reindex(data.index).ffill()
    else:
        data['gold_oil_ratio'] = 0.0

    # Time Features
    data['day_of_week'] = data.index.dayofweek
    data['month'] = data.index.month

    # Drop initial NaNs from rolling/indicator calculations
    features = data.dropna()
    return features
