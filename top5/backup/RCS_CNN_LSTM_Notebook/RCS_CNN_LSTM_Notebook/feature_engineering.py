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

    # Forex-Specific Time Features
    data['day_of_week'] = data.index.dayofweek
    data['month'] = data.index.month
    data['hour'] = data.index.hour
    
    # Trading Session Features (UTC times)
    # Asian Session: 23:00-08:00 UTC
    # London Session: 08:00-17:00 UTC  
    # NY Session: 13:00-22:00 UTC
    asian_session = ((data.index.hour >= 23) | (data.index.hour < 8))
    london_session = ((data.index.hour >= 8) & (data.index.hour < 17))
    ny_session = ((data.index.hour >= 13) & (data.index.hour < 22))
    
    data['asian_session'] = asian_session.astype(int)
    data['london_session'] = london_session.astype(int)
    data['ny_session'] = ny_session.astype(int)
    data['session_overlap'] = ((london_session & ny_session).astype(int))
    
    # Session-based volatility (use 20-period rolling ATR as baseline)
    atr_baseline = data['atr'].rolling(20).mean()
    data['session_volatility_ratio'] = data['atr'] / atr_baseline
    
    # Currency Correlation Features (if EURUSD)
    if symbol == "EURUSD":
        # Correlation with major EUR pairs - approximate using price momentum
        eur_momentum = data['return_1d']
        data['eur_strength_proxy'] = eur_momentum.rolling(5).mean()
        data['eur_strength_trend'] = data['eur_strength_proxy'].diff(3)
    
    # USD Index proxy (inverted for non-USD base currencies)
    if 'dxy' in data.columns and data['dxy'].sum() != 0:
        dxy_momentum = data['dxy'].pct_change(1)
        if symbol.endswith('USD'):  # USD is quote currency
            data['usd_strength_impact'] = -dxy_momentum  # Inverse relationship
        elif symbol.startswith('USD'):  # USD is base currency
            data['usd_strength_impact'] = dxy_momentum   # Direct relationship
        else:
            data['usd_strength_impact'] = 0
    else:
        data['usd_strength_impact'] = 0
    
    # Volatility clustering features
    data['volatility_regime'] = (data['atr'] > data['atr'].rolling(50).quantile(0.8)).astype(int)
    data['volatility_persistence'] = data['atr'].rolling(10).corr(data['atr'].shift(1))
    
    # Interest rate proxy using gold/bond relationship
    if 'gc=f' in data.columns and data['gc=f'].sum() != 0:
        gold_momentum = data['gc=f'].pct_change(5)
        # Gold often inversely correlated with real rates
        data['risk_sentiment'] = gold_momentum.rolling(10).mean()
    else:
        data['risk_sentiment'] = 0
    
    # Market structure features
    data['range_ratio'] = ((ohlc[(symbol, "high")] - ohlc[(symbol, "low")]) / 
                          ohlc[(symbol, "close")]).reindex(data.index)
    data['close_position'] = ((ohlc[(symbol, "close")] - ohlc[(symbol, "low")]) / 
                             (ohlc[(symbol, "high")] - ohlc[(symbol, "low")])).reindex(data.index)
    
    # Weekend effect (Friday close to Monday open gaps)
    data['is_friday'] = (data.index.dayofweek == 4).astype(int)
    data['is_monday'] = (data.index.dayofweek == 0).astype(int)
    
    # Drop initial NaNs from rolling/indicator calculations
    features = data.dropna()
    return features
