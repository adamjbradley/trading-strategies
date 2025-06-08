
import MetaTrader5 as mt5
import pandas as pd

symbols = ["EURUSD", "GBPUSD", "USDJPY"]
timeframe = mt5.TIMEFRAME_H1
bars = 5000

if not mt5.initialize():
    print("MetaTrader5 initialization failed")
    mt5.shutdown()
    exit()

for symbol in symbols:
    print(f"Fetching data for {symbol}")
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)

    if rates is not None:
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.to_csv(f"{symbol}_H1.csv", index=False)
        print(f"{symbol}: {len(df)} rows saved.")
    else:
        print(f"Failed to fetch {symbol}")

mt5.shutdown()
