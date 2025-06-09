import asyncio
import aiohttp
import pandas as pd
import yfinance as yf
from datetime import datetime
import MetaTrader5 as mt5

from .data_loader import normalize_symbol

def parse_date(ts, fmt="%Y-%m-%d %H:%M:%S"):
    return datetime.strptime(ts, fmt)

async def fetch_json(session, url, headers=None):
    async with session.get(url, headers=headers) as response:
        return await response.json()

async def fetch_twelve_data(session, symbol, api_key, interval="1min", outputsize=500):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize={outputsize}&apikey={api_key}"
    data = await fetch_json(session, url)
    values = data.get("values", [])
    return pd.DataFrame([{
        "timestamp": parse_date(d["datetime"]),
        "open": float(d["open"]), "high": float(d["high"]),
        "low": float(d["low"]), "close": float(d["close"]), "volume": float(d.get("volume", 0))
    } for d in reversed(values)])

async def fetch_polygon_data(session, symbol, api_key, interval="minute", limit=500):
    symbol_clean = normalize_symbol(symbol)
    url = (
        "https://api.polygon.io/v2/aggs/ticker/C:"
        f"{symbol_clean}/range/1/{interval}/2023-01-01/2023-12-31"
        f"?adjusted=true&sort=asc&limit={limit}&apiKey={api_key}"
    )
    data = await fetch_json(session, url)
    results = data.get("results", [])
    return pd.DataFrame([
        {
            "timestamp": datetime.fromtimestamp(d["t"] / 1000),
            "open": d["o"],
            "high": d["h"],
            "low": d["l"],
            "close": d["c"],
            "volume": d["v"],
        }
        for d in results
    ])

async def fetch_yfinance(symbol, interval="1m", period="1y"):
    """Fetch data from Yahoo Finance using a thread executor."""
    loop = asyncio.get_event_loop()
    df = await loop.run_in_executor(None, lambda: yf.download(symbol, interval=interval, period=period, progress=False))
    if df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    df.rename(columns={"Datetime": "timestamp", "Date": "timestamp", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Adj Close": "close", "Volume": "volume"}, inplace=True)
    return df[["timestamp", "open", "high", "low", "close", "volume"]]

async def fetch_metatrader_data(symbol, timeframe=mt5.TIMEFRAME_M1, start=None, end=None):
    """Fetch OHLC data from a running MetaTrader 5 terminal."""
    loop = asyncio.get_event_loop()

    def _load():
        if not mt5.initialize():
            raise RuntimeError("MetaTrader5 initialization failed")
        try:
            records = mt5.copy_rates_range(symbol, timeframe, start, end)
            return records
        finally:
            mt5.shutdown()

    records = await loop.run_in_executor(None, _load)
    if records is None:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["time"], unit="s")
    df.rename(columns={"tick_volume": "volume"}, inplace=True)
    return df[["timestamp", "open", "high", "low", "close", "volume"]]

async def fetch_all_data(symbols, provider, api_key, **kwargs):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for symbol in symbols:
            if provider == "twelvedata":
                tasks.append(fetch_twelve_data(session, symbol, api_key, **kwargs))
            elif provider == "polygon":
                tasks.append(fetch_polygon_data(session, symbol, api_key, **kwargs))
            elif provider == "yfinance":
                tasks.append(fetch_yfinance(symbol, **kwargs))
            elif provider == "metatrader":
                tasks.append(fetch_metatrader_data(symbol, **kwargs))
        results = await asyncio.gather(*tasks)
        return dict(zip(symbols, results))
