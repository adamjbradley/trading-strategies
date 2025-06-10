import os
import asyncio
import aiohttp
import pandas as pd
import requests
import yfinance as yf
from datetime import datetime
import MetaTrader5 as mt5

# Supported provider names for fetch_all_data
VALID_PROVIDERS = {"twelvedata", "polygon", "yfinance", "metatrader"}

# Normalize symbols like "EUR/USD" -> "EURUSD" for providers such as Polygon
def normalize_symbol(symbol: str) -> str:
    """Return an uppercase symbol without separator characters."""
    return symbol.replace("/", "").upper()

def parse_date(ts, fmt="%Y-%m-%d %H:%M:%S"):
    return datetime.strptime(ts, fmt)

def _resolve_key(key: str, env_var: str) -> str:
    """Return the given key or fall back to an environment variable."""
    return key if key and not key.startswith("YOUR_") else os.getenv(env_var, "")

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

async def fetch_polygon_data(
    session,
    symbol,
    api_key,
    interval="minute",
    limit=500,
    start="2023-01-01",
    end="2023-12-31",
):
    """Fetch OHLC data from Polygon.io within a date range."""
    symbol_clean = normalize_symbol(symbol)
    url = (
        "https://api.polygon.io/v2/aggs/ticker/C:"
        f"{symbol_clean}/range/1/{interval}/{start}/{end}"
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

async def fetch_yfinance(symbol, interval="1m", period="1y", **_):
    """Fetch data from Yahoo Finance using a thread executor."""
    loop = asyncio.get_event_loop()
    df = await loop.run_in_executor(
        None,
        lambda: yf.download(
            symbol,
            interval=interval,
            period=period,
            progress=False,
        ),
    )
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
    """Fetch data for multiple symbols from the given provider."""
    if provider not in VALID_PROVIDERS:
        raise ValueError(f"Unsupported provider: {provider}")

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

# ---------------------------------------------------------------------------
# Synchronous helpers copied from the old `data_loader` module
# ---------------------------------------------------------------------------

def load_polygon_data(symbol, api_key, interval="minute", limit=500, start="2023-01-01", end="2023-12-31"):
    """Synchronously fetch Polygon.io data for a date range."""
    api_key = _resolve_key(api_key, "POLYGON_API_KEY")
    symbol_clean = normalize_symbol(symbol)
    url = (
        "https://api.polygon.io/v2/aggs/ticker/C:"
        f"{symbol_clean}/range/1/{interval}/{start}/{end}"
        f"?adjusted=true&sort=asc&limit={limit}&apiKey={api_key}"
    )
    resp = requests.get(url)
    data = resp.json().get("results", [])
    return pd.DataFrame([
        {
            "timestamp": datetime.fromtimestamp(d["t"] / 1000),
            "open": d["o"],
            "high": d["h"],
            "low": d["l"],
            "close": d["c"],
            "volume": d["v"],
        }
        for d in data
    ])

def load_twelve_data(symbol, api_key, interval="1min", outputsize=500):
    url = (
        f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}"
        f"&outputsize={outputsize}&apikey={api_key}"
    )
    resp = requests.get(url)
    data = resp.json().get("values", [])
    return pd.DataFrame([
        {
            "timestamp": parse_date(d["datetime"]),
            "open": float(d["open"]),
            "high": float(d["high"]),
            "low": float(d["low"]),
            "close": float(d["close"]),
            "volume": float(d.get("volume", 0)),
        }
        for d in reversed(data)
    ])

def load_alpha_vantage(symbol, api_key, interval="1min"):
    url = (
        f"https://www.alphavantage.co/query?function=FX_INTRADAY&from_symbol={symbol[:3]}"
        f"&to_symbol={symbol[3:]}&interval={interval}&apikey={api_key}&outputsize=compact"
    )
    resp = requests.get(url)
    time_series = resp.json().get(f"Time Series FX ({interval})", {})
    return pd.DataFrame([
        {
            "timestamp": parse_date(ts),
            "open": float(d["1. open"]),
            "high": float(d["2. high"]),
            "low": float(d["3. low"]),
            "close": float(d["4. close"]),
            "volume": 0,
        }
        for ts, d in sorted(time_series.items())
    ])

def load_currencystack(symbol, api_key):
    url = f"https://api.currencystack.io/forex?base={symbol[:3]}&target={symbol[3:]}&apikey={api_key}"
    resp = requests.get(url)
    data = resp.json()
    return pd.DataFrame([
        {
            "timestamp": datetime.now(),
            "open": data["rate"],
            "high": data["rate"],
            "low": data["rate"],
            "close": data["rate"],
            "volume": 0,
        }
    ])

def load_tiingo(symbol, api_key, limit=500):
    headers = {"Content-Type": "application/json", "Authorization": f"Token {api_key}"}
    url = f"https://api.tiingo.com/tiingo/fx/{symbol}/prices?resampleFreq=1min&limit={limit}"
    resp = requests.get(url, headers=headers)
    data = resp.json()
    return pd.DataFrame([
        {
            "timestamp": parse_date(d["date"]),
            "open": d["open"],
            "high": d["high"],
            "low": d["low"],
            "close": d["close"],
            "volume": d.get("volume", 0),
        }
        for d in data
    ])

def load_yfinance(symbol, interval="1m", period="1y", **_):
    """Load data from Yahoo Finance."""
    df = yf.download(symbol, interval=interval, period=period, progress=False)
    if df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    df.rename(
        columns={
            "Datetime": "timestamp",
            "Date": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "close",
            "Volume": "volume",
        },
        inplace=True,
    )
    return df[["timestamp", "open", "high", "low", "close", "volume"]]

def save_optimized(df, symbol, provider):
    symbol_clean = symbol.replace("/", "").upper()
    os.makedirs("data", exist_ok=True)
    filename = f"data/{provider}_{symbol_clean}"
    df.to_parquet(f"{filename}.parquet", index=False)
    df.to_hdf(f"{filename}.h5", key="df", mode="w")
    print(f"📦 Saved to {filename}.parquet and {filename}.h5")

def load_or_fetch(symbol, provider, loader_func, api_key, force_refresh=False, **kwargs):
    symbol_clean = symbol.replace("/", "").upper()
    base_path = f"data/{provider}_{symbol_clean}"
    parquet_path = f"{base_path}.parquet"
    h5_path = f"{base_path}.h5"

    if not force_refresh:
        if os.path.exists(parquet_path):
            print(f"✅ Loaded cached data from {parquet_path}")
            return pd.read_parquet(parquet_path)
        elif os.path.exists(h5_path):
            print(f"✅ Loaded cached data from {h5_path}")
            return pd.read_hdf(h5_path)

    print(f"🔄 Fetching fresh data from {provider} API")
    df = loader_func(symbol, api_key, **kwargs)
    save_optimized(df, symbol, provider)
    return df
