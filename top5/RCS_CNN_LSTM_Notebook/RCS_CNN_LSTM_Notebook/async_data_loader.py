import os
import asyncio
import aiohttp
import pandas as pd
import requests
import yfinance as yf
from datetime import datetime
from typing import Optional
from mt5_downloader import MT5Downloader
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

def _map_interval_to_timeframe(interval: str) -> str:
    mapping = {
        "1m": "M1",
        "1min": "M1",
        "5m": "M5",
        "5min": "M5",
        "15m": "M15",
        "30m": "M30",
        "1h": "H1",
        "60m": "H1",
        "4h": "H4",
    }
    return mapping.get(interval.lower(), interval.upper())


def _load_metatrader(symbol: str, timeframe: str = "H1", bars: int = 5000, path: Optional[str] = None):
    if mt5 is None:
        raise ImportError("MetaTrader5 package is not installed")
    if path:
        initialized = mt5.initialize(path)
    else:
        initialized = mt5.initialize()
    if not initialized:
        raise RuntimeError("MetaTrader5 initialization failed")
    try:
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"Symbol {symbol} not available")
        tf = getattr(mt5, f"TIMEFRAME_{timeframe.upper()}", mt5.TIMEFRAME_H1)
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)
    finally:
        mt5.shutdown()
    if rates is None:
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    df.rename(columns={"time": "timestamp", "tick_volume": "volume"}, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    return df[["timestamp", "open", "high", "low", "close", "volume"]]

async def fetch_metatrader_data(symbol, timeframe="H1", start=None, end=None, broker="default"):
    """
    Fetch OHLC data from MetaTrader 5 using the robust MT5Downloader class.
    This function runs the synchronous MT5Downloader logic in a thread executor for async compatibility.
    """
    loop = asyncio.get_event_loop()
    def _download():
        downloader = MT5Downloader(broker=broker)
        if not downloader.initialize():
            return pd.DataFrame()
        try:
            symbol_info = downloader.ensure_symbol(symbol)
            if symbol_info is None:
                return pd.DataFrame()
            # Use the downloader's download_symbol_timeframe method
            df_path = downloader.download_symbol_timeframe(symbol, timeframe, start, end)
            if df_path:
                # Always read the Parquet file for robustness
                base_filename = os.path.splitext(df_path)[0]
                parquet_filename = base_filename + ".parquet"
                try:
                    df = pd.read_parquet(parquet_filename)
                except Exception as e:
                    print(f"    Error reading {parquet_filename}: {e}. Will re-download.")
                    try:
                        os.remove(parquet_filename)
                    except Exception as del_e:
                        print(f"    Error deleting corrupt file {parquet_filename}: {del_e}")
                    # Re-download
                    df_path = downloader.download_symbol_timeframe(symbol, timeframe, start, end)
                    base_filename = os.path.splitext(df_path)[0]
                    parquet_filename = base_filename + ".parquet"
                    df = pd.read_parquet(parquet_filename)
                return df
            else:
                return pd.DataFrame()
        finally:
            downloader.shutdown()
    df = await loop.run_in_executor(None, _download)
    return df

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
                # Map 'interval' to 'timeframe' for MetaTrader provider and filter valid args
                mt_kwargs = kwargs.copy()
                if "interval" in mt_kwargs:
                    mt_kwargs["timeframe"] = mt_kwargs.pop("interval")
                # Only keep valid arguments for fetch_metatrader_data
                valid_mt_args = {"timeframe", "start", "end", "broker"}
                mt_kwargs = {k: v for k, v in mt_kwargs.items() if k in valid_mt_args}
                tasks.append(fetch_metatrader_data(symbol, **mt_kwargs))
        results = await asyncio.gather(*tasks)
        return dict(zip(symbols, results))

# ---------------------------------------------------------------------------
# Synchronous helpers copied from the old `data_loader` module
# ---------------------------------------------------------------------------

def load_polygon_data(symbol, api_key, interval="minute", limit=500):
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

def load_metatrader_data(symbol, timeframe="H1", bars=5000, interval=None, path=None, **_):
    """Load data from a running MetaTrader 5 terminal."""
    if interval and not timeframe:
        timeframe = _map_interval_to_timeframe(interval)
    return _load_metatrader(symbol, timeframe, bars, path)

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
