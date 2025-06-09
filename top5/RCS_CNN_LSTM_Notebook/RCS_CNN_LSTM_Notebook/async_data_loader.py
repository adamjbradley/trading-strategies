import asyncio
import aiohttp
import pandas as pd
from datetime import datetime

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
    url = f"https://api.polygon.io/v2/aggs/ticker/C:{symbol}/range/1/{interval}/2023-01-01/2023-12-31?adjusted=true&sort=asc&limit={limit}&apiKey={api_key}"
    data = await fetch_json(session, url)
    results = data.get("results", [])
    return pd.DataFrame([{
        "timestamp": datetime.fromtimestamp(d["t"] / 1000),
        "open": d["o"], "high": d["h"], "low": d["l"], "close": d["c"], "volume": d["v"]
    } for d in results])

async def fetch_all_data(symbols, provider, api_key, **kwargs):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for symbol in symbols:
            if provider == "twelvedata":
                tasks.append(fetch_twelve_data(session, symbol, api_key, **kwargs))
            elif provider == "polygon":
                tasks.append(fetch_polygon_data(session, symbol, api_key, **kwargs))
        results = await asyncio.gather(*tasks)
        return dict(zip(symbols, results))
