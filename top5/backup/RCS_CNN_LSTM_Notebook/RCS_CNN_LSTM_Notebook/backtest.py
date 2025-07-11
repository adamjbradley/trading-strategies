import yaml
import asyncio
import argparse
from async_data_loader import (
    fetch_all_data,
    load_or_fetch,
    load_twelve_data,
    load_polygon_data,
    load_yfinance,
    load_metatrader_data,
)

def run_backtest(config):
    symbol = config['symbol']
    provider = config.get('provider', 'twelvedata')
    api_key = config['api_keys'].get(provider, "")
    interval = config.get("interval", "1min")
    outputsize = config.get("outputsize", 500)
    start = config.get("start", "2023-01-01")
    end = config.get("end", "2023-12-31")

    if config.get("use_async", False):
        print(f"⚡ Async loading for {symbol}...")
        df_dict = asyncio.run(
            fetch_all_data(
                [symbol],
                provider,
                api_key,
                interval=interval,
                outputsize=outputsize,
                start=start,
                end=end,
            )
        )
        df = df_dict[symbol]
    else:
        print(f"⏳ Sync loading for {symbol}...")
        loader_map = {
            "twelvedata": load_twelve_data,
            "polygon": load_polygon_data,
            "yfinance": load_yfinance,
            "metatrader": load_metatrader_data,
        }
        loader = loader_map.get(provider, load_twelve_data)
        df = load_or_fetch(
            symbol=symbol,
            provider=provider,
            loader_func=loader,
            api_key=api_key,
            interval=interval,
            outputsize=outputsize,
            start=start,
            end=end,
        )

    print(f"✅ Loaded {len(df)} rows for {symbol} — simulate training or backtest here")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="batch_config.yaml", help="Path to batch config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        configs = yaml.safe_load(f)

    for i, cfg in enumerate(configs):
        print(f"=== Running config #{i+1} ===")
        run_backtest(cfg)

if __name__ == "__main__":
    main()
