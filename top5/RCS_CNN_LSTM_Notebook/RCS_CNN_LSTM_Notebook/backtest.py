import yaml
import asyncio
import argparse
from async_data_loader import fetch_all_data
from data_loader import load_or_fetch, load_twelve_data

def run_backtest(config):
    symbol = config['symbol']
    provider = config.get('provider', 'twelvedata')
    api_key = config['api_keys'][provider]
    interval = config.get("interval", "1min")
    outputsize = config.get("outputsize", 500)

    if config.get("use_async", False):
        print(f"⚡ Async loading for {symbol}...")
        df_dict = asyncio.run(fetch_all_data([symbol], provider, api_key, interval=interval, outputsize=outputsize))
        df = df_dict[symbol]
    else:
        print(f"⏳ Sync loading for {symbol}...")
        df = load_or_fetch(symbol=symbol, provider=provider, loader_func=load_twelve_data, api_key=api_key,
                           interval=interval, outputsize=outputsize)

    print(f"✅ Loaded {len(df)} rows for {symbol} — simulate training or backtest here")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="batch_config.yaml", help="Path to batch config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        configs = yaml.safe_load(f)

    for i, cfg in enumerate(configs):
        print(f"
=== Running config #{i+1} ===")
        run_backtest(cfg)

if __name__ == "__main__":
    main()
