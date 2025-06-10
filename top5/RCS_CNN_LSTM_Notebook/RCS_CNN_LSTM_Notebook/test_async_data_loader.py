from async_data_loader import load_or_fetch

symbol = "EURUSD"
provider = "metatrader"
broker = "amp_global"
interval = "H1"

df = load_or_fetch(
    symbol=symbol,
    provider=provider,
    loader_func=None,
    api_key="",
    interval=interval,
    broker=broker,
    force_refresh=False,
)

print(df.head())
print("Data shape:", df.shape)
