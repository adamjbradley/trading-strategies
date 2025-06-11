import os
import json
from hashlib import sha256
from pathlib import Path

MANIFEST_PATH = Path("data_manifest.json")

def compute_hash(text):
    return sha256(text.encode("utf-8")).hexdigest()

def write_manifest_entry(symbol, provider, interval, outputsize, version_file):
    entry = {
        "symbol": symbol,
        "provider": provider,
        "interval": interval,
        "outputsize": outputsize,
        "file": version_file
    }
    manifest = {}
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text())
    key = compute_hash(f"{symbol}_{provider}_{interval}_{outputsize}")
    manifest[key] = entry
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))

def get_cached_file(symbol, provider, interval, outputsize):
    key = compute_hash(f"{symbol}_{provider}_{interval}_{outputsize}")
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text())
        if key in manifest:
            return manifest[key].get("file")
    return None
