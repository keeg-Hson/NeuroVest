# Archived Download Scripts

These download scripts have been archived as part of the codebase consolidation.

## Canonical Scripts (Use These)

| Script | Purpose |
|--------|---------|
| `update_data.py` | Main data update interface (stocks, crypto, CLI) |
| `download_crypto_comprehensive.py` | Multi-source crypto data (CryptoCompare, Binance, CoinGecko) |

## Archived Scripts

| File | Reason | Replacement |
|------|--------|-------------|
| `download_assets_simple.py` | Simple ETF downloader | Use `update_data.py` |
| `download_cross_asset_simple.py` | Cross-asset downloader | Use `update_data.py` |
| `download_crypto_data.py` | Basic crypto downloader | Use `download_crypto_comprehensive.py` |
| `download_crypto_enhanced.py` | Enhanced crypto downloader | Use `download_crypto_comprehensive.py` |
| `download_equity_etfs.py` | ETF downloader | Use `update_data.py` |
| `download_equity_etfs_alternative.py` | Alternative ETF downloader | Use `update_data.py` |
| `download_multi_asset_data.py` | Multi-asset downloader (yfinance) | Use `update_data.py` |
| `download_spy_data.py` | SPY-specific downloader | Use `update_data.py` |
| `update_spy_data.py` | SPY updater | Use `update_data.py` |

## Usage Examples

```bash
# Update all assets (stocks and crypto)
python update_data.py update

# Query specific asset
python update_data.py query SPY --start 2024-01-01

# Export to CSV
python update_data.py export SPY spy_data.csv

# Show database stats
python update_data.py stats

# For comprehensive crypto data with multiple sources
python download_crypto_comprehensive.py
```

*Archived: 2026-02-11*
