# SMA Derivative Stock Bot

## Run a workflow

Use the root CLI to run a scanner, backtest, or backtest analysis:

```powershell
python main.py --strategy sma_slope
python main.py --strategy stochastic
python main.py --strategy backtest
python main.py --strategy analyze
```

Generated scanner CSVs are saved under `outputs/scans/`. Backtest CSVs and
analysis charts are saved under `outputs/backtests/`.

## Saved market data

Every market data download is saved under `data/downloads/` (a `.pkl` file
plus a `.json` metadata file; the folder is git-ignored). When you run a
backtest from `main.py`, you are asked to either:

- pick one of the saved downloads (listed by date range, ticker count, and
  download time), or
- enter a specific start/end date range. Extra history before the start date
  is loaded so long indicators like the 200-day SMA are ready, and signals are
  trimmed back to your range. A saved download covering the range and tickers
  is reused if one exists; otherwise the data is downloaded and saved.

## Project layout

- `config/`: strategy parameters and ticker universes.
- `src/`: shared data download, indicator, and output-path helpers.
- `scanners/`: live SMA-slope and stochastic scanners.
- `backtesting/`: the portfolio backtest, strategy signals, and analysis.
- `outputs/scans/`: historical scanner exports.
- `outputs/backtests/`: historical backtest results and artifacts.
- `archive/`: obsolete scripts retained for reference.
