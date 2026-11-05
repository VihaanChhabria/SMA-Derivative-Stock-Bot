# SMA Derivative Stock Bot

## Run a workflow

Run a backtest interactively:

```powershell
python main.py
```

You pick one or more buy strategies (a ticker is bought only when every
selected buy strategy signals it), one or more sell strategies, risk settings
if you chose `risk_reward`, starting capital, max open positions, and the data
to test on.

The backtest engine (`backtesting/engine.py`):

- buys at the signal day's close, ranking candidates by the SMA-slope score
  when available, and splits available cash evenly across each day's new buys
  up to the max open positions (fractional shares allowed, see
  `ALLOW_FRACTIONAL_SHARES` in `config/settings.py`);
- applies each sell strategy to the positions it actually holds and exits on
  whichever rule fires first (`x_day_hold` at the close after N days,
  `risk_reward` at the stop or target price, or the open on a gap);
- values positions still open at the end at the last close.

Each run is saved to `outputs/backtests/<timestamp>_<buys>_<sells>/`:
`trades.csv` (every round trip with P/L and exit reason), `daily_equity.csv`
(cash, positions value, equity, open positions, drawdown), `summary.json`
(return, win rate, profit factor, max drawdown, ...) and `settings.json`.

Compare a saved run against the S&P 500 (chart saved into the run folder):

```powershell
python -m backtesting.analysis
```

Live scanners can still run on their own, e.g.
`python -m scanners.sma_slope_buy_signal`; their CSVs go to `outputs/scans/`.

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
