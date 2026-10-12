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

## Project layout

- `config/`: strategy parameters and ticker universes.
- `src/`: shared data download, indicator, and output-path helpers.
- `scanners/`: live SMA-slope and stochastic scanners.
- `backtesting/`: the portfolio backtest, strategy signals, and analysis.
- `outputs/scans/`: historical scanner exports.
- `outputs/backtests/`: historical backtest results and artifacts.
- `archive/`: obsolete scripts retained for reference.
