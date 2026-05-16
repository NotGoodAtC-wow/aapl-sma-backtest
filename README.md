# AAPL SMA Robustness Research

Educational quantitative analysis project for Apple Inc. (`AAPL`). It downloads
daily market data, tests a long-only SMA 20 / SMA 100 crossover strategy, applies
10 bps transaction costs on position changes, and compares the strategy with a
buy-and-hold benchmark.

Version 2 expands the project into a robustness research framework: transaction
cost sensitivity, parameter sweeps, train/test validation, walk-forward tests,
multi-asset checks, and long-only return enhancement variants.

This project is for research and education only. It is not investment advice.

## Project Contents

- `main.py` - CLI entrypoint for downloading data and running the backtest.
- `research.py` - CLI entrypoint for the v2 robustness research workflow.
- `configs/research_v2.yaml` - default research experiment configuration.
- `src/quant_backtest/` - data loading, strategy, metrics, and charting code.
- `scripts/create_visual_report.py` - creates the model forecast PNG and Excel-ready CSV.
- `scripts/create_excel_report.py` - creates the Excel dashboard from generated CSV files.
- `outputs/` - generated reports and sample output from the AAPL run.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```

## Run

```powershell
.\.venv\Scripts\python main.py --ticker AAPL --start 2015-01-01 --short-window 20 --long-window 100 --cost-bps 10 --initial-capital 10000
```

The command writes:

- `outputs/equity_curve.csv`
- `outputs/metrics.csv`
- `outputs/aapl_sma_backtest.png`

## Run Robustness Research

Full run using Yahoo Finance data:

```powershell
.\.venv\Scripts\python research.py --config configs\research_v2.yaml
```

Offline smoke run using deterministic fixture data:

```powershell
.\.venv\Scripts\python research.py --config configs\research_v2.yaml --no-download --output-dir outputs_fixture
```

The v2 research command writes:

- `outputs/base_backtest.csv`
- `outputs/cost_sensitivity.csv`
- `outputs/parameter_sweep.csv`
- `outputs/train_test_results.csv`
- `outputs/walk_forward_results.csv`
- `outputs/multi_asset_results.csv`
- `outputs/model_leaderboard.csv`
- `outputs/research_report.xlsx`
- PNG charts for baseline, costs, heatmaps, train/test, multi-asset, and leaderboard.

## Create Reports

```powershell
.\.venv\Scripts\python scripts\create_visual_report.py
.\.venv\Scripts\python scripts\create_excel_report.py
```

The report scripts write:

- `outputs/aapl_model_forecast_visual.png`
- `outputs/excel_model_data.csv`
- `outputs/aapl_model_forecast_report.xlsx`

## Test

```powershell
.\.venv\Scripts\python -m pytest
```

## Latest Sample Result

For the AAPL run from `2015-01-02` through `2026-05-15`, starting with
`$10,000`:

- SMA strategy ending equity: `$34,979.36`
- SMA strategy total return: `249.79%`
- Buy-and-hold ending equity: `$124,099.93`
- Buy-and-hold total return: `1141.00%`

The strategy made money historically, but it did not outperform buy-and-hold for
this AAPL period.

## Notes

- The strategy uses adjusted close when available.
- Signals are shifted by one day to avoid lookahead bias.
- The `--end` argument is exclusive because Yahoo Finance treats it that way.
- v2 intentionally does not add shorts; it focuses on validating and improving long-only signals first.
