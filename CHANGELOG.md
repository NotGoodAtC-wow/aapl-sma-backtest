# Changelog

All notable project changes are documented here.

## 0.3.0 - Turnover-Aware Trend Allocation

### Added

- `TrendAllocationStrategy` with SMA hysteresis, minimum holding periods, and
  cooldown periods.
- Regime-aware fallback variants for SPY and QQQ.
- Hybrid allocation variants that can split weak-trend exposure between AAPL and
  a market fallback asset.
- Turnover-aware `selection_score` that penalizes excessive trading and unstable
  parameter choices.
- `configs/research_v3.yaml` as the default research configuration.
- New research outputs:
  - `hysteresis_sweep.csv`;
  - `allocation_leaderboard.csv`;
  - `capture_analysis.csv`;
  - `turnover_analysis.csv`;
  - `v03_comparison.csv`;
  - `v03_cost_sensitivity.csv`;
  - `v03_selected_curve.csv`.
- New charts for v0.3 equity/drawdown, turnover vs Sharpe, capture ratios,
  allocation exposure, selected-model cost sensitivity, and entry/exit signals.
- Additional metrics: upside capture, downside capture, missed return while
  underweight AAPL, holding-period stats, trade frequency, and fallback exposure.

### Changed

- `research.py` now defaults to `configs/research_v3.yaml`.
- The model selection process now prefers robust low-turnover candidates over
  marginally higher Sharpe candidates.
- Yahoo Finance timezone cache is stored under the project-local `.cache/`
  directory to avoid Windows cache permission failures.
- Market-regime window settings from the YAML config are now passed through to
  the fallback and hybrid allocation evaluators.
- `research_v2.yaml` remains runnable for compatibility.

### Findings

- The selected v0.3 model is `SMA 5/200` with `1%` entry threshold, `-1%` exit
  threshold, 10-day minimum hold, and 5-day cooldown.
- On the test period, the selected model improves versus v2 `SMA 5/50`
  long/cash: higher CAGR, higher Sharpe, lower turnover, lower cost drag, and
  similar drawdown control.
- The aggressive SPY/QQQ fallback variants still produce higher raw CAGR, but
  they fail the current turnover filter.
- The selected v0.3 model remains below AAPL buy-and-hold by raw CAGR, so it is
  a risk-managed trend baseline rather than a proven alpha model.

## 0.2.0 - Robustness Research Framework

### Added

- Research CLI: `research.py --config configs/research_v2.yaml`.
- Config-driven research workflow for repeatable experiments.
- Transaction cost sensitivity across `0`, `5`, `10`, `20`, and `50` bps.
- SMA parameter sweep with Sharpe/CAGR heatmaps and local stability fields.
- Train/test validation and walk-forward testing.
- Multi-asset validation across AAPL, MSFT, NVDA, AMZN, META, GOOGL, SPY, and QQQ.
- Long-only return enhancement variants:
  - long/cash baseline;
  - SPY fallback;
  - QQQ fallback;
  - partial exposure;
  - SMA spread threshold;
  - 3-month momentum filter;
  - 6-month momentum filter.
- Expanded metrics: Sortino, Calmar, exposure, turnover, trade distribution,
  cost drag, excess CAGR, and drawdown improvement versus benchmark.
- Research outputs:
  - `research_report.xlsx`;
  - cost sensitivity table and chart;
  - parameter sweep table and heatmaps;
  - train/test and walk-forward result tables;
  - multi-asset comparison;
  - model leaderboard.
- GitHub Actions workflow for running the test suite.

### Changed

- Split the original single-function backtest into strategy, cost, engine,
  experiment, and reporting modules.
- Kept the original `main.py` command compatible with the first version.
- Updated generated sample outputs to include the expanded metric set.
- Improved project documentation and result interpretation.

### Findings

- The original `SMA 20/100` strategy was profitable, but underperformed AAPL
  buy-and-hold by a wide margin.
- The faster `SMA 5/50` region improved the full-sample strategy profile:
  higher CAGR, better Sharpe, and lower max drawdown than `SMA 20/100`.
- Out-of-sample results are still weaker than buy-and-hold by CAGR and Sharpe.
- Fallback variants using SPY or QQQ improve raw return but currently trade too
  much to pass the robustness filter.

## 0.1.0 - Initial AAPL SMA Backtest

### Added

- AAPL daily price download via Yahoo Finance.
- Long-only SMA crossover strategy.
- Basic backtest metrics and buy-and-hold comparison.
- CSV, PNG, and Excel report outputs.
- Unit tests for signal shifting, transaction costs, metrics, and smoke runs.
