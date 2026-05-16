# Research Summary

This project evaluates a long-only SMA timing model for AAPL and related assets.
The current objective is not to force a profitable-looking chart, but to test
whether the signal remains useful across reasonable assumptions.

## Baseline

The first version used `SMA 20/100` on AAPL with 10 bps transaction costs.

| Metric | Strategy | Buy and Hold |
| --- | ---: | ---: |
| Total return | 249.79% | 1141.00% |
| CAGR | 11.65% | 24.81% |
| Sharpe | 0.62 | 0.92 |
| Max drawdown | -35.64% | -38.52% |

The baseline was profitable and slightly reduced drawdown, but it gave up too
much upside.

## Parameter Sweep Result

The robustness workflow found a better full-sample SMA region around `SMA 5/50`.
At 10 bps, this selected setup improved the model profile:

| Metric | SMA 20/100 | SMA 5/50 |
| --- | ---: | ---: |
| Total return | 249.79% | 467.47% |
| CAGR | 11.65% | 16.50% |
| Sharpe | 0.62 | 0.91 |
| Max drawdown | -35.64% | -25.04% |

This is a real improvement over the original rule, but it comes with higher
turnover.

## Cost Sensitivity

The selected `SMA 5/50` strategy remains profitable across the tested cost range.

| Cost | CAGR | Sharpe | Max drawdown |
| ---: | ---: | ---: | ---: |
| 0 bps | 17.42% | 0.95 | -24.59% |
| 5 bps | 16.96% | 0.93 | -24.82% |
| 10 bps | 16.50% | 0.91 | -25.04% |
| 20 bps | 15.60% | 0.87 | -25.49% |
| 50 bps | 12.91% | 0.74 | -28.64% |

The model does not depend on exactly 10 bps, but transaction costs matter because
the faster rule trades more often.

## Out-of-Sample Check

The selected parameters perform worse out of sample than on the training period.

| Period | Strategy CAGR | Benchmark CAGR | Strategy Sharpe | Benchmark Sharpe |
| --- | ---: | ---: | ---: | ---: |
| Train | 26.41% | 32.19% | 1.29 | 1.09 |
| Test | 7.39% | 17.64% | 0.51 | 0.73 |

This is the main limitation. The strategy improves risk control, but it has not
yet proven persistent alpha versus buy-and-hold.

## Variant Review

The SPY and QQQ fallback variants improve raw return, especially on the test
period, but they also raise turnover. They are promising research directions,
not finished models.

The best current interpretation:

- `SMA 5/50 long/cash` is a stronger baseline than `SMA 20/100`.
- Fallback assets may reduce cash drag.
- Turnover control is the next important research target.
- Shorts should still wait until the long-only signal is stronger.

## Next Research Steps

1. Add turnover-aware selection criteria directly into fallback variants.
2. Test minimum holding periods and signal hysteresis to reduce whipsaw trades.
3. Compare fallback assets with Treasury ETFs or cash yield assumptions.
4. Expand regime analysis by bull, bear, and sideways markets.
5. Only consider shorts after long-only robustness improves.
