# Market-Making Simulator (Python)

Simulates a market maker quoting a bid and an ask. It compares three quoting strategies on identical simulated markets, across 81 combinations of volatility, order flow and risk aversion.

## Strategies

| Strategy | How it quotes |
|---|---|
| `fixed` | Symmetric quotes at a constant half-spread around the mid, and ignores inventory. The spread is set to the average width of the inventory strategy, so the comparison is fair. |
| `inventory` | Avellaneda-Stoikov (2008). It skews both quotes against its current position to shed risk, and **widens** the spread when volatility or remaining time is high. |
| `adaptive` | Same as `inventory`, but it estimates volatility online (EWMA of mid-price moves) instead of assuming it is known, so it reacts when the market regime changes. |

## Model

- **Mid price:** a random walk with volatility σ. It can optionally switch at random between a calm regime and a stressed one.
- **Order flow:** the chance that a market order hits our quote falls off exponentially with the quote's distance δ from the mid: `λ(δ) = A·e^(−kδ)`. A is the amount of order flow, and k is how fast it thins out.
- **Inventory-aware quoting:** the strategy quotes around a reservation price `r = mid − q·γ·σ²·(T−t)` instead of the mid. Here q is inventory and γ is risk aversion. When long, it lowers both quotes, which makes it more likely to sell and less likely to buy. The half-spread is `½[γσ²(T−t) + (2/γ)·ln(1 + γ/k)]`.
- **Position limit:** 50 units. At the limit the strategy stops quoting that side.
- **P&L:** cash plus inventory marked to the final mid. Drawdown is measured on mark-to-market P&L at every step.

## Run

```bash
pip install numpy pandas
python test_mmsim.py          # tests
python run_experiments.py     # all experiments (~20s), writes results/*.csv
```

## Results

**Scale:** 81 market configurations (volatility × order flow × fill decay × risk aversion) × 3 strategies × 5,000 sessions gives 1.2M simulated sessions and 243M time steps, in about 18 seconds.

**Baseline** (σ=2, A=140, k=1.5, γ=0.1, the parameters from the original paper):

| Strategy | Mean P&L | P&L std | Sharpe (mean/std) | Avg max drawdown | Final inventory std |
|---|---:|---:|---:|---:|---:|
| fixed | 60.8 | 12.7 | 4.8 | 5.8 | 8.0 |
| inventory | 57.1 | 6.4 | 8.9 | 1.2 | 2.9 |

These results match the paper: inventory-aware quoting gives up a little profit and roughly halves the risk.

Across all 81 configurations, compared with fixed quoting, inventory-aware quoting:
- had a higher Sharpe ratio in 81 of 81 configurations (median 1.8×)
- cut P&L standard deviation by a median of 50%
- cut max drawdown by a median of 76%
- earned a median of 93.5% of fixed quoting's mean P&L

Regime-switching market (volatility flips at random between 1 and 4; `inventory` assumes σ=1):

| γ | Strategy | Sharpe | Avg max drawdown |
|---|---|---:|---:|
| 0.1 | fixed | 3.4 | 9.9 |
| 0.1 | inventory | 6.6 | 3.7 |
| 0.1 | adaptive | **7.6** | **1.8** |
| 0.5 | inventory | **8.5** | 1.8 |
| 0.5 | adaptive | 3.5 | 1.6 |

When volatility is estimated online, the strategy gains Sharpe at low and moderate risk aversion (γ = 0.05 and 0.1). At high risk aversion (γ = 0.5) it backfires: noise in the volatility estimate gets amplified by γσ², so spreads blow out and fills dry up. The lesson is that an adaptive estimator needs smoothing or caps when the strategy is sensitive to it.

**Performance:** vectorizing across paths with NumPy is 25× faster than a pure-Python loop (0.66s → 0.026s for 1,000 sessions), with identical results (max difference 3e-13).

## Layout

```
mmsim.py             simulator: model, three strategies, vectorized + reference loop versions
run_experiments.py   parameter sweep, regime test, speed benchmark
test_mmsim.py        tests (vectorized == loop, position limits, zero-vol sanity, risk reduction)
results/             CSV output
```

## Limitations and next steps
- There is no adverse selection: order flow doesn't predict future price moves. Adding informed flow would penalize tight quoting.
- Fills are all-or-nothing at one unit, with no queue position or partial fills.
- A natural next step is to replay real order-book data instead of a simulated mid.

Reference: Avellaneda, M. & Stoikov, S. (2008). *High-frequency trading in a limit order book.* Quantitative Finance 8(3).
