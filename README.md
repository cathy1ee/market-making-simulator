# Market-Making Simulator (Python)

![tests](https://github.com/cathy1ee/market-making-simulator/actions/workflows/tests.yml/badge.svg)

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
pip install -r requirements.txt
python test_mmsim.py          # tests
python test_live.py           # live-data tests (offline, no network)
python run_experiments.py     # all experiments (~20s), writes results/*.csv
python live.py --product ETH-USD --minutes 5   # paper-trade on live data
```

## Live data

`live.py` runs the same three strategies against Coinbase's public market data feed (no account or API key). Nothing is ever sent to the exchange: quotes are virtual and filled from real trades.

- Mid: the real best-bid/best-ask midpoint, from the websocket `ticker` channel.
- Fills: a real trade strictly *through* our bid fills us at our bid (and likewise for the ask), for no more than the size that really traded. A trade exactly at our price does not, since we would be at the back of the queue. Each side fills at most one lot per quote refresh.
- Quotes are post-only on the exchange's price grid, so they never cross the real book.
- Calibration: the first `--warmup` seconds (default 60) are used to estimate σ (1-second mid changes) and A, k (a fit of trade intensity against distance from the mid). γ defaults to the value where the inventory-risk and order-flow parts of the spread are equal at the start (γσ²T = 2/k); set it with `--gamma`.
- Markout: the mid 5s after each fill minus our fill price, in our favour, averaged per lot filled. It includes the spread we earned, so a negative value means the market moved against us by more than that spread. This is adverse selection, the effect the simulator leaves out.

```bash
python live.py --product ETH-USD --minutes 5 --record ticks.csv   # live, and save the feed
python live.py --replay ticks.csv --minutes 5                      # replay it offline (same result)
python live.py --help                                              # lot size, refresh, strategies, limits
```

Binance's API is blocked from US IP addresses, which is why the feed is Coinbase. Any Coinbase product id works; very tight books such as BTC-USD will fill rarely, because quotes wider than one tick need a trade to sweep through them.

In one short live sample (ETH-USD, 30s warm-up, 2 min session, 1 ETH lots) all three strategies finished slightly negative (`fixed` −0.78, `inventory` −0.47, `adaptive` −0.44). `inventory` carried less risk, with a final position of 1.4 lots against 4.3 and a max drawdown of 1.7 against 4.6. Two minutes is far too short to draw a conclusion; run longer sessions and several `--replay` files before comparing strategies.

## Results

Scale: 81 market configurations (volatility × order flow × fill decay × risk aversion) × 3 strategies × 5,000 sessions gives 1.2M simulated sessions and 243M time steps, in about 18 seconds.

Baseline (σ=2, A=140, k=1.5, γ=0.1, the parameters from the original paper):

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

| γ | Strategy | Mean P&L | Sharpe | Avg max drawdown |
|---|---|---:|---:|---:|
| 0.05 | fixed | 60.5 | 3.3 | 10.2 |
| 0.05 | inventory | 60.1 | 5.4 | 5.1 |
| 0.05 | adaptive | 57.7 | 7.7 | 2.4 |
| 0.1 | fixed | 60.8 | 3.4 | 9.9 |
| 0.1 | inventory | 59.7 | 6.6 | 3.7 |
| 0.1 | adaptive | 54.9 | 7.6 | 1.8 |
| 0.5 | fixed | 60.7 | 3.4 | 9.9 |
| 0.5 | inventory | 55.2 | 8.5 | 1.8 |
| 0.5 | adaptive | 37.7 | 3.5 | 1.6 |

When volatility is estimated online, the strategy gains Sharpe at low and moderate risk aversion (γ = 0.05 and 0.1). At high risk aversion (γ = 0.5) it backfires: noise in the volatility estimate gets amplified by γσ², so spreads blow out and fills dry up. The lesson is that an adaptive estimator needs smoothing or caps when the strategy is sensitive to it.

Performance: vectorizing across paths with NumPy is 28× faster than a pure-Python loop (0.68s → 0.024s for 1,000 sessions), with identical results (max difference 3e-13).

## Layout

```
mmsim.py             simulator: model, three strategies, vectorized + reference loop versions
live.py              paper trading on live Coinbase data: feed, calibration, record/replay
run_experiments.py   parameter sweep, regime test, speed benchmark
test_mmsim.py        tests (vectorized == loop, position limits, zero-vol sanity, risk reduction)
test_live.py         offline tests for live.py on synthetic ticks
results/             CSV output
```

## Limitations and next steps
- There is no adverse selection: order flow doesn't predict future price moves. Adding informed flow would penalize tight quoting.
- Fills are all-or-nothing at one unit, with no queue position or partial fills.
- `live.py` uses real trades, but sees only the top of the book at each trade, with no full depth or queue position. It assumes we get the whole size of any trade through our quote, up to one lot per side per refresh, which is still optimistic.

Reference: Avellaneda, M. & Stoikov, S. (2008). *High-frequency trading in a limit order book.* Quantitative Finance 8(3).
