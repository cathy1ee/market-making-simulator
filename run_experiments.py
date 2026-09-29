"""
Runs every experiment in the README and writes results/*.csv.

    python run_experiments.py            # full run
    python run_experiments.py --quick    # fewer paths, for a fast sanity check
"""

import argparse
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd

from mmsim import STRATEGIES, Params, make_draws, simulate, simulate_loop, summarize, with_

OUT = Path(__file__).parent / "results"


def parameter_sweep(n_paths: int) -> pd.DataFrame:
    """Every strategy on the *same* simulated markets, across a grid of market conditions."""
    base = Params()
    grid = {
        "sigma": [1.0, 2.0, 3.0],          # volatility
        "A": [70.0, 140.0, 280.0],         # order-flow intensity
        "k": [1.0, 1.5, 3.0],              # how fast fills thin out away from the mid
        "gamma": [0.05, 0.1, 0.5],         # risk aversion (inventory/adaptive quoting)
    }
    rows = []
    for idx, values in enumerate(itertools.product(*grid.values())):
        p = with_(base, **dict(zip(grid.keys(), values)), sigma_high=values[0])
        draws = make_draws(p, n_paths, seed=idx)
        for strat in STRATEGIES:
            rows.append({**dict(zip(grid.keys(), values)), "strategy": strat, **summarize(simulate(strat, p, draws))})
    return pd.DataFrame(rows)


def regime_test(n_paths: int) -> pd.DataFrame:
    """Market that randomly flips between calm (sigma=1) and stressed (sigma=4) volatility.
    `inventory` still assumes sigma=1; `adaptive` estimates vol online."""
    rows = []
    for gamma in [0.05, 0.1, 0.5]:
        p = Params(sigma=1.0, sigma_high=4.0, p_switch=0.02, gamma=gamma)
        draws = make_draws(p, n_paths, seed=999 + int(gamma * 100))
        for strat in STRATEGIES:
            rows.append({"gamma": gamma, "strategy": strat, **summarize(simulate(strat, p, draws))})
    return pd.DataFrame(rows)


def speed_and_correctness(n_paths: int) -> dict:
    p = Params()
    draws = make_draws(p, n_paths, seed=123)
    t0 = time.perf_counter()
    ref = simulate_loop("inventory", p, draws)
    t_loop = time.perf_counter() - t0
    t0 = time.perf_counter()
    vec = simulate("inventory", p, draws)["pnl"]
    t_vec = time.perf_counter() - t0
    return {
        "paths": n_paths,
        "max_abs_diff": float(np.max(np.abs(ref - vec))),
        "loop_sec": t_loop,
        "vectorized_sec": t_vec,
        "speedup": t_loop / t_vec,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    n = 500 if args.quick else 5000
    OUT.mkdir(exist_ok=True)
    pd.set_option("display.width", 160)

    t0 = time.perf_counter()
    sweep = parameter_sweep(n)
    sweep.to_csv(OUT / "sweep.csv", index=False)
    n_cfg = len(sweep) // len(STRATEGIES)
    sessions = len(sweep) * n
    print(f"Parameter sweep: {n_cfg} market configs x {len(STRATEGIES)} strategies x {n} sessions "
          f"= {sessions:,} simulated sessions ({sessions * Params().n_steps:,} steps) "
          f"in {time.perf_counter() - t0:.1f}s\n")

    # Head-to-head at the Avellaneda-Stoikov paper's baseline parameters.
    base = sweep[(sweep.sigma == 2.0) & (sweep.A == 140.0) & (sweep.k == 1.5) & (sweep.gamma == 0.1)]
    print("Baseline (sigma=2, A=140, k=1.5, gamma=0.1):")
    print(base[["strategy", "mean_pnl", "std_pnl", "sharpe", "mean_max_drawdown", "std_final_inventory", "mean_trades"]]
          .round(3).to_string(index=False), "\n")

    # How often does inventory-aware quoting beat fixed quoting on the same markets?
    piv = sweep.pivot_table(index=["sigma", "A", "k", "gamma"], columns="strategy",
                            values=["sharpe", "std_pnl", "mean_max_drawdown", "mean_pnl"])
    wins = (piv["sharpe"]["inventory"] > piv["sharpe"]["fixed"]).mean()
    med_sharpe = (piv["sharpe"]["inventory"] / piv["sharpe"]["fixed"]).median()
    med_std = 1 - (piv["std_pnl"]["inventory"] / piv["std_pnl"]["fixed"]).median()
    med_dd = 1 - (piv["mean_max_drawdown"]["inventory"] / piv["mean_max_drawdown"]["fixed"]).median()
    med_pnl = (piv["mean_pnl"]["inventory"] / piv["mean_pnl"]["fixed"]).median()
    summary = pd.Series({
        "configs": n_cfg,
        "inventory_beats_fixed_sharpe_pct": 100 * wins,
        "median_sharpe_ratio_inventory_vs_fixed": med_sharpe,
        "median_pnl_std_reduction_pct": 100 * med_std,
        "median_drawdown_reduction_pct": 100 * med_dd,
        "median_mean_pnl_ratio": med_pnl,
    })
    summary.to_csv(OUT / "sweep_summary.csv", header=["value"])
    print("Inventory-aware vs fixed across all configs:")
    print(summary.round(3).to_string(), "\n")

    reg = regime_test(n)
    reg.to_csv(OUT / "regime.csv", index=False)
    print("Regime-switching market (vol flips between 1 and 4):")
    print(reg[["gamma", "strategy", "mean_pnl", "std_pnl", "sharpe", "mean_max_drawdown"]].round(3).to_string(index=False), "\n")

    sp = speed_and_correctness(200 if args.quick else 1000)
    pd.Series(sp).to_csv(OUT / "speed.csv", header=["value"])
    print(f"Vectorized vs pure-Python loop on {sp['paths']} paths: {sp['loop_sec']:.2f}s -> "
          f"{sp['vectorized_sec']:.3f}s ({sp['speedup']:.0f}x faster), max P&L difference {sp['max_abs_diff']:.1e}")


if __name__ == "__main__":
    main()
