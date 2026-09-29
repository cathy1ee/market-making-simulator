"""
Market-making simulator.

A market maker posts a bid and an ask around the mid price. Market orders arrive at random
and hit our quotes with a probability that falls off exponentially the further our quote is
from the mid (the Avellaneda-Stoikov 2008 order-flow model):

    fill intensity  lambda(delta) = A * exp(-k * delta)

where delta is the distance from our quote to the mid, A is how much order flow there is,
and k is how quickly that flow thins out away from the mid.

Three quoting strategies are compared:

  fixed     - symmetric quotes at a constant half-spread around the mid; ignores inventory.
  inventory - Avellaneda-Stoikov: shift both quotes against current inventory (to shed risk)
              and widen the spread when volatility or remaining time is high.
  adaptive  - same as `inventory`, but volatility is *estimated* online from recent mid-price
              moves (EWMA) instead of assumed known, so quotes react when the market regime shifts.

Everything is vectorized across paths with NumPy: one time loop, with every simulated
session updated at once as an array.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

STRATEGIES = ("fixed", "inventory", "adaptive")


@dataclass(frozen=True)
class Params:
    s0: float = 100.0         # starting mid price
    T: float = 1.0            # session length (one simulated trading session)
    n_steps: int = 200        # time steps per session
    sigma: float = 2.0        # mid-price volatility (calm regime)
    sigma_high: float = 2.0   # volatility in the stressed regime (== sigma means no regime switching)
    p_switch: float = 0.0     # per-step probability of flipping regime
    A: float = 140.0          # order-flow intensity at the mid
    k: float = 1.5            # decay of fill intensity with distance from mid
    gamma: float = 0.1        # risk aversion for inventory/adaptive strategies
    fixed_half_spread: float | None = None  # if None, matched to the AS average spread (fair comparison)
    max_inventory: int = 50   # hard position limit (stop quoting that side at the limit)
    ewma_lambda: float = 0.94 # decay for the adaptive strategy's volatility estimate

    @property
    def dt(self) -> float:
        return self.T / self.n_steps

    def as_half_spread(self, sigma: float, tau: float) -> float:
        """Avellaneda-Stoikov optimal half-spread for a given vol and time remaining."""
        g, k = self.gamma, self.k
        return 0.5 * (g * sigma**2 * tau + (2.0 / g) * np.log1p(g / k))

    def matched_fixed_half_spread(self) -> float:
        """Average AS half-spread over the session, so `fixed` quotes equally wide on average."""
        taus = self.T - self.dt * np.arange(self.n_steps)
        return float(np.mean([self.as_half_spread(self.sigma, t) for t in taus]))


@dataclass
class Draws:
    """All randomness for a batch, generated up front so any strategy can replay the same market."""
    z: np.ndarray        # (n_steps, n_paths) standard normals for mid-price moves
    u_bid: np.ndarray    # (n_steps, n_paths) uniforms deciding bid fills
    u_ask: np.ndarray    # (n_steps, n_paths) uniforms deciding ask fills
    u_regime: np.ndarray # (n_steps, n_paths) uniforms deciding regime switches


def make_draws(p: Params, n_paths: int, seed: int) -> Draws:
    rng = np.random.default_rng(seed)
    shape = (p.n_steps, n_paths)
    return Draws(rng.standard_normal(shape), rng.random(shape), rng.random(shape), rng.random(shape))


def simulate(strategy: str, p: Params, draws: Draws) -> dict[str, np.ndarray]:
    """Run one strategy over every path in `draws`. Returns per-path results as arrays."""
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}")
    n_steps, n_paths = draws.z.shape
    dt, sqrt_dt = p.dt, np.sqrt(p.dt)
    fixed_h = p.fixed_half_spread if p.fixed_half_spread is not None else p.matched_fixed_half_spread()

    mid = np.full(n_paths, p.s0)
    cash = np.zeros(n_paths)
    inv = np.zeros(n_paths)
    high_vol = np.zeros(n_paths, dtype=bool)
    var_hat = np.full(n_paths, p.sigma**2)  # adaptive strategy's running variance estimate (per unit time)
    peak = np.zeros(n_paths)
    max_dd = np.zeros(n_paths)
    n_trades = np.zeros(n_paths)
    abs_inv_sum = np.zeros(n_paths)

    for i in range(n_steps):
        tau = p.T - i * dt

        # --- 1. choose quotes ---
        if strategy == "fixed":
            bid, ask = mid - fixed_h, mid + fixed_h
        else:
            sig2 = p.sigma**2 if strategy == "inventory" else var_hat
            reservation = mid - inv * p.gamma * sig2 * tau          # skew quotes against inventory
            half = 0.5 * (p.gamma * sig2 * tau + (2.0 / p.gamma) * np.log1p(p.gamma / p.k))
            bid, ask = reservation - half, reservation + half

        # --- 2. market orders arrive and may fill us ---
        lam_bid = p.A * np.exp(-p.k * np.maximum(mid - bid, 0.0))
        lam_ask = p.A * np.exp(-p.k * np.maximum(ask - mid, 0.0))
        fill_bid = (draws.u_bid[i] < 1.0 - np.exp(-lam_bid * dt)) & (inv < p.max_inventory)
        fill_ask = (draws.u_ask[i] < 1.0 - np.exp(-lam_ask * dt)) & (inv > -p.max_inventory)
        cash += np.where(fill_ask, ask, 0.0) - np.where(fill_bid, bid, 0.0)
        inv += fill_bid.astype(float) - fill_ask.astype(float)
        n_trades += fill_bid + fill_ask

        # --- 3. mid price moves; volatility regime may switch ---
        high_vol ^= draws.u_regime[i] < p.p_switch
        sigma_now = np.where(high_vol, p.sigma_high, p.sigma)
        dS = sigma_now * sqrt_dt * draws.z[i]
        mid += dS
        var_hat = p.ewma_lambda * var_hat + (1 - p.ewma_lambda) * dS**2 / dt

        # --- 4. bookkeeping: mark-to-market P&L and drawdown ---
        pnl = cash + inv * mid
        peak = np.maximum(peak, pnl)
        max_dd = np.maximum(max_dd, peak - pnl)
        abs_inv_sum += np.abs(inv)

    return {
        "pnl": cash + inv * mid,        # terminal P&L, open inventory marked at the final mid
        "final_inventory": inv,
        "max_drawdown": max_dd,
        "n_trades": n_trades,
        "avg_abs_inventory": abs_inv_sum / n_steps,
    }


def simulate_loop(strategy: str, p: Params, draws: Draws) -> np.ndarray:
    """Plain-Python reference implementation (one path at a time, scalar math).
    Used to verify the vectorized version gives identical results and to measure the speedup.
    Returns terminal P&L per path."""
    import math

    n_steps, n_paths = draws.z.shape
    dt, sqrt_dt = p.dt, math.sqrt(p.dt)
    fixed_h = p.fixed_half_spread if p.fixed_half_spread is not None else p.matched_fixed_half_spread()
    out = np.empty(n_paths)
    for j in range(n_paths):
        mid, cash, inv, high, var_hat = p.s0, 0.0, 0.0, False, p.sigma**2
        for i in range(n_steps):
            tau = p.T - i * dt
            if strategy == "fixed":
                bid, ask = mid - fixed_h, mid + fixed_h
            else:
                sig2 = p.sigma**2 if strategy == "inventory" else var_hat
                r = mid - inv * p.gamma * sig2 * tau
                h = 0.5 * (p.gamma * sig2 * tau + (2.0 / p.gamma) * math.log1p(p.gamma / p.k))
                bid, ask = r - h, r + h
            lb = p.A * math.exp(-p.k * max(mid - bid, 0.0))
            la = p.A * math.exp(-p.k * max(ask - mid, 0.0))
            fb = draws.u_bid[i, j] < 1.0 - math.exp(-lb * dt) and inv < p.max_inventory
            fa = draws.u_ask[i, j] < 1.0 - math.exp(-la * dt) and inv > -p.max_inventory
            if fa:
                cash += ask
            if fb:
                cash -= bid
            inv += int(fb) - int(fa)
            if draws.u_regime[i, j] < p.p_switch:
                high = not high
            dS = (p.sigma_high if high else p.sigma) * sqrt_dt * draws.z[i, j]
            mid += dS
            var_hat = p.ewma_lambda * var_hat + (1 - p.ewma_lambda) * dS * dS / dt
        out[j] = cash + inv * mid
    return out


def summarize(res: dict[str, np.ndarray]) -> dict[str, float]:
    pnl = res["pnl"]
    return {
        "mean_pnl": float(pnl.mean()),
        "std_pnl": float(pnl.std(ddof=1)),
        "sharpe": float(pnl.mean() / pnl.std(ddof=1)),  # per-session risk-adjusted P&L
        "mean_max_drawdown": float(res["max_drawdown"].mean()),
        "mean_trades": float(res["n_trades"].mean()),
        "std_final_inventory": float(res["final_inventory"].std(ddof=1)),
        "mean_abs_inventory": float(res["avg_abs_inventory"].mean()),
    }


def with_(p: Params, **kw) -> Params:
    return replace(p, **kw)
