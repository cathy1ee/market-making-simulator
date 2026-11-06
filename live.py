"""
Paper-trade the three quoting strategies against live market data.

Instead of a simulated random-walk mid and model fills, this listens to real trades from
Coinbase's public websocket feed (no account or API key needed). Every strategy sees the
same feed at the same time, so the comparison is still like-for-like:

  - the mid is the real best-bid/best-ask midpoint;
  - our quotes are virtual: no orders are ever sent. A real trade *through* our bid (strictly
    below it) fills us at our bid, and a trade through our ask fills us at our ask, for no more
    than the size that really traded. A trade exactly at our price does not fill us, since we
    would be at the back of the queue. Each side fills at most one lot per quote refresh, so
    small trades give partial fills.
  - quotes are post-only on the exchange's price grid: we never cross the real book.

Because fills come from real order flow, live data has the adverse selection the simulator
lacks: a trade that sweeps through our bid is often followed by the mid moving lower. The
`markout` column measures it: the mid a few seconds after a fill minus our fill price, in our
favour, averaged per lot filled. It includes the spread we earned, so negative means the market
moved against us by more than that spread.

Model parameters are calibrated from a warm-up window of the same feed:

  sigma  stdev of 1-second mid changes (price per sqrt(second))
  A, k   fit of trade intensity against distance from the mid: lambda(delta) = A * exp(-k * delta)
  gamma  risk aversion is a preference, not something the data shows. By default it is set so
         that, at the start of the session, the inventory-risk part of the spread equals the
         order-flow part (gamma * sigma^2 * T = 2 / k). Override with --gamma.

    python live.py --product ETH-USD --minutes 5                       # live
    python live.py --product ETH-USD --minutes 5 --record ticks.csv    # live, and save the feed
    python live.py --replay ticks.csv                                  # replay a saved feed offline
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import AsyncIterator, Iterable

import numpy as np

from mmsim import STRATEGIES, Params, quote

WS_URL = "wss://ws-feed.exchange.coinbase.com"
TICK_FIELDS = ("t", "bid", "ask", "price", "size")


@dataclass(frozen=True)
class Tick:
    """One trade, with the best bid/ask the exchange reported alongside it."""
    t: float      # exchange timestamp, seconds since the epoch
    bid: float
    ask: float
    price: float  # trade price
    size: float   # trade size, in the base asset

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask)


# ---------------------------------------------------------------------------------------------
# Feeds
# ---------------------------------------------------------------------------------------------

def parse_time(s: str) -> float:
    """Coinbase timestamps look like 2026-11-03T16:10:08.934779Z, sometimes with nanoseconds."""
    s = s.rstrip("Z")
    if "." in s:
        head, frac = s.split(".")
        s = f"{head}.{frac[:6]}"
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp()


async def coinbase_ticks(product: str) -> AsyncIterator[Tick]:
    """Live trades from Coinbase's public `ticker` channel. Reconnects with backoff if dropped."""
    import websockets

    backoff = 1.0
    while True:
        try:
            async with websockets.connect(WS_URL, ping_interval=20) as ws:
                await ws.send(json.dumps({"type": "subscribe", "product_ids": [product], "channels": ["ticker"]}))
                backoff = 1.0
                async for raw in ws:
                    m = json.loads(raw)
                    if m.get("type") == "error":
                        raise SystemExit(f"Coinbase rejected the subscription: {m.get('message')} {m.get('reason', '')}")
                    if m.get("type") != "ticker" or not m.get("best_bid") or not m.get("best_ask"):
                        continue
                    yield Tick(parse_time(m["time"]), float(m["best_bid"]), float(m["best_ask"]),
                               float(m["price"]), float(m.get("last_size") or 0.0))
        except (OSError, websockets.ConnectionClosed) as e:
            print(f"feed disconnected ({e}); reconnecting in {backoff:.0f}s", file=sys.stderr)
            await asyncio.sleep(backoff)
            backoff = min(2 * backoff, 30.0)


async def replay_ticks(path: str) -> AsyncIterator[Tick]:
    """Ticks saved with --record, replayed as fast as they can be read."""
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            yield Tick(*(float(row[k]) for k in TICK_FIELDS))


# ---------------------------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Calibration:
    sigma: float     # price per sqrt(second)
    A: float         # trades per second at the mid
    k: float         # per unit of price
    tick: float      # price grid
    n_trades: int
    seconds: float


def calibrate(ticks: list[Tick]) -> Calibration:
    """Estimate sigma, A and k from a window of ticks (see the module docstring)."""
    t = np.array([x.t for x in ticks])
    bid = np.array([x.bid for x in ticks])
    ask = np.array([x.ask for x in ticks])
    price = np.array([x.price for x in ticks])
    mid = 0.5 * (bid + ask)
    seconds = max(t[-1] - t[0], 1e-9)

    # The tightest spread seen is the best guess at the price grid without asking the exchange.
    spreads = ask - bid
    tick = float(np.round(spreads[spreads > 0].min(), 10)) if (spreads > 0).any() else 0.01

    # Volatility of the mid sampled once a second. Floor it at one tick per window so a quiet
    # warm-up doesn't produce sigma = 0 (and an infinite automatic gamma).
    grid = np.arange(t[0], t[-1], 1.0)
    sampled = mid[np.searchsorted(t, grid, side="right") - 1]
    sigma = float(np.std(np.diff(sampled))) if len(sampled) > 2 else 0.0
    sigma = max(sigma, tick / math.sqrt(seconds))

    # Trade intensity at each distance from the mid just before the trade, fitted on a log scale.
    dist = np.abs(price[1:] - mid[:-1])
    deltas = np.unique(np.quantile(dist, np.linspace(0.0, 0.9, 10)))
    rates = np.array([(dist >= d).sum() / seconds for d in deltas])
    k = 0.0
    if len(deltas) >= 2 and (rates > 0).all():
        slope, intercept = np.polyfit(deltas, np.log(rates), 1)
        k, A = -slope, math.exp(intercept)
    if k <= 0:
        # Too few distinct distances to fit (common on very tight books): use the exponential MLE.
        k = 1.0 / max(float(dist.mean()), tick / 2)
        A = len(dist) / seconds
    return Calibration(sigma, float(A), float(k), tick, len(ticks), seconds)


def auto_gamma(sigma: float, k: float, T: float) -> float:
    """gamma such that gamma * sigma^2 * T = 2 / k: inventory risk and order flow weigh equally."""
    return 2.0 / (k * sigma**2 * T)


# ---------------------------------------------------------------------------------------------
# Paper trading
# ---------------------------------------------------------------------------------------------

@dataclass
class PaperMarketMaker:
    """One strategy quoting against the live feed. Inventory is in lots (fractional after a
    partial fill), cash in the quote currency."""
    strategy: str
    p: Params
    lot: float
    tick: float
    t0: float
    cash: float = 0.0
    inv: float = 0.0
    bid: float | None = None
    ask: float | None = None
    peak: float = 0.0
    max_dd: float = 0.0
    fills: list[tuple[float, int, float, float]] = field(default_factory=list)  # (time, +1 buy / -1 sell, price, lots)
    _bid_left: float = 0.0  # lots still open on each resting quote
    _ask_left: float = 0.0
    _var_hat: float = 0.0
    _last: tuple[float, float] | None = None  # (time, mid) at the previous requote

    def __post_init__(self):
        self._var_hat = self.p.sigma**2
        self._fixed_h = (self.p.fixed_half_spread if self.p.fixed_half_spread is not None
                         else self.p.matched_fixed_half_spread())

    def on_trade(self, t: float, price: float, size: float) -> None:
        """A real trade of `size` printed at `price`: fill our resting quotes if it went through them."""
        if self.bid is not None and price < self.bid:
            lots = min(size / self.lot, self._bid_left)
            self.cash -= self.bid * lots * self.lot
            self.inv += lots
            self._bid_left -= lots
            self.fills.append((t, +1, self.bid, lots))
            if self._bid_left <= 0:
                self.bid = None
        if self.ask is not None and price > self.ask:
            lots = min(size / self.lot, self._ask_left)
            self.cash += self.ask * lots * self.lot
            self.inv -= lots
            self._ask_left -= lots
            self.fills.append((t, -1, self.ask, lots))
            if self._ask_left <= 0:
                self.ask = None

    def requote(self, t: float, best_bid: float, best_ask: float) -> None:
        mid = 0.5 * (best_bid + best_ask)
        if self._last is not None and t > self._last[0]:
            dS, dt = mid - self._last[1], t - self._last[0]
            self._var_hat = self.p.ewma_lambda * self._var_hat + (1 - self.p.ewma_lambda) * dS**2 / dt
        self._last = (t, mid)

        tau = max(self.p.T - (t - self.t0), 0.0)
        sig2 = self.p.sigma**2 if self.strategy != "adaptive" else self._var_hat
        bid, ask = quote(self.strategy, self.p, mid, self.inv, sig2, tau, self._fixed_h)
        # Post-only, on the price grid: round away from the mid and never cross the real book.
        # round() strips float noise like 100.10000000000001 so fill comparisons stay exact.
        bid = round(min(math.floor(bid / self.tick + 1e-9) * self.tick, best_ask - self.tick), 10)
        ask = round(max(math.ceil(ask / self.tick - 1e-9) * self.tick, best_bid + self.tick), 10)
        # One lot per side per refresh, or less if that would pass the position limit.
        self._bid_left = min(1.0, self.p.max_inventory - self.inv)
        self._ask_left = min(1.0, self.p.max_inventory + self.inv)
        self.bid = bid if self._bid_left > 0 else None
        self.ask = ask if self._ask_left > 0 else None

        pnl = self.mark(mid)
        self.peak = max(self.peak, pnl)
        self.max_dd = max(self.max_dd, self.peak - pnl)

    def mark(self, mid: float) -> float:
        return self.cash + self.inv * self.lot * mid

    def summary(self, mid: float, mids: tuple[np.ndarray, np.ndarray], horizon: float) -> dict[str, float]:
        """`mids` is the (times, mids) history, used for the markout `horizon` seconds after each fill."""
        times, values = mids
        markout = float("nan")
        if self.fills:
            ft, side, px, lots = (np.array(c) for c in zip(*self.fills))
            later = values[np.minimum(np.searchsorted(times, ft + horizon), len(values) - 1)]
            markout = float(np.average(side * (later - px), weights=lots) * self.lot)
        return {
            "strategy": self.strategy,
            "pnl": self.mark(mid),
            "max_drawdown": self.max_dd,
            "n_trades": len(self.fills),
            "lots_filled": sum(f[3] for f in self.fills),
            "final_inventory": self.inv,
            f"markout_{horizon:g}s": markout,
        }


async def run(feed: AsyncIterator[Tick], strategies: Iterable[str], session: float, warmup: float,
              refresh: float, lot: float, gamma: float | None, max_inventory: int,
              record: str | None = None, verbose: bool = True, markout_horizon: float = 5.0) -> list[dict]:
    """Calibrate on the first `warmup` seconds of `feed`, then paper-trade for `session` seconds."""
    log = print if verbose else (lambda *a, **k: None)
    rec_file = open(record, "w", newline="") if record else None
    rec = csv.writer(rec_file) if rec_file else None
    if rec:
        rec.writerow(TICK_FIELDS)

    warm: list[Tick] = []
    makers: list[PaperMarketMaker] = []
    times: list[float] = []
    mids: list[float] = []
    last = None
    try:
        async for tick in feed:
            if rec:
                rec.writerow([tick.t, tick.bid, tick.ask, tick.price, tick.size])
            last = tick
            if not makers:
                warm.append(tick)
                if tick.t - warm[0].t < warmup or len(warm) < 20:
                    continue
                cal = calibrate(warm)
                g = gamma if gamma is not None else auto_gamma(cal.sigma, cal.k, session)
                p = Params(s0=tick.mid, T=session, n_steps=max(int(session / refresh), 1), sigma=cal.sigma,
                           sigma_high=cal.sigma, A=cal.A, k=cal.k, gamma=g, max_inventory=max_inventory)
                log(f"Calibrated on {cal.n_trades} trades over {cal.seconds:.0f}s: sigma={cal.sigma:.4g}/sqrt(s), "
                    f"A={cal.A:.3g}/s, k={cal.k:.4g}, tick={cal.tick:g}, gamma={g:.4g}"
                    f"{' (auto)' if gamma is None else ''}")
                log(f"Starting half-spread: {p.as_half_spread(cal.sigma, session):.4g} "
                    f"(fixed strategy: {p.matched_fixed_half_spread():.4g}). Mid {tick.mid:.6g}.\n")
                makers = [PaperMarketMaker(s, p, lot, cal.tick, t0=tick.t) for s in strategies]
                t_start = t_next = t_log = tick.t
                continue

            for m in makers:
                m.on_trade(tick.t, tick.price, tick.size)  # trades hit the quotes that were resting before them
            times.append(tick.t)
            mids.append(tick.mid)
            if tick.t >= t_next:
                for m in makers:
                    m.requote(tick.t, tick.bid, tick.ask)
                t_next = tick.t + refresh
            if verbose and tick.t - t_log >= 10:
                t_log = tick.t
                log(f"[{tick.t - t_start:5.0f}s] mid {tick.mid:.6g}  " + "  ".join(
                    f"{m.strategy}: inv {m.inv:+.3g} pnl {m.mark(tick.mid):+.4g}" for m in makers))
            if tick.t - t_start >= session:
                break
    finally:
        if rec_file:
            rec_file.close()

    if not makers:
        raise RuntimeError(f"feed ended during warm-up ({len(warm)} ticks); need {warmup:g}s and at least 20 trades")
    hist = (np.array(times), np.array(mids))
    return [m.summary(last.mid, hist, markout_horizon) for m in makers]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--product", default="BTC-USD", help="Coinbase product id (default BTC-USD)")
    ap.add_argument("--replay", metavar="CSV", help="replay ticks saved with --record instead of going live")
    ap.add_argument("--record", metavar="CSV", help="save every tick received, for --replay later")
    ap.add_argument("--minutes", type=float, default=5.0, help="trading session length (default 5)")
    ap.add_argument("--warmup", type=float, default=60.0, help="seconds of data to calibrate on first (default 60)")
    ap.add_argument("--refresh", type=float, default=1.0, help="seconds between quote updates (default 1)")
    ap.add_argument("--lot", type=float, default=1.0, help="largest fill per side per refresh, in the base asset (default 1)")
    ap.add_argument("--gamma", type=float, help="risk aversion per lot (default: chosen from the calibration)")
    ap.add_argument("--max-inventory", type=int, default=50, help="position limit in lots (default 50)")
    ap.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES))
    args = ap.parse_args()

    feed = replay_ticks(args.replay) if args.replay else coinbase_ticks(args.product)
    print(f"{'Replaying ' + args.replay if args.replay else 'Live: Coinbase ' + args.product}. "
          f"Warm-up {args.warmup:g}s, then {args.minutes:g} min of paper trading.")
    try:
        rows = asyncio.run(run(feed, args.strategies, args.minutes * 60, args.warmup, args.refresh, args.lot,
                               args.gamma, args.max_inventory, record=args.record))
    except KeyboardInterrupt:
        sys.exit("\nstopped")

    import pandas as pd
    print("\n" + pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
