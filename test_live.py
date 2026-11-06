"""Offline tests for live.py, on synthetic ticks (no network). Run with: python test_live.py"""
import asyncio
import math

import numpy as np

from live import PaperMarketMaker, Tick, calibrate, run
from mmsim import Params


def synthetic_ticks(n=3000, sigma=0.5, tick=0.01, seed=0):
    """One trade a second; the mid is a random walk and trades print around the touch."""
    rng = np.random.default_rng(seed)
    mid = 100.0 + np.cumsum(sigma * rng.standard_normal(n))
    out = []
    for i, m in enumerate(mid):
        bid = math.floor(m / tick) * tick
        ask = bid + tick
        side = rng.choice([-1, 1])
        price = (ask if side > 0 else bid) + side * rng.exponential(0.2)
        out.append(Tick(float(i), bid, ask, round(price, 2), 1.0))
    return out


async def as_feed(ticks):
    for t in ticks:
        yield t


def maker(strategy="fixed", **kw):
    p = Params(T=100.0, n_steps=100, sigma=0.1, k=10.0, gamma=0.1, **kw)
    return PaperMarketMaker(strategy, p, lot=1.0, tick=0.01, t0=0.0)


def test_fill_only_when_trade_goes_through_quote():
    m = maker(fixed_half_spread=0.10)
    m.requote(0.0, 99.99, 100.01)
    assert (m.bid, m.ask) == (99.90, 100.10)
    m.on_trade(0.1, 99.90, 5.0)       # at our price: behind the queue, no fill
    assert m.inv == 0
    m.on_trade(0.2, 99.89, 5.0)       # through our price: filled one lot at our bid
    assert m.inv == 1 and math.isclose(m.cash, -99.90)
    m.on_trade(0.3, 99.50, 5.0)       # that side is used up until the next requote
    assert m.inv == 1


def test_fill_is_capped_by_the_real_trade_size():
    m = maker(fixed_half_spread=0.10)
    m.requote(0.0, 99.99, 100.01)
    m.on_trade(0.1, 99.80, 0.25)      # a small trade only fills part of the lot
    assert math.isclose(m.inv, 0.25) and math.isclose(m.cash, -0.25 * 99.90) and m.bid == 99.90
    m.on_trade(0.2, 99.80, 5.0)       # a big one fills what is left, and no more
    assert math.isclose(m.inv, 1.0) and m.bid is None


def test_markout_is_weighted_by_size():
    m = maker(fixed_half_spread=0.10)
    m.requote(0.0, 99.99, 100.01)
    m.on_trade(1.0, 99.80, 0.25)
    m.requote(2.0, 99.99, 100.01)
    m.on_trade(3.0, 99.80, 0.75)      # bought 0.25 and 0.75, both at 99.90
    hist = (np.array([1.0, 3.0, 7.0, 9.0]), np.array([100.0, 100.0, 100.5, 99.5]))
    r = m.summary(100.0, hist, horizon=5.0)
    assert math.isclose(r["markout_5s"], 0.25 * (100.5 - 99.90) + 0.75 * (99.5 - 99.90))
    assert math.isclose(r["lots_filled"], 1.0)


def test_quotes_are_post_only_on_the_price_grid():
    m = maker(fixed_half_spread=0.001)   # model wants a spread narrower than one tick
    m.requote(0.0, 99.99, 100.01)
    assert m.bid < 100.01 and m.ask > 99.99
    assert math.isclose(m.bid * 100, round(m.bid * 100)) and math.isclose(m.ask * 100, round(m.ask * 100))


def test_position_limit_stops_quoting_that_side():
    m = maker(fixed_half_spread=0.10, max_inventory=2)
    for i in range(5):
        m.requote(float(i), 99.99, 100.01)
        m.on_trade(i + 0.5, 90.0, 0.75)   # fractional fills must not overshoot the limit
    assert math.isclose(m.inv, 2.0) and m.bid is None and m.ask is not None


def test_calibration_recovers_volatility():
    cal = calibrate(synthetic_ticks(sigma=0.5))
    assert abs(cal.sigma / 0.5 - 1) < 0.1
    assert cal.k > 0 and cal.A > 0 and math.isclose(cal.tick, 0.01)


def test_run_end_to_end_is_deterministic():
    ticks = synthetic_ticks()
    kw = dict(strategies=("fixed", "inventory", "adaptive"), session=1500.0, warmup=300.0,
              refresh=1.0, lot=1.0, gamma=None, max_inventory=50, verbose=False)
    a = asyncio.run(run(as_feed(ticks), **kw))
    b = asyncio.run(run(as_feed(ticks), **kw))
    assert a == b and [r["strategy"] for r in a] == list(kw["strategies"])
    assert all(r["n_trades"] > 0 and abs(r["final_inventory"]) <= 50 for r in a)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
