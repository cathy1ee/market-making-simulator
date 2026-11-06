"""Offline tests for live.py, on synthetic ticks (no network). Run with: python test_live.py"""
import asyncio
import json
import math

import numpy as np

import websockets

import live
from live import PaperMarketMaker, Tick, calibrate, coinbase_ticks, run
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
    m.on_trade(0.1, 99.90)            # at our price: behind the queue, no fill
    assert m.inv == 0
    m.on_trade(0.2, 99.89)            # through our price: filled one lot at our bid
    assert m.inv == 1 and math.isclose(m.cash, -99.90)
    m.on_trade(0.3, 99.50)            # that side is used up until the next requote
    assert m.inv == 1


def test_quotes_are_post_only_on_the_price_grid():
    m = maker(fixed_half_spread=0.001)   # model wants a spread narrower than one tick
    m.requote(0.0, 99.99, 100.01)
    assert m.bid < 100.01 and m.ask > 99.99
    assert math.isclose(m.bid * 100, round(m.bid * 100)) and math.isclose(m.ask * 100, round(m.ask * 100))


def test_position_limit_stops_quoting_that_side():
    m = maker(fixed_half_spread=0.10, max_inventory=2)
    for i in range(5):
        m.requote(float(i), 99.99, 100.01)
        m.on_trade(i + 0.5, 90.0)
    assert m.inv == 2 and m.bid is None and m.ask is not None


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


def test_feed_retries_when_the_handshake_fails():
    # e.g. an HTTP 503 while reconnecting, which is not an OSError or ConnectionClosed.
    message = json.dumps({"type": "ticker", "time": "2026-11-03T16:10:08.934779Z", "best_bid": "99.99",
                          "best_ask": "100.01", "price": "100.00", "last_size": "0.5"})

    class FakeSocket:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def send(self, _):
            pass

        async def __aiter__(self):
            yield message

    attempts = []

    def connect(*_, **__):
        attempts.append(1)
        if len(attempts) == 1:
            raise websockets.InvalidHandshake("server rejected the connection")
        return FakeSocket()

    async def no_wait(_):
        pass

    async def first_tick():
        async for tick in coinbase_ticks("ETH-USD"):
            return tick

    real_connect, real_sleep = websockets.connect, live.asyncio.sleep
    websockets.connect, live.asyncio.sleep = connect, no_wait
    try:
        tick = asyncio.run(first_tick())
    finally:
        websockets.connect, live.asyncio.sleep = real_connect, real_sleep
    assert len(attempts) == 2 and tick.price == 100.0


def test_stopping_early_still_returns_a_summary():
    async def stalls_after(ticks):
        for t in ticks:
            yield t
        await asyncio.Event().wait()

    async def main():
        task = asyncio.create_task(run(stalls_after(synthetic_ticks(600)), ("fixed",), session=1500.0, warmup=300.0,
                                       refresh=1.0, lot=1.0, gamma=None, max_inventory=50, verbose=False))
        await asyncio.sleep(0.2)
        task.cancel()
        return await task

    rows = asyncio.run(main())
    assert rows[0]["strategy"] == "fixed" and rows[0]["n_trades"] > 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
