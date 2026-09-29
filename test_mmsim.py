"""Run with: python -m pytest -q   (or just: python test_mmsim.py)"""
import numpy as np

from mmsim import Params, make_draws, simulate, simulate_loop


def test_vectorized_matches_reference_loop():
    for strat in ("fixed", "inventory", "adaptive"):
        p = Params(sigma=1.0, sigma_high=4.0, p_switch=0.02)
        d = make_draws(p, 50, seed=1)
        assert np.allclose(simulate(strat, p, d)["pnl"], simulate_loop(strat, p, d), atol=1e-9)


def test_zero_volatility_fixed_spread_earns_spread():
    # With a frozen mid, every round trip earns exactly 2 * half_spread, and P&L can never be negative.
    p = Params(sigma=1e-12, sigma_high=1e-12, fixed_half_spread=0.5)
    res = simulate("fixed", p, make_draws(p, 200, seed=2))
    assert (res["pnl"] > -1e-6).all()


def test_position_limit_respected():
    p = Params(max_inventory=3, A=2000.0)
    res = simulate("fixed", p, make_draws(p, 200, seed=3))
    assert np.abs(res["final_inventory"]).max() <= 3


def test_inventory_quoting_reduces_risk():
    p = Params()
    d = make_draws(p, 2000, seed=4)
    assert simulate("inventory", p, d)["pnl"].std() < simulate("fixed", p, d)["pnl"].std()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
