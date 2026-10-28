import numpy as np
import pytest

from nls.flume import design, flap_transfer, paddle_signal, paddle_stroke, piston_transfer, surface


def test_breather_peaks_at_three_a0_at_the_focus():
    d = design(0.7, 0.1, 1.0, 20.0)
    assert d["kh"] > 3 and d["a0"] == pytest.approx(0.1 / d["k0"])
    t, eta, tf, ph = paddle_signal(d, fs=100.0)
    tt = np.linspace(tf - 2 * d["T0"], tf + 2 * d["T0"], 4001)
    assert surface(20.0, tt, d, tf, ph).max() == pytest.approx(3 * d["a0"], rel=0.002)
    assert surface(20.0, tf, d, tf, ph) == pytest.approx(3 * d["a0"], rel=1e-9)
    # far from the focus the train is nearly uniform at a0
    assert surface(-200.0, tt, d, tf, ph).max() == pytest.approx(d["a0"], rel=0.05)
    # the paddle record is ramped in and out
    assert abs(eta[0]) < 1e-6 and abs(eta[-1]) < 1e-6


def test_wavemaker_transfer_functions_deep_and_shallow_limits():
    assert flap_transfer(np.array([50.0]), 1.0)[0] == pytest.approx(2 * 49 / 50, rel=1e-6)   # deep flap: 2 (kh - 1) / kh
    assert np.isfinite(flap_transfer(np.array([1e4]), 1.0)).all()
    assert piston_transfer(np.array([50.0]), 1.0)[0] == pytest.approx(2.0, rel=1e-3)
    kh = 0.05
    assert piston_transfer(np.array([kh]), 1.0)[0] == pytest.approx(kh, rel=0.01)     # shallow piston: H/S = kh
    assert flap_transfer(np.array([kh]), 1.0)[0] == pytest.approx(kh / 2, rel=0.01)   # shallow flap: H/S = kh/2


def test_stroke_follows_the_flap_transfer_at_the_carrier():
    d = design(0.7, 0.1, 1.0, 20.0)
    t, eta, _, _ = paddle_signal(d, fs=50.0)
    s = paddle_stroke(t, eta, 1.0, "flap")
    assert np.abs(s).max() == pytest.approx(np.abs(eta).max() / flap_transfer(d["k0"], 1.0), rel=0.05)
