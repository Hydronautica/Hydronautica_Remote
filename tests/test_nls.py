import numpy as np
import pytest

import nls
from nls import solutions as sol


def pde_residual(f, x, t, h=1e-4):
    """|i psi_t + psi_xx/2 + |psi|^2 psi| via centred finite differences."""
    psi = f(x, t)
    psi_t = (f(x, t + h) - f(x, t - h)) / (2 * h)
    psi_xx = (f(x + h, t) - 2 * psi + f(x - h, t)) / h**2
    return np.max(np.abs(1j * psi_t + 0.5 * psi_xx + np.abs(psi) ** 2 * psi))


@pytest.mark.parametrize("f", [
    sol.peregrine,
    lambda x, t: sol.soliton(x, t, A=1.3, v=0.4),
    lambda x, t: sol.akhmediev(x, t, a=0.25),
    lambda x, t: sol.akhmediev(x, t, a=0.45),
    lambda x, t: sol.kuznetsov_ma(x, t, a=0.8),
    lambda x, t: sol.kuznetsov_ma(x, t, a=2.0),
])
@pytest.mark.parametrize("t", [-1.3, 0.0, 0.7])
def test_exact_solutions_satisfy_pde(f, t):
    x = np.linspace(-6, 6, 241) + 0.0123
    assert pde_residual(f, x, t) < 1e-4


@pytest.mark.parametrize("a", [0.1, 0.3, 0.45, 0.7, 1.5])
def test_breather_peak_amplitude(a):
    _, _, peak = sol.breather_params(a)
    assert abs(abs(sol.breather(0.0, 0.0, a)) - peak) < 1e-12


def test_peregrine_peak_is_three():
    assert abs(abs(sol.peregrine(0.0, 0.0)) - 3) < 1e-14


def test_soliton_propagation():
    # L must keep the moving soliton's exponential tail clear of the boundary.
    s = nls.NLSSolver(L=80, N=512)
    f = lambda x, t: sol.soliton(x, t, A=1.0, v=0.5)
    res = s.run(lambda x: f(x, 0.0), 0.0, 10.0, dt=0.005, n_out=5)
    err = np.max(np.abs(res.psi[-1] - f(res.x, 10.0)))
    assert err < 1e-8


def test_akhmediev_breather_matches_exact():
    a = 0.25
    Om, _, peak = sol.breather_params(a)
    s = nls.NLSSolver(L=2 * np.pi / Om, N=128)
    f = lambda x, t: sol.akhmediev(x, t, a)
    res = s.run(lambda x: f(x, -6.0), -6.0, 0.0, dt=0.0025, n_out=7)
    err = np.max(np.abs(res.psi[-1] - f(res.x, 0.0)))
    assert err < 1e-6
    assert abs(res.max_amplitude()[-1] - peak) < 1e-6


def test_kuznetsov_ma_full_period():
    a = 1.0
    _, b, _ = sol.breather_params(a)
    period = 2 * np.pi / b
    s = nls.NLSSolver(L=40, N=512)
    f = lambda x, t: sol.kuznetsov_ma(x, t, a)
    res = s.run(lambda x: f(x, 0.0), 0.0, period, dt=0.001, n_out=3)
    assert np.max(np.abs(res.psi[-1] - f(res.x, period))) < 2e-5


def test_peregrine_focusing():
    # Peregrine decays only algebraically, so needs a wide domain; MI then
    # amplifies truncation error, so the tolerance is looser.
    s = nls.NLSSolver(L=200, N=4096)
    res = s.run(lambda x: sol.peregrine(x, -2.0), -2.0, 0.0, dt=0.002, n_out=3)
    err = np.max(np.abs(res.psi[-1] - sol.peregrine(res.x, 0.0)))
    assert err < 1e-2
    assert abs(res.max_amplitude()[-1] - 3.0) < 1e-2


@pytest.mark.parametrize("order,expected", [(2, 2), (4, 4)])
def test_convergence_order(order, expected):
    a = 0.3
    Om, _, _ = sol.breather_params(a)
    s = nls.NLSSolver(L=2 * np.pi / Om, N=128, order=order)
    f = lambda x, t: sol.akhmediev(x, t, a)
    errs = []
    for dt in (0.02, 0.01):
        res = s.run(lambda x: f(x, -1.0), -1.0, 1.0, dt=dt, n_out=2)
        errs.append(np.max(np.abs(res.psi[-1] - f(res.x, 1.0))))
    rate = np.log2(errs[0] / errs[1])
    assert abs(rate - expected) < 0.3


def test_invariants_conserved():
    s = nls.NLSSolver(L=2 * np.pi / np.sqrt(2) * 4, N=256)
    res = s.run(lambda x: sol.modulated_background(x, eps=0.05), 0, 20, dt=0.005, n_out=40)
    drift = nls.invariant_drift(res)
    assert drift["mass"] < 1e-10
    assert drift["hamiltonian"] < 1e-5


def test_find_rogue_events_on_akhmediev():
    a = 0.4
    Om, _, peak = sol.breather_params(a)
    s = nls.NLSSolver(L=2 * 2 * np.pi / Om, N=256)
    f = lambda x, t: sol.akhmediev(x, t, a)
    res = s.run(lambda x: f(x, -5.0), -5.0, 5.0, dt=0.005, n_out=201)
    events = nls.find_rogue_events(res, threshold=2.0, reference=1.0)
    assert len(events) == 2  # two spatial periods in the box
    for e in events:
        assert abs(e["t"]) < 0.06
        assert abs(e["amplitude"] - peak) < 1e-2


def test_deep_water_scaling_roundtrip():
    dw = nls.DeepWater(a0=2.0, period=10.0)
    assert np.isclose(dw.w0, 2 * np.pi / 10.0)
    x = np.array([0.0, 100.0, 500.0])
    assert np.allclose(dw.to_physical_x(dw.to_normalised_x(x, 30.0), 30.0), x)
    assert np.isclose(dw.to_physical_t(dw.to_normalised_t(42.0)), 42.0)


def test_deep_water_envelope_solves_physical_nls():
    """A = a0 conj(psi(X, T)) must satisfy the dimensional envelope equation."""
    dw = nls.DeepWater(a0=1.5, wavelength=150.0)
    k0, w0, cg = dw.k0, dw.w0, dw.cg

    def A(x, t):
        return dw.envelope(sol.peregrine(dw.to_normalised_x(x, t), dw.to_normalised_t(t) - 0.4))

    x = np.linspace(-300, 300, 101)
    t, hx, ht = 5.0, 1e-2, 1e-2
    a = A(x, t)
    a_t = (A(x, t + ht) - A(x, t - ht)) / (2 * ht)
    a_x = (A(x + hx, t) - A(x - hx, t)) / (2 * hx)
    a_xx = (A(x + hx, t) - 2 * a + A(x - hx, t)) / hx**2
    res = 1j * (a_t + cg * a_x) - w0 / (8 * k0**2) * a_xx - 0.5 * w0 * k0**2 * np.abs(a) ** 2 * a
    scale = 0.5 * w0 * k0**2 * dw.a0**3
    assert np.max(np.abs(res)) / scale < 1e-4


def test_deep_water_rejects_steep_waves():
    with pytest.raises(ValueError):
        nls.DeepWater(a0=10.0, wavelength=100.0)
