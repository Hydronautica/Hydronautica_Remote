import numpy as np
import pytest

import nls
from nls.dysthe import DystheSolver, dysthe_for_height, mass, peregrine_domain, peregrine_initial
from nls.kinematics import G, SpectralWaveField
from nls.second_order import SecondOrderWaveField, pair_kernels
from nls.solver import Result


# --- second-order theory -------------------------------------------------------

def test_stokes_second_order_finite_depth():
    T, d, a = 8.0, 15.0, 0.5
    om = 2 * np.pi / T
    k = float(nls.wavenumber(om, d))
    f = SecondOrderWaveField(SpectralWaveField([k], [a], depth=d))
    x = np.linspace(0, 2 * np.pi / k, 9)
    exact = (k * a**2 / 4) * np.cosh(k * d) * (2 + np.cosh(2 * k * d)) \
        / np.sinh(k * d) ** 3 * np.cos(2 * k * x)
    assert np.allclose(f.eta_second(x), exact, atol=1e-14)
    z = np.array([-12.0, -5.0, 0.0])
    u2 = f.kinematics(0.0, z, "none")["u"] - f.first.kinematics(0.0, z, "none")["u"]
    u2_exact = 0.75 * om * k * a**2 * np.cosh(2 * k * (z + d)) / np.sinh(k * d) ** 4
    assert np.allclose(u2, u2_exact, rtol=1e-10)


def test_deep_water_kernels_match_tayfun():
    k = np.array([0.02, 0.035, 0.05])
    kn = pair_kernels(k, np.inf)
    Kp, Km = k[:, None] + k[None, :], k[:, None] - k[None, :]
    assert np.allclose(kn["Ep"], Kp / 4)
    off = ~np.eye(3, dtype=bool)
    assert np.allclose(kn["Em"][off], -np.abs(Km[off]) / 4)
    assert np.allclose(kn["Cp"], 0)


def _bichromatic(scale, t, d=20.0):
    k = nls.wavenumber(2 * np.pi / np.array([9.0, 7.0]), d)
    om = nls.frequency(k, d)
    amp = scale * np.array([1.0, 0.6]) * np.exp(-1j * om * t)
    return SecondOrderWaveField(SpectralWaveField(k, amp, depth=d))


def _bc_residuals(scale, order):
    x = np.linspace(0, 150, 13)
    t, h = 1.7, 1e-4
    f = _bichromatic(scale, t)
    field = f if order == 2 else f.first
    eta = field.eta(x)
    eta_t = (_bichromatic(scale, t + h).eta(x) - _bichromatic(scale, t - h).eta(x)) / (2 * h) \
        if order == 2 else (_bichromatic(scale, t + h).first.eta(x)
                            - _bichromatic(scale, t - h).first.eta(x)) / (2 * h)
    eta_x = (field.eta(x + h) - field.eta(x - h)) / (2 * h)
    kin = field.kinematics(x, eta - 1e-12, "none")
    kinematic = eta_t + kin["u"] * eta_x - kin["w"]
    fp = _bichromatic(scale, t + h)
    fm = _bichromatic(scale, t - h)
    fp, fm = (fp, fm) if order == 2 else (fp.first, fm.first)
    phi_t = (fp.potential(x, eta) - fm.potential(x, eta)) / (2 * h)
    dynamic = G * eta + phi_t + 0.5 * (kin["u"] ** 2 + kin["w"] ** 2)
    return np.abs(kinematic).max(), np.abs(dynamic - dynamic.mean()).max()


def test_second_order_satisfies_surface_conditions_to_third_order():
    """Halving the amplitude cuts the residual ~8x (O(a^3)) at second order,
    but only ~4x (O(a^2)) for the linear field."""
    for order, expected in ((1, 4.0), (2, 8.0)):
        r1 = _bc_residuals(0.4, order)
        r2 = _bc_residuals(0.2, order)
        for a, b in zip(r1, r2):
            assert abs(a / b / expected - 1) < 0.25, (order, a / b)


def test_newwave_second_order():
    nw1 = nls.NewWave(Tp=10.0, depth=100.0, H=12.0)
    nw2 = nls.NewWave(Tp=10.0, depth=100.0, H=12.0, order=2, n_modes=256)
    t = np.linspace(-15, 15, 601)
    eta2 = np.array([nw2.eta(0.0, ti) for ti in t]).ravel()
    assert np.isclose(nls.wave_height(t, eta2), 12.0, rtol=1e-6)
    assert eta2.max() > nw1.eta(0.0, 0.0) + 0.3      # bound harmonics lift the crest
    kin = nw2.kinematics(0.0, np.array([-20.0, 0.0, eta2.max() - 0.01]), 0.0)
    assert np.all(np.isfinite(kin["u"]))


def test_envelope_second_order_matches_pair_sums():
    """In deep water the envelope formulas use the exact second-order kernels,
    so eta2 must agree with the general pair-sum theory."""
    dw = nls.DeepWater(a0=1.0, period=10.0)
    X = -20 + 40 / 64 * np.arange(64)
    psi = 1 + 0.3 * np.cos(2 * np.pi / 40 * X) + 0.1j * np.sin(4 * np.pi / 40 * X)
    res = Result(X, np.array([0.0]), psi[None, :])
    wave = nls.NLSWave(res, dw, order=2, phase=0.4)
    f = wave.field(0.0)
    pair = SecondOrderWaveField(f.first, rel_tol=0.0)
    x = np.linspace(-500, 500, 23)
    assert np.allclose(f.eta_second(x), pair.eta_second(x), atol=1e-10)


def test_peregrine_second_order_height():
    dw2 = nls.peregrine_for_height(12.0, period=10.0, order=2)
    t = np.linspace(-15, 15, 601)
    eta = nls.physical.peregrine_series(dw2, t, order=2)
    assert np.isclose(nls.wave_height(t, eta), 12.0, rtol=1e-6)
    assert eta.max() > 3 * dw2.a0                 # crest lifted above linear 3 a0


# --- Dysthe ---------------------------------------------------------------------

def test_dysthe_stokes_wave():
    s = DystheSolver(2 * np.pi * 20, 128)
    eps, K = 0.1, s.K[3]
    nu = -K**2 / 8 + K**3 / 16 + eps**2 / 2 + 1.25 * eps**2 * K
    res = s.run(eps * np.exp(1j * K * s.x), 0, 50, dt=0.05, n_out=2)
    assert np.abs(res.psi[-1] - eps * np.exp(1j * (K * s.x - nu * 50))).max() < 1e-12
    # Agrees with the exact Stokes frequency of a wave of wavenumber 1 + K to
    # O(eps^2 K^2); the 3/2 and 1/4 coefficients give the eps^2 K term.
    nu_exact = np.sqrt(1 + K) * (1 + 0.5 * ((1 + K) * eps) ** 2) - 1 - K / 2
    assert abs(nu - nu_exact) < 1.5 * eps**2 * K**2


def test_dysthe_reduces_to_nls():
    eps = 0.1
    L, N = peregrine_domain(eps)
    s = DystheSolver(L, N, higher_order=False)
    res = s.run(peregrine_initial(s.x, eps, -2.0), 0, 4 / eps**2, dt=1.0, n_out=2)
    assert np.abs(res.psi[-1] - peregrine_initial(s.x, eps, 0.0)).max() / eps < 1e-4


def test_dysthe_conserves_mass_and_converges():
    eps = 0.1
    L, N = peregrine_domain(eps)
    s = DystheSolver(L, N)
    B0 = peregrine_initial(s.x, eps, -2.0)
    errs = []
    # (dt is capped near 0.7 by the stability estimate, so stay below it.)
    ref = s.run(B0, 0, 400, dt=0.05, n_out=2).psi[-1]
    for dt in (0.3, 0.15):
        errs.append(np.abs(s.run(B0, 0, 400, dt=dt, n_out=2).psi[-1] - ref).max())
    assert 12 < errs[0] / errs[1] < 20          # 4th order
    res = s.run(B0, 0, 400, dt=0.2, n_out=5)
    m = mass(res.psi, s.dx)
    assert np.abs(m - m[0]).max() / m[0] < 1e-7


def test_dysthe_lowers_breather_peak():
    eps = 0.1
    L, N = peregrine_domain(eps)
    s = DystheSolver(L, N)
    res = s.run(peregrine_initial(s.x, eps, -2.0), 0, 8 / eps**2, dt=1.0, n_out=201)
    peak = np.abs(res.psi).max() / eps
    assert 2.5 < peak < 2.9          # NLS gives exactly 3


def test_mean_flow_stabilises_modulational_instability():
    eps, K = 0.1, 0.1               # inside the unstable band 0 < K < 2 sqrt(2) eps
    L = 2 * np.pi / K
    growth = {}
    for mf in (False, True):
        s = DystheSolver(L, 64, mean_flow=mf)
        B0 = eps * (1 + 1e-4 * np.cos(K * s.x))
        res = s.run(B0, 0, 300, dt=0.5, n_out=2)
        side = np.abs(np.fft.fft(res.psi[-1]))[1]
        growth[mf] = side
    assert growth[True] < growth[False]


def test_dysthe_for_height():
    wave, t_f, x_f, eps = dysthe_for_height(12.0, 10.0, depth=100.0)
    t = wave.physical_times()
    eta = np.array([wave.eta(x_f, ti) for ti in t]).ravel()
    assert abs(nls.wave_height(t, eta) - 12.0) < 0.02
    assert np.isclose(wave.eta(x_f, t_f), eta.max(), rtol=1e-3)
