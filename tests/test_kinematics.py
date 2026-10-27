import numpy as np
import pytest

import nls
from nls.kinematics import G


def test_dispersion_roundtrip():
    om = np.linspace(0.1, 3, 20)
    for d in (5.0, 50.0, np.inf):
        assert np.allclose(nls.frequency(nls.wavenumber(om, d), d), om, rtol=1e-12)


def test_single_mode_matches_airy():
    H, T, d = 2.0, 8.0, 30.0
    wave = nls.LinearWave(H, T, d)
    k, om, a = wave.k, wave.omega, H / 2
    x = np.array([0.0, 10.0, 25.0])[:, None]
    z = np.array([-25.0, -10.0, -1.0])[None, :]
    t = 1.3
    th = k * x - om * t
    kin = wave.kinematics(x, z, t, stretching="none")
    ch = np.cosh(k * (z + d)) / np.sinh(k * d)
    sh = np.sinh(k * (z + d)) / np.sinh(k * d)
    assert np.allclose(kin["u"], a * om * ch * np.cos(th))
    assert np.allclose(kin["w"], a * om * sh * np.sin(th))
    assert np.allclose(kin["ut"], a * om**2 * ch * np.sin(th))
    assert np.allclose(kin["wt"], -a * om**2 * sh * np.cos(th))


@pytest.mark.parametrize("stretching", ["wheeler", "linear", "none"])
def test_points_above_surface_are_nan(stretching):
    wave = nls.LinearWave(2.0, 8.0, 30.0)
    kin = wave.kinematics(0.0, np.array([0.5, 0.99, 1.01, 2.0]), 0.0, stretching)
    assert np.all(np.isfinite(kin["u"][:2])) and np.all(np.isnan(kin["u"][2:]))


def test_wheeler_surface_velocity_equals_mean_level_velocity():
    wave = nls.LinearWave(2.0, 8.0, 30.0)
    crest = wave.kinematics(0.0, 1.0, 0.0, "wheeler")["u"]
    mwl = wave.kinematics(0.0, 0.0, 0.0, "none")["u"]
    assert np.isclose(crest, mwl)


def test_stream_function_small_amplitude_is_airy():
    H, T, d = 0.01, 9.0, 40.0
    sf = nls.StreamFunctionWave(H, T, d)
    lw = nls.LinearWave(H, T, d)
    assert np.isclose(sf.wavelength, 2 * np.pi / lw.k, rtol=1e-5)
    z = np.array([-30.0, -10.0, -1.0])
    assert np.allclose(sf.kinematics(0.0, z)["u"], lw.kinematics(0.0, z, stretching="none")["u"],
                       rtol=1e-3)


def test_stream_function_steep_wave():
    H, T, d = 12.0, 10.0, 100.0
    sf = nls.StreamFunctionWave(H, T, d)
    kin_res, dyn_res = sf.surface_residuals()
    assert kin_res < 1e-8 and dyn_res < 1e-8
    assert np.isclose(sf.crest - sf.trough, H)
    x = np.linspace(0, sf.wavelength, 2000, endpoint=False)
    assert abs(sf.eta(x).mean()) < 1e-9
    # Third-order Stokes, deep water: c = c0 (1 + (ka)^2 / 2 ... ) with crest > H/2
    assert sf.crest > H / 2 and sf.c > np.sqrt(G / nls.wavenumber(2 * np.pi / T, d))
    # No-current Eulerian criterion: mean velocity at a fixed point below troughs ~ 0
    t = np.linspace(0, T, 400, endpoint=False)
    assert abs(np.mean(sf.kinematics(0.0, -20.0, t)["u"])) < 1e-6


def test_stream_function_accelerations_consistent():
    sf = nls.StreamFunctionWave(10.0, 9.0, 60.0)
    x, z, t, h = 13.0, -3.0, 0.4, 1e-5
    k0 = sf.kinematics(x, z, t)
    kp = sf.kinematics(x, z, t + h)
    km = sf.kinematics(x, z, t - h)
    assert np.isclose(k0["ut"], (kp["u"] - km["u"]) / (2 * h), rtol=1e-6)
    assert np.isclose(k0["wt"], (kp["w"] - km["w"]) / (2 * h), rtol=1e-6)
    # Continuity
    assert np.isclose(k0["ux"] + k0["wz"], 0.0, atol=1e-12)


def test_newwave_focus():
    nw = nls.NewWave(Tp=12.0, depth=80.0, H=14.0)
    assert np.isclose(nw.eta(0.0, 0.0), nw.crest)
    t = np.linspace(-18, 18, 3001)
    assert np.isclose(nls.wave_height(t, nw.eta(0.0, t)), 14.0, rtol=1e-6)
    f = nw.field(0.0)
    assert np.isclose(f.eta(0.0), nw.crest)


def test_peregrine_for_height():
    dw = nls.peregrine_for_height(10.0, period=11.0)
    t = np.linspace(-20, 20, 4001)
    assert np.isclose(nls.wave_height(t, nls.peregrine_focus_series(dw, t)), 10.0, rtol=1e-6)


def _peregrine_run(dw, n_out=81, half_window=0.02):
    solver = nls.NLSSolver(L=200, N=4096)
    T_start = -2.0
    pre = solver.run(lambda X: nls.peregrine(X, T_start), T_start, -half_window, dt=0.002, n_out=2)
    return solver.run(pre.psi[-1], -half_window, half_window, dt=0.0005, n_out=n_out)


def test_nls_wave_surface_matches_exact():
    dw = nls.peregrine_for_height(12.0, period=10.0)
    res = _peregrine_run(dw)
    wave = nls.NLSWave(res, dw, depth=np.inf, phase=np.pi)
    t = wave.physical_times()[40]
    x = np.linspace(-200, 200, 41)
    from nls.solutions import peregrine
    psi = peregrine(dw.to_normalised_x(x, t), dw.to_normalised_t(t))
    exact = np.real(dw.envelope(psi) * np.exp(1j * (dw.k0 * x - dw.w0 * t + np.pi)))
    assert np.max(np.abs(wave.eta(x, t) - exact)) < 1e-3 * dw.a0


def test_nls_wave_uniform_background_is_stokes_wave():
    dw = nls.DeepWater(a0=1.0, period=10.0)
    solver = nls.NLSSolver(L=20, N=64)
    res = solver.run(lambda X: nls.background(X), 0.0, 0.0, dt=0.01, n_out=1)
    f = nls.NLSWave(res, dw).field(0.0)
    assert f.k.size == 1 and np.isclose(f.k[0], dw.k0)
    assert np.isclose(abs(f.amp[0]), dw.a0)
    # Nonlinear frequency correction w = w0 (1 + eps^2 / 2)
    assert np.isclose(f.amp_dot[0] / f.amp[0], -1j * dw.w0 * (1 + 0.5 * dw.eps**2))


def test_nls_wave_local_acceleration_matches_time_derivative():
    dw = nls.peregrine_for_height(12.0, period=10.0)
    res = _peregrine_run(dw, n_out=401)
    wave = nls.NLSWave(res, dw, depth=np.inf, phase=np.pi)
    times = wave.physical_times()
    i = 200
    h = times[i + 1] - times[i]
    x, z = 20.0, -5.0
    u = [wave.kinematics(x, z, times[j], "none")["u"] for j in (i - 1, i + 1)]
    ut = wave.kinematics(x, z, times[i], "none")["ut"]
    scale = dw.a0 * dw.w0**2
    assert abs((u[1] - u[0]) / (2 * h) - ut) < 2e-3 * scale


def test_linear_extrapolation_above_mwl():
    wave = nls.LinearWave(2.0, 8.0, 30.0)
    t = 0.3
    zc = 0.6
    above = wave.kinematics(0.0, zc, t, "linear")
    at0 = wave.kinematics(0.0, 0.0, t, "none")
    k, om = wave.k, wave.omega
    assert np.isclose(above["u"], at0["u"] + zc * at0["uz"])
    assert np.isclose(above["w"], at0["w"] + zc * at0["wz"])
    # d(ut)/dz = k om^2 a Sh sin(th); at z = 0, Sh = 1
    th = -om * t
    assert np.isclose(above["ut"], at0["ut"] + zc * k * om**2 * 1.0 * np.sin(th))


def test_breaking_height_deep_water_limit():
    from nls.stream_function import breaking_height
    T = 10.0
    exact = 0.141063 * 1.0923**2 * G * T**2 / (2 * np.pi)  # Williams highest wave
    assert abs(breaking_height(T, 1000.0) / exact - 1) < 0.01


def test_stream_function_near_breaking():
    from nls.stream_function import breaking_height
    T, d = 10.0, 100.0
    Hb = breaking_height(T, d)
    sf = nls.StreamFunctionWave(0.95 * Hb, T, d)
    assert sf.residual < 1e-4
    assert np.isclose(sf.crest - sf.trough, 0.95 * Hb)
    assert 0.6 < sf.crest / sf.H < 0.7          # strongly crest-dominated
    u_crest = sf.kinematics(0.0, sf.crest - 1e-9)["u"]
    assert 0.6 < u_crest / sf.c < 1.0


def test_stream_function_analytic_jacobian():
    sf = nls.StreamFunctionWave(12.0, 10.0, 100.0, N=16)
    v = sf._v * (1 + 1e-3 * np.random.default_rng(0).standard_normal(sf._v.size))
    r, J = sf._residual(v, 0.12, jacobian=True)
    F = np.empty_like(J)
    for i in range(v.size):
        h = 1e-6 * max(1.0, abs(v[i]))
        vp, vm = v.copy(), v.copy()
        vp[i] += h
        vm[i] -= h
        F[:, i] = (sf._residual(vp, 0.12) - sf._residual(vm, 0.12)) / (2 * h)
    assert np.abs(J - F).max() / np.abs(F).max() < 1e-8
