import numpy as np
import pytest

from nls.kinematics import G, wavenumber
from nls.variable_depth import (Transect, VariableDepthNLS, critical_kh, group_velocity,
                                irregular_over_transect, jonswap_envelope, nonlinear_coefficient)


def test_nonlinear_coefficient_limits():
    k = 0.05
    assert nonlinear_coefficient(k, 1e4) == pytest.approx(np.sqrt(G * k) * k**2 / 2, rel=1e-6)
    assert nonlinear_coefficient(k, 10 / k) == pytest.approx(np.sqrt(G * k) * k**2 / 2, rel=0.25)
    assert critical_kh() == pytest.approx(1.363, abs=2e-3)
    assert nonlinear_coefficient(1.0, 2.0) > 0 > nonlinear_coefficient(1.0, 1.0)


def test_jonswap_envelope_hs():
    tau, A, w0 = jonswap_envelope(4.0, 10.0, seed=2)
    eta = np.real(A * np.exp(-1j * w0 * tau))
    assert 4 * eta.std() == pytest.approx(4.0, rel=0.02)


@pytest.mark.parametrize("dispersion", ["exact", "nls"])
def test_constant_depth_linear_keeps_hs(dispersion):
    r = irregular_over_transect(Transect.constant(40, 3000), Hs=1.0, Tp=9, seed=1,
                                nonlinear=False, dispersion=dispersion)
    assert np.allclose(r.hs(), r.hs()[0], rtol=1e-6)


@pytest.mark.parametrize("dispersion", ["exact", "nls"])
def test_linear_shoaling_conserves_energy_flux(dispersion):
    tr = Transect([0, 1000, 5000, 6000], [80, 80, 12, 12])
    r = irregular_over_transect(tr, Hs=0.3, Tp=8, gamma=7, seed=4, nonlinear=False, dispersion=dispersion)
    w = 2 * np.pi / 8
    ks = [wavenumber(w, h) for h in (80, 12)]
    ratio = np.sqrt(group_velocity(ks[0], 80) / group_velocity(ks[1], 12))
    assert r.hs()[-1] / r.hs()[0] == pytest.approx(ratio, rel=0.02)


def test_deep_water_soliton_is_steady():
    T0 = 10.0
    w0 = 2 * np.pi / T0
    k0 = w0**2 / G
    a = 0.05 / k0                                # steepness 0.05
    Q = k0**3
    tau0 = np.sqrt(2 / (G * Q * a**2))
    n, T = 1024, 60 * tau0
    tau = (np.arange(n) - n / 2) * T / n
    A0 = a / np.cosh(tau / tau0)
    r = VariableDepthNLS(Transect.constant(5000, 20 * 2 * np.pi / k0), w0, dispersion="nls",
                         breaking=None).run(tau, A0)
    assert np.abs(r.A[-1]).max() == pytest.approx(a, rel=1e-3)
    assert np.allclose(np.abs(r.A[-1]), A0, atol=2e-3 * a)


def _peak_growth(depth):
    T0 = 8.0
    w0 = 2 * np.pi / T0
    k0 = float(wavenumber(w0, depth))
    a = 0.1 / k0
    n, T = 256, 40 * T0
    tau = np.arange(n) * T / n
    A0 = a * (1 + 0.02 * np.cos(2 * np.pi * tau / T * 2))
    r = VariableDepthNLS(Transect.constant(depth, 150 * 2 * np.pi / k0), w0, dispersion="nls",
                         breaking=None, kh_min=0).run(tau, A0)
    return np.abs(r.A).max() / a          # peak envelope over the whole run


def test_focusing_switches_off_below_critical_depth():
    assert _peak_growth(100.0) > 2.5      # kh ~ 6.3: Benjamin-Feir growth
    assert _peak_growth(8.0) < 1.1        # kh ~ 1.0: stable


def test_shallow_stations_are_flagged_linear():
    tr = Transect([0, 4000], [50, 6])
    r = irregular_over_transect(tr, Hs=1.0, Tp=10, seed=1)
    assert r.nonlinear[0] and not r.nonlinear[-1]


def test_breaking_cap_limits_height():
    tr = Transect([0, 3000], [30, 4])
    r = irregular_over_transect(tr, Hs=3.0, Tp=10, seed=1)
    assert r.crest_max()[-1] <= 0.39 * 4 + 1e-9
    assert r.broken[-1] > 0
