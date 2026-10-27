"""Exact solutions and initial conditions for i psi_t + psi_xx/2 + |psi|^2 psi = 0.

All breathers sit on a unit background, psi -> exp(i t), using the
parametrisation of Akhmediev, Ankiewicz & Taki (2009):

    psi = [1 + (2(1-2a) cosh(b t) + i b sinh(b t)) /
               (sqrt(2a) cos(Omega x) - cosh(b t))] exp(i t)

    Omega = 2 sqrt(1 - 2a),   b = sqrt(8a(1 - 2a))

* 0 < a < 1/2 : Akhmediev breather (periodic in x, localised in t)
* a = 1/2     : Peregrine soliton (localised in both)
* a > 1/2     : Kuznetsov-Ma breather (localised in x, periodic in t)

Lengths scale with the background: if psi(x, t) is a solution, so is
A psi(A x, A^2 t).
"""

import numpy as np


def background(x, t=0.0, A=1.0):
    """Uniform Stokes-wave background A exp(i A^2 t)."""
    return A * np.exp(1j * A**2 * t) * np.ones_like(np.asarray(x, dtype=float))


def soliton(x, t=0.0, A=1.0, v=0.0, x0=0.0):
    """Bright envelope soliton with peak amplitude A moving at speed v."""
    x = np.asarray(x, dtype=float)
    xi = x - x0 - v * t
    return A / np.cosh(A * xi) * np.exp(1j * (v * xi + 0.5 * (A**2 + v**2) * t))


def peregrine(x, t=0.0):
    """Peregrine soliton; peak |psi| = 3 at x = t = 0."""
    x = np.asarray(x, dtype=float)
    return (1 - 4 * (1 + 2j * t) / (1 + 4 * x**2 + 4 * t**2)) * np.exp(1j * t)


def breather(x, t=0.0, a=0.25):
    """General Akhmediev/Kuznetsov-Ma breather with parameter a > 0 (a != 1/2)."""
    if a <= 0:
        raise ValueError("a must be positive")
    if np.isclose(a, 0.5):
        return peregrine(x, t)
    x = np.asarray(x, dtype=float)
    # Complex square roots handle both regimes: for a > 1/2, Omega and b are
    # imaginary, so cos(Omega x) -> cosh and cosh(b t) -> cos.
    Om = 2 * np.sqrt(complex(1 - 2 * a))
    b = np.sqrt(complex(8 * a * (1 - 2 * a)))
    num = 2 * (1 - 2 * a) * np.cosh(b * t) + 1j * b * np.sinh(b * t)
    den = np.sqrt(2 * a) * np.cos(Om * x) - np.cosh(b * t)
    return (1 + num / den) * np.exp(1j * t)


def akhmediev(x, t=0.0, a=0.25):
    """Akhmediev breather (0 < a < 1/2), spatial period 2 pi / Omega."""
    if not 0 < a < 0.5:
        raise ValueError("Akhmediev breather requires 0 < a < 1/2")
    return breather(x, t, a)


def kuznetsov_ma(x, t=0.0, a=1.0):
    """Kuznetsov-Ma breather (a > 1/2), temporal period 2 pi / |b|."""
    if a <= 0.5:
        raise ValueError("Kuznetsov-Ma breather requires a > 1/2")
    return breather(x, t, a)


def breather_params(a):
    """Return (Omega, b, peak amplitude) for breather parameter a.

    For a > 1/2 Omega and b are returned as their (real) moduli, i.e. the
    spatial decay rate and the temporal frequency of the Kuznetsov-Ma breather.
    """
    Om = 2 * np.sqrt(abs(1 - 2 * a))
    b = np.sqrt(abs(8 * a * (1 - 2 * a)))
    peak = 1 + 2 * np.sqrt(2 * a)
    return Om, b, peak


def akhmediev_a_for_period(L_period):
    """Breather parameter a whose spatial period is L_period (must exceed pi)."""
    Om = 2 * np.pi / L_period
    if Om >= 2:
        raise ValueError("period must exceed pi (modulation must be unstable)")
    return 0.5 * (1 - Om**2 / 4)


def modulated_background(x, eps=1e-3, K=np.sqrt(2), A=1.0):
    """Background with a small harmonic seed, A (1 + eps cos(K x)).

    Seeds modulational (Benjamin-Feir) instability; growth is fastest for
    K = sqrt(2) A and the instability band is 0 < K < 2 A.
    """
    x = np.asarray(x, dtype=float)
    return A * (1 + eps * np.cos(K * x)).astype(complex)


def random_wave_field(x, k_width=0.5, mean_amplitude=1.0, k_center=0.0, seed=None):
    """Random-phase wave envelope with a Gaussian wavenumber spectrum.

    Gives a statistically homogeneous field whose RMS |psi| equals
    ``mean_amplitude``. Narrow spectra (small ``k_width`` relative to the
    instability band ~2*amplitude) correspond to a large Benjamin-Feir index
    and readily produce rogue waves.
    """
    x = np.asarray(x, dtype=float)
    N = x.size
    dx = x[1] - x[0]
    k = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    rng = np.random.default_rng(seed)
    spec = np.exp(-((k - k_center) ** 2) / (2 * k_width**2))
    phases = np.exp(2j * np.pi * rng.random(N))
    psi = np.fft.ifft(np.sqrt(spec) * phases)
    psi *= mean_amplitude / np.sqrt(np.mean(np.abs(psi) ** 2))
    return psi
