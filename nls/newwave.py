"""NewWave (Tromans, Anaturk & Hagemeijer 1991): the expected shape of a
large linear wave in a random sea, proportional to the autocorrelation of the
spectrum:

    eta(x, t) = alpha / sigma^2 * sum_n S(w_n) dw cos(k_n (x - x0) - w_n (t - t0))

where sigma^2 = sum S dw and alpha is the crest elevation at the focus.
"""

import numpy as np

from .kinematics import G, SpectralWaveField, wave_height, wavenumber


def jonswap(omega, Tp, gamma=3.3):
    """JONSWAP spectral shape (arbitrary normalisation) at angular frequency omega."""
    omega = np.asarray(omega, dtype=float)
    wp = 2 * np.pi / Tp
    sigma = np.where(omega <= wp, 0.07, 0.09)
    peak = gamma ** np.exp(-((omega - wp) ** 2) / (2 * sigma**2 * wp**2))
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        S = omega**-5.0 * np.exp(-1.25 * (wp / omega) ** 4) * peak
    return np.where(omega > 0, S, 0.0)


class NewWave:
    """Linear focused NewWave group.

    Parameters
    ----------
    H : float, optional
        Target crest-to-trough height (m) of the focused wave, measured in the
        time series at the focus point. Give either H or ``crest``.
    crest : float, optional
        Crest elevation (m) at the focus instead of H.
    Tp : float
        Spectral peak period (s).
    depth : float
        Water depth (m).
    gamma : float
        JONSWAP peak enhancement (1 gives Pierson-Moskowitz). Ignored if a
        custom ``spectrum(omega)`` callable is given.
    x0, t0 : float
        Focus position (m) and time (s).
    n_modes : int
        Number of frequency components.
    omega_range : (float, float)
        Frequency band as multiples of the peak frequency.
    """

    def __init__(self, Tp, depth=np.inf, H=None, crest=None, gamma=3.3,
                 spectrum=None, x0=0.0, t0=0.0, n_modes=512,
                 omega_range=(0.2, 5.0), g=G):
        if (H is None) == (crest is None):
            raise ValueError("give exactly one of H or crest")
        self.Tp, self.depth, self.g = Tp, depth, g
        self.x0, self.t0 = x0, t0
        wp = 2 * np.pi / Tp
        self.omega = np.linspace(omega_range[0] * wp, omega_range[1] * wp, n_modes)
        dw = self.omega[1] - self.omega[0]
        S = spectrum(self.omega) if spectrum else jonswap(self.omega, Tp, gamma)
        self.k = wavenumber(self.omega, depth, g)
        self._weights = S * dw / np.sum(S * dw)  # unit crest

        if crest is None:
            t = t0 + np.linspace(-1.5 * Tp, 1.5 * Tp, 3001)
            unit_height = wave_height(t, self._eta_unit(x0, t))
            crest = H / unit_height
        self.crest = float(crest)
        self.H = float(wave_height(*self._probe())) if H is None else float(H)

    def _eta_unit(self, x, t):
        x, t = np.broadcast_arrays(np.asarray(x, float), np.asarray(t, float))
        phase = (x - self.x0)[..., None] * self.k - (t - self.t0)[..., None] * self.omega
        return np.cos(phase) @ self._weights

    def _probe(self):
        t = self.t0 + np.linspace(-1.5 * self.Tp, 1.5 * self.Tp, 3001)
        return t, self.crest * self._eta_unit(self.x0, t)

    def field(self, t):
        """SpectralWaveField at time t (s)."""
        amp = self.crest * self._weights * np.exp(
            -1j * (self.k * self.x0 + self.omega * (t - self.t0)))
        return SpectralWaveField(self.k, amp, depth=self.depth, g=self.g)

    def eta(self, x, t=0.0):
        return self.crest * self._eta_unit(x, t)

    def kinematics(self, x, z, t=0.0, stretching="wheeler"):
        return self.field(t).kinematics(x, z, stretching)
