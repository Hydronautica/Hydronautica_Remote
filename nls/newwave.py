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
    """Focused NewWave group, linear or with second-order bound waves.

    Parameters
    ----------
    H : float, optional
        Target crest-to-trough height (m) of the focused wave, measured in the
        time series at the focus point. Give either H or ``crest``.
    crest : float, optional
        Linear (first-order) crest amplitude alpha (m) instead of H. For
        ``order=2`` the actual crest is higher by the bound harmonics.
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
    order : {1, 2}
        1: linear NewWave. 2: add Sharma-Dean second-order bound waves
        (:class:`nls.second_order.SecondOrderWaveField`); H then refers to the
        second-order surface.
    rel_tol : float
        Pair-sum truncation for order 2 (see SecondOrderWaveField).
    """

    def __init__(self, Tp, depth=np.inf, H=None, crest=None, gamma=3.3,
                 spectrum=None, x0=0.0, t0=0.0, n_modes=512,
                 omega_range=(0.2, 5.0), order=1, rel_tol=1e-3, g=G):
        if order not in (1, 2):
            raise ValueError("order must be 1 or 2")
        if (H is None) == (crest is None):
            raise ValueError("give exactly one of H or crest")
        self.Tp, self.depth, self.g, self.order = Tp, depth, g, order
        self.x0, self.t0 = x0, t0
        wp = 2 * np.pi / Tp
        self.omega = np.linspace(omega_range[0] * wp, omega_range[1] * wp, n_modes)
        dw = self.omega[1] - self.omega[0]
        S = spectrum(self.omega) if spectrum else jonswap(self.omega, Tp, gamma)
        self.k = wavenumber(self.omega, depth, g)
        self._weights = S * dw / np.sum(S * dw)  # unit crest

        if order == 2:
            from .second_order import SecondOrderWaveField
            base = SecondOrderWaveField(self._linear_field(t0, 1.0), rel_tol=rel_tol)
            self._keep, self._kern = base.keep, base.kern

        if crest is None:
            t = t0 + np.linspace(-1.5 * Tp, 1.5 * Tp, 3001 if order == 1 else 601)
            e1 = self._eta_unit(x0, t)
            if order == 1:
                crest = H / wave_height(t, e1)
            else:
                # eta = c e1 + c^2 e2 for linear crest amplitude c.
                e2 = np.array([self._second(ti, 1.0).eta_second(x0)[0] for ti in t])
                c = H / wave_height(t, e1)
                for _ in range(50):
                    c_new = c * H / wave_height(t, c * e1 + c**2 * e2)
                    if abs(c_new - c) < 1e-10 * c:
                        break
                    c = c_new
                crest = c
        self.crest = float(crest)
        self.H = float(wave_height(*self._probe())) if H is None else float(H)

    def _eta_unit(self, x, t):
        x, t = np.broadcast_arrays(np.asarray(x, float), np.asarray(t, float))
        phase = (x - self.x0)[..., None] * self.k - (t - self.t0)[..., None] * self.omega
        return np.cos(phase) @ self._weights

    def _probe(self):
        t = self.t0 + np.linspace(-1.5 * self.Tp, 1.5 * self.Tp, 3001)
        if self.order == 1:
            return t, self.crest * self._eta_unit(self.x0, t)
        return t, np.array([self.eta(self.x0, ti) for ti in t]).ravel()

    def _linear_field(self, t, crest):
        amp = crest * self._weights * np.exp(
            -1j * (self.k * self.x0 + self.omega * (t - self.t0)))
        return SpectralWaveField(self.k, amp, depth=self.depth, g=self.g)

    def _second(self, t, crest):
        from .second_order import SecondOrderWaveField
        return SecondOrderWaveField(self._linear_field(t, crest), keep=self._keep,
                                    kernels=self._kern)

    def field(self, t):
        """Wave field at time t (s): SpectralWaveField (order 1) or
        SecondOrderWaveField (order 2)."""
        if self.order == 1:
            return self._linear_field(t, self.crest)
        return self._second(t, self.crest)

    def eta(self, x, t=0.0):
        if self.order == 1:
            return self.crest * self._eta_unit(x, t)
        return self.field(t).eta(x)

    def kinematics(self, x, z, t=0.0, stretching=None):
        """Kinematics; stretching defaults to "wheeler" (order 1) or the
        consistent "linear" Taylor option (order 2)."""
        if stretching is None:
            stretching = "wheeler" if self.order == 1 else "linear"
        return self.field(t).kinematics(x, z, stretching)
