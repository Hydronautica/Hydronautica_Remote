"""Pseudo-spectral split-step solver for the 1D cubic NLS.

Solves, on a periodic domain x in [-L/2, L/2),

    i psi_t + alpha * psi_xx + beta * |psi|^2 psi = 0

The default (alpha=1/2, beta=1) is the normalised focusing NLS used throughout
the rogue-wave literature, for which the Peregrine soliton and the Akhmediev /
Kuznetsov-Ma breathers in :mod:`nls.solutions` are exact solutions.

Both sub-problems are solved exactly:

* linear:     psi_hat(k) <- psi_hat(k) * exp(-i alpha k^2 dt)
* nonlinear:  psi <- psi * exp(i beta |psi|^2 dt)      (|psi| is conserved)

and combined with Strang splitting (2nd order) or the Yoshida triple-jump
composition of Strang steps (4th order).
"""

from dataclasses import dataclass, field

import numpy as np

# Yoshida (1990) 4th-order composition coefficients.
_CBRT2 = 2.0 ** (1.0 / 3.0)
_W1 = 1.0 / (2.0 - _CBRT2)
_W0 = -_CBRT2 / (2.0 - _CBRT2)


@dataclass
class Result:
    """Output of :meth:`NLSSolver.run`."""

    x: np.ndarray
    t: np.ndarray
    psi: np.ndarray  # shape (len(t), len(x))
    params: dict = field(default_factory=dict)

    @property
    def amplitude(self):
        return np.abs(self.psi)

    def max_amplitude(self):
        """Peak |psi| over x at each stored time."""
        return self.amplitude.max(axis=1)


class NLSSolver:
    """Periodic-domain split-step Fourier solver.

    Parameters
    ----------
    L : float
        Domain length; the grid is x in [-L/2, L/2).
    N : int
        Number of grid points (a power of two is fastest).
    alpha, beta : float
        Dispersion and nonlinearity coefficients. beta*alpha > 0 is focusing.
    order : {2, 4}
        Splitting order: Strang (2) or Yoshida composition (4).
    """

    def __init__(self, L, N, alpha=0.5, beta=1.0, order=4):
        if order not in (2, 4):
            raise ValueError("order must be 2 or 4")
        self.L = float(L)
        self.N = int(N)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.order = order
        self.dx = self.L / self.N
        self.x = -self.L / 2 + self.dx * np.arange(self.N)
        self.k = 2 * np.pi * np.fft.fftfreq(self.N, d=self.dx)
        self._lin_cache = {}

    def _linear(self, psi, dt):
        key = round(dt, 15)
        phase = self._lin_cache.get(key)
        if phase is None:
            phase = np.exp(-1j * self.alpha * self.k**2 * dt)
            self._lin_cache[key] = phase
        return np.fft.ifft(np.fft.fft(psi) * phase)

    def _nonlinear(self, psi, dt):
        return psi * np.exp(1j * self.beta * np.abs(psi) ** 2 * dt)

    def _strang(self, psi, dt):
        psi = self._nonlinear(psi, dt / 2)
        psi = self._linear(psi, dt)
        return self._nonlinear(psi, dt / 2)

    def step(self, psi, dt):
        """Advance psi by one time step dt."""
        if self.order == 2:
            return self._strang(psi, dt)
        psi = self._strang(psi, _W1 * dt)
        psi = self._strang(psi, _W0 * dt)
        return self._strang(psi, _W1 * dt)

    def run(self, psi0, t0, t1, dt, n_out=200, callback=None):
        """Integrate from t0 to t1, storing n_out snapshots (including both ends).

        psi0 may be an array on ``self.x`` or a callable ``psi0(x)``.
        dt is adjusted downwards so every output interval holds a whole number
        of steps. ``callback(t, psi)`` is called at each output time.
        """
        psi = np.asarray(psi0(self.x) if callable(psi0) else psi0, dtype=complex)
        if psi.shape != self.x.shape:
            raise ValueError(f"psi0 has shape {psi.shape}, expected {self.x.shape}")

        t_out = np.linspace(t0, t1, n_out)
        interval = t_out[1] - t_out[0] if n_out > 1 else 0.0
        steps = max(1, int(np.ceil(abs(interval) / dt))) if interval else 0
        h = interval / steps if steps else 0.0

        out = np.empty((n_out, self.N), dtype=complex)
        out[0] = psi
        if callback:
            callback(t_out[0], psi)
        for j in range(1, n_out):
            for _ in range(steps):
                psi = self.step(psi, h)
            out[j] = psi
            if callback:
                callback(t_out[j], psi)

        params = dict(L=self.L, N=self.N, alpha=self.alpha, beta=self.beta,
                      order=self.order, dt=h)
        return Result(x=self.x.copy(), t=t_out, psi=out, params=params)
