"""Map normalised NLS solutions onto deep-water surface gravity waves.

The envelope A(x, t) of a narrow-banded deep-water wave train,
eta = Re[A exp(i(k0 x - w0 t))], obeys to leading order

    i (A_t + cg A_x) - w0/(8 k0^2) A_xx - (w0 k0^2 / 2) |A|^2 A = 0,

with w0^2 = g k0 and cg = w0 / (2 k0). With background amplitude a0 and
steepness eps = k0 a0, the substitution

    A(x, t) = a0 * conj(psi(X, T)),
    X = sqrt(2) k0 eps (x - cg t),   T = w0 eps^2 t / 2

turns this into the normalised i psi_T + psi_XX/2 + |psi|^2 psi = 0 solved by
:class:`nls.solver.NLSSolver`.
"""

import numpy as np

G = 9.81


class DeepWater:
    """Scaling between normalised NLS variables and a physical wave train.

    Parameters
    ----------
    a0 : float
        Background (carrier) wave amplitude in metres.
    k0 : float, optional
        Carrier wavenumber in rad/m. Give exactly one of k0, wavelength, period.
    wavelength, period : float, optional
        Carrier wavelength (m) or period (s).
    g : float
        Gravitational acceleration.
    """

    def __init__(self, a0, k0=None, wavelength=None, period=None, g=G):
        given = [v is not None for v in (k0, wavelength, period)]
        if sum(given) != 1:
            raise ValueError("specify exactly one of k0, wavelength, period")
        if wavelength is not None:
            k0 = 2 * np.pi / wavelength
        elif period is not None:
            k0 = (2 * np.pi / period) ** 2 / g
        self.g = g
        self.a0 = float(a0)
        self.k0 = float(k0)
        self.w0 = np.sqrt(g * self.k0)
        self.cg = self.w0 / (2 * self.k0)
        self.eps = self.k0 * self.a0
        if self.eps > 0.3:
            raise ValueError(f"steepness k0*a0 = {self.eps:.3f} is too large "
                             "for the weakly nonlinear NLS model")
        self.x_scale = np.sqrt(2) * self.k0 * self.eps   # X per metre
        self.t_scale = 0.5 * self.w0 * self.eps**2       # T per second

    @property
    def wavelength(self):
        return 2 * np.pi / self.k0

    @property
    def period(self):
        return 2 * np.pi / self.w0

    def to_physical_x(self, X, t=0.0):
        """Physical position (m) of normalised coordinate X at time t (s)."""
        return np.asarray(X) / self.x_scale + self.cg * t

    def to_physical_t(self, T):
        return np.asarray(T) / self.t_scale

    def to_normalised_x(self, x, t=0.0):
        return (np.asarray(x) - self.cg * t) * self.x_scale

    def to_normalised_t(self, t):
        return np.asarray(t) * self.t_scale

    def envelope(self, psi):
        """Physical complex envelope A (m) from normalised psi."""
        return self.a0 * np.conj(psi)

    def surface(self, psi, X, T, phase=0.0):
        """Surface elevation eta (m) at normalised (X, T).

        ``phase`` shifts the carrier, eta = Re[A exp(i(k0 x - w0 t + phase))];
        it decides whether the envelope peak falls on a crest or a trough.

        psi may be 1D (one time T, scalar) or 2D with shape (len(T), len(X)).
        Returns (x, t, eta) in physical units, x having the same shape as eta
        because the frame moves with the group velocity.
        """
        X = np.asarray(X, dtype=float)
        T = np.atleast_1d(np.asarray(T, dtype=float))
        psi2 = np.atleast_2d(psi)
        t = self.to_physical_t(T)[:, None]
        x = self.to_physical_x(X[None, :], t)
        eta = np.real(self.envelope(psi2) * np.exp(1j * (self.k0 * x - self.w0 * t + phase)))
        if np.ndim(psi) == 1:
            return x[0], t[0, 0], eta[0]
        return x, t[:, 0], eta

    def describe(self):
        return (f"carrier: lambda={self.wavelength:.1f} m, T={self.period:.2f} s, "
                f"a0={self.a0:.2f} m, k0*a0={self.eps:.3f}; "
                f"1 unit X = {1 / self.x_scale:.1f} m, "
                f"1 unit T = {1 / self.t_scale:.1f} s")


def peregrine_focus_series(dw, t, phase=np.pi):
    """Surface elevation (m) at the focal point x = 0 of a Peregrine wave
    focusing at t = 0, from the exact solution. phase = pi puts a crest at the
    focus (psi(0, 0) = -3)."""
    from .solutions import peregrine
    t = np.asarray(t, dtype=float)
    psi = peregrine(dw.to_normalised_x(0.0, t), dw.to_normalised_t(t))
    return np.real(dw.envelope(psi) * np.exp(1j * (phase - dw.w0 * t)))


def peregrine_for_height(H, k0=None, wavelength=None, period=None, phase=np.pi,
                         g=G):
    """DeepWater scaling whose Peregrine breather has focal wave height H.

    H is crest-to-trough, measured in the time series at the focus as for
    :func:`nls.kinematics.wave_height`. Since the envelope width depends on the
    steepness, a0 is found by fixed-point iteration (H is roughly 5.1 a0).
    """
    from .kinematics import wave_height
    a0 = H / 5.1
    for _ in range(20):
        dw = DeepWater(a0, k0=k0, wavelength=wavelength, period=period, g=g)
        t = np.linspace(-1.5 * dw.period, 1.5 * dw.period, 3001)
        h = wave_height(t, peregrine_focus_series(dw, t, phase))
        a0_new = a0 * H / h
        if abs(a0_new - a0) < 1e-10 * a0:
            break
        a0 = a0_new
    return DeepWater(a0, k0=k0, wavelength=wavelength, period=period, g=g)
