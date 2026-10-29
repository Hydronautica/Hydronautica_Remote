"""Spatial NLS for irregular waves travelling over a varying sea bed.

A wave train of carrier frequency w0 travels in +x over a depth h(x) that
changes slowly compared with a wavelength. The surface is

    eta(x, t) = Re[ A(x, tau) exp(i (int k0 dx - w0 t)) ],   tau = t - int dx / cg0

where k0(x) and cg0(x) are the local linear wavenumber and group velocity of
the carrier. The envelope obeys the spatial NLS with varying coefficients
(Djordjevic & Redekopp 1978):

    i (A_x + cg0_x / (2 cg0) A) + P(x) A_tautau - Q(x) |A|^2 A = 0

    P = w''(k0) / (2 cg0^3)         (-1/g in deep water)
    Q = nu(k0 h) / cg0              ( k0^3 in deep water)

with the finite-depth nonlinear coefficient (Mei 1989; mean flow included)

    nu = w k^2 / (16 s^4) [9 - 10 s^2 + 9 s^4
           - 2 s^2 / (g h - cg^2) (4 cp^2 + 4 cp cg (1 - s^2) + g h (1 - s^2)^2)],
    s = tanh(k h),

which tends to w k^2 / 2 in deep water and changes sign at k h = 1.363: deeper
than that, groups focus (Benjamin-Feir instability, rogue waves); shallower,
they spread out.

The cg0_x term keeps the wave energy flux |A|^2 cg0 constant, i.e. linear
shoaling. By default the linear part is solved exactly instead: every
frequency component is advanced with its own wavenumber k(w, h(x)) and
shoaled with its own group velocity, so linear dispersion and shoaling are
exact for a broad spectrum and only the nonlinear term relies on the
narrow-band assumption ("exact linear dispersion" NLS, cf. Trulsen et al.
2000). ``dispersion="nls"`` gives the classical quadratic form.

Validity. The NLS assumes moderate steepness, a narrow-ish spectrum and an
Ursell number that is not large. Where the carrier k0 h falls below
``kh_min`` the nonlinear term is switched off (it grows like (k h)^-4 and the
equation stops describing the physics); the result marks those stations. A
simple depth-limited breaking cap (H < gamma h) clips the envelope.
"""

from dataclasses import dataclass, field

import numpy as np

from .kinematics import G, wavenumber
from .newwave import jonswap


def group_velocity(k, h, g=G):
    """Linear group velocity at wavenumber k and depth h."""
    k = np.asarray(k, dtype=float)
    kh = k * h
    w = np.sqrt(g * k * np.tanh(kh))
    with np.errstate(over="ignore"):
        f = np.where(kh > 20, 0.0, 2 * kh / np.sinh(np.minimum(2 * kh, 700)))
    return w / k / 2 * (1 + f)


def dispersion_second_derivative(k, h, g=G):
    """w''(k) for w^2 = g k tanh(k h), by differentiating cg."""
    dk = 1e-5 * k
    return (group_velocity(k + dk, h, g) - group_velocity(k - dk, h, g)) / (2 * dk)


def nonlinear_coefficient(k, h, g=G):
    """Finite-depth NLS coefficient nu (frequency shift = nu |A|^2)."""
    kh = k * h
    if kh > 30:
        return np.sqrt(g * k) * k**2 / 2
    s = np.tanh(kh)
    w = np.sqrt(g * k * s)
    cp = w / k
    cg = group_velocity(k, h, g)
    mean = 2 * s**2 / (g * h - cg**2) * (4 * cp**2 + 4 * cp * cg * (1 - s**2) + g * h * (1 - s**2) ** 2)
    return w * k**2 / (16 * s**4) * (9 - 10 * s**2 + 9 * s**4 - mean)


def critical_kh():
    """k h at which the finite-depth NLS changes from focusing to defocusing."""
    lo, hi = 0.8, 3.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if nonlinear_coefficient(1.0, mid) > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


@dataclass
class Transect:
    """Depth along the wave path: x (m) increasing, h (m) positive."""

    x: np.ndarray
    h: np.ndarray

    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=float)
        self.h = np.asarray(self.h, dtype=float)
        if self.x.ndim != 1 or self.x.shape != self.h.shape or len(self.x) < 2:
            raise ValueError("x and h must be 1-D arrays of the same length (at least 2)")
        if np.any(np.diff(self.x) <= 0):
            raise ValueError("x must increase")
        if np.any(self.h <= 0):
            raise ValueError("depth must be positive everywhere on the transect")

    @classmethod
    def constant(cls, depth, length):
        return cls([0.0, float(length)], [float(depth), float(depth)])

    def depth(self, x):
        return np.interp(x, self.x, self.h)


@dataclass
class TransectResult:
    """Envelope at the output stations, with summary statistics along x."""

    x: np.ndarray            # stations (m)
    t: np.ndarray            # time in the group frame, tau (s)
    A: np.ndarray            # complex envelope, shape (len(x), len(t))
    depth: np.ndarray
    k0: np.ndarray
    cg0: np.ndarray
    phase: np.ndarray        # int k0 dx at each station
    delay: np.ndarray        # int dx / cg0 at each station
    w0: float
    nonlinear: np.ndarray    # False where the NLS term was switched off (too shallow)
    broken: np.ndarray       # fraction of the record clipped by the breaking cap
    params: dict = field(default_factory=dict)

    @property
    def kh(self):
        return self.k0 * self.depth

    def hs(self):
        """Significant wave height 4 sqrt(m0) at each station (m0 = <|A|^2> / 2)."""
        return 4 * np.sqrt(0.5 * np.mean(np.abs(self.A) ** 2, axis=1))

    def eta(self, i, t=None):
        """Surface elevation time series at station i, at absolute times t."""
        tt = self.t + self.delay[i] if t is None else np.asarray(t, dtype=float)
        tau = tt - self.delay[i]
        T = self.t[-1] - self.t[0] + (self.t[1] - self.t[0])
        Ai = np.interp(np.mod(tau - self.t[0], T) + self.t[0], self.t, self.A[i], period=T)
        return tt, np.real(Ai * np.exp(1j * (self.phase[i] - self.w0 * tt)))

    def crest_max(self):
        """Largest crest at each station (linear carrier, envelope peak)."""
        return np.abs(self.A).max(axis=1)

    def kurtosis(self):
        out = np.empty(len(self.x))
        for i in range(len(self.x)):
            _, e = self.eta(i)
            e = e - e.mean()
            out[i] = np.mean(e**4) / np.mean(e**2) ** 2
        return out


def jonswap_envelope(Hs, Tp, gamma=3.3, duration=1200.0, dt=None, band=(0.5, 2.5), seed=None):
    """Random-phase JONSWAP sea at x = 0 as a periodic envelope about wp.

    Returns (tau, A, w0). The record repeats with period ``duration`` so it
    suits the periodic solver; components outside band * wp are dropped.
    """
    wp = 2 * np.pi / Tp
    if dt is None:
        dt = Tp / 16
    n = int(2 ** np.ceil(np.log2(duration / dt)))
    dt = duration / n
    tau = dt * np.arange(n)
    dw = 2 * np.pi / duration
    w = wp + dw * np.fft.fftfreq(n, d=1.0 / n)   # absolute frequencies of the envelope modes
    keep = (w > band[0] * wp) & (w < band[1] * wp)
    S = np.where(keep, jonswap(np.where(w > 0, w, 1.0), Tp, gamma), 0.0)
    m0 = np.sum(S) * dw
    S *= (Hs / 4) ** 2 / m0
    rng = np.random.default_rng(seed)
    amp = np.sqrt(2 * S * dw) * np.exp(2j * np.pi * rng.random(n))
    # eta = Re sum a_n exp(-i w_n t) = Re[A exp(-i wp t)],  A = sum a_n exp(-i (w_n - wp) t)
    A = n * np.fft.ifft(amp[_conj_order(n)])  # mode with offset d sits at FFT frequency -d
    return tau, A, wp


def _conj_order(n):
    """Index map putting envelope offset d at FFT bin -d."""
    return (-np.arange(n)) % n


class VariableDepthNLS:
    """Split-step solver for the spatial NLS over a depth transect.

    Parameters
    ----------
    transect : Transect
    w0 : float
        Carrier angular frequency (usually the spectral peak).
    dispersion : {"exact", "nls"}
        Exact linear dispersion and per-component shoaling (default), or the
        classical quadratic NLS operator with energy-flux shoaling.
    nonlinear : bool
        False gives the linear (refraction-free, normal incidence) result.
    kh_min : float
        Below this k0 h the nonlinear term is switched off.
    breaking : float or None
        Depth-limited breaking index gamma (H_max = gamma h); None disables it.
    """

    def __init__(self, transect, w0, dispersion="exact", nonlinear=True, kh_min=0.7, breaking=0.78, g=G):
        if dispersion not in ("exact", "nls"):
            raise ValueError("dispersion must be 'exact' or 'nls'")
        self.tr = transect
        self.w0 = float(w0)
        self.dispersion = dispersion
        self.nonlinear = bool(nonlinear)
        self.kh_min = float(kh_min)
        self.breaking = breaking
        self.g = g

    def _carrier(self, h):
        k = float(wavenumber(self.w0, h, self.g))
        return k, float(group_velocity(k, h, self.g))

    def run(self, tau, A0, dx=None, stations=None):
        """Propagate the envelope A0(tau) given at x = transect.x[0].

        ``stations`` are the x positions to store (default: about 200 evenly
        spaced along the transect). Returns a :class:`TransectResult`.
        """
        tau = np.asarray(tau, dtype=float)
        A = np.asarray(A0, dtype=complex).copy()
        n = len(tau)
        T = (tau[1] - tau[0]) * n
        x0, x1 = self.tr.x[0], self.tr.x[-1]
        k_first = self._carrier(self.tr.h[0])[0]
        if dx is None:
            dx = 0.5 * 2 * np.pi / k_first          # half a carrier wavelength
        nsteps = max(1, int(np.ceil((x1 - x0) / dx)))
        dx = (x1 - x0) / nsteps
        if stations is None:
            stations = np.linspace(x0, x1, min(nsteps + 1, 201))
        stations = np.asarray(stations, dtype=float)
        store_at = np.clip(np.round((stations - x0) / dx).astype(int), 0, nsteps)

        # envelope FFT mode m has offset d = -W[m] from w0 (see jonswap_envelope)
        W = 2 * np.pi * np.fft.fftfreq(n, d=T / n)
        wabs = self.w0 - W
        valid = wabs > 0.05 * self.w0

        def comp(h):
            k0, cg0 = self._carrier(h)
            if self.dispersion == "exact":
                kn = np.where(valid, wavenumber(np.where(valid, wabs, self.w0), h, self.g), 0.0)
                cgn = np.where(valid, group_velocity(np.where(valid, kn, 1.0), h, self.g), 1.0)
            else:
                kn = cgn = None
            P = dispersion_second_derivative(k0, h, self.g) / (2 * cg0**3)
            Q = nonlinear_coefficient(k0, h, self.g) / cg0 if (self.nonlinear and k0 * h >= self.kh_min) else 0.0
            return k0, cg0, kn, cgn, P, Q

        def linear(Ah, ca, cm, cb):
            # advance half a step (dx/2) from coefficients ca (start) via cm (mid) to cb (end)
            k0, cg0, kn, cgn, P, _ = cm
            L = 0.5 * dx
            if self.dispersion == "exact":
                dk = np.where(valid, kn - k0 + W / cg0, 0.0)   # k(w0 - W) - k0 - (-W)/cg0
                Ah = Ah * np.exp(1j * dk * L)
                Ah = Ah * np.where(valid, np.sqrt(ca[3] / cb[3]), 0.0)
            else:
                Ah = Ah * np.exp(-1j * P * W**2 * L) * np.sqrt(ca[1] / cb[1])
            return Ah

        xs = x0 + dx * np.arange(nsteps + 1)
        h_nodes = self.tr.depth(xs)
        h_mid = self.tr.depth(xs[:-1] + 0.5 * dx)
        out_A = np.empty((len(stations), n), dtype=complex)
        out_brk = np.zeros(len(stations))
        coef_node = comp(h_nodes[0])
        if np.any(store_at == 0):
            out_A[store_at == 0] = A
        for j in range(nsteps):
            cmid = comp(h_mid[j])
            cend = comp(h_nodes[j + 1])
            # Strang: half linear (node -> mid), full nonlinear at mid, half linear (mid -> node)
            Ah = np.fft.fft(A)
            Ah = linear(Ah, coef_node, cmid, cmid)
            A = np.fft.ifft(Ah)
            Q = cmid[5]
            if Q:
                A = A * np.exp(-1j * Q * np.abs(A) ** 2 * dx)
            Ah = np.fft.fft(A)
            Ah = linear(Ah, cmid, cmid, cend)
            A = np.fft.ifft(Ah)
            broken_frac = 0.0
            if self.breaking:
                amax = 0.5 * self.breaking * h_nodes[j + 1]
                mag = np.abs(A)
                over = mag > amax
                if np.any(over):
                    A[over] *= amax / mag[over]
                    broken_frac = over.mean()
            coef_node = cend
            hit = store_at == j + 1
            if np.any(hit):
                out_A[hit] = A
                out_brk[hit] = broken_frac

        hs_ = self.tr.depth(stations)
        car = [self._carrier(h) for h in hs_]
        k0s = np.array([c[0] for c in car])
        cgs = np.array([c[1] for c in car])
        # phase and delay integrals on the fine grid, sampled at the stations
        kk = np.array([self._carrier(h)[0] for h in h_nodes])
        cc = np.array([self._carrier(h)[1] for h in h_nodes])
        ph = np.concatenate([[0.0], np.cumsum(0.5 * (kk[1:] + kk[:-1]) * dx)])
        de = np.concatenate([[0.0], np.cumsum(0.5 * (1 / cc[1:] + 1 / cc[:-1]) * dx)])
        return TransectResult(
            x=stations, t=tau, A=out_A, depth=hs_, k0=k0s, cg0=cgs,
            phase=ph[store_at], delay=de[store_at], w0=self.w0,
            nonlinear=np.array([self.nonlinear and k * h >= self.kh_min for k, h in zip(k0s, hs_)]),
            broken=out_brk,
            params=dict(dx=dx, dispersion=self.dispersion, kh_min=self.kh_min, breaking=self.breaking),
        )


def irregular_over_transect(transect, Hs, Tp, gamma=3.3, duration=1200.0, seed=None, **kwargs):
    """Convenience: a JONSWAP sea at the start of the transect, propagated along it."""
    run_kw = {k: kwargs.pop(k) for k in ("dx", "stations") if k in kwargs}
    tau, A0, w0 = jonswap_envelope(Hs, Tp, gamma, duration=duration, seed=seed)
    return VariableDepthNLS(transect, w0, **kwargs).run(tau, A0, **run_kw)
