"""Dysthe (1979) modified NLS equation for deep-water wave envelopes.

The cubic NLS is the leading-order (O(eps^3)) envelope equation. Dysthe
carried the multiple-scale expansion one order further, to O(eps^4), which
adds third-order dispersion, nonlinear terms that depend on the envelope
slope, and the coupling to the wave-induced mean flow. In units
t' = w0 t, x' = k0 x, B = k0 A and in a frame moving with the linear group
velocity (xi = x' - t'/2):

    B_t + i/8 B_xixi - 1/16 B_xixixi + i/2 |B|^2 B
        + 3/2 |B|^2 B_xi + 1/4 B^2 B*_xi + i B phi_xi = 0,
    phi_xi = -1/2 F^-1[|K| F(|B|^2)]                    (at z = 0)

with eta = Re[B/k0 exp(i(k0 x - w0 t))] + ... (Fedele & Dutykh 2012, eq. 1;
the mean flow follows from phi_z = 1/2 (|B|^2)_xi). The extra terms make the
evolution asymmetric (the wave group's front steepens), limit the growth of
the Benjamin-Feir instability and shift its band, and generally lower the
peak amplification of breathers compared with the NLS.

``dispersion="exact"`` replaces the polynomial linear operator by the exact
deep-water dispersion relation (the "broader bandwidth MNLS" of Trulsen,
Kliakhandler, Dysthe & Velarde 2000).
"""

import warnings

import numpy as np

from .kinematics import G, EnvelopeWave
from .solutions import peregrine
from .solver import Result


class DystheSolver:
    """Integrating-factor RK4 (RK4IP) solver for the Dysthe equation.

    Parameters
    ----------
    L, N : float, int
        Periodic domain length in xi = k0 (x - cg t) and number of points.
    dispersion : {"dysthe", "exact"}
        Polynomial (third-order) or exact linear dispersion.
    higher_order : bool
        Include the O(eps^4) nonlinear terms (False reduces to the NLS).
    mean_flow : bool
        Include the mean-flow term.
    """

    def __init__(self, L, N, dispersion="dysthe", higher_order=True, mean_flow=True):
        if dispersion not in ("dysthe", "exact"):
            raise ValueError("dispersion must be 'dysthe' or 'exact'")
        self.L, self.N = float(L), int(N)
        self.dx = self.L / self.N
        self.x = -self.L / 2 + self.dx * np.arange(self.N)
        self.K = 2 * np.pi * np.fft.fftfreq(self.N, d=self.dx)
        K = self.K
        if dispersion == "dysthe" or not higher_order:
            lin = 1j * K**2 / 8 - (1j * K**3 / 16 if higher_order else 0.0)
        else:
            # B_t = -i (sqrt(1+K) - 1 - K/2) B for K > -1 (waves with k > 0).
            lin = 1j * K**2 / 8 - 1j * K**3 / 16
            ok = K > -1
            lin[ok] = -1j * (np.sqrt(1 + K[ok]) - 1 - K[ok] / 2)
        self.lin = lin
        self.dispersion = dispersion
        self.higher_order = higher_order
        self.mean_flow = mean_flow and higher_order

    def rhs_nonlinear(self, B):
        """Nonlinear part of dB/dt in physical space."""
        I = np.abs(B) ** 2
        out = -0.5j * I * B
        if self.higher_order:
            B_x = np.fft.ifft(1j * self.K * np.fft.fft(B))
            out += -1.5 * I * B_x - 0.25 * B**2 * np.conj(B_x)
        if self.mean_flow:
            out += 0.5j * B * np.fft.ifft(np.abs(self.K) * np.fft.fft(I))
        return out

    def time_derivative(self, B):
        """dB/dt in the moving frame (linear + nonlinear)."""
        return np.fft.ifft(self.lin * np.fft.fft(B)) + self.rhs_nonlinear(B)

    def stable_dt(self, B, amp_factor=1.0):
        """Rough explicit stability limit of the nonlinear terms for RK4, for
        an envelope that may grow to amp_factor times its current maximum."""
        Bmax2 = (amp_factor * np.max(np.abs(B))) ** 2
        rate = 0.5 * Bmax2 + (1.75 * Bmax2 * np.abs(self.K).max() if self.higher_order else 0)
        if self.mean_flow:
            rate += 0.5 * Bmax2 * np.abs(self.K).max()
        return 2.0 / max(rate, 1e-300)

    def _step(self, Bh, h, E2):
        """One RK4IP step in Fourier space (Hult 2007)."""
        Nf = lambda Bh_: np.fft.fft(self.rhs_nonlinear(np.fft.ifft(Bh_)))
        BI = E2 * Bh
        k1 = E2 * Nf(Bh)
        k2 = Nf(BI + 0.5 * h * k1)
        k3 = Nf(BI + 0.5 * h * k2)
        k4 = Nf(E2 * (BI + h * k3))
        return E2 * (BI + h / 6 * (k1 + 2 * k2 + 2 * k3)) + h / 6 * k4

    def run(self, B0, t0, t1, dt, n_out=200, callback=None, amp_factor=3.0):
        """Integrate from t0 to t1 (units of 1/w0), storing n_out snapshots.

        dt is reduced if needed so every output interval holds a whole number
        of steps, and to respect :meth:`stable_dt` for an envelope growing to
        ``amp_factor`` times its initial maximum (3 suits breathers).
        """
        B = np.asarray(B0(self.x) if callable(B0) else B0, dtype=complex)
        dt = min(dt, 0.5 * self.stable_dt(B, amp_factor))
        t_out = np.linspace(t0, t1, n_out)
        interval = t_out[1] - t_out[0] if n_out > 1 else 0.0
        steps = max(1, int(np.ceil(abs(interval) / dt))) if interval else 0
        h = interval / steps if steps else 0.0
        E2 = np.exp(self.lin * h / 2)
        out = np.empty((n_out, self.N), dtype=complex)
        out[0] = B
        Bh = np.fft.fft(B)
        if callback:
            callback(t_out[0], B)
        for j in range(1, n_out):
            for _ in range(steps):
                Bh = self._step(Bh, h, E2)
            out[j] = np.fft.ifft(Bh)
            if callback:
                callback(t_out[j], out[j])
        params = dict(L=self.L, N=self.N, dt=h, model="dysthe",
                      dispersion=self.dispersion, higher_order=self.higher_order,
                      mean_flow=self.mean_flow)
        return Result(x=self.x.copy(), t=t_out, psi=out, params=params)


def mass(B, dx):
    return np.sum(np.abs(B) ** 2, axis=-1) * dx


def peregrine_initial(xi, eps, T0):
    """Peregrine breather of background steepness eps at NLS time T0 (< 0),
    in Dysthe units: B = eps conj(psi_P(sqrt(2) eps xi, T0)).

    Under the NLS this focuses at t' = -2 T0 / eps^2 after the start."""
    return eps * np.conj(peregrine(np.sqrt(2) * eps * np.asarray(xi, float), T0))


def peregrine_domain(eps, L_nls=200.0, N=2048):
    """(L, N) in xi matching the NLS grid used for the Peregrine examples."""
    return L_nls / (np.sqrt(2) * eps), N


class DystheWave(EnvelopeWave):
    """Physical wave field from a Dysthe run (see :class:`EnvelopeWave`).

    Parameters
    ----------
    result : Result
        Output of :meth:`DystheSolver.run`.
    solver : DystheSolver
        The solver used (for the envelope time derivative).
    k0 / wavelength / period : float
        Carrier wavenumber (rad/m), wavelength (m) or period (s).
    order : {1, 2}
        Reconstruction order; 2 (default) is the consistent one for Dysthe.
    """

    def __init__(self, result, solver, k0=None, wavelength=None, period=None,
                 depth=np.inf, phase=0.0, order=2, g=G):
        given = [v is not None for v in (k0, wavelength, period)]
        if sum(given) != 1:
            raise ValueError("specify exactly one of k0, wavelength, period")
        if wavelength is not None:
            k0 = 2 * np.pi / wavelength
        elif period is not None:
            k0 = (2 * np.pi / period) ** 2 / g
        w0 = np.sqrt(g * k0)
        super().__init__(np.asarray(result.t) / w0, depth, phase, order, g)
        self.result, self.solver = result, solver
        self.k0, self.w0 = float(k0), float(w0)
        self.cg = 0.5 * w0 / k0

    def _envelope(self, i):
        t_i = self.times[i]
        B = self.result.psi[i]
        s = self.solver
        # Fixed-frame derivative: dB/dt'|_x' = dB/dt'|_xi - 1/2 dB/dxi.
        B_xi = np.fft.ifft(1j * s.K * np.fft.fft(B))
        B_t = s.time_derivative(B) - 0.5 * B_xi
        A = B / self.k0
        A_t = B_t * self.w0 / self.k0
        dx = s.dx / self.k0
        x0 = s.x[0] / self.k0 + self.cg * t_i
        return x0, dx, A, A_t, t_i

    def focus(self):
        """(time s, position m, index) of the largest envelope amplitude."""
        amp = np.abs(self.result.psi)
        i, j = np.unravel_index(np.argmax(amp), amp.shape)
        t = self.times[i]
        return t, self.solver.x[j] / self.k0 + self.cg * t, i


def _solve_height(fun, target, x0, tol=1e-3, max_iter=30):
    """Solve fun(x) = target for an increasing, slightly noisy fun.

    Brackets the root by stepping x up or down by 10%, then uses regula falsi
    (Illinois variant) so every iterate stays inside the bracket.
    """
    lo, flo = x0, fun(x0) - target
    if abs(flo) < tol * target:
        return lo
    step = 1.1 if flo < 0 else 1 / 1.1
    hi, fhi = lo, flo
    for _ in range(30):
        hi = hi * step
        fhi = fun(hi) - target
        if abs(fhi) < tol * target:
            return hi
        if np.sign(fhi) != np.sign(flo):
            break
        lo, flo = hi, fhi
    else:
        raise RuntimeError("could not bracket the target height")
    side = 0
    for _ in range(max_iter):
        if abs(hi - lo) < 1e-6 * abs(lo):
            # The bracket has collapsed onto a jump in fun: no exact root.
            x = lo if abs(flo) < abs(fhi) else hi
            warnings.warn(f"target height not reachable exactly (jump in H(eps)); "
                          f"using eps={x:.5g}, off by {min(abs(flo), abs(fhi)):.3g}")
            return x
        x = (lo * fhi - hi * flo) / (fhi - flo)
        fx = fun(x) - target
        if abs(fx) < tol * target:
            return x
        if np.sign(fx) == np.sign(flo):
            lo, flo = x, fx
            if side == -1:
                fhi *= 0.5
            side = -1
        else:
            hi, fhi = x, fx
            if side == 1:
                flo *= 0.5
            side = 1
    raise RuntimeError("height calibration did not converge")


def first_focus_index(peak, rel=0.85, smooth=5):
    """Index of the first major focusing event in a peak-amplitude history:
    the first local maximum (after light smoothing) reaching ``rel`` times the
    overall maximum. At large steepness a Dysthe group can focus more than
    once and the largest event switches between them as eps changes, so the
    first event gives a height that varies continuously with eps."""
    peak = np.asarray(peak, float)
    if smooth > 1:
        ker = np.ones(smooth) / smooth
        ps = np.convolve(np.pad(peak, smooth // 2, mode="edge"), ker, mode="valid")
    else:
        ps = peak
    thresh = rel * ps.max()
    for i in range(1, ps.size - 1):
        if ps[i] >= thresh and ps[i] >= ps[i - 1] and ps[i] >= ps[i + 1]:
            # refine on the raw history around the smoothed peak
            lo, hi = max(0, i - smooth), min(peak.size, i + smooth + 1)
            return lo + int(np.argmax(peak[lo:hi]))
    return int(np.argmax(peak))


def dysthe_focused_wave(eps, period, depth=np.inf, order=2, T0=-2.0, g=G,
                        dispersion="dysthe", window_periods=1.5, samples_per_period=100,
                        focus="first"):
    """Evolve a Peregrine initial condition of steepness eps under Dysthe and
    return a :class:`DystheWave` sampled finely around its focus.

    ``focus="first"`` (default) takes the first major focusing event (see
    :func:`first_focus_index`); ``"max"`` takes the largest one. The carrier
    phase is chosen so that a crest sits at the envelope maximum.
    Returns (wave, focus_time_s, focus_x_m).
    """
    L, N = peregrine_domain(eps)
    solver = DystheSolver(L, N, dispersion=dispersion)
    t_nls = -2 * T0 / eps**2
    dt = 0.004 / eps**2
    coarse = solver.run(peregrine_initial(solver.x, eps, T0), 0.0, 2.0 * t_nls, dt, n_out=401)
    peak = np.abs(coarse.psi).max(axis=1)
    i = first_focus_index(peak) if focus == "first" else int(np.argmax(peak))
    half = window_periods * 2 * np.pi          # window in units of 1/w0
    t_start = coarse.t[i] - half
    j = max(0, int(np.searchsorted(coarse.t, t_start)) - 1)
    pre = solver.run(coarse.psi[j], coarse.t[j], t_start, dt, n_out=2)
    n_out = int(2 * window_periods * samples_per_period) + 1
    fine = solver.run(pre.psi[-1], t_start, t_start + 2 * half, dt, n_out=n_out)
    amp = np.abs(fine.psi)
    it, ix = np.unravel_index(np.argmax(amp), amp.shape)
    # Refine the peak position between grid nodes (spectral interpolation of
    # the envelope around the discrete maximum).
    B_hat = np.fft.fft(fine.psi[it]) / solver.N
    xi_loc = solver.x[ix] + solver.dx * np.linspace(-1, 1, 201)
    B_loc = np.exp(1j * np.outer(xi_loc - solver.x[0], solver.K)) @ B_hat
    m = int(np.argmax(np.abs(B_loc)))
    xi_f, B_f = xi_loc[m], B_loc[m]
    k0 = (2 * np.pi / period) ** 2 / g
    w0 = np.sqrt(g * k0)
    t_f = fine.t[it] / w0
    x_f = xi_f / k0 + 0.5 * w0 / k0 * t_f
    phase = -np.angle(B_f) - k0 * x_f + w0 * t_f
    wave = DystheWave(fine, solver, period=period, depth=depth, phase=phase, order=order, g=g)
    return wave, t_f, x_f


def dysthe_for_height(H, period, depth=np.inf, order=2, eps_guess=None, **kwargs):
    """Dysthe focused wave whose largest wave has crest-to-trough height H.

    At large steepness the focused Dysthe group is multi-peaked, so the height
    at a single "focal point" jumps as eps changes. Instead H is the largest
    local wave height (crest minus deepest trough within 0.75 periods, from
    the time series) over positions within 1.5 wavelengths of the envelope
    focus. This is continuous in eps, and it is the engineering quantity: the
    biggest wave in the group, measured where it is biggest. The returned
    (t_f, x_f) are that wave's crest time and position.

    Returns (wave, focus_time_s, focus_x_m, eps).
    """
    from .kinematics import wave_height
    k0 = (2 * np.pi / period) ** 2 / G
    lam = 2 * np.pi / k0
    cache = {}

    def height(eps):
        wave, t_f, x_f = dysthe_focused_wave(eps, period, depth, order, **kwargs)
        t = wave.physical_times()
        xs = x_f + lam * np.linspace(-1.5, 1.5, 61)
        eta = np.array([wave.eta(xs, ti) for ti in t])          # (time, position)
        Hx = [wave_height(t, eta[:, j], window=0.75 * period) for j in range(xs.size)]
        j = int(np.argmax(Hx))
        cache[eps] = (wave, float(t[np.argmax(eta[:, j])]), float(xs[j]))
        return Hx[j]

    e0 = eps_guess if eps_guess is not None else k0 * H / 5.0
    eps = _solve_height(height, H, e0, tol=1e-3)
    if eps not in cache:
        height(eps)
    wave, t_f, x_f = cache[eps]
    return wave, t_f, x_f, eps
