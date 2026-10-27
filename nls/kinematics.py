"""Wave kinematics from spectral (sum-of-modes) surface descriptions.

A unidirectional wave field travelling in +x is represented at one instant as

    eta(x) = Re sum_n a_n exp(i k_n x),     k_n > 0,

together with the time derivatives of the amplitudes, adot_n. Each mode is
given linear (Airy) kinematics,

    u = Re sum w_n Ch(k_n, z) a_n e^{i k_n x}
    w = Re sum -i w_n Sh(k_n, z) a_n e^{i k_n x}

with Ch = cosh k(z+d) / sinh kd, Sh = sinh k(z+d) / sinh kd and w_n the linear
frequency. Local accelerations use adot_n instead of a_n, so they follow the
true evolution of the field (e.g. the NLS envelope dynamics including the
Stokes frequency correction) rather than assuming free linear waves.

All classes in this package expose the same interface,

    wave.eta(x, t)
    wave.kinematics(x, z, t, stretching=...)   -> dict of arrays

with z measured upward from the still water level and points above the
surface (or below the bed) returned as NaN.
"""

import numpy as np

G = 9.81

STRETCHING = ("wheeler", "linear", "none")


def wavenumber(omega, depth=np.inf, g=G):
    """Solve the linear dispersion relation omega^2 = g k tanh(k d) for k."""
    omega = np.asarray(omega, dtype=float)
    k = omega**2 / g
    if np.isinf(depth):
        return k
    k = np.where(k * depth < 1, omega / np.sqrt(g * depth), k)  # shallow start
    for _ in range(50):
        th = np.tanh(k * depth)
        f = g * k * th - omega**2
        df = g * th + g * k * depth * (1 - th**2)
        k_new = k - f / df
        if np.allclose(k_new, k, rtol=1e-14, atol=0):
            return k_new
        k = k_new
    return k


def frequency(k, depth=np.inf, g=G):
    """Linear dispersion omega(k) = sqrt(g k tanh(k d))."""
    k = np.asarray(k, dtype=float)
    return np.sqrt(g * k * (1.0 if np.isinf(depth) else np.tanh(k * depth)))


def _depth_functions(k, z, depth):
    """Ch = cosh k(z+d)/sinh kd and Sh = sinh k(z+d)/sinh kd, overflow-safe."""
    ez = np.exp(k * z)
    if np.isinf(depth):
        return ez, ez
    e_bed = np.exp(-2 * k * (z + depth))
    den = 1 - np.exp(-2 * k * depth)
    return ez * (1 + e_bed) / den, ez * (1 - e_bed) / den


def wave_height(t, eta):
    """Height of the largest wave in a record: crest minus deeper adjacent trough.

    The troughs are the lowest points between the highest crest and the
    neighbouring zero crossings on either side.
    """
    t = np.asarray(t)
    eta = np.asarray(eta)
    i = int(np.argmax(eta))
    sign = np.sign(eta)
    # first zero crossings left/right of the crest, then the next ones beyond
    def trough(direction):
        j = i
        crossings = 0
        lo = np.inf
        while 0 <= j + direction < eta.size:
            if sign[j + direction] != sign[j]:
                crossings += 1
                if crossings == 2:
                    break
            j += direction
            if crossings == 1:
                lo = min(lo, eta[j])
        return lo
    lo = min(trough(-1), trough(+1))
    if not np.isfinite(lo):
        lo = eta.min()
    return float(eta[i] - lo)


class SpectralWaveField:
    """Instantaneous linear wave field defined by mode amplitudes.

    Parameters
    ----------
    k : array
        Wavenumbers (rad/m), all positive.
    amp : complex array
        Amplitudes a_n such that eta(x) = Re sum a_n exp(i k_n x).
    amp_dot : complex array, optional
        d a_n / dt. Defaults to free linear waves, -i omega_n a_n.
    depth : float
        Water depth (m); np.inf for deep water.
    """

    def __init__(self, k, amp, amp_dot=None, depth=np.inf, g=G):
        self.k = np.asarray(k, dtype=float)
        self.amp = np.asarray(amp, dtype=complex)
        self.depth = float(depth)
        self.g = g
        self.omega = frequency(self.k, self.depth, g)
        if amp_dot is None:
            amp_dot = -1j * self.omega * self.amp
        self.amp_dot = np.asarray(amp_dot, dtype=complex)
        if np.any(self.k <= 0):
            raise ValueError("all wavenumbers must be positive")

    def truncated(self, rel_tol=1e-9):
        """Copy keeping only modes with |a| or |adot| above rel_tol of the max."""
        keep = (np.abs(self.amp) > rel_tol * np.abs(self.amp).max()) | (
            np.abs(self.amp_dot) > rel_tol * np.abs(self.amp_dot).max())
        return SpectralWaveField(self.k[keep], self.amp[keep], self.amp_dot[keep],
                                 self.depth, self.g)

    def eta(self, x):
        x = np.asarray(x, dtype=float)
        E = np.exp(1j * np.multiply.outer(x, self.k))
        return np.real(E @ self.amp)

    def kinematics(self, x, z, stretching="wheeler"):
        """Velocities and accelerations at points (x, z).

        stretching controls how the modes are evaluated in the crest region:

        * ``"wheeler"``: z is mapped to z' = d (z - eta) / (d + eta) (or
          z - eta in deep water), so the surface maps to z' = 0;
        * ``"linear"``: kinematics above z = 0 are extrapolated linearly from
          their value and vertical gradient at z = 0;
        * ``"none"``: plain linear theory evaluated up to the surface (over-
          predicts crest kinematics for broad spectra).

        Returns a dict with eta, u, w, ut, wt (local accelerations),
        ax, az (total, including convective terms) and the gradients
        ux, uz, wx, wz. For the Wheeler and linear options the gradients and
        convective terms are evaluated at the stretched coordinate, which is
        the usual engineering approximation.
        """
        if stretching not in STRETCHING:
            raise ValueError(f"stretching must be one of {STRETCHING}")
        x, z = np.broadcast_arrays(np.asarray(x, float), np.asarray(z, float))
        shape = x.shape
        x = x.ravel()
        z = z.ravel()
        keys = ("eta", "u", "w", "ut", "wt", "ax", "az", "ux", "uz", "wx", "wz")
        out = {key: np.empty(x.size) for key in keys}
        # Chunk the points so the (points x modes) work arrays stay ~<64 MB.
        chunk = max(1, int(4e6 // max(self.k.size, 1)))
        for s in range(0, x.size, chunk):
            part = self._kinematics_flat(x[s:s + chunk], z[s:s + chunk], stretching)
            for key in keys:
                out[key][s:s + chunk] = part[key]
        return {key: v.reshape(shape) for key, v in out.items()}

    def _kinematics_flat(self, x, z, stretching):
        k, om, d = self.k, self.omega, self.depth
        E = np.exp(1j * np.multiply.outer(x, k))
        eta = np.real(E @ self.amp)

        if stretching == "wheeler":
            zs = (z - eta) if np.isinf(d) else d * (z - eta) / (d + eta)
        elif stretching == "linear":
            zs = np.minimum(z, 0.0)
        else:
            zs = z
        Ch, Sh = _depth_functions(k[None, :], zs[:, None], d)
        ChE, ShE = Ch * E, Sh * E

        # Mode weights: u, w, gradients, and their local time derivatives.
        u = np.real(ChE @ (om * self.amp))
        w = np.real(ShE @ (-1j * om * self.amp))
        ut = np.real(ChE @ (om * self.amp_dot))
        wt = np.real(ShE @ (-1j * om * self.amp_dot))
        ux = np.real(ChE @ (1j * k * om * self.amp))
        uz = np.real(ShE @ (k * om * self.amp))
        if stretching == "linear":
            # Above z = 0: f(z) = f(0) + z f'(0), with dCh/dz = k Sh, dSh/dz = k Ch.
            e = np.maximum(z, 0.0)
            u = u + e * uz
            w = w + e * np.real(ChE @ (-1j * k * om * self.amp))
            ut = ut + e * np.real(ShE @ (k * om * self.amp_dot))
            wt = wt + e * np.real(ChE @ (-1j * k * om * self.amp_dot))
        wx, wz = uz, -ux

        out = dict(eta=eta, u=u, w=w, ut=ut, wt=wt,
                   ax=ut + u * ux + w * uz, az=wt + u * wx + w * wz,
                   ux=ux, uz=uz, wx=wx, wz=wz)
        outside = (z > eta + 1e-12) | (z < -d)
        for key, v in out.items():
            if key != "eta":
                v[outside] = np.nan
        return out


class LinearWave:
    """A regular Airy wave, eta = (H/2) cos(k x - omega t)."""

    def __init__(self, H, T, depth=np.inf, g=G):
        self.H, self.T, self.depth, self.g = H, T, depth, g
        self.omega = 2 * np.pi / T
        self.k = float(wavenumber(self.omega, depth, g))

    def field(self, t):
        a = 0.5 * self.H * np.exp(-1j * self.omega * t)
        return SpectralWaveField([self.k], [a], depth=self.depth, g=self.g)

    def eta(self, x, t=0.0):
        return self.field(t).eta(x)

    def kinematics(self, x, z, t=0.0, stretching="wheeler"):
        return self.field(t).kinematics(x, z, stretching)


class NLSWave:
    """Physical wave field reconstructed from an NLS simulation.

    Parameters
    ----------
    result : nls.solver.Result
        Normalised NLS run (alpha=1/2, beta=1).
    deep_water : nls.physical.DeepWater
        Scaling to physical units.
    depth : float
        Depth used for the vertical structure of the kinematics. The NLS model
        itself is deep water, so this should satisfy k0 d >~ 3.
    phase : float
        Carrier phase offset (see :meth:`DeepWater.surface`).

    ``eta`` and ``kinematics`` use the stored snapshot nearest to the
    requested physical time t, so run the solver with output times at the
    resolution you need (see :meth:`physical_times`).
    """

    def __init__(self, result, deep_water, depth=np.inf, phase=0.0):
        self.result = result
        self.dw = deep_water
        self.depth = float(depth)
        self.phase = float(phase)
        self.times = self.dw.to_physical_t(result.t)

    def physical_times(self):
        return self.times

    def snapshot_index(self, t):
        return int(np.argmin(np.abs(self.times - t)))

    def field(self, t):
        """SpectralWaveField at the stored snapshot nearest to t (s)."""
        dw = self.dw
        i = self.snapshot_index(t)
        t_i = self.times[i]
        A = dw.envelope(self.result.psi[i])
        N = A.size
        dx = (self.result.x[1] - self.result.x[0]) / dw.x_scale
        x0 = dw.to_physical_x(self.result.x[0], t_i)
        kappa = 2 * np.pi * np.fft.fftfreq(N, d=dx)

        A_hat = np.fft.fft(A) / N
        # Fixed-frame envelope equation:
        #   A_t = -cg A_x - i p A_xx - i q |A|^2 A,  p = w0/(8k0^2), q = w0 k0^2/2
        p = dw.w0 / (8 * dw.k0**2)
        q = 0.5 * dw.w0 * dw.k0**2
        A_hat_t = (-1j * dw.cg * kappa + 1j * p * kappa**2) * A_hat \
            - 1j * q * np.fft.fft(np.abs(A) ** 2 * A) / N

        carrier = np.exp(-1j * kappa * x0) * np.exp(1j * (self.phase - dw.w0 * t_i))
        amp = A_hat * carrier
        amp_dot = A_hat_t * carrier - 1j * dw.w0 * amp
        k = dw.k0 + kappa
        # Short envelopes (steep breathers) leak into k < 0. Re[a e^{ikx}] =
        # Re[conj(a) e^{i|k|x}], so fold those modes onto |k|: eta and eta_t
        # stay exact, but they are then treated as travelling in +x.
        neg = k < 0
        k = np.abs(k)
        amp = np.where(neg, np.conj(amp), amp)
        amp_dot = np.where(neg, np.conj(amp_dot), amp_dot)
        keep = k > 0
        f = SpectralWaveField(k[keep], amp[keep], amp_dot[keep], self.depth, dw.g)
        return f.truncated()

    def negative_wavenumber_fraction(self, t=0.0):
        """Fraction of envelope energy at k < 0 -- a check on the narrow-band
        assumption (should be small, say < 1e-3)."""
        i = self.snapshot_index(t)
        A_hat = np.fft.fft(self.result.psi[i])
        K = 2 * np.pi * np.fft.fftfreq(A_hat.size, d=self.result.x[1] - self.result.x[0])
        # A = a0 conj(psi), so envelope wavenumber kappa corresponds to -K / x_scale.
        kappa = -K * self.dw.x_scale
        E = np.abs(A_hat) ** 2
        return float(E[self.dw.k0 + kappa < 0].sum() / E.sum())

    def eta(self, x, t=0.0):
        return self.field(t).eta(x)

    def kinematics(self, x, z, t=0.0, stretching="wheeler"):
        return self.field(t).kinematics(x, z, stretching)
