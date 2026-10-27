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


def wave_height(t, eta, window=None):
    """Height of the largest wave in a record: crest minus deeper adjacent trough.

    By default the troughs are the lowest points between the highest crest
    and the neighbouring zero crossings on either side. If ``window`` is given
    (same units as t, e.g. 0.75 of the period), the trough is instead the
    lowest point within +-window of the crest; this does not jump when small
    ripples near the trough cross zero (e.g. in Dysthe wave groups).
    """
    t = np.asarray(t)
    eta = np.asarray(eta)
    i = int(np.argmax(eta))
    if window is not None:
        # Parabolic refinement of the sampled extrema removes the sampling
        # jitter (~(pi dt/T)^2 / 2 of the amplitude) from crest and trough.
        def vertex(j):
            if 0 < j < eta.size - 1:
                a, b, c = eta[j - 1], eta[j], eta[j + 1]
                den = a - 2 * b + c
                if den != 0:
                    return b - 0.125 * (a - c) ** 2 / den
            return eta[j]
        near = np.flatnonzero(np.abs(t - t[i]) <= window)
        j = near[int(np.argmin(eta[near]))]
        return float(vertex(i) - vertex(j))
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

    def potential(self, x, z):
        """Velocity potential phi at points (x, z), evaluated without stretching."""
        x, z = np.broadcast_arrays(np.asarray(x, float), np.asarray(z, float))
        Ch, _ = _depth_functions(self.k, z[..., None], self.depth)
        # phi = Re sum -i (g / w) a cosh k(z+d)/cosh kd e^{ikx};  g/w * tanh = w/k
        C = Ch * (np.tanh(self.k * self.depth) if np.isfinite(self.depth) else 1.0)
        E = np.exp(1j * x[..., None] * self.k)
        return np.real((C * E) @ (-1j * self.g / self.omega * self.amp))

    def kinematics(self, x, z, stretching="wheeler", surface=None):
        """Velocities and accelerations at points (x, z).

        stretching controls how the modes are evaluated in the crest region:

        * ``"wheeler"``: z is mapped to z' = d (z - eta) / (d + eta) (or
          z - eta in deep water), so the surface maps to z' = 0;
        * ``"linear"``: kinematics above z = 0 are extrapolated linearly from
          their value and vertical gradient at z = 0;
        * ``"none"``: plain linear theory evaluated up to the surface (over-
          predicts crest kinematics for broad spectra).

        ``surface`` optionally overrides the surface elevation at the points
        (used by the second-order fields, whose surface differs from this
        field's own eta) for masking and Wheeler stretching.

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
        if surface is not None:
            surface = np.broadcast_to(np.asarray(surface, float), shape).ravel()
        out = {key: np.empty(x.size) for key in KIN_KEYS}
        # Chunk the points so the (points x modes) work arrays stay ~<64 MB.
        chunk = max(1, int(4e6 // max(self.k.size, 1)))
        for s in range(0, x.size, chunk):
            sl = slice(s, s + chunk)
            part = self._kinematics_flat(x[sl], z[sl], stretching,
                                         None if surface is None else surface[sl])
            for key in KIN_KEYS:
                out[key][sl] = part[key]
        return {key: v.reshape(shape) for key, v in out.items()}

    def _kinematics_flat(self, x, z, stretching, surface=None):
        k, om, d = self.k, self.omega, self.depth
        E = np.exp(1j * np.multiply.outer(x, k))
        eta = np.real(E @ self.amp) if surface is None else surface

        zs = stretched_z(z, eta, d, stretching)
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
        return finish_kinematics(dict(eta=eta, u=u, w=w, ut=ut, wt=wt,
                                      ux=ux, uz=uz, wx=wx, wz=wz), z, d)


KIN_KEYS = ("eta", "u", "w", "ut", "wt", "ax", "az", "ux", "uz", "wx", "wz")


def stretched_z(z, eta, depth, stretching):
    """Coordinate at which modes are evaluated for each stretching option.

    For ``"linear"`` this is min(z, 0); the caller adds the extrapolation.
    """
    if stretching == "wheeler":
        return (z - eta) if np.isinf(depth) else depth * (z - eta) / (depth + eta)
    if stretching == "linear":
        return np.minimum(z, 0.0)
    return z


def finish_kinematics(out, z, depth):
    """Add total accelerations and blank points above the surface / below the bed."""
    out["ax"] = out["ut"] + out["u"] * out["ux"] + out["w"] * out["uz"]
    out["az"] = out["wt"] + out["u"] * out["wx"] + out["w"] * out["wz"]
    outside = (z > out["eta"] + 1e-12) | (z < -depth)
    for key, v in out.items():
        if key != "eta":
            v = np.array(v, dtype=float)
            v[outside] = np.nan
            out[key] = v
    return out


def add_kinematics(first, second, z, depth):
    """Sum first- and second-order contributions (eta taken from ``second``)."""
    out = {key: first[key] + second[key]
           for key in ("u", "w", "ut", "wt", "ux", "uz", "wx", "wz")}
    out["eta"] = second["eta"]
    return finish_kinematics(out, z, depth)


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


class EnvelopeSecondOrderField:
    """Deep-water second-order reconstruction of a narrow-band envelope field.

    The surface is eta = eta1 + eta2 + eta_mean with (exact second-order
    deep-water kernels, written in envelope form)

        eta2     = Re[(k0/2 A^2 - i/2 A A_x) e^{2 i theta}]
        eta_mean = -1/4 F^-1[|kappa| F(|A|^2)]           (set-down)

    and the kinematics add the wave-induced mean flow of Dysthe (1979),
    phi_mean = F^-1[(i w0 / 2) sign(kappa) F(|A|^2)] e^{|kappa| z}. The
    second-order sum-frequency potential vanishes in deep water.

    ``first`` is the linear :class:`SpectralWaveField` of the same envelope.
    The envelope A (without carrier) and its fixed-frame time derivative A_t
    are given on the uniform grid x_j = x0 + j dx at time t.
    """

    def __init__(self, first, x0, dx, A, A_t, k0, w0, phase, t, depth=np.inf):
        self.first = first
        self.depth = float(depth)
        self.k0, self.w0, self.x0 = k0, w0, x0
        self.theta0 = phase - w0 * t      # carrier phase at x = 0
        N = A.size
        kappa = 2 * np.pi * np.fft.fftfreq(N, d=dx)
        A_x = np.fft.ifft(1j * kappa * np.fft.fft(A))
        A2 = 0.5 * k0 * A**2 - 0.5j * A * A_x
        I_hat = np.fft.fft(np.abs(A) ** 2)
        I_t_hat = np.fft.fft(2 * np.real(np.conj(A) * A_t))
        S2 = np.fft.fft(A2) / N
        Sm = -0.25 * np.abs(kappa) * I_hat / N
        Cm = 0.5j * w0 * np.sign(kappa) * I_hat / N
        Cm_t = 0.5j * w0 * np.sign(kappa) * I_t_hat / N
        # Keep only the significant modes for evaluation at arbitrary points.
        def keep(*arrays):
            m = np.zeros(N, bool)
            for a in arrays:
                m |= np.abs(a) > 1e-12 * max(np.abs(a).max(), 1e-300)
            return m
        k2 = keep(S2)
        self._kap2, self._S2 = kappa[k2], S2[k2]
        km = keep(Sm, Cm, Cm_t)
        self._kapm, self._Sm, self._Cm, self._Cm_t = kappa[km], Sm[km], Cm[km], Cm_t[km]

    def _phases(self, x, kap):
        return np.exp(1j * np.multiply.outer(x - self.x0, kap))

    def eta_second(self, x):
        """Second-order part of the surface (bound harmonic + set-down)."""
        x = np.asarray(x, float)
        carrier = np.exp(2j * (self.k0 * x + self.theta0))
        eta2 = np.real(carrier * (self._phases(x, self._kap2) @ self._S2))
        return eta2 + np.real(self._phases(x, self._kapm) @ self._Sm)

    def eta(self, x):
        x = np.asarray(x, float)
        return self.first.eta(x) + self.eta_second(x)

    def kinematics(self, x, z, stretching="linear"):
        """Kinematics to second order.

        ``"linear"`` (default) is the consistent second-order choice: the
        first-order field is Taylor-extrapolated above z = 0 and the
        second-order mean flow is evaluated at min(z, 0). ``"wheeler"`` and
        ``"none"`` are available for comparison.
        """
        if stretching not in STRETCHING:
            raise ValueError(f"stretching must be one of {STRETCHING}")
        x, z = np.broadcast_arrays(np.asarray(x, float), np.asarray(z, float))
        shape = x.shape
        x, z = x.ravel(), z.ravel()
        out = {key: np.empty(x.size) for key in KIN_KEYS}
        chunk = max(1, int(2e6 // max(self._kapm.size + self._kap2.size, 1)))
        for s in range(0, x.size, chunk):
            sl = slice(s, s + chunk)
            eta = self.eta(x[sl])
            k1 = self.first._kinematics_flat(x[sl], z[sl], stretching, surface=eta)
            zs = stretched_z(z[sl], eta, self.depth, stretching)
            kap = self._kapm
            P = self._phases(x[sl], kap) * np.exp(np.abs(kap) * zs[:, None])
            two = dict(eta=eta,
                       u=np.real(P @ (1j * kap * self._Cm)),
                       w=np.real(P @ (np.abs(kap) * self._Cm)),
                       ut=np.real(P @ (1j * kap * self._Cm_t)),
                       wt=np.real(P @ (np.abs(kap) * self._Cm_t)),
                       ux=np.real(P @ (-kap**2 * self._Cm)),
                       uz=np.real(P @ (1j * kap * np.abs(kap) * self._Cm)))
            two["wx"], two["wz"] = two["uz"], -two["ux"]
            part = add_kinematics(k1, two, z[sl], self.depth)
            for key in KIN_KEYS:
                out[key][sl] = part[key]
        return {key: v.reshape(shape) for key, v in out.items()}


class EnvelopeWave:
    """Base class for physical wave fields reconstructed from envelope runs.

    Subclasses implement ``_envelope(i)`` returning, for stored snapshot i,
    ``(x0, dx, A, A_t, t)``: the physical grid origin and spacing (m), the
    complex envelope A (m, without carrier), its fixed-frame time derivative
    and the physical time (s). The surface is
    eta = Re[A exp(i(k0 x - w0 t + phase))] (+ second-order terms if
    ``order=2``).
    """

    k0 = w0 = None

    def __init__(self, times, depth=np.inf, phase=0.0, order=1, g=G):
        if order not in (1, 2):
            raise ValueError("order must be 1 or 2")
        self.times = np.asarray(times, float)
        self.depth = float(depth)
        self.phase = float(phase)
        self.order = order
        self.g = g

    def physical_times(self):
        return self.times

    def snapshot_index(self, t):
        return int(np.argmin(np.abs(self.times - t)))

    def _linear_field(self, x0, dx, A, A_t, t_i):
        N = A.size
        kappa = 2 * np.pi * np.fft.fftfreq(N, d=dx)
        A_hat = np.fft.fft(A) / N
        A_hat_t = np.fft.fft(A_t) / N
        carrier = np.exp(-1j * kappa * x0) * np.exp(1j * (self.phase - self.w0 * t_i))
        amp = A_hat * carrier
        amp_dot = A_hat_t * carrier - 1j * self.w0 * amp
        k = self.k0 + kappa
        # Short envelopes (steep breathers) leak into k < 0. Re[a e^{ikx}] =
        # Re[conj(a) e^{i|k|x}], so fold those modes onto |k|: eta and eta_t
        # stay exact, but they are then treated as travelling in +x.
        neg = k < 0
        k = np.abs(k)
        amp = np.where(neg, np.conj(amp), amp)
        amp_dot = np.where(neg, np.conj(amp_dot), amp_dot)
        keep = k > 0
        f = SpectralWaveField(k[keep], amp[keep], amp_dot[keep], self.depth, self.g)
        return f.truncated()

    def field(self, t):
        """Wave field at the stored snapshot nearest to t (s): a
        :class:`SpectralWaveField` (order 1) or :class:`EnvelopeSecondOrderField`."""
        x0, dx, A, A_t, t_i = self._envelope(self.snapshot_index(t))
        first = self._linear_field(x0, dx, A, A_t, t_i)
        if self.order == 1:
            return first
        return EnvelopeSecondOrderField(first, x0, dx, A, A_t, self.k0, self.w0,
                                        self.phase, t_i, self.depth)

    def negative_wavenumber_fraction(self, t=0.0):
        """Fraction of envelope energy at k < 0 -- a check on the narrow-band
        assumption (should be small, say < 1e-3)."""
        x0, dx, A, _, _ = self._envelope(self.snapshot_index(t))
        kappa = 2 * np.pi * np.fft.fftfreq(A.size, d=dx)
        E = np.abs(np.fft.fft(A)) ** 2
        return float(E[self.k0 + kappa < 0].sum() / E.sum())

    def eta(self, x, t=0.0):
        return self.field(t).eta(x)

    def kinematics(self, x, z, t=0.0, stretching=None):
        """Kinematics at (x, z, t). stretching defaults to "wheeler" for
        order 1 and the consistent "linear" (Taylor) option for order 2."""
        if stretching is None:
            stretching = "wheeler" if self.order == 1 else "linear"
        return self.field(t).kinematics(x, z, stretching)


class NLSWave(EnvelopeWave):
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
    order : {1, 2}
        1: linear reconstruction. 2: add the bound second harmonic, set-down
        and mean flow (:class:`EnvelopeSecondOrderField`).

    ``eta`` and ``kinematics`` use the stored snapshot nearest to the
    requested physical time t, so run the solver with output times at the
    resolution you need (see :meth:`physical_times`).
    """

    def __init__(self, result, deep_water, depth=np.inf, phase=0.0, order=1):
        super().__init__(deep_water.to_physical_t(result.t), depth, phase, order,
                         deep_water.g)
        self.result = result
        self.dw = deep_water
        self.k0, self.w0 = deep_water.k0, deep_water.w0

    def _envelope(self, i):
        dw = self.dw
        t_i = self.times[i]
        A = dw.envelope(self.result.psi[i])
        N = A.size
        dx = (self.result.x[1] - self.result.x[0]) / dw.x_scale
        x0 = dw.to_physical_x(self.result.x[0], t_i)
        kappa = 2 * np.pi * np.fft.fftfreq(N, d=dx)
        # Fixed-frame envelope equation:
        #   A_t = -cg A_x - i p A_xx - i q |A|^2 A,  p = w0/(8k0^2), q = w0 k0^2/2
        p = dw.w0 / (8 * dw.k0**2)
        q = 0.5 * dw.w0 * dw.k0**2
        A_t = np.fft.ifft((-1j * dw.cg * kappa + 1j * p * kappa**2) * np.fft.fft(A)) \
            - 1j * q * np.abs(A) ** 2 * A
        return x0, dx, A, A_t, t_i
