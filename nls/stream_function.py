"""Steady nonlinear regular waves by Fenton's Fourier stream function method.

Rienecker & Fenton (1981), Fenton (1988). In a frame moving with the wave
(X = x - c t, Y = z + d measured from the bed) the stream function is

    psi = -U Y + sum_j B_j sinh(j k Y) / cosh(j k d) cos(j k X),

and the unknowns k, U, Q, R, the surface nodes eta_m and the coefficients B_j
are found by Newton iteration so that the free surface is a streamline
(psi = -Q) with constant Bernoulli head R. Everything is solved in units
where g = d = 1 and scaled back at the end.
"""

import numpy as np

from .kinematics import G, wavenumber


def _ratio_functions(jk, Y):
    """C = cosh(jk Y)/cosh(jk), S = sinh(jk Y)/cosh(jk) with d = 1, overflow-safe."""
    e = np.exp(jk * (Y - 1.0))
    eb = np.exp(-2 * jk * Y)
    den = 1 + np.exp(-2 * jk)
    return e * (1 + eb) / den, e * (1 - eb) / den


class StreamFunctionWave:
    """Fully nonlinear steady wave of height H and period T in depth d.

    Parameters
    ----------
    H, T, depth : float
        Wave height (m), period (s) and water depth (m).
    N : int
        Number of Fourier modes (more for steeper / shallower waves).
    current : float
        Current magnitude (m/s) for the chosen criterion.
    criterion : {"eulerian", "stokes"}
        Whether ``current`` is the time-mean Eulerian velocity (no current:
        mean horizontal velocity at a fixed point below the troughs is zero)
        or the depth-averaged mass-transport velocity.
    height_steps : int, optional
        Number of steps used to ramp the height up from linear theory.
    """

    def __init__(self, H, T, depth, N=24, current=0.0, criterion="eulerian",
                 height_steps=None, g=G, tol=1e-12, max_iter=60):
        if criterion not in ("eulerian", "stokes"):
            raise ValueError("criterion must be 'eulerian' or 'stokes'")
        self.H, self.T, self.depth, self.N, self.g = H, T, depth, N, g
        self.current, self.criterion = current, criterion
        self._Hn = H / depth
        self._Tn = T * np.sqrt(g / depth)
        self._cn = current / np.sqrt(g * depth)
        self._solve(height_steps, tol, max_iter)

    # --- nondimensional problem -------------------------------------------
    def _unpack(self, v):
        N = self.N
        return v[0], v[1], v[2], v[3], v[4:5 + N], v[5 + N:]

    def _residual(self, v, Hn):
        N = self.N
        k, U, Q, R, eta, B = self._unpack(v)
        m = np.arange(N + 1)
        j = np.arange(1, N + 1)
        jk = j * k
        X = m * np.pi / (k * N)
        C, S = _ratio_functions(jk[None, :], eta[:, None])
        cos = np.cos(np.outer(X, jk))
        sin = np.sin(np.outer(X, jk))
        psi = -U * eta + (B * S * cos).sum(1) + Q
        u = -U + (jk * B * C * cos).sum(1)
        w = (jk * B * S * sin).sum(1)
        bern = 0.5 * (u**2 + w**2) + eta - R
        mean = (0.5 * eta[0] + eta[1:-1].sum() + 0.5 * eta[-1]) / N - 1.0
        height = eta[0] - eta[-1] - Hn
        c = 2 * np.pi / (k * self._Tn)
        if self.criterion == "eulerian":
            cur = U - (c - self._cn)
        else:
            cur = Q - (c - self._cn)
        return np.concatenate([psi, bern, [mean, height, cur]])

    def _linear_guess(self, Hn):
        N = self.N
        om = 2 * np.pi / self._Tn
        k = float(wavenumber(om, 1.0, 1.0))
        c = om / k
        U = c - self._cn
        m = np.arange(N + 1)
        eta = 1 + 0.5 * Hn * np.cos(m * np.pi / N)
        B = np.zeros(N)
        B[0] = 0.5 * Hn * c / np.tanh(k)
        return np.concatenate([[k, U, U, 0.5 * U**2 + 1], eta, B])

    def _newton(self, v, Hn, tol, max_iter):
        n = v.size
        for _ in range(max_iter):
            r = self._residual(v, Hn)
            J = np.empty((n, n))
            for i in range(n):
                h = 1e-7 * max(1.0, abs(v[i]))
                vp = v.copy()
                vp[i] += h
                J[:, i] = (self._residual(vp, Hn) - r) / h
            dv = np.linalg.solve(J, -r)
            v = v + dv
            if np.max(np.abs(dv)) < tol and np.max(np.abs(r)) < 1e-9:
                return v
        if np.max(np.abs(self._residual(v, Hn))) > 1e-8:
            raise RuntimeError("stream function iteration did not converge; "
                               "try more height_steps or modes, or the wave "
                               "may be beyond the breaking limit")
        return v

    def _solve(self, height_steps, tol, max_iter):
        Hn = self._Hn
        k_lin = wavenumber(2 * np.pi / self._Tn, 1.0, 1.0)
        H_break = 0.141063 * 2 * np.pi / k_lin * np.tanh(k_lin)  # approx. limit
        if height_steps is None:
            height_steps = max(1, int(np.ceil(10 * Hn / H_break)))
        prev = []
        v = self._linear_guess(Hn / height_steps)
        for s in range(1, height_steps + 1):
            Hs = Hn * s / height_steps
            if len(prev) >= 2:
                v = 2 * prev[-1] - prev[-2]  # linear extrapolation in height
            v = self._newton(v, Hs, tol, max_iter)
            prev.append(v)
        self._v = v

        k, U, Q, R, eta, B = self._unpack(v)
        d, g = self.depth, self.g
        vs = np.sqrt(g * d)
        self.k = k / d
        self.wavelength = 2 * np.pi / self.k
        self.c = 2 * np.pi / (k * self._Tn) * vs
        self._U, self._Q, self._R = U, Q, R
        self._B, self._eta_nodes, self._kn = B, eta, k
        # Cosine series of the surface from the nodes (discrete cosine transform).
        N = self.N
        m = np.arange(N + 1)
        wts = np.ones(N + 1)
        wts[[0, -1]] = 0.5
        jj = np.arange(N + 1)
        E = (2.0 / N) * (wts * eta * np.cos(np.outer(jj, m) * np.pi / N)).sum(1)
        E[[0, -1]] *= 0.5
        self._E = E
        self.crest = (eta[0] - 1) * d
        self.trough = (eta[-1] - 1) * d
        self.eulerian_current = (self.c / vs - U) * vs
        self.stokes_drift = (self.c / vs - Q) * vs

    # --- dimensional interface --------------------------------------------
    def eta(self, x, t=0.0):
        """Surface elevation above still water level (m)."""
        X = (np.asarray(x, float) - self.c * t) / self.depth
        jk = np.arange(self.N + 1) * self._kn
        return (np.cos(np.multiply.outer(X, jk)) @ self._E - 1.0) * self.depth

    def kinematics(self, x, z, t=0.0, stretching=None):
        """Exact kinematics at (x, z, t); ``stretching`` is ignored.

        Returns the same keys as :meth:`SpectralWaveField.kinematics`.
        """
        x, z = np.broadcast_arrays(np.asarray(x, float), np.asarray(z, float))
        d, g = self.depth, self.g
        vs = np.sqrt(g * d)
        eta = self.eta(x, t)
        X = (x - self.c * t) / d
        Y = (z + d) / d
        jk = np.arange(1, self.N + 1) * self._kn
        C, S = _ratio_functions(jk, Y[..., None])
        cos = np.cos(X[..., None] * jk)
        sin = np.sin(X[..., None] * jk)
        B = self._B
        cn = self.c / vs
        U = -self._U + (jk * B * C * cos).sum(-1)   # moving-frame velocities
        W = (jk * B * S * sin).sum(-1)
        uX = -(jk**2 * B * C * sin).sum(-1)
        uY = (jk**2 * B * S * cos).sum(-1)
        wX = uY
        wY = -uX
        u = cn + U
        # Steady in the moving frame: d/dt = -c d/dX, and D/Dt = U d/dX + W d/dY.
        ut = -cn * uX
        wt = -cn * wX
        ax = U * uX + W * uY
        az = U * wX + W * wY
        out = dict(eta=eta, u=u * vs, w=W * vs, ut=ut * g, wt=wt * g,
                   ax=ax * g, az=az * g,
                   ux=uX * vs / d, uz=uY * vs / d, wx=wX * vs / d, wz=wY * vs / d)
        outside = (z > eta + 1e-12) | (z < -d)
        for key in out:
            if key != "eta":
                out[key] = np.where(outside, np.nan, out[key])
        return out

    def surface_residuals(self, n=200):
        """Max violation of the free-surface conditions between the nodes.

        Returns (kinematic, dynamic) residuals, nondimensional.
        """
        X = np.linspace(0, np.pi / self._kn, n)
        eta = self.eta(X * self.depth) / self.depth + 1
        jk = np.arange(1, self.N + 1) * self._kn
        C, S = _ratio_functions(jk[None, :], eta[:, None])
        cos, sin = np.cos(np.outer(X, jk)), np.sin(np.outer(X, jk))
        B = self._B
        psi = -self._U * eta + (B * S * cos).sum(1) + self._Q
        u = -self._U + (jk * B * C * cos).sum(1)
        w = (jk * B * S * sin).sum(1)
        bern = 0.5 * (u**2 + w**2) + eta - self._R
        return float(np.abs(psi).max()), float(np.abs(bern).max())
