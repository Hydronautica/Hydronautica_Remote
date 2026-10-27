"""Steady nonlinear regular waves by Fenton's Fourier stream function method.

Rienecker & Fenton (1981), Fenton (1988). In a frame moving with the wave
(X = x - c t, Y = z + d measured from the bed) the stream function is

    psi = -U Y + sum_j B_j sinh(j k Y) / cosh(j k Y_ref) cos(j k X),

and the unknowns k, U, Q, R, the surface nodes eta_m and the coefficients B_j
are found by Newton iteration so that the free surface is a streamline
(psi = -Q) with constant Bernoulli head R. Everything is solved in units
where g = d = 1 and scaled back at the end.

Unlike Fenton, who normalises by cosh(j k d), the basis here is normalised
near the crest (Y_ref = d + 0.6 H). In deep water the high harmonics are
otherwise ~exp(j k (eta_crest - d)) times smaller in B_j than their effect at
the crest, which makes the Newton system badly conditioned for steep waves.
"""

import numpy as np

from .kinematics import G, wavenumber


def _ratio_functions(jk, Y, Yref):
    """C = cosh(jk Y)/cosh(jk Yref), S = sinh(jk Y)/cosh(jk Yref), overflow-safe."""
    e = np.exp(jk * (Y - Yref))
    eb = np.exp(-2 * jk * Y)
    den = 1 + np.exp(-2 * jk * Yref)
    return e * (1 + eb) / den, e * (1 - eb) / den


class StreamFunctionWave:
    """Fully nonlinear steady wave of height H and period T in depth d.

    Parameters
    ----------
    H, T, depth : float
        Wave height (m), period (s) and water depth (m).
    N : int or None
        Number of Fourier modes. None (default) tries several values and keeps
        the one that best satisfies the surface conditions between the
        collocation nodes; see ``self.N`` and ``self.residual`` afterwards.
        In deep water too many modes is as bad as too few, because the high
        harmonics are exponentially weak in the troughs.
    current : float
        Current magnitude (m/s) for the chosen criterion.
    criterion : {"eulerian", "stokes"}
        Whether ``current`` is the time-mean Eulerian velocity (no current:
        mean horizontal velocity at a fixed point below the troughs is zero)
        or the depth-averaged mass-transport velocity.
    height_steps : int, optional
        Number of steps used to ramp the height up from linear theory.
    """

    N_CANDIDATES = (16, 24, 32, 40)

    def __init__(self, H, T, depth, N=None, current=0.0, criterion="eulerian",
                 height_steps=None, g=G, tol=1e-12, max_iter=40):
        if criterion not in ("eulerian", "stokes"):
            raise ValueError("criterion must be 'eulerian' or 'stokes'")
        self.H, self.T, self.depth, self.N, self.g = H, T, depth, N, g
        self.current, self.criterion = current, criterion
        self._Hn = H / depth
        self._Tn = T * np.sqrt(g / depth)
        self._cn = current / np.sqrt(g * depth)
        self._Yref = 1.0 + 0.6 * self._Hn
        if N is not None:
            self._solve(height_steps, tol, max_iter)
            self.residual = max(self.surface_residuals())
            return
        best, err = None, None
        for n in self.N_CANDIDATES:
            self.N = n
            try:
                self._solve(height_steps, tol, max_iter)
            except RuntimeError as e:
                err = e
                continue
            res = max(self.surface_residuals())
            if best is None or res < best[0]:
                best = (res, n, dict(self.__dict__))
            if res < 1e-10:
                break
        if best is None:
            raise err
        self.__dict__.update(best[2])
        self.residual = best[0]

    # --- nondimensional problem -------------------------------------------
    def _unpack(self, v):
        N = self.N
        return v[0], v[1], v[2], v[3], v[4:5 + N], v[5 + N:]

    def _residual(self, v, Hn, jacobian=False):
        N = self.N
        k, U, Q, R, eta, B = self._unpack(v)
        j = np.arange(1, N + 1)
        a = j * k
        # Node positions jkX_m = j m pi / N do not depend on k.
        phase = np.outer(np.arange(N + 1), j) * np.pi / N
        cos, sin = np.cos(phase), np.sin(phase)
        C, S = _ratio_functions(a[None, :], eta[:, None], self._Yref)
        BC, BS = B * C * cos, B * S * sin
        psi = -U * eta + (B * S * cos).sum(1) + Q
        u = -U + (a * BC).sum(1)
        w = (a * BS).sum(1)
        bern = 0.5 * (u**2 + w**2) + eta - R
        mean = (0.5 * eta[0] + eta[1:-1].sum() + 0.5 * eta[-1]) / N - 1.0
        height = eta[0] - eta[-1] - Hn
        c = 2 * np.pi / (k * self._Tn)
        cur_var = U if self.criterion == "eulerian" else Q
        r = np.concatenate([psi, bern, [mean, height, cur_var - (c - self._cn)]])
        if not jacobian:
            return r

        # Analytic Jacobian. dS/dY = aC, dC/dY = aS, and with a = jk,
        # dS/dk = j (Y C - S Yr tanh(a Yr)), dC/dk = j (Y S - C Yr tanh(a Yr)).
        M = N + 1
        n = v.size
        Jm = np.zeros((n, n))
        Y = eta[:, None]
        th = self._Yref * np.tanh(a * self._Yref)
        dS_dk = j * (Y * C - S * th)
        dC_dk = j * (Y * S - C * th)
        ie, ib = slice(4, 4 + M), slice(4 + M, 4 + M + N)
        rows_psi, rows_b = slice(0, M), slice(M, 2 * M)
        # psi equations
        Jm[rows_psi, 0] = (B * cos * dS_dk).sum(1)
        Jm[rows_psi, 1] = -eta
        Jm[rows_psi, 2] = 1.0
        Jm[rows_psi, ie] = np.diag(u)              # d psi / d eta = u
        Jm[rows_psi, ib] = S * cos
        # velocities at the nodes
        du_dk = (B * cos * (j * C + a * dC_dk)).sum(1)
        dw_dk = (B * sin * (j * S + a * dS_dk)).sum(1)
        du_deta = (a**2 * B * S * cos).sum(1)
        dw_deta = (a**2 * B * C * sin).sum(1)
        du_dB = a * C * cos
        dw_dB = a * S * sin
        # Bernoulli equations
        Jm[rows_b, 0] = u * du_dk + w * dw_dk
        Jm[rows_b, 1] = -u
        Jm[rows_b, 3] = -1.0
        Jm[rows_b, ie] = np.diag(u * du_deta + w * dw_deta + 1.0)
        Jm[rows_b, ib] = u[:, None] * du_dB + w[:, None] * dw_dB
        # mean level, height, current
        wts = np.full(M, 1.0 / N)
        wts[[0, -1]] = 0.5 / N
        Jm[2 * M, ie] = wts
        Jm[2 * M + 1, 4] = 1.0
        Jm[2 * M + 1, 4 + N] = -1.0
        Jm[2 * M + 2, 0] = c / k
        Jm[2 * M + 2, 1 if self.criterion == "eulerian" else 2] = 1.0
        return r, Jm

    def _linear_guess(self, Hn):
        N = self.N
        om = 2 * np.pi / self._Tn
        k = float(wavenumber(om, 1.0, 1.0))
        c = om / k
        U = c - self._cn
        m = np.arange(N + 1)
        eta = 1 + 0.5 * Hn * np.cos(m * np.pi / N)
        B = np.zeros(N)
        # linear theory: B_1 cosh(k)/... -> renormalise from cosh(k) to cosh(k Yref)
        B[0] = 0.5 * Hn * c / np.tanh(k) * np.cosh(k * self._Yref) / np.cosh(k)
        return np.concatenate([[k, U, U, 0.5 * U**2 + 1], eta, B])

    def _newton(self, v, Hn, tol, max_iter):
        """Damped Newton with a column-scaled analytic Jacobian.

        Returns the converged vector, or None if the iteration fails.
        """
        n = v.size
        r = self._residual(v, Hn)
        norm = np.max(np.abs(r))
        for _ in range(max_iter):
            if not np.isfinite(norm):
                return None
            if norm < tol:
                return v
            _, J = self._residual(v, Hn, jacobian=True)
            # Columns for the high harmonics are tiny; equilibrate them.
            scale = np.linalg.norm(J, axis=0)
            scale[scale == 0] = 1.0
            try:
                dv = np.linalg.solve(J / scale, -r) / scale
            except np.linalg.LinAlgError:
                return None
            # Backtracking line search on the max residual.
            step = 1.0
            while step > 1e-3:
                v_new = v + step * dv
                r_new = self._residual(v_new, Hn)
                norm_new = np.max(np.abs(r_new))
                if np.isfinite(norm_new) and norm_new < norm:
                    break
                step *= 0.5
            else:
                # No decrease possible: accept if already near the noise floor.
                return v if norm < 1e3 * tol else None
            v, r, norm = v_new, r_new, norm_new
        return v if norm < 1e3 * tol else None

    def _solve(self, height_steps, tol, max_iter):
        Hn = self._Hn
        if height_steps is None:
            lam = 2 * np.pi / wavenumber(2 * np.pi / self._Tn, 1.0, 1.0)
            height_steps = max(1, int(np.ceil(10 * Hn / _williams_fit(lam, 1.0))))
        # Continuation in height with step halving on failure.
        dH = Hn / height_steps
        H_done = 0.0
        hist = []   # (height, solution) of the last converged steps
        v = None
        while H_done < Hn * (1 - 1e-12):
            H_try = min(Hn, H_done + dH)
            if len(hist) >= 2:
                (H0, v0), (H1, v1) = hist[-2], hist[-1]
                guess = v1 + (v1 - v0) * (H_try - H1) / (H1 - H0)
            elif hist:
                guess = hist[-1][1].copy()
                guess[4:5 + self.N] = 1 + (guess[4:5 + self.N] - 1) * H_try / hist[-1][0]
                guess[5 + self.N:] *= H_try / hist[-1][0]
            else:
                guess = self._linear_guess(H_try)
            v = self._newton(guess, H_try, tol, max_iter)
            if v is None:
                dH *= 0.5
                if dH < Hn * 1e-4:
                    raise RuntimeError(
                        f"stream function iteration failed at H = {H_try * self.depth:.3f} m "
                        f"(target {self.H} m); the wave may be beyond the breaking "
                        "limit, or try more Fourier modes N")
                continue
            hist = (hist + [(H_try, v)])[-2:]
            H_done = H_try
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
        C, S = _ratio_functions(jk, Y[..., None], self._Yref)
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
        C, S = _ratio_functions(jk[None, :], eta[:, None], self._Yref)
        cos, sin = np.cos(np.outer(X, jk)), np.sin(np.outer(X, jk))
        B = self._B
        psi = -self._U * eta + (B * S * cos).sum(1) + self._Q
        u = -self._U + (jk * B * C * cos).sum(1)
        w = (jk * B * S * sin).sum(1)
        bern = 0.5 * (u**2 + w**2) + eta - self._R
        return float(np.abs(psi).max()), float(np.abs(bern).max())


def _williams_fit(lam, depth):
    """Fenton's (1990) fit to Williams' (1981) highest waves, H_max(lambda)."""
    x = lam / depth
    num = 0.141063 * x + 0.0095721 * x**2 + 0.0077829 * x**3
    den = 1 + 0.0788340 * x + 0.0317567 * x**2 + 0.0093407 * x**3
    return num / den * depth


def breaking_height(T, depth, g=G, N=32):
    """Approximate height (m) of the highest steady wave of period T (s).

    The limiting height for a given *wavelength* is known accurately
    (Williams 1981, as fitted by Fenton 1990), but at fixed period the
    wavelength itself grows with height. It is estimated here by solving the
    stream function problem at 60% and 80% of the limit, fitting
    lambda(H) = lambda_lin + b H^2 + c H^4 and iterating to self-consistency.
    In deep water this gives H_max ~ 0.0268 g T^2.
    """
    lam_lin = 2 * np.pi / wavenumber(2 * np.pi / T, depth, g)
    Hm = _williams_fit(lam_lin, depth)
    pts = []
    for frac in (0.6, 0.8):
        H = frac * Hm
        pts.append((H, StreamFunctionWave(H, T, depth, N=N, g=g).wavelength))
    (H1, l1), (H2, l2) = pts
    A = np.array([[H1**2, H1**4], [H2**2, H2**4]])
    b, c = np.linalg.solve(A, [l1 - lam_lin, l2 - lam_lin])
    for _ in range(100):
        Hm_new = _williams_fit(lam_lin + b * Hm**2 + c * Hm**4, depth)
        if abs(Hm_new - Hm) < 1e-10 * Hm:
            break
        Hm = Hm_new
    return float(Hm)
