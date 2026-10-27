"""Second-order (Sharma & Dean 1981) bound waves for unidirectional fields.

Given a linear field eta1 = Re sum_m A_m exp(i k_m x) (amplitudes at the
current time, k_m > 0) in depth d, the second-order solution is a double sum
over ordered pairs of sum- and difference-frequency terms,

    eta2 = Re sum_mn [E+_mn A_m A_n e^{i(k_m+k_n)x} + E-_mn A_m A_n* e^{i(k_m-k_n)x}]
    phi2 = Re sum_mn [C+_mn A_m A_n ch(K+, z) e^{i(k_m+k_n)x} + C-_mn A_m A_n* ch(K-, z) ...]

with ch(K, z) = cosh K(z+d) / cosh Kd, K+- = |k_m +- k_n|. The kernels come
from the second-order free-surface conditions

    phi2_tt + g phi2_z = -d/dt |grad phi1|^2 - eta1 d/dz (phi1_tt + g phi1_z)
    eta2 = -(1/g) [phi2_t + 1/2 |grad phi1|^2 + eta1 phi1_tz]            at z = 0.

In deep water they reduce to E+ = (k_m+k_n)/4, E- = -|k_m-k_n|/4, C+ = 0.
The constant (m = n difference) terms -- a uniform set-down and Bernoulli
constant -- are omitted so that the mean level far from a wave group is zero.
"""

import numpy as np

from .kinematics import (G, KIN_KEYS, STRETCHING, SpectralWaveField,
                         add_kinematics, stretched_z)


def _tanh_kd(k, depth):
    return np.ones_like(k) if np.isinf(depth) else np.tanh(k * depth)


def pair_kernels(k, depth=np.inf, g=G):
    """Second-order transfer functions for wavenumbers k (all > 0).

    Returns a dict of (M, M) arrays: Kp, Km (signed wavenumbers k_m + k_n,
    k_m - k_n), Ep, Em (elevation) and Cp, Cm (potential) kernels.
    """
    k = np.asarray(k, float)
    T = _tanh_kd(k, depth)
    om = np.sqrt(g * k * T)
    b = g / om                       # phi1 amplitude per unit eta amplitude
    U = k * b                        # phi_x at z = 0      (times A)
    W = -1j * b * k * T              # phi_z at z = 0
    Gz = -1j * b * g * k**2 * (1 - T**2)   # d/dz (phi_tt + g phi_z) at z = 0
    V = -om * b * k * T              # phi_tz at z = 0

    Kp = k[:, None] + k[None, :]
    Km = k[:, None] - k[None, :]
    Wp = om[:, None] + om[None, :]
    Wm = om[:, None] - om[None, :]

    Pp = 0.5 * (U[:, None] * U[None, :] + W[:, None] * W[None, :])
    Pm = 0.5 * (U[:, None] * np.conj(U)[None, :] + W[:, None] * np.conj(W)[None, :])
    Fp = 1j * Wp * Pp - 0.5 * Gz[None, :]
    Fm = 1j * Wm * Pm - 0.5 * np.conj(Gz)[None, :]
    # Symmetrise the sum kernel (only the symmetric part survives A_m A_n);
    # the difference kernel must be Hermitian for a real field.
    Fp = 0.5 * (Fp + Fp.T)
    Fm = 0.5 * (Fm + np.conj(Fm.T))
    Vn = 0.5 * (V[None, :] + V[:, None]) * 0.5   # symmetrised 1/2 eta1 phi1_tz

    def denom(K, Om):
        aK = np.abs(K)
        return g * aK * _tanh_kd(aK, depth) - Om**2

    Cp = Fp / denom(Kp, Wp)
    with np.errstate(divide="ignore", invalid="ignore"):
        Cm = np.where(np.abs(Km) > 0, Fm / denom(Km, Wm), 0.0)
    Ep = -(1 / g) * (-1j * Wp * Cp + 0.5 * Pp + Vn)
    Em = -(1 / g) * (-1j * Wm * Cm + 0.5 * Pm + Vn)
    np.fill_diagonal(Em, 0.0)
    np.fill_diagonal(Cm, 0.0)
    return dict(Kp=Kp, Km=Km, Ep=Ep, Em=Em, Cp=Cp, Cm=Cm)


def _ch_sh(K, z, depth):
    """cosh(|K|(z+d))/cosh(|K|d) and sinh(...)/cosh(...), overflow-safe."""
    aK = np.abs(K)
    e = np.exp(aK * z)
    if np.isinf(depth):
        return e, e
    eb = np.exp(-2 * aK * (z + depth))
    den = 1 + np.exp(-2 * aK * depth)
    return e * (1 + eb) / den, e * (1 - eb) / den


class SecondOrderWaveField:
    """Linear field plus its second-order bound waves.

    Parameters
    ----------
    first : SpectralWaveField
        The linear field (any amplitudes / amplitude derivatives).
    rel_tol : float
        Modes with |A| below rel_tol * max|A| are left out of the pair sums
        (they still contribute at first order). The pair sums cost
        O(M^2) per evaluation point.
    keep, kernels : optional
        Reuse a mode selection and :func:`pair_kernels` result from another
        field with the same wavenumbers (e.g. the same wave at another time).
    """

    def __init__(self, first, rel_tol=1e-3, keep=None, kernels=None):
        self.first = first
        self.depth, self.g = first.depth, first.g
        if keep is None:
            keep = np.abs(first.amp) > rel_tol * np.abs(first.amp).max()
        self.keep = keep
        self.k = first.k[keep]
        self.A = first.amp[keep]
        self.A_dot = first.amp_dot[keep]
        self.kern = kernels if kernels is not None else pair_kernels(self.k, self.depth, self.g)

    def _pairs(self, x):
        """Pair products at points x: (P, M, M) arrays for the sum and difference."""
        E = np.exp(1j * np.multiply.outer(x, self.k))
        a = E * self.A
        ad = E * self.A_dot
        sum_ = a[:, :, None] * a[:, None, :]
        dif = a[:, :, None] * np.conj(a)[:, None, :]
        sum_t = ad[:, :, None] * a[:, None, :] + a[:, :, None] * ad[:, None, :]
        dif_t = ad[:, :, None] * np.conj(a)[:, None, :] + a[:, :, None] * np.conj(ad)[:, None, :]
        return sum_, dif, sum_t, dif_t

    def _chunk(self):
        return max(1, int(2e6 // max(self.k.size**2, 1)))

    def eta_second(self, x):
        x = np.atleast_1d(np.asarray(x, float))
        shape = x.shape
        x = x.ravel()
        out = np.empty(x.size)
        kn = self.kern
        for s in range(0, x.size, self._chunk()):
            sl = slice(s, s + self._chunk())
            sum_, dif, _, _ = self._pairs(x[sl])
            out[sl] = np.real((sum_ * kn["Ep"]).sum((1, 2)) + (dif * kn["Em"]).sum((1, 2)))
        return out.reshape(shape)

    def eta(self, x):
        return self.first.eta(x) + self.eta_second(x)

    def potential(self, x, z):
        """Velocity potential to second order (no stretching)."""
        x, z = np.broadcast_arrays(np.asarray(x, float), np.asarray(z, float))
        phi1 = self.first.potential(x, z).ravel()
        x, z = x.ravel(), z.ravel()
        kn = self.kern
        out = np.empty(x.size)
        for s in range(0, x.size, self._chunk()):
            sl = slice(s, s + self._chunk())
            sum_, dif, _, _ = self._pairs(x[sl])
            zz = z[sl][:, None, None]
            chp, _ = _ch_sh(kn["Kp"], zz, self.depth)
            chm, _ = _ch_sh(kn["Km"], zz, self.depth)
            out[sl] = np.real((sum_ * kn["Cp"] * chp).sum((1, 2))
                              + (dif * kn["Cm"] * chm).sum((1, 2)))
        return (phi1 + out).reshape(np.shape(np.broadcast_arrays(x, z)[0]))

    def kinematics(self, x, z, stretching="linear"):
        """Kinematics to second order.

        ``"linear"`` (default) is the consistent second-order choice: the
        first-order field is Taylor-extrapolated above z = 0 and the
        second-order terms are evaluated at min(z, 0). ``"wheeler"`` and
        ``"none"`` are available for comparison.
        """
        if stretching not in STRETCHING:
            raise ValueError(f"stretching must be one of {STRETCHING}")
        x, z = np.broadcast_arrays(np.asarray(x, float), np.asarray(z, float))
        shape = x.shape
        x, z = x.ravel(), z.ravel()
        kn = self.kern
        out = {key: np.empty(x.size) for key in KIN_KEYS}
        for s in range(0, x.size, self._chunk()):
            sl = slice(s, s + self._chunk())
            sum_, dif, sum_t, dif_t = self._pairs(x[sl])
            eta = self.first.eta(x[sl]) + np.real(
                (sum_ * kn["Ep"]).sum((1, 2)) + (dif * kn["Em"]).sum((1, 2)))
            k1 = self.first._kinematics_flat(x[sl], z[sl], stretching, surface=eta)
            zs = stretched_z(z[sl], eta, self.depth, stretching)[:, None, None]
            two = dict(eta=eta)
            for key in ("u", "w", "ut", "wt", "ux", "uz"):
                two[key] = np.zeros(eta.size)
            for K, C, prod, prod_t in ((kn["Kp"], kn["Cp"], sum_, sum_t),
                                       (kn["Km"], kn["Cm"], dif, dif_t)):
                ch, sh = _ch_sh(K, zs, self.depth)
                aK = np.abs(K)
                c, ct = C * prod, C * prod_t
                two["u"] += np.real((1j * K * ch * c).sum((1, 2)))
                two["w"] += np.real((aK * sh * c).sum((1, 2)))
                two["ut"] += np.real((1j * K * ch * ct).sum((1, 2)))
                two["wt"] += np.real((aK * sh * ct).sum((1, 2)))
                two["ux"] += np.real((-K**2 * ch * c).sum((1, 2)))
                two["uz"] += np.real((1j * K * aK * sh * c).sum((1, 2)))
            two["wx"], two["wz"] = two["uz"], -two["ux"]
            part = add_kinematics(k1, two, z[sl], self.depth)
            for key in KIN_KEYS:
                out[key][sl] = part[key]
        return {key: v.reshape(shape) for key, v in out.items()}
