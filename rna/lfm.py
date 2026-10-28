"""Bayesian virtual sensing with Gaussian-process latent forces, and uncertainty carried to fatigue.

State-space model, sampled every dt:
  tower     first fore-aft and side-side modes (displacement, velocity), driven by
            the rotor thrust and the wave force;
  waves     wave force on each axis as a second-order stochastic process whose spectrum
            matches JONSWAP for the measured Hs, Tp;
  aero      the turbulent part of the 18 blade-root loads, as K principal components, each a
            second-order stochastic process fitted to the turbulence spectrum the rotor sees.
Every sensor is linear in the state once the rotor azimuth is known, so a Kalman filter and
smoother give the posterior of the whole load chain. Fatigue is nonlinear and depends on time
correlation, so the smoother draws whole load histories (forward filter, backward sample) and
each history goes through the chain, rainflow and S-N: the result is a distribution of damage.
"""
from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm, solve_discrete_lyapunov

from .chain import rotx
from .fatigue import DESIGN_LIFE, SN, rainflow
from .records import G, _band, synthetic
from .virtual import chain_maps, sensors


# ---------------------------------------------------------------------------------------------
@dataclass
class Tower:
    m: float = 1.0e6            # modal mass, kg (rotor-nacelle plus tower participation)
    f1: float = 0.167           # Hz
    zeta_op: float = 0.04       # aerodynamic damping while operating
    zeta_idle: float = 0.008
    c_wave: float = 0.035e6     # wave force at hub-equivalent, N per m of wave amplitude (inertia regime)

    def kc(self, kind, f_err=1.0, z_err=1.0):
        w = 2 * np.pi * self.f1 * f_err
        z = (self.zeta_op if kind == "op" else self.zeta_idle) * z_err
        return self.m * w * w, 2 * z * self.m * w


def jonswap(w, Hs, Tp, gamma=3.3):
    wp = 2 * np.pi / Tp
    sig = np.where(w <= wp, 0.07, 0.09)
    S = w ** -5.0 * np.exp(-1.25 * (wp / w) ** 4) * gamma ** np.exp(-((w - wp) ** 2) / (2 * sig ** 2 * wp ** 2))
    return S * (Hs / 4) ** 2 / np.trapezoid(S, w)


def wave_force_spectrum(w, Hs, Tp, tw):
    """Inertia-regime force spectrum at the hub-equivalent point: (c ω²/ωp²)² S_η."""
    wp = 2 * np.pi / Tp
    return (tw.c_wave * (w / wp) ** 2) ** 2 * jonswap(w, Hs, Tp)


def tower_truth(tw, Hs, Tp, wave_dir=20.0, slam=None):
    """Callable for records.synthetic(tower=...): nacelle acceleration from the tower modes,
    driven by the rotor's aerodynamic force and a random JONSWAP wave force (plus an optional slam)."""
    def run(r):
        t, dt, rng, kind = r["t"], r["dt"], r["rng"], r["kind"]
        n = t.size
        w = np.linspace(0.3, 3.0, 120) * 2 * np.pi / Tp
        dw = w[1] - w[0]
        S = wave_force_spectrum(w, Hs, Tp, tw)
        def series():
            ph = 2 * np.pi * rng.random(w.size)
            return (np.sqrt(2 * S * dw)[None, :] * np.cos(np.outer(t, w) + ph)).sum(1)
        W1, W2 = series(), series()
        th = np.radians(wave_dir)
        Ffa = np.cos(th) * W1 + r["aero_F"][:, 0]
        Fss = np.sin(th) * W1 + 0.4 * W2 + r["aero_F"][:, 2]
        if slam is not None:
            t0, amp = slam
            Ffa = Ffa + amp * np.exp(-0.5 * ((t - t0) / 0.3) ** 2)
        k, c = tw.kc(kind)
        Ac = np.array([[0, 1], [-k / tw.m, -c / tw.m]]); Bc = np.array([0, 1 / tw.m])
        Ad = expm(Ac * dt); Bd = np.linalg.solve(Ac, (Ad - np.eye(2))) @ Bc
        acc = []
        for F in (Ffa, Fss):
            x = np.array([F[0] / k, 0.0]); a = np.empty(n)
            for i in range(n):
                a[i] = (F[i] - c * x[1] - k * x[0]) / tw.m
                x = Ad @ x + Bd * F[i]
            acc.append(a)
        ay = _band(rng, t, 0.5, 0.3, 0.02 if kind == "op" else 0.05)
        return np.stack([acc[0], G + ay, acc[1]], 1)
    return run


def truth_record(kind, tb, Hs, Tp, seed=None, tw=None, slam=None):
    tw = tw or Tower()
    return synthetic(kind, tb, seed=seed, tower=tower_truth(tw, Hs, Tp, slam=slam))


# ---------------------------------------------------------------------------------------------
def aero_mean(rec, tb):
    """What the model expects of the aerodynamic blade-root loads without turbulence (blade frame, MN)."""
    n = rec["t"].size
    psi, op = rec["psi"], rec["kind"] == "op"
    out = np.zeros((3, 6, n))
    r_f, r_e = tb.r_flap - tb.r_root, tb.r_edge - tb.r_root
    for k in range(3):
        pk = psi + 2 * np.pi * k / 3
        ff = 0.70e6 * (1 + 0.05 * np.cos(pk)) if op else np.full(n, 0.23e6)
        fe = tb.rated_torque / 3 / tb.r_edge if op else 0.0
        out[k] = np.array([ff, np.full(n, fe), np.zeros(n), ff * r_f, np.full(n, fe * r_e), np.full(n, -0.3e6)]) / 1e6
    return out.reshape(18, n)


def inertial_map(psi, tb):
    """J (n × 21 × 3): blade-root loads (MN) and rotor-frame acceleration rows per unit nacelle acceleration."""
    n = psi.size
    J = np.zeros((n, 21, 3))
    for j in range(3):
        A = np.zeros((n, 3)); A[:, j] = 1.0
        for k in range(3):
            pk = psi + 2 * np.pi * k / 3
            er = np.stack([np.zeros(n), np.cos(pk), np.sin(pk)], 1); et = np.stack([np.zeros(n), -np.sin(pk), np.cos(pk)], 1)
            F = -tb.blade_mass * A
            M = np.cross(tb.blade_cg * er, F)
            J[:, 6 * k:6 * k + 6, j] = np.stack([F[:, 0], np.sum(F * et, 1), np.sum(F * er, 1),
                                                -np.sum(M * et, 1), M[:, 0], np.sum(M * er, 1)], 1) / 1e6
        J[:, 18:, j] = np.einsum("nji,nj->ni", rotx(psi), A)
    return J


def centrifugal(rec, tb):
    n = rec["t"].size
    om = rec["rpm"] / 60 * 2 * np.pi
    out = np.zeros((21, n))
    for k in range(3):
        out[6 * k + 2] = tb.blade_mass * om ** 2 * (tb.r_root + tb.blade_cg) / 1e6
    return out


def fit_second_order(f, S):
    """Fit S(ω) ∝ 1/((ω0² - ω²)² + (2ζω0ω)²) to a one-sided PSD by grid search on log spectra."""
    w = 2 * np.pi * f
    ok = (S > S.max() * 1e-4) & (w > 0)
    best = None
    for w0 in np.geomspace(0.01, 6.0, 90):
        for z in np.geomspace(0.05, 10.0, 48):
            M = 1 / ((w0 ** 2 - w[ok] ** 2) ** 2 + (2 * z * w0 * w[ok]) ** 2)
            c = np.mean(np.log(S[ok]) - np.log(M))
            e = np.mean((np.log(S[ok]) - np.log(M) - c) ** 2)
            if best is None or e < best[0]:
                best = (e, w0, z)
    return best[1], best[2]


def aero_prior(tb, kind, seeds=range(201, 216), K=None):
    """Principal components of the turbulent blade-root loads and a second-order process for each."""
    D = []
    for s in seeds:
        r = synthetic(kind, tb, seed=s)
        D.append(r["aero"].reshape(18, -1) / 1e6 - aero_mean(r, tb))
    dt = r["dt"]
    X = np.hstack(D)
    lam, U = np.linalg.eigh(np.cov(X))
    lam, U = lam[::-1], U[:, ::-1]
    K = K or int(np.sum(lam > 1e-9 * lam[0]))                 # keep every component that carries variance
    L = U[:, :K] * np.sqrt(lam[:K])
    proc = []
    for j in range(K):
        Ss = []
        for d in D:
            s = U[:, j] @ d / np.sqrt(lam[j]); s -= s.mean()
            F = np.fft.rfft(s * np.hanning(s.size)); Ss.append(np.abs(F) ** 2)
        f = np.fft.rfftfreq(D[0].shape[1], dt)
        S = np.convolve(np.mean(Ss, 0), np.ones(5) / 5, "same")
        proc.append(fit_second_order(f[1:], S[1:]))
    return dict(L=L, proc=proc, K=K, energy=float(np.cumsum(lam)[K - 1] / lam.sum()))


def wave_prior(Hs, Tp, tw):
    w = np.linspace(0.2, 4.0, 400) * 2 * np.pi / Tp
    S = wave_force_spectrum(w, Hs, Tp, tw)
    w0, z = fit_second_order(w / (2 * np.pi), S)
    return w0, z, float(np.sqrt(np.trapezoid(S, w)))


# ---------------------------------------------------------------------------------------------
class LFM:
    """The state-space model for one record's conditions."""

    def __init__(self, tb, kind, ap, Hs, Tp, tw=None, dt=0.25, f_err=1.01, z_err=1.2, wave_dir=20.0):
        self.tb, self.kind, self.ap, self.tw = tb, kind, ap, tw or Tower()
        K = ap["K"]
        self.nz = nz = 10 + 2 * K                            # tower 4, waves 4, vertical acceleration 2, aero 2K
        k, c = self.tw.kc(kind, f_err, z_err)          # the model's tower is a little off, as a real one would be
        m = self.tw.m
        w0w, zw, sw = wave_prior(Hs, Tp, self.tw)
        Ac = np.zeros((nz, nz)); Qc = np.zeros((nz, nz))
        thrust = 1e6 * ap["L"][0::6].sum(0)                  # thrust (N) per unit latent: sum of out-of-plane forces
        th = np.radians(wave_dir)
        share = (np.cos(th), np.sqrt(np.sin(th) ** 2 + 0.16))     # wave force per axis (main direction plus spreading)
        for a, (iq, iw) in enumerate(((0, 4), (2, 6))):
            Ac[iq, iq + 1] = 1
            Ac[iq + 1, iq] = -k / m; Ac[iq + 1, iq + 1] = -c / m; Ac[iq + 1, iw] = share[a] * sw / m
            if a == 0:
                Ac[iq + 1, 10::2] = thrust / m
            Ac[iw, iw + 1] = 1; Ac[iw + 1, iw] = -w0w ** 2; Ac[iw + 1, iw + 1] = -2 * zw * w0w
            Qc[iw + 1, iw + 1] = 4 * zw * w0w ** 3                     # unit-variance process
        say = 0.02 if kind == "op" else 0.05                  # small vertical nacelle acceleration
        Ac[8, 9] = 1; Ac[9, 8] = -np.pi ** 2; Ac[9, 9] = -np.pi; Qc[9, 9] = 4 * 0.5 * np.pi ** 3
        self.say = say
        for j, (w0, z) in enumerate(ap["proc"]):
            i = 10 + 2 * j
            Ac[i, i + 1] = 1; Ac[i + 1, i] = -w0 ** 2; Ac[i + 1, i + 1] = -2 * z * w0
            Qc[i + 1, i + 1] = 4 * z * w0 ** 3
        M = expm(np.block([[-Ac, Qc], [np.zeros_like(Ac), Ac.T]]) * dt)      # Van Loan
        self.Ad = M[nz:, nz:].T
        self.Qd = self.Ad @ M[:nz, nz:]
        self.Qd = 0.5 * (self.Qd + self.Qd.T) + 1e-10 * np.eye(nz)
        self.P0 = solve_discrete_lyapunov(self.Ad, self.Qd)
        ay = np.zeros(nz); ay[8] = say
        self.rowA = np.vstack([Ac[1], ay, Ac[3]])                          # nacelle acceleration (x, y, z) per state
        self.Ac = Ac

    def x_map(self, rec):
        """x = mx + Gx z, with x the 21 chain inputs (MN, MN m, m/s²)."""
        n = rec["t"].size
        J = inertial_map(rec["psi"], self.tb)
        mx = np.vstack([aero_mean(rec, self.tb), np.zeros((3, n))]) + centrifugal(rec, self.tb) + J[:, :, 1].T * G
        Gx = np.einsum("nij,jk->nik", J, self.rowA)
        Gx[:, :18, 10::2] += self.ap["L"]
        return mx, Gx

    def smooth(self, rec, fe, which, n_samples=100, rng=None):
        rng = rng or np.random.default_rng(3)
        n, nz = rec["t"].size, self.nz
        maps = chain_maps(fe, rec["psi"])
        Hx, sd, names = sensors(fe, maps, which)
        x_true = np.vstack([rec["blade"].reshape(18, n) / 1e6, np.einsum("nji,nj->in", rotx(rec["psi"]), rec["A"])])
        y = np.einsum("nmk,kn->nm", Hx, x_true) + sd * np.random.default_rng(5).standard_normal((n, len(sd)))
        mx, Gx = self.x_map(rec)
        H = np.einsum("nmk,nkj->nmj", Hx, Gx)
        y0 = np.einsum("nmk,kn->nm", Hx, mx)
        V = np.diag(sd ** 2)
        mf = np.zeros((n, nz)); Pf = np.zeros((n, nz, nz))
        m_, P_ = np.zeros(nz), self.P0.copy()
        for t in range(n):
            if t:
                m_ = self.Ad @ mf[t - 1]; P_ = self.Ad @ Pf[t - 1] @ self.Ad.T + self.Qd
            Ht = H[t]; S = Ht @ P_ @ Ht.T + V
            Kt = np.linalg.solve(S, Ht @ P_).T
            mf[t] = m_ + Kt @ (y[t] - y0[t] - Ht @ m_)
            Pf[t] = P_ - Kt @ Ht @ P_
            Pf[t] = 0.5 * (Pf[t] + Pf[t].T)
        # RTS smoother and backward sampling
        ms = mf.copy(); Ps = Pf.copy()
        Z = np.empty((n_samples, n, nz))
        Lc = np.linalg.cholesky(Pf[-1] + 1e-12 * np.eye(nz))
        Z[:, -1] = mf[-1] + rng.standard_normal((n_samples, nz)) @ Lc.T
        for t in range(n - 2, -1, -1):
            Pp = self.Ad @ Pf[t] @ self.Ad.T + self.Qd
            Jt = np.linalg.solve(Pp, self.Ad @ Pf[t]).T
            ms[t] = mf[t] + Jt @ (ms[t + 1] - self.Ad @ mf[t])
            Ps[t] = Pf[t] + Jt @ (Ps[t + 1] - Pp) @ Jt.T
            Cb = Pf[t] - Jt @ self.Ad @ Pf[t]
            Lb = np.linalg.cholesky(0.5 * (Cb + Cb.T) + 1e-9 * np.eye(nz))
            Z[:, t] = mf[t] + (Z[:, t + 1] - self.Ad @ mf[t]) @ Jt.T + rng.standard_normal((n_samples, nz)) @ Lb.T
        X = mx.T[None] + np.einsum("nij,snj->sni", Gx, Z)                 # samples of x: S × n × 21
        Xm = mx.T + np.einsum("nij,nj->ni", Gx, ms)
        Xsd = np.sqrt(np.einsum("nij,njk,nik->ni", Gx, Ps, Gx))
        return dict(x_true=x_true.T, X=X, Xmean=Xm, Xsd=Xsd, maps=maps, Gx=Gx, ms=ms, Ps=Ps, sensors=names)


# ---------------------------------------------------------------------------------------------
CAST_SN = SN(80e6, 2e6, 7, 13)
SHAFT_SN = SN(320e6, 1e6, 5, 9)


def spots_from(comps):
    C = {c["key"]: c for c in comps}
    s = {k: (C[k]["part"], C[k]["vertex"], np.array(C[k]["normal"])) for k in ("hub", "housing", "bedplate")}
    s["shaft"] = (np.array(C["main_shaft"]["ring"]), np.array(C["main_shaft"]["ring_angles"]), C["main_shaft"]["phi0"])
    s["bolt_beta"] = C["bolts"]["beta"]
    return s


def outputs(x, maps, fe, spots):
    """Chain outputs for one load history x (n × 21)."""
    o = {"fl": np.einsum("nij,nj->ni", maps["fl"], x), "mb": np.einsum("nij,nj->ni", maps["mb"], x),
         "armL": np.einsum("nij,nj->ni", maps["armL"], x), "yaw": np.einsum("nij,nj->ni", maps["yaw"], x)}
    Ldt = np.einsum("nij,nj->ni", maps["L_dt"], x); Lbed = np.einsum("nij,nj->ni", maps["L_bed"], x)
    L = {"hub": x, "drivetrain": Ldt, "bedplate": Lbed}
    for k in ("hub", "housing", "bedplate"):
        part, v, nr = spots[k]
        w = np.array([nr[0] ** 2, nr[1] ** 2, nr[2] ** 2, 2 * nr[0] * nr[1], 2 * nr[1] * nr[2], 2 * nr[2] * nr[0]])
        o[k] = L[part] @ (fe[f"{part}_stress"][v].astype(float) @ w) / 1e6
    o["Ldt"] = Ldt
    return o


def shaft_series(o, fe, spots, psi):
    ring, ang, phi0 = spots["shaft"]
    field = o["Ldt"] @ (fe["drivetrain_stress"][ring, :, 0].astype(float).T) / 1e6       # n × ring
    a = np.mod(phi0 + psi + np.pi, 2 * np.pi) - np.pi
    tq = np.r_[ang - 2 * np.pi, ang, ang + 2 * np.pi]
    return np.array([np.interp(a[t], tq, np.r_[field[t], field[t], field[t]]) for t in range(psi.size)])


def damages(x, maps, fe, spots, rec, tb):
    """Ageing (design-life hours per hour) of each component for one load history."""
    dt, n = rec["dt"], rec["t"].size
    dur = n * dt
    o = outputs(x, maps, fe, spots)
    fat = lambda s, sn: sum(c / sn.cycles(r * 1e6) for r, c in zip(*rainflow(s)[::2])) * DESIGN_LIFE / dur
    d = {k: fat(o[k], CAST_SN) for k in ("hub", "housing", "bedplate")}
    d["main_shaft"] = fat(shaft_series(o, fe, spots, rec["psi"]), SHAFT_SN)
    Fr, Fa = np.hypot(o["mb"][:, 1], o["mb"][:, 2]), np.abs(o["mb"][:, 0])
    P = np.where(Fa <= 0.3 * Fr, Fr + 2.3 * Fa, 0.67 * Fr + 3.4 * Fa)
    d["main_bearing"] = float(np.sum(np.abs(rec["rpm"]) / 60 * dt / ((48 / np.maximum(P, 1e-3)) ** (10 / 3) * 1e6))) * DESIGN_LIFE / dur
    from .components import _design_rate
    T = np.abs(o["fl"][:, 3]) * 1e6
    d["gears"] = max(np.mean(np.abs(rec["rpm"]) * T ** m) / _design_rate(m, tb) for m in (8.7, 6.6))
    d["gear_bearings"] = np.mean(np.abs(rec["rpm"]) * T ** (10 / 3)) / _design_rate(10 / 3, tb)
    beta, ks = spots["bolt_beta"], (30 / 42) ** 0.25
    sd5 = ks * 50e6 * 0.4 ** (1 / 3)
    worst = 0.0
    for k in range(3):
        Z = 4 * (x[:, 6 * k + 3] * np.cos(beta) + x[:, 6 * k + 4] * np.sin(beta)) * 1e6 / (144 * 4.25) + x[:, 6 * k + 2] * 1e6 / 144
        sig = np.maximum(738e3 + 0.1 * Z, Z) / 1121e-6
        rg, _, ct, _ = rainflow(sig)
        Nc = np.where(rg >= sd5, 2e6 * (ks * 50e6 / np.maximum(rg, 1)) ** 3, 5e6 * (sd5 / np.maximum(rg, 1)) ** 5)
        worst = max(worst, float(np.sum(ct / Nc)))
    d["bolts"] = worst * DESIGN_LIFE / dur
    return d, o


# model and S-N uncertainty layered on top of the estimation samples
MODEL_UQ = {  # (stress-factor CoV, S-N slope for that factor, log10 scatter of the S-N life)
    "hub": (0.05, 7, 0.20), "housing": (0.05, 7, 0.20), "bedplate": (0.05, 7, 0.20),
    "main_shaft": (0.05, 5, 0.18), "bolts": (0.04, 3, 0.20),
    "main_bearing": (0.05, 10 / 3, 0.30), "gears": (0.03, 8.7, 0.15), "gear_bearings": (0.03, 10 / 3, 0.30),
}


def with_model_uq(samples, rng):
    """Multiply each estimation sample by a draw of stress-model error and S-N life scatter."""
    out = {}
    for k, v in samples.items():
        cov, m, s10 = MODEL_UQ[k]
        f = np.exp(rng.normal(0, cov, v.size)) ** m * 10 ** rng.normal(0, s10, v.size)
        out[k] = np.asarray(v) * f
    return out
