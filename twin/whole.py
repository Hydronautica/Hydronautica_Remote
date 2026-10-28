"""Whole-turbine state estimation with uncertainty: blades, drivetrain, tower, monopile and pile.

Truth: the nonlinear beam model (pile on p-y springs, monopile, transition piece, tower,
RNA mass) in the fore-aft and side-side planes, driven by the rotor loads built blade by
blade (records.synthetic) and by Morison wave forces with drag, plus aerodynamic damping.

Estimator: one linear state-space model and Kalman smoother for everything:
  tower and foundation  the first modes in each plane, linearised about the mean deflection
                        (soil tangent stiffness), with the rotor force and moment at the top
                        and a wave force with the peak wave's depth profile;
  waves                 one second-order process per plane fitted to the JONSWAP force spectrum;
  rotor                 the turbulent blade-root loads as principal components (rna.lfm), which
                        both load the tower (through the rotor azimuth) and feed the drivetrain;
  nacelle               vertical acceleration as a small process.
Sensors are linear in the state at each time step. Load histories drawn from the smoother go
through the drivetrain FE chain, the tower and foundation hot spots and the blades, giving
damage distributions everywhere, including below the seabed where nothing is measured.
"""
import numpy as np
from scipy.linalg import eigh, expm, solve_discrete_lyapunov

from rna.chain import rotx
from rna.fatigue import DESIGN_LIFE, SN, rainflow
from rna.lfm import aero_mean, centrifugal, fit_second_order, inertial_map, jonswap
from rna.records import G, synthetic
from rna.virtual import chain_maps

RHO_W, CM, CD = 1025.0, 2.0, 1.0


def wavenumber(w, d):
    k = w * w / G
    for _ in range(40):
        k = w * w / (G * np.tanh(k * d))
    return k


# ---------------------------------------------------------------------------------------------
class Beam:
    """Helpers on a structure.Model: load vectors, moment operators, modes about a mean state."""

    def __init__(self, model):
        self.m = model
        z = model.z
        self.n = z.size
        self.top = 2 * (self.n - 1)
        self.Ltop = z[-1] - z[-2]

    def top_load(self, F, Mb):
        """Nodal lateral forces (… × n_nodes) for a force F and a bending moment Mb at the top (as a couple)."""
        F, Mb = np.atleast_1d(F), np.atleast_1d(Mb)
        out = np.zeros(F.shape + (self.n,))
        out[..., -1] = F + Mb / self.Ltop
        out[..., -2] = -Mb / self.Ltop
        return out

    def moment_row(self, zt):
        """Row (ndof) giving the bending moment at elevation zt from the displacement vector."""
        z = self.m.z
        e = int(np.clip(np.searchsorted(z, zt) - 1, 0, self.n - 2))
        L = z[e + 1] - z[e]; xi = (zt - z[e]) / L
        r = np.zeros(self.m.ndof)
        r[2 * e:2 * e + 4] = self.m.EI[e] * np.array([(-6 + 12 * xi) / L ** 2, (-4 + 6 * xi) / L, (6 - 12 * xi) / L ** 2, (-2 + 6 * xi) / L])
        return r

    def moment_op(self):
        zc = 0.5 * (self.m.z[1:] + self.m.z[:-1])
        return zc, np.array([self.moment_row(z) for z in zc])

    def modes(self, F_mean, Mb_mean, n_modes, zeta, c_top=0.0):
        F = np.zeros(self.m.ndof)
        F[0::2] = self.top_load(F_mean, Mb_mean)[0]
        u0 = self.m.static(F)
        _, kt = self.m.restoring(u0)
        w2, Phi = eigh(self.m.K + np.diag(kt), self.m.M)
        w = np.sqrt(w2[:n_modes]); Phi = Phi[:, :n_modes]
        z = zeta + c_top * Phi[self.top] ** 2 / (2 * w)               # aerodynamic damping at the hub
        return dict(u0=u0, w=w, Phi=Phi, zeta=z, F0=F)


def morison_truth(model, t, Hs, Tp, rng, wave_dir=20.0, spread=0.4, ncomp=60):
    """Nodal wave forces (n × nodes) in the fore-aft and side-side planes, inertia and drag."""
    tb, z = model.tb, model.z
    d = tb.depth
    wet = np.flatnonzero((z > -d) & (z < 0))
    wp = 2 * np.pi / Tp
    w = np.linspace(0.5, 3.0, ncomp) * wp
    dw = w[1] - w[0]
    a = np.sqrt(2 * jonswap(w, Hs, Tp) * dw)
    k = wavenumber(w, d)
    C = np.cosh(np.outer(z[wet] + d, k)) / np.sinh(k * d)               # wet × comp
    th = np.radians(wave_dir)
    vel, acc = [], []
    for _ in range(2):
        ph = 2 * np.pi * rng.random(ncomp)
        arg = np.outer(t, w) + ph
        vel.append(np.einsum("tc,zc->tz", a * w * np.cos(arg), C))
        acc.append(np.einsum("tc,zc->tz", -a * w * w * np.sin(arg), C))
    ux, uz = np.cos(th) * vel[0], np.sin(th) * vel[0] + spread * vel[1]
    ax, az = np.cos(th) * acc[0], np.sin(th) * acc[0] + spread * acc[1]
    D = tb.D_pile
    sp = np.hypot(ux, uz)
    tr = model.trib[wet]
    fx = (RHO_W * CM * np.pi * D ** 2 / 4 * ax + 0.5 * RHO_W * CD * D * sp * ux) * tr
    fz = (RHO_W * CM * np.pi * D ** 2 / 4 * az + 0.5 * RHO_W * CD * D * sp * uz) * tr
    Ffa = np.zeros((t.size, z.size)); Fss = np.zeros_like(Ffa)
    Ffa[:, wet] = fx; Fss[:, wet] = fz
    return Ffa, Fss


def simulate(kind, tb, models, Hs, Tp, seed=None, wave_dir=20.0, c_aero=0.12e6):
    """Truth record: rotor loads blade by blade, nonlinear beam in both planes, nacelle motion back to the rotor."""
    held = {}
    def tower(r):
        t, dt, rng = r["t"], r["dt"], r["rng"]
        Ffa_w, Fss_w = morison_truth(models["fa"], t, Hs, Tp, rng, wave_dir)
        out = {}
        for p, Fw, F, Mb in (("fa", Ffa_w, r["aero_F"][:, 0], -r["aero_M"][:, 2]),
                             ("ss", Fss_w, r["aero_F"][:, 2], r["aero_M"][:, 0])):
            m = models[p]; bm = Beam(m)
            Fd = Fw + bm.top_load(F, Mb)
            m.C = None; m.setup_damping()
            if p == "fa" and kind == "op":
                m.C[bm.top, bm.top] += c_aero
            F0 = np.zeros(m.ndof); F0[0::2] = Fd[0]
            U, _ = m.run(t, Fd, u0=m.static(F0))
            acc = np.gradient(np.gradient(U[:, bm.top], dt), dt)
            out[p] = (U, acc)
        held.update(U_fa=out["fa"][0], U_ss=out["ss"][0])
        ay = 0.02 * np.sin(2 * np.pi * 0.5 * t + 1.0)
        return np.stack([out["fa"][1], G + ay, out["ss"][1]], 1)
    rec = synthetic(kind, tb, seed=seed, tower=tower)
    rec.update(held)
    return rec


# ---------------------------------------------------------------------------------------------
def hub_aero_map(psi, tb):
    """T (n × 6 × 18): hub-centre aerodynamic loads (fixed frame) per blade-root load (blade frame)."""
    n = psi.size
    T = np.zeros((n, 6, 18))
    x = np.array([1.0, 0, 0])
    for k in range(3):
        pk = psi + 2 * np.pi * k / 3
        er = np.stack([np.zeros(n), np.cos(pk), np.sin(pk)], 1); et = np.stack([np.zeros(n), -np.sin(pk), np.cos(pk)], 1)
        dirsF = [np.broadcast_to(x, (n, 3)), et, er]
        for j, dvec in enumerate(dirsF):
            T[:, :3, 6 * k + j] = dvec
            T[:, 3:, 6 * k + j] = np.cross(tb.r_root * er, dvec)
        for j, dvec in enumerate([-et, np.broadcast_to(x, (n, 3)), er]):
            T[:, 3:, 6 * k + 3 + j] = dvec
    return T


class WholeLFM:
    def __init__(self, tb, kind, ap, models, Hs, Tp, n_modes=4, zeta=0.015, c_aero=0.12e6, dt=0.25,
                 err=dict(stiff=0.97, aero=1.2), wave_dir=20.0, model_noise=0.3):
        self.tb, self.kind, self.ap, self.dt = tb, kind, ap, dt
        self.models = models
        self.beam = {p: Beam(models[p]) for p in ("fa", "ss")}
        op = kind == "op"
        T0 = 2.1e6 if op else 0.7e6
        Mx0 = tb.rated_torque if op else 0.0
        self.md = {"fa": self.beam["fa"].modes(T0, 0.0, n_modes, zeta, c_aero * err["aero"] if op else 0.0),
                   "ss": self.beam["ss"].modes(0.0, Mx0, n_modes, zeta)}
        for p in self.md:                              # the model's structure is a little stiffer or softer than the real one
            self.md[p]["w"] = self.md[p]["w"] * np.sqrt(err["stiff"])
        nm, K = n_modes, ap["K"]
        self.nm = nm
        self.i = dict(fa=0, ss=2 * nm, wfa=4 * nm, wss=4 * nm + 2, ay=4 * nm + 4, aero=4 * nm + 6, bias=4 * nm + 6 + 2 * K)
        self.nz = nz = 4 * nm + 10 + 2 * K                    # + slowly varying static force and moment at the top, per plane
        # wave force: peak-wave depth profile, second-order process fitted to the force spectrum
        m = models["fa"]; d = m.tb.depth; z = m.z
        wp = 2 * np.pi / Tp; kp = wavenumber(wp, d)
        wet = (z > -d) & (z < 0)
        shape = np.zeros(z.size)
        shape[wet] = RHO_W * CM * np.pi * m.tb.D_pile ** 2 / 4 * wp ** 2 * np.cosh(kp * (z[wet] + d)) / np.sinh(kp * d) * m.trib[wet]
        w = np.linspace(0.3, 3.5, 300) * wp
        S = (w / wp) ** 4 * jonswap(w, Hs, Tp)
        self.w0w, self.zw = fit_second_order(w / (2 * np.pi), S)
        sw = np.sqrt(np.trapezoid(S, w))
        th = np.radians(wave_dir)
        Ac = np.zeros((nz, nz)); Qc = np.zeros((nz, nz))
        self.Bw = {}
        for p, iw, share in (("fa", self.i["wfa"], np.cos(th)), ("ss", self.i["wss"], np.sqrt(np.sin(th) ** 2 + 0.16))):
            md, i0 = self.md[p], self.i[p]
            Ac[i0:i0 + nm, i0 + nm:i0 + 2 * nm] = np.eye(nm)
            Ac[i0 + nm:i0 + 2 * nm, i0:i0 + nm] = -np.diag(md["w"] ** 2)
            Ac[i0 + nm:i0 + 2 * nm, i0 + nm:i0 + 2 * nm] = -np.diag(2 * md["zeta"] * md["w"])
            fw = np.zeros(m.ndof); fw[0::2] = shape * sw * share
            Ac[i0 + nm:i0 + 2 * nm, iw] = md["Phi"].T @ fw
            Ac[iw, iw + 1] = 1; Ac[iw + 1, iw] = -self.w0w ** 2; Ac[iw + 1, iw + 1] = -2 * self.zw * self.w0w
            Qc[iw + 1, iw + 1] = 4 * self.zw * self.w0w ** 3
        # static mismatch between model and turbine (mean loads the model leaves out, stiffness): random walks
        for j, p in enumerate(("fa", "ss")):
            md, i0, bm = self.md[p], self.i[p], self.beam[p]
            ib = self.i["bias"] + 2 * j
            phiF = md["Phi"][bm.top]; phiM = (md["Phi"][bm.top] - md["Phi"][bm.top - 2]) / bm.Ltop
            Ac[i0 + nm:i0 + 2 * nm, ib] = 1e6 * phiF
            Ac[i0 + nm:i0 + 2 * nm, ib + 1] = 1e6 * phiM
            Qc[ib, ib], Qc[ib + 1, ib + 1] = 1e-4, 1e-2                     # MN²/s, (MN m)²/s
        self.bias_prior = (0.5 ** 2, 10.0 ** 2)
        # what the linear model leaves out (soil nonlinearity, wave drag and profile, higher modes): white modal forces
        # with a spectral density set relative to the wave forcing of each mode
        for p in ("fa", "ss"):
            i0 = self.i[p]
            iw = self.i["wfa" if p == "fa" else "wss"]
            for j in range(nm):
                gw = Ac[i0 + nm + j, iw]
                Qc[i0 + nm + j, i0 + nm + j] = (model_noise * gw) ** 2 * 4 * self.zw / self.w0w
        ia = self.i["ay"]
        Ac[ia, ia + 1] = 1; Ac[ia + 1, ia] = -np.pi ** 2; Ac[ia + 1, ia + 1] = -np.pi; Qc[ia + 1, ia + 1] = 2 * np.pi ** 3
        for j, (w0, zz) in enumerate(ap["proc"]):
            i = self.i["aero"] + 2 * j
            Ac[i, i + 1] = 1; Ac[i + 1, i] = -w0 ** 2; Ac[i + 1, i + 1] = -2 * zz * w0
            Qc[i + 1, i + 1] = 4 * zz * w0 ** 3
        self.Ac0, self.Qc = Ac, Qc
        self.say = 0.02 if op else 0.05

    def Ac_at(self, Tpsi):
        """Continuous system matrix at one azimuth: the rotor's turbulent loads drive the tower modes."""
        Ac = self.Ac0.copy()
        nm, L = self.nm, self.ap["L"]
        R = 1e6 * Tpsi @ L                              # hub aero loads (N, N m) per unit latent
        for p, Frow, Mrow in (("fa", R[0], -R[5]), ("ss", R[2], R[3])):
            md, i0, bm = self.md[p], self.i[p], self.beam[p]
            phiF = md["Phi"][bm.top]
            phiM = (md["Phi"][bm.top] - md["Phi"][bm.top - 2]) / bm.Ltop
            Ac[i0 + nm:i0 + 2 * nm, self.i["aero"]:self.i["bias"]:2] = np.outer(phiF, Frow) + np.outer(phiM, Mrow)
        return Ac

    def smooth(self, rec, fe, which, n_samples=100, rng=None):
        rng = rng or np.random.default_rng(3)
        tb, n, nz, dt, nm = self.tb, rec["t"].size, self.nz, self.dt, self.nm
        psi = rec["psi"]
        Tm = hub_aero_map(psi, tb)
        # transition matrices per step (time-varying through the azimuth); process noise from the mean system
        Acs = np.array([self.Ac_at(Tm[k]) for k in range(n)])
        Ads = np.array([expm(A * dt) for A in Acs])
        A0 = Acs[0]
        Mv = expm(np.block([[-A0, self.Qc], [np.zeros_like(A0), A0.T]]) * dt)
        Ad0 = Mv[nz:, nz:].T
        Qd = Ad0 @ Mv[:nz, nz:]; Qd = 0.5 * (Qd + Qd.T) + 1e-10 * np.eye(nz)
        st = np.r_[0:self.i["bias"]]                                  # stationary part
        P0 = np.zeros((nz, nz))
        P0[np.ix_(st, st)] = solve_discrete_lyapunov(Ad0[np.ix_(st, st)], Qd[np.ix_(st, st)])
        for j in range(2):
            ib = self.i["bias"] + 2 * j
            P0[ib, ib], P0[ib + 1, ib + 1] = self.bias_prior
        # rows: nacelle acceleration (from the modal accelerations), chain inputs x = mx + Gx z
        rowA = np.zeros((n, 3, nz))
        for j, p in ((0, "fa"), (2, "ss")):
            i0, top = self.i[p], self.beam[p].top
            rowA[:, j] = np.einsum("m,nmk->nk", self.md[p]["Phi"][top], Acs[:, i0 + nm:i0 + 2 * nm])
        rowA[:, 1, self.i["ay"]] = self.say
        J = inertial_map(psi, tb)
        mx = np.vstack([aero_mean(rec, tb), np.zeros((3, n))]) + centrifugal(rec, tb) + J[:, :, 1].T * G
        Gx = np.einsum("nij,njk->nik", J, rowA)
        Gx[:, :18, self.i["aero"]:self.i["bias"]:2] += self.ap["L"]
        maps = chain_maps(fe, psi)
        # sensors
        rows, offs, sd, names = [], [], [], []
        def add(r, o, s, nm_):
            rows.append(r); offs.append(o); sd.append(s); names.append(nm_)
        torque = np.einsum("nk,nkj->nj", maps["fl"][:, 3], Gx)
        add(torque, np.einsum("nk,kn->n", maps["fl"][:, 3], mx), 0.4e0, "SCADA torque")
        for j, nmv in ((0, "nacelle accelerometer fore-aft"), (2, "nacelle accelerometer side-side")):
            add(rowA[:, j], np.zeros(n), 0.01, nmv)
        tbm = self.models["fa"].tb
        def gauge(zg, label, s):
            for p in ("fa", "ss"):
                bm, md, i0 = self.beam[p], self.md[p], self.i[p]
                r = bm.moment_row(zg)
                row = np.zeros((n, nz)); row[:, i0:i0 + nm] = r @ md["Phi"] / 1e6
                add(row, np.full(n, r @ md["u0"] / 1e6), s, f"{label} ({p})")
        def node_acc(zg, label):
            for j, p in ((0, "fa"), (2, "ss")):
                bm, md, i0 = self.beam[p], self.md[p], self.i[p]
                node = 2 * self.models[p].node(zg)
                row = np.einsum("m,nmk->nk", md["Phi"][node], Acs[:, i0 + nm:i0 + 2 * nm])
                add(row, np.zeros(n), 0.01, f"{label} ({p})")
        if which >= 1:
            gauge(tbm.z_interface + 3, "tower-base strain gauges", 1.5)
        if which >= 2:
            gauge(tbm.z_hub - 6, "tower-top strain gauges", 0.5)
            node_acc(tbm.z_interface, "TP accelerometer")
        if which >= 3:
            for k in range(3):
                for j, lab in ((3, "flap"), (4, "edge")):
                    add(Gx[:, 6 * k + j], mx[6 * k + j], 0.5, f"blade {'ABC'[k]} root {lab}")
        H = np.stack([np.atleast_2d(r) if r.ndim == 2 else r for r in rows], 1)
        y0 = np.stack(offs, 1)
        sd = np.array(sd)
        # measurements from the truth
        rng_m = np.random.default_rng(5)
        x_true = np.vstack([rec["blade"].reshape(18, n) / 1e6, np.einsum("nji,nj->in", rotx(psi), rec["A"])])
        y = np.empty((n, len(sd)))
        y[:, 0] = np.einsum("nk,kn->n", maps["fl"][:, 3], x_true)
        y[:, 1], y[:, 2] = rec["A"][:, 0], rec["A"][:, 2]
        c = 3
        for zg, ok in ((tbm.z_interface + 3, which >= 1), (tbm.z_hub - 6, which >= 2)):
            if ok:
                for p in ("fa", "ss"):
                    y[:, c] = rec[f"U_{p}"] @ self.beam[p].moment_row(zg) / 1e6; c += 1
        if which >= 2:
            for p in ("fa", "ss"):
                node = 2 * self.models[p].node(tbm.z_interface)
                y[:, c] = np.gradient(np.gradient(rec[f"U_{p}"][:, node], dt), dt); c += 1
        if which >= 3:
            for k in range(3):
                for j in (3, 4):
                    y[:, c] = x_true[6 * k + j]; c += 1
        y += sd * rng_m.standard_normal(y.shape)
        V = np.diag(sd ** 2)
        # filter
        mf = np.zeros((n, nz)); Pf = np.zeros((n, nz, nz))
        m_, P_ = np.zeros(nz), P0.copy()
        for t in range(n):
            if t:
                m_ = Ads[t - 1] @ mf[t - 1]; P_ = Ads[t - 1] @ Pf[t - 1] @ Ads[t - 1].T + Qd
            Ht = H[t]; S = Ht @ P_ @ Ht.T + V
            Kt = np.linalg.solve(S, Ht @ P_).T
            mf[t] = m_ + Kt @ (y[t] - y0[t] - Ht @ m_)
            Pf[t] = P_ - Kt @ Ht @ P_; Pf[t] = 0.5 * (Pf[t] + Pf[t].T)
        # smoother + backward sampling
        ms, Ps = mf.copy(), Pf.copy()
        Z = np.empty((n_samples, n, nz))
        Z[:, -1] = mf[-1] + rng.standard_normal((n_samples, nz)) @ np.linalg.cholesky(Pf[-1] + 1e-12 * np.eye(nz)).T
        for t in range(n - 2, -1, -1):
            Ad = Ads[t]
            Pp = Ad @ Pf[t] @ Ad.T + Qd
            Jt = np.linalg.solve(Pp, Ad @ Pf[t]).T
            ms[t] = mf[t] + Jt @ (ms[t + 1] - Ad @ mf[t])
            Ps[t] = Pf[t] + Jt @ (Ps[t + 1] - Pp) @ Jt.T
            Cb = Pf[t] - Jt @ Ad @ Pf[t]
            Z[:, t] = mf[t] + (Z[:, t + 1] - Ad @ mf[t]) @ Jt.T + rng.standard_normal((n_samples, nz)) @ np.linalg.cholesky(0.5 * (Cb + Cb.T) + 1e-9 * np.eye(nz)).T
        return dict(Z=Z, ms=ms, Ps=Ps, mx=mx, Gx=Gx, maps=maps, x_true=x_true.T, sensors=names)

    def displacement(self, p, Z):
        """Displacement vectors (… × ndof) of plane p from states."""
        md, i0 = self.md[p], self.i[p]
        return md["u0"] + Z[..., i0:i0 + self.nm] @ md["Phi"].T


# ---------------------------------------------------------------------------------------------
# tower and foundation hot spots (DNV-RP-C203 curves, stress range in Pa)
SN_AIR_D = SN(52.6e6, 1e7, 3, 5)
SN_SEA_D = SN(83.4e6, 1e6, 3, 5)


def hot_spots(models):
    tb = models["fa"].tb
    zs = models["fa"].z
    return {
        "tower_top": dict(z=tb.z_hub - 10, scf=1.15, sn=SN_AIR_D, name="Tower weld below the yaw flange"),
        "tower_mid": dict(z=80.0, scf=1.15, sn=SN_AIR_D, name="Tower girth weld, +80 m"),
        "tower_base": dict(z=tb.z_interface + 2, scf=1.15, sn=SN_AIR_D, name="Tower base weld"),
        "door": dict(z=tb.z_door, scf=2.4, sn=SN_AIR_D, name="Door frame"),
        "flange": dict(z=tb.z_interface, bolts=True, name="Tower-base flange bolts"),
        "mudline": dict(z=-tb.depth + 1.0, scf=1.25, sn=SN_SEA_D, name="Monopile weld at the mudline"),
        "pile": dict(z=-tb.depth - 6.0, scf=1.0, sn=SN_SEA_D, name="Pile below the seabed (−6 m)"),
    }


def section_props(tb, z):
    D, t = float(tb.diameter(z)), float(tb.thickness(z))
    A = np.pi * D * t; I = np.pi * D ** 3 * t / 8
    return A, I / (D / 2), t


def spot_damage(Mfa, Mss, spot, model, dur, az=np.radians(np.arange(0, 360, 45))):
    """Ageing at a hot spot from fore-aft and side-side moment histories (N m), worst of 8 positions round the wall."""
    tb = model.tb
    A, W, t = section_props(tb, spot["z"])
    N = model.weight_above(spot["z"])
    worst = 0.0
    if spot.get("bolts"):
        n_b, Dbc, As = tb.n_bolts, tb.D_bolt_circle, tb.bolt_As
        FV = 0.7 * tb.bolt_fub * As * tb.preload_frac
        ks = (30 / (tb.bolt_d * 1e3)) ** 0.25
        sd5 = ks * 50e6 * 0.4 ** (1 / 3)
        for a in az:
            Z = 4 * (Mfa * np.cos(a) + Mss * np.sin(a)) / (n_b * Dbc) - N / n_b
            sig = np.maximum(FV + 0.15 * Z, Z) / As
            rg, _, ct, _ = rainflow(sig)
            Nc = np.where(rg >= sd5, 2e6 * (ks * 50e6 / np.maximum(rg, 1)) ** 3, 5e6 * (sd5 / np.maximum(rg, 1)) ** 5)
            worst = max(worst, float(np.sum(ct / Nc)))
    else:
        tc = max(1.0, (t * 1e3 / 25) ** 0.2)
        for a in az:
            sig = spot["scf"] * tc * ((Mfa * np.cos(a) + Mss * np.sin(a)) / W - N / A)
            rg, _, ct, _ = rainflow(sig)
            worst = max(worst, float(np.sum(ct / spot["sn"].cycles(rg))))
    return worst * DESIGN_LIFE / dur


def blade_del(x, dur, m=10.0, f_ref=1.0):
    """Damage-equivalent root moments (MN m, m = 10 at 1 Hz) for flap and edge, worst blade."""
    out = {}
    for j, lab in ((3, "flap"), (4, "edge")):
        v = []
        for k in range(3):
            rg, _, ct, _ = rainflow(x[:, 6 * k + j])
            v.append((np.sum(ct * rg ** m) / (f_ref * dur)) ** (1 / m))
        out[lab] = max(v)
    return out
