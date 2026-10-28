"""Virtual sensing for the rotor-nacelle load chain.

Everything downstream of the blade roots is linear in the 21 chain inputs
x = [18 blade-root loads (MN, MN m, blade frame), 3 rotor-frame accelerations (m/s²)]
once the rotor azimuth is known. So every sensor on the turbine is a row of a linear
measurement equation y = H(psi) x + v, and every unmeasured load, stress or damage follows
from the estimate of x.

The estimator is the static best linear unbiased estimate at each time step:
  prior  x ~ N(mu, P): mu from the global model and SCADA (gravity, centrifugal, mean thrust
         and torque); P the covariance of what the model cannot know (turbulence),
  update x = mu + K (y - H mu), K = P H' (H P H' + V)^-1.
A Kalman filter adds time correlation on top; the static form shows what each sensor set
can and cannot observe.
"""
import numpy as np

from .chain import rotx, run
from .records import synthetic

SENSOR_SETS = {
    "S0": "SCADA + nacelle accelerometer",
    "S1": "S0 + tower-top strain gauges",
    "S2": "S1 + blade-root strain gauges",
    "S3": "S2 + two local gauges on the drivetrain",
}


def chain_maps(fe, psi):
    """Linear maps from x (21) to every link, per time step: dicts of n × rows × 21 arrays."""
    n = len(psi)
    Rm = rotx(psi)
    Rb = np.zeros((n, 6, 6)); Rb[:, :3, :3] = Rm; Rb[:, 3:, 3:] = Rm
    T_fl = np.einsum("nij,jk->nik", Rb, fe["R_hub"]) / 1e6                   # shaft flange, fixed frame, MN
    Racc = np.zeros((n, 3, 21)); Racc[:, :, 18:] = Rm                        # fixed-frame acceleration
    L_dt = np.concatenate([T_fl, Racc], 1)
    mb = np.einsum("ij,njk->nik", fe["R_mb"], L_dt) / 1e6
    aL = np.einsum("ij,njk->nik", fe["R_armL"], L_dt) / 1e6
    aR = np.einsum("ij,njk->nik", fe["R_armR"], L_dt) / 1e6
    L_bed = np.concatenate([mb, aL[:, :3], aR[:, :3], Racc], 1)
    yaw = np.einsum("ij,njk->nik", fe["R_yaw"], L_bed) / 1e6
    L_hub = np.broadcast_to(np.eye(21), (n, 21, 21))
    return dict(fl=T_fl, acc=Racc, L_dt=L_dt, L_bed=L_bed, L_hub=L_hub, mb=mb, armL=aL, armR=aR, yaw=yaw)


def gauges(fe):
    """Two local strain gauges: torque-arm root on the gearbox housing, main-bearing seat on the bedplate.

    Each reads normal stress along the direction of largest unit-case response at a vertex away
    from supports. Returns (part, vertex, unit normal) per gauge.
    """
    out = []
    for part, pick in (("drivetrain", lambda X, S: (~fe["drivetrain_is_shaft"]) & (X[:, 0] > 0.1) & (X[:, 0] < 0.7) & (np.abs(X[:, 2]) > 0.9) & (np.abs(X[:, 2]) < 1.2)),
                       ("bedplate", lambda X, S: (X[:, 0] > -3.6) & (X[:, 0] < -2.8) & (X[:, 1] < 1.6))):
        X = fe[f"{part}_origin"] + fe[f"{part}_grid"] * float(fe["h"])
        S = fe[f"{part}_stress"].astype(float)
        ok = pick(X, S) & ~fe[f"{part}_near_bc"]
        mag = np.abs(S[:, :, 0]).max(1) + np.abs(S[:, :, 1]).max(1) + np.abs(S[:, :, 2]).max(1)
        v = np.flatnonzero(ok)[np.argmax(mag[ok])]
        k = int(np.argmax([np.abs(S[v, :, i]).max() for i in range(3)]))
        out.append((part, int(v), np.eye(3)[k]))
    return out


def gauge_row(fe, part, v, nrm):
    """Normal stress (MPa) at a gauge per unit case of that part."""
    s = fe[f"{part}_stress"][v].astype(float)                                # cases × 6
    w = np.array([nrm[0] ** 2, nrm[1] ** 2, nrm[2] ** 2, 2 * nrm[0] * nrm[1], 2 * nrm[1] * nrm[2], 2 * nrm[2] * nrm[0]])
    return s @ w / 1e6


def sensors(fe, maps, which):
    """Measurement rows H (n × m × 21) and noise standard deviations (m) for a sensor set."""
    rows, sd, names = [], [], []
    def add(r, s, nm):
        rows.append(r); sd.append(s); names.append(nm)
    add(maps["fl"][:, 3:4], 0.4, "SCADA torque")
    add(maps["acc"], np.full(3, 0.01), "nacelle accelerometer x y z")
    if which >= 1:
        add(maps["yaw"][:, 3:6], np.full(3, 0.5), "tower-top gauges Mx My Mz")
    if which >= 2:
        I = np.zeros((maps["fl"].shape[0], 6, 21))
        for k in range(3):
            I[:, 2 * k, 6 * k + 3] = 1; I[:, 2 * k + 1, 6 * k + 4] = 1
        add(I, np.full(6, 0.5), "blade-root flap and edge moments ×3")
    if which >= 3:
        for part, v, nrm in gauges(fe):
            g = gauge_row(fe, part, v, nrm)
            L = maps["L_dt"] if part == "drivetrain" else maps["L_bed"]
            add(np.einsum("c,nck->nk", g, L)[:, None, :], np.array([0.5]), f"strain gauge on the {part}")
    return np.concatenate(rows, 1), np.concatenate([np.atleast_1d(s) for s in sd]), names


def prior_mean(rec, tb):
    """What the global model knows without turbulence: gravity, centrifugal, mean thrust and shear,
    and torque split equally between blades, from measured azimuth, speed and generator torque."""
    n = rec["t"].size
    psi, rpm, A = rec["psi"], rec["rpm"], rec["A"]
    om = rpm / 60 * 2 * np.pi
    mu = np.zeros((21, n))
    op = rec["kind"] == "op"
    torque = rec["torque_aero"]
    for k in range(3):
        pk = psi + 2 * np.pi * k / 3
        er = np.stack([np.zeros(n), np.cos(pk), np.sin(pk)], 1); et = np.stack([np.zeros(n), -np.sin(pk), np.cos(pk)], 1)
        xh = np.array([1.0, 0, 0])
        f_flap = (0.70e6 * (1 + 0.05 * np.cos(pk))) if op else np.full(n, 0.23e6)
        Ff = f_flap[:, None] * xh
        f_in = (torque / 3 / tb.r_edge)[:, None] * et
        F_inert = -tb.blade_mass * A
        F = Ff + f_in + F_inert + (tb.blade_mass * om ** 2 * (tb.r_root + tb.blade_cg))[:, None] * er
        M = (np.cross((tb.r_flap - tb.r_root) * er, Ff) + np.cross((tb.r_edge - tb.r_root) * er, f_in)
             + np.cross(tb.blade_cg * er, F_inert) - 0.3e6 * er)
        mu[6 * k:6 * k + 6] = np.array([np.sum(F * xh, 1), np.sum(F * et, 1), np.sum(F * er, 1),
                                        -np.sum(M * et, 1), np.sum(M * xh, 1), np.sum(M * er, 1)]) / 1e6
    Rm = rotx(psi)
    mu[18:] = np.einsum("nji,nj->in", Rm, A)
    return mu


def prior_cov(tb, kind="op", seeds=range(101, 113)):
    """Covariance of (truth - prior mean) over independent turbulence seeds."""
    E = []
    for s in seeds:
        r = synthetic(kind, tb, seed=s)
        x = np.vstack([r["blade"].reshape(18, -1) / 1e6, np.einsum("nji,nj->in", rotx(r["psi"]), r["A"])])
        E.append(x - prior_mean(r, tb))
    E = np.hstack(E)
    P = np.cov(E)
    P[18:, 18:] += np.eye(3) * 1.0                                            # accelerations: loose prior
    return P + 1e-6 * np.eye(21)


def estimate(rec, fe, P, which, tb, rng=None):
    """Simulated measurements (truth + noise) -> estimate of x at every time step."""
    rng = rng or np.random.default_rng(5)
    n = rec["t"].size
    x_true = np.vstack([rec["blade"].reshape(18, n) / 1e6, np.einsum("nji,nj->in", rotx(rec["psi"]), rec["A"])])
    maps = chain_maps(fe, rec["psi"])
    H, sd, names = sensors(fe, maps, which)
    y = np.einsum("nmk,kn->nm", H, x_true) + sd * rng.standard_normal((n, len(sd)))
    mu = prior_mean(rec, tb)
    V = np.diag(sd ** 2)
    x_hat = np.empty_like(x_true)
    post_sd = np.empty_like(x_true)
    for t in range(n):
        Ht = H[t]
        S = Ht @ P @ Ht.T + V
        K = np.linalg.solve(S, Ht @ P).T
        x_hat[:, t] = mu[:, t] + K @ (y[t] - Ht @ mu[:, t])
        post_sd[:, t] = np.sqrt(np.clip(np.diag(P - K @ Ht @ P), 0, None))
    return x_true, x_hat, post_sd, names


def as_record(rec, x, fe):
    """A record with the blade-root loads and accelerations replaced by x, ready for the chain."""
    n = rec["t"].size
    r = dict(rec)
    r["blade"] = (x[:18] * 1e6).reshape(3, 6, n)
    r["A"] = np.einsum("nij,jn->ni", rotx(rec["psi"]), x[18:])
    r["torque_aero"] = run(r, fe)["flange"][3]
    return r


def r2(truth, est):
    d = truth - truth.mean()
    return float(1 - np.sum((est - truth) ** 2) / max(np.sum(d ** 2), 1e-30))
