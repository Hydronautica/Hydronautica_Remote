"""The load chain: blade-root loads through the hub, drivetrain and bedplate FE models to the tower.

Each FE model (examples/rna_fe_chain.py) was solved once for unit loads at its inputs. Its
support reactions per unit case form a matrix, so the loads a part passes on are a matrix
product of the loads it receives:

    blade roots (18) + hub accel (3)  --R_hub-->  shaft-flange loads (6, rotor frame)
    rotate by azimuth                 ------->    shaft-flange loads (6, fixed frame)
    + nacelle accel (3)               --R_mb, R_arm-->  main bearing (6), torque arms (3 + 3)
    + nacelle accel (3)               --R_yaw-->  tower top (6)

At each link the result can be checked against plain rigid-body statics of the global loads.
"""
import numpy as np

from .components import (SN, _result, blade_root_bolts, gearbox_bearings, gearbox_gears, nacelle_equipment,
                         not_modelled, pitch_bearings, unit_load_hotspot)
from .fatigue import rainflow, place
from .fe import von_mises
from .records import Turbine

CAST_SN = SN(80e6, 2e6, 7, 13)          # representative ductile cast iron, stress range
SHAFT_SN = SN(320e6, 1e6, 5, 9)         # forged 42CrMo4 incl. size and surface, stress range


def load(path):
    d = np.load(path)
    return {k: d[k] for k in d.files}


def turbine_for(fe, **kw):
    """Turbine whose masses match the FE models, so the chain and the statics can be compared."""
    m = np.array([fe["drivetrain_mass"], fe["bedplate_mass"], fe["rest_mass"]])
    cg = np.array([fe["drivetrain_cg"], fe["bedplate_cg"], fe["rest_cg"]])
    return Turbine(hub_mass=float(fe["hub_mass"]), hub=tuple(fe["hub_c"]), yaw_centre=tuple(fe["yaw"]),
                   x_main_bearing=float(fe["x_mb"]), nacelle_mass=float(m.sum()),
                   nacelle_cg=tuple((m[:, None] * cg).sum(0) / m.sum()), **kw)


def rotx(psi):
    c, s = np.cos(psi), np.sin(psi)
    R = np.zeros((len(psi), 3, 3))
    R[:, 0, 0] = 1; R[:, 1, 1] = c; R[:, 1, 2] = -s; R[:, 2, 1] = s; R[:, 2, 2] = c
    return R


def run(rec, fe):
    n = rec["t"].size
    Rm = rotx(rec["psi"])                                         # rotor frame -> fixed frame
    A = rec["A"]
    A_rot = np.einsum("nji,nj->ni", Rm, A)
    L_hub = np.vstack([rec["blade"].reshape(18, n) / 1e6, A_rot.T])
    fl = fe["R_hub"] @ L_hub
    Ff = np.einsum("nij,jn->in", Rm, fl[:3]); Mf = np.einsum("nij,jn->in", Rm, fl[3:])
    L_dt = np.vstack([Ff / 1e6, Mf / 1e6, A.T])
    mb, aL, aR = fe["R_mb"] @ L_dt, fe["R_armL"] @ L_dt, fe["R_armR"] @ L_dt
    L_bed = np.vstack([mb / 1e6, aL[:3] / 1e6, aR[:3] / 1e6, A.T])
    yaw = fe["R_yaw"] @ L_bed
    return dict(L_hub=L_hub, L_dt=L_dt, L_bed=L_bed, flange=np.vstack([Ff, Mf]), mb=mb, armL=aL, armR=aR, yaw=yaw)


def closure(rec, ch):
    """How well each link matches rigid-body statics of the global loads (relative error)."""
    hub = np.vstack([rec["hub_F"].T, rec["hub_M"].T])
    tt = np.vstack([rec["tt_F"].T, rec["tt_M"].T])
    rel = lambda a, b: float(np.max(np.abs(a - b)) / np.max(np.abs(b)))
    return dict(hub=rel(ch["flange"], hub), tower=rel(ch["yaw"], tt))


# ---------------------------------------------------------------------------------------------
def _candidates(S, L, k):
    st = np.einsum("pck,cn->pnk", S, L[:, ::10])
    return np.argsort(von_mises(st).max(1))[-k:]


def fe_hotspot(fe, part, L, rec, key, name, note, mask=None, k=60):
    S = fe[f"{part}_stress"].astype(float)
    ok = ~fe[f"{part}_near_bc"] if f"{part}_near_bc" in fe else np.ones(len(S), bool)
    idx = np.flatnonzero(ok & (mask if mask is not None else True))
    cand = idx[_candidates(S[idx], L, k)]
    r = unit_load_hotspot(S[cand], L, CAST_SN, key=key, name=name, rec=rec)
    r["note"] = note
    r["part"], r["vertex"] = part, int(cand[r["point"]])
    return r


def shaft_rotating(fe, L_dt, rec, n_points=12):
    """Rotating-bending fatigue of the shaft from the FE field.

    The drivetrain is solved in the fixed frame; the shaft is axisymmetric, so a material point
    at angle phi0 sees the fixed-frame axial stress at angle phi0 + psi(t).
    """
    S = fe["drivetrain_stress"].astype(float)
    isS = fe["drivetrain_is_shaft"]
    g = fe["drivetrain_grid"].astype(float)
    P = fe["drivetrain_origin"] + g * float(fe["h"])
    sy = float(fe["hub_c"][1])
    rho = np.hypot(P[:, 1] - sy, P[:, 2]); th = np.arctan2(P[:, 2], P[:, 1] - sy)
    psi = rec["psi"]
    best = None
    phi0 = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    for xi in np.unique(g[isS, 0]):
        ring = np.flatnonzero(isS & (g[:, 0] == xi) & ~fe["drivetrain_near_bc"])
        if len(ring):
            ring = ring[rho[ring] > rho[ring].max() - 0.11]
        if len(ring) < 8:
            continue
        o = np.argsort(th[ring]); ring = ring[o]
        field = S[ring, :, 0] @ L_dt                                # axial stress at the ring nodes, nodes × n
        tq = np.r_[th[ring] - 2 * np.pi, th[ring], th[ring] + 2 * np.pi]
        fq = np.vstack([field, field, field])
        ang = np.mod(phi0[:, None] + psi[None, :] + np.pi, 2 * np.pi) - np.pi
        series = np.array([np.interp(ang[:, t], tq, fq[:, t]) for t in range(field.shape[1])]).T
        for p in range(n_points):
            rg, _, ct, ends = rainflow(series[p])
            dc = ct / SHAFT_SN.cycles(rg)
            if best is None or dc.sum() > best[0]:
                best = (dc.sum(), xi, p, series[p], rg, ct, ends, dc, ring)
    _, xi, p, sig, rg, ct, ends, dc, ring = best
    dmg = place(sig.size, ends, dc)
    x = float(fe["drivetrain_origin"][0] + xi * float(fe["h"]))
    m = [("Critical section", f"x = {x:.1f} m"), ("Axial stress range (max)", f"{rg.max() / 1e6:.0f} MPa"),
         ("Cycles in record", f"{ct.sum():.0f}"), ("Range / endurance range", f"{rg.max() / SHAFT_SN.s_knee:.2f}")]
    r = _result("main_shaft", "Main shaft", "Drivetrain FE (fixed frame) → material point turning with the rotor → rainflow → S-N",
                "DIN 743 / FKM", dmg, rec, m,
                "The FE gives the bending stress field in space; a point on the turning shaft passes through tension and compression once per revolution. Its worst section is found from the FE rather than assumed.",
                ("Axial stress at a material point on the critical section", "MPa", sig / 1e6))
    r["part"], r["vertex"] = "drivetrain", int(ring[0])
    return r


def main_bearing_chain(rec, ch, C=48e6, C0=95e6, e=0.30, Y1=2.3, X2=0.67, Y2=3.4, Y0=2.2):
    Fr = np.hypot(ch["mb"][1], ch["mb"][2]); Fa = np.abs(ch["mb"][0])
    P = np.where(Fa <= e * Fr, Fr + Y1 * Fa, X2 * Fr + Y2 * Fa)
    dmg = np.abs(rec["rpm"]) / 60 * rec["dt"] / ((C / P) ** (10 / 3) * 1e6)
    s0 = C0 / np.max(Fr + Y0 * Fa)
    moving = rec["rpm"] > 1.0
    life = 1e6 * np.mean((C / P[moving]) ** (10 / 3)) / (np.mean(rec["rpm"][moving]) * 60 * 8766) if moving.any() else None
    m = [("Radial load (mean)", f"{Fr.mean() / 1e6:.1f} MN"), ("Axial load (mean)", f"{Fa.mean() / 1e6:.1f} MN"),
         ("Equivalent load P/C", f"{P.mean() / C:.2f}"), ("L10 if this lasted", f"{life:.0f} years" if life else "not turning"),
         ("Static safety s0", f"{s0:.1f}"), ("Time under 2 % of C (slip risk)", f"{100 * np.mean(P[moving] < 0.02 * C) if moving.any() else 0:.0f} %")]
    return _result("main_bearing", "Main bearing", "Reaction from the drivetrain FE → equivalent load → L10 every step",
                   "ISO 281, ISO/TS 16281", dmg, rec, m,
                   "Its load is the drivetrain model's main-bearing reaction, so the split between bearing and torque arms comes from the stiffness of the shaft, gearbox and elastomers rather than a hand calculation.",
                   ("Equivalent load P", "MN", P / 1e6), check=1 / s0)


def yaw_bearing_chain(rec, ch, D=4.4, Ca=60e6, C0a=160e6):
    F, M = ch["yaw"][:3], ch["yaw"][3:]
    Fa = -F[1]; Fr = np.hypot(F[0], F[2]); Mt = np.hypot(M[0], M[2])
    P = 0.75 * Fr + np.abs(Fa) + 2.0 * Mt / D
    revs = np.abs(np.diff(rec["yaw"], prepend=rec["yaw"][0])) / 360
    dmg = revs / ((Ca / P) ** 3 * 1e6)
    s0 = C0a / P.max()
    m = [("Axial load (weight)", f"{Fa.mean() / 1e6:.1f} MN"), ("Tilting moment (max)", f"{Mt.max() / 1e6:.0f} MN·m"),
         ("Yaw torque (max)", f"{np.abs(M[1]).max() / 1e6:.1f} MN·m"), ("Yawed in record", f"{np.abs(np.diff(rec['yaw'])).sum():.0f}°"),
         ("Static safety s0", f"{s0:.1f}")]
    return _result("yaw_bearing", "Yaw bearing", "Bedplate FE reaction at the yaw bearing → equivalent load",
                   "NREL DG03", dmg, rec, m,
                   "Its load is the bedplate model's reaction at the yaw bearing: the end of the chain, checked against the tower-top load from the global model. It turns so rarely that the static margin and standstill wear decide its design.",
                   ("Equivalent load P", "MN", P / 1e6), check=1 / s0)


def assess_chain(rec, fe, tb=None):
    tb = tb or turbine_for(fe)
    ch = run(rec, fe)
    shaft = fe["drivetrain_is_shaft"]
    out = [
        fe_hotspot(fe, "hub", ch["L_hub"], rec, "hub", "Hub casting",
                   "Hub FE with 18 blade-root unit cases plus its own inertia, in the rotor frame. Blade-root bending and the hub's shape put the hot spots round the pitch-bearing flanges."),
        shaft_rotating(fe, ch["L_dt"], rec),
        main_bearing_chain(rec, ch),
        fe_hotspot(fe, "drivetrain", ch["L_dt"], rec, "housing", "Gearbox housing and torque arms",
                   "Drivetrain FE: the ring gear passes the rotor torque into the housing, which carries it out through the torque arms as a push-pull pair. The arm roots are where housings crack.", mask=~shaft),
        gearbox_gears(rec, tb), gearbox_bearings(rec, tb),
        fe_hotspot(fe, "bedplate", ch["L_bed"], rec, "bedplate", "Bedplate casting",
                   "Bedplate FE loaded by the drivetrain's reactions (main-bearing seat and torque-arm pads) and its own and the rear equipment's inertia."),
        pitch_bearings(rec, tb), blade_root_bolts(rec, tb), yaw_bearing_chain(rec, ch), nacelle_equipment(rec, tb),
        not_modelled("generator", "Generator bearings", "Bearing life from rotor weight and magnetic pull", "ISO 281",
                     "Its loads are internal (rotor weight, magnetic pull, speed) rather than from the global model. Better tracked by vibration and temperature monitoring."),
    ]
    return out, ch
