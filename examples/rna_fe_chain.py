"""Unit-load FE of every structural part of a geared rotor-nacelle assembly, chained by reactions.

Three voxel FE models in ductile iron / forged steel (representative 14 MW class geometry):

  hub         cast spherical hub with three blade flanges, held at the shaft flange.
              21 unit cases: 6 per blade root (blade frame) + 3 hub-frame accelerations.
              It turns with the rotor, so it is solved in the rotor frame.
  drivetrain  forged main shaft + cast gearbox housing in one model. Shaft and housing are
              joined only by the front carrier bearing (radial springs) and the planet
              carrier (radial + tangential springs: rear carrier bearing and gear mesh).
              Held by the main bearing (radial + axial springs, one plane, self-aligning)
              and the two torque-arm elastomers. 9 unit cases: 6 shaft-flange loads about
              the hub centre (fixed frame) + 3 accelerations.
  bedplate    cast bedplate held at the yaw bearing. 15 unit cases: main-bearing seat
              load (6), each torque-arm pad force (3 + 3), 3 accelerations.

Every model's support reactions per unit case are stored as a matrix, so the reactions of
one part become the loads on the next: blade roots -> hub -> drivetrain -> bedplate -> tower.

Writes demo/rna/fe_chain.npz (full precision, for fatigue in Python and the page build).
"""
import os
import time

import numpy as np
from scipy.spatial import cKDTree

import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from rna.fe import VoxelModel, von_mises  # noqa: E402

OUT = os.path.join(ROOT, "demo", "rna", "fe_chain.npz")
H = 0.10
SY = 2.55                                   # shaft axis height above the yaw bearing
HUB = np.array([-7.6, SY, 0.0])
R_HUB, T_HUB, R_ROOT = 2.84, 0.35, 1.85     # hub outer radius, wall, blade-flange plane distance
X_FLANGE = HUB[0] + 2.70                    # hub/shaft flange plane (x = -4.90)
X_MB = -3.2                                 # main bearing plane
PADS = np.array([[0.4, 2.35, 1.5], [0.4, 2.35, -1.5]])   # torque-arm elastomer pads (left, right)
YAW = np.array([-0.2, 0.0, 0.0])
REST_MASS, REST_CG = 170e3, np.array([3.2, 2.2, 0.0])   # generator, rear frame, cover: lumped on the bedplate rear face
CAST = dict(E=169e9, nu=0.275, rho=7100.0)
STEEL = dict(E=206e9, nu=0.30, rho=7850.0)


def blade_axes(k):
    a = 2 * np.pi * k / 3
    er = np.array([0, np.cos(a), np.sin(a)]); et = np.array([0, -np.sin(a), np.cos(a)])
    return er, et


# ---------------------------------------------------------------------------------------------
def hub_solid(p):
    q = p - HUB
    r = np.linalg.norm(q, axis=-1)
    shell = (r < R_HUB) & (r > R_HUB - T_HUB)
    keep = q[..., 0] < 2.70                                       # rear opening for the shaft flange
    keep &= q[..., 0] > -2.55                                     # front (spinner) opening
    rho_x = np.hypot(q[..., 1], q[..., 2])
    flange = (q[..., 0] > 2.45) & (q[..., 0] < 2.70) & (rho_x > 0.55) & (rho_x < 1.1)
    out = (shell & keep) | flange
    for k in range(3):
        er, _ = blade_axes(k)
        s = q @ er
        rho = np.linalg.norm(q - s[..., None] * er, axis=-1)
        out &= s < R_ROOT                                         # blade opening
        out |= (s > R_ROOT - 0.40) & (s < R_ROOT) & (rho > 1.60) & (rho < 2.30)   # pitch-bearing flange and bearing rings
    return out


def build_hub():
    m = VoxelModel(hub_solid, ((-10.5, -4.8), (SY - 2.9, SY + 2.9), (-2.9, 2.9)), H, **CAST)
    q = m.X - HUB
    m.fix("shaft_flange", np.flatnonzero((q[:, 0] > 2.70 - 0.5 * H) & (np.hypot(q[:, 1], q[:, 2]) < 1.1 + 1e-6)))
    m.bc = [m.fixed["shaft_flange"]]
    F, names = [], []
    for k in range(3):
        er, et = blade_axes(k)
        s = q @ er
        nodes = np.flatnonzero((s > R_ROOT - 0.75 * H) & (s < R_ROOT + 1e-6))
        m.bc.append(nodes)
        c = HUB + R_ROOT * er
        x = np.array([1.0, 0, 0])
        for nm, Fv, Mv in (("Foop", x, 0 * x), ("Fedge", et, 0 * x), ("Frad", er, 0 * x),
                           ("Moop", 0 * x, -et), ("Medge", 0 * x, x), ("Mpitch", 0 * x, er)):
            F.append(m.load(nodes, c, 1e6 * Fv, 1e6 * Mv)); names.append(f"{'ABC'[k]} {nm}")
    for i, a in enumerate("xyz"):
        F.append(m.accel(np.eye(3)[i])); names.append(f"a{a}")
    return m, np.array(F).T, names


# ---------------------------------------------------------------------------------------------
def shaft_region(p):
    x, rho = p[..., 0], np.hypot(p[..., 1] - SY, p[..., 2])
    flange = (x > X_FLANGE) & (x < X_FLANGE + 0.2) & (rho > 0.25) & (rho < 1.0)
    body = (x >= X_FLANGE + 0.2) & (x < -2.6) & (rho > 0.25) & (rho < 0.725)
    tail = (x >= -2.6) & (x < -0.45) & (rho > 0.25) & (rho < 0.6)
    carrier = (x >= -0.45) & (x < -0.25) & (rho > 0.25) & (rho < 1.15)         # planet carrier, joined to the ring gear (the mesh)
    return flange | body | tail | carrier


def housing_region(p):
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    rho = np.hypot(y - SY, z)
    cover = (x > -1.45) & (x < -1.25) & (rho > 0.8) & (rho < 1.3)
    ring = (x > -1.45) & (x < -0.05) & (rho > 1.1) & (rho < 1.3)                # stage 1 / ring gear
    step = (x > -0.05) & (x < 0.15) & (rho > 0.8) & (rho < 1.3)
    stage2 = (x > -0.05) & (x < 0.85) & (rho > 0.8) & (rho < 1.0)
    back = (x > 0.65) & (x < 0.85) & (rho > 0.3) & (rho < 1.0)
    arms = (x > 0.1) & (x < 0.7) & (y > 2.35) & (y < 2.85) & (np.abs(z) > 0.85) & (np.abs(z) < 1.75)
    return cover | ring | step | stage2 | back | arms


def pair(m, a, b, rb):
    """Pair each node of a with the nearest node of b after pushing a radially out to radius rb."""
    pa = m.X[a].copy(); d = pa[:, 1:] - [SY, 0]
    pa[:, 1:] = [SY, 0] + d / np.linalg.norm(d, axis=1, keepdims=True) * rb
    _, j = cKDTree(m.X[b]).query(pa)
    return b[j]


def radial_tangential(m, nodes, wr, wt, wa=0.0):
    d = m.X[nodes][:, 1:] - [SY, 0]; d /= np.linalg.norm(d, axis=1, keepdims=True)
    n = np.column_stack([np.zeros(len(nodes)), d]); t = np.column_stack([np.zeros(len(nodes)), -d[:, 1], d[:, 0]])
    x = np.array([1.0, 0, 0])
    return wr * n[:, :, None] * n[:, None, :] + wt * t[:, :, None] * t[:, None, :] + wa * np.outer(x, x)


def build_drivetrain():
    solid = lambda p: shaft_region(p) | housing_region(p)
    internals = lambda p: np.where(housing_region(p), 2.5, 1.0)           # gears and carrier smeared into the housing
    m = VoxelModel(solid, ((-5.0, 1.0), (SY - 1.4, SY + 1.4), (-1.9, 1.9)), H, extra_mass=internals, **STEEL)
    m.cast = housing_region(m.origin + (m.el + 0.5) * H)                   # element flags
    x, rho = m.X[:, 0], np.hypot(m.X[:, 1] - SY, m.X[:, 2])
    on_shaft = np.zeros(m.nn, bool); on_shaft[np.unique(m.conn[~m.cast])] = True
    on_house = np.zeros(m.nn, bool); on_house[np.unique(m.conn[m.cast])] = True
    # shaft and housing meet only where the planet carrier joins the ring gear
    assert not (on_shaft & on_house & (x < -0.5)).any(), "shaft and housing touch away from the gear mesh"
    m.shaft_nodes = np.flatnonzero(on_shaft)
    # main bearing: one plane of shaft surface nodes, radial + axial
    mb = np.flatnonzero(on_shaft & (np.abs(x - X_MB) < 0.5 * H) & (rho > 0.6))
    m.add_springs("main_bearing", mb, 2.0e10, radial_tangential(m, mb, 1, 0))
    m.add_mean_spring("main_bearing", mb, 1.0e10, [1, 0, 0])        # axial on the ring average: self-aligning
    # front carrier bearing: shaft (rho ~0.6) to cover bore (rho ~0.8), radial
    a = np.flatnonzero(on_shaft & (np.abs(x + 1.35) < 0.11) & (rho > 0.5) & (rho < 0.65))
    b = np.flatnonzero(on_house & (np.abs(x + 1.35) < 0.11) & (rho > 0.75) & (rho < 0.85))
    m.add_ties("carrier_front", a, pair(m, a, b, 0.8), 1.0e10, radial_tangential(m, a, 1, 0))
    for i, c in enumerate(PADS):
        pad = np.flatnonzero(on_house & (np.abs(m.X[:, 1] - c[1]) < 0.4 * H) & (np.abs(x - c[0]) < 0.31) & (np.abs(m.X[:, 2] - c[2]) < 0.26))
        m.add_springs("arm_" + "LR"[i], pad, 2.5e9, np.diag([0.2, 1.0, 0.2]))
    front = np.flatnonzero(on_shaft & (x < X_FLANGE + 0.5 * H))
    m.bc = [front, mb] + [v[0] for v in m.springs.values()] + [np.r_[t[0], t[1]] for t in m.ties.values()]
    F, names = [], []
    for i, nm in enumerate(("Fx", "Fy", "Fz")):
        F.append(m.load(front, HUB, F=1e6 * np.eye(3)[i])); names.append(nm)
    for i, nm in enumerate(("Mx", "My", "Mz")):
        F.append(m.load(front, HUB, M=1e6 * np.eye(3)[i])); names.append(nm)
    for i, a in enumerate("xyz"):
        F.append(m.accel(np.eye(3)[i])); names.append(f"a{a}")
    return m, np.array(F).T, names


# ---------------------------------------------------------------------------------------------
BEAR_X, BEAR_R, ARMS_X = (-3.5, -2.9), (0.62, 1.05), (0.1, 0.7)


def bedplate_solid(p):
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    r = np.hypot(x + 0.2, z)
    ring = (y < 0.35) & (r > 1.85) & (r < 2.5)
    u = np.clip((y - 0.35) / 1.85, 0, 1)
    cx = -0.2 - 1.0 * u; ro = 2.5 - 0.75 * u
    rr = np.hypot(x - cx, z)
    body = (y >= 0.35) & (y < 2.2) & (rr < ro) & (rr > ro - 0.27)
    deck = (y >= 1.95) & (y < 2.2) & (rr < ro) & (np.abs(z) > 0.85)
    snout_x = (x > -3.5) & (x < -1.0)
    floor = snout_x & (y > 1.2) & (y < 1.45) & (np.abs(z) < 1.25)
    walls = snout_x & (y > 1.2) & (y < 2.9) & (np.abs(z) > 1.0) & (np.abs(z) < 1.25)
    rho = np.hypot(y - SY, z)
    housing = (x > BEAR_X[0]) & (x < BEAR_X[1]) & (rho > BEAR_R[0]) & (rho < BEAR_R[1] + 0.25) & (y > 1.2)
    web = (x > BEAR_X[0]) & (x < BEAR_X[1]) & (np.abs(z) < 1.25) & (y > 1.2) & (y < SY) & (rho > BEAR_R[0])
    lugs = (x > ARMS_X[0] - 0.1) & (x < ARMS_X[1] + 0.1) & (np.abs(z) > 1.15) & (np.abs(z) < 1.85) & (y > 1.3) & (y < 2.35)
    return ring | body | deck | floor | walls | housing | web | lugs


def build_bedplate():
    m = VoxelModel(bedplate_solid, ((-3.6, 1.0), (0.0, 3.2), (-2.6, 2.6)), H, **CAST)
    m.fix("yaw_bearing", np.flatnonzero(m.X[:, 1] < 1e-6))
    rho = np.hypot(m.X[:, 1] - SY, m.X[:, 2])
    seat = np.flatnonzero((m.X[:, 0] >= BEAR_X[0] - 1e-6) & (m.X[:, 0] <= BEAR_X[1] + 1e-6) & (np.abs(rho - BEAR_R[0]) < 0.75 * H))
    cmb = np.array([X_MB, SY, 0])
    m.bc = [m.fixed["yaw_bearing"], seat]
    F, names = [], []
    for i, nm in enumerate(("Fx", "Fy", "Fz")):
        F.append(m.load(seat, cmb, F=1e6 * np.eye(3)[i])); names.append("MB " + nm)
    for i, nm in enumerate(("Mx", "My", "Mz")):
        F.append(m.load(seat, cmb, M=1e6 * np.eye(3)[i])); names.append("MB " + nm)
    for s, c in zip("LR", PADS):
        pad = np.flatnonzero((np.abs(m.X[:, 1] - c[1]) < 0.051) & (np.abs(m.X[:, 0] - c[0]) < 0.31) & (np.abs(m.X[:, 2] - c[2]) < 0.26))
        for i, nm in enumerate(("Fx", "Fy", "Fz")):
            F.append(m.load(pad, c, F=1e6 * np.eye(3)[i])); names.append(f"Arm {s} {nm}")
        m.bc.append(pad)
    rear = np.flatnonzero(m.X[:, 0] > 1.0 - 0.5 * H)
    m.bc.append(rear)
    m.rest = (REST_MASS, REST_CG)
    for i, a in enumerate("xyz"):
        F.append(m.accel(np.eye(3)[i]) + m.load(rear, REST_CG, F=-REST_MASS * np.eye(3)[i])); names.append(f"a{a}")
    return m, np.array(F).T, names


def summary(tag, m, names, t):
    s = m.surface()
    vm = von_mises(s["stress"]).max(0) / 1e6
    print(f"{tag}: {len(m.el)} voxels, {m.mass / 1e3:.0f} t, {m.ndof} dof, solved in {t:.0f} s, {len(s['nodes'])} surface vertices")
    print("   max von Mises per unit case (MPa):", ", ".join(f"{n} {v:.2f}" for n, v in zip(names, vm)))
    return s


def main():
    res = {}
    for tag, build in (("hub", build_hub), ("drivetrain", build_drivetrain), ("bedplate", build_bedplate)):
        t0 = time.time()
        m, F, names = build()
        m.solve(F)
        s = summary(tag, m, names, time.time() - t0)
        res[tag] = (m, names, s)
    hub, dt, bed = (res[k][0] for k in ("hub", "drivetrain", "bedplate"))
    R_hub = hub.reaction("shaft_flange", HUB)                     # load the hub puts on the shaft, rotor frame
    R_mb = dt.reaction("main_bearing", np.array([X_MB, SY, 0]))
    R_aL, R_aR = dt.reaction("arm_L", PADS[0]), dt.reaction("arm_R", PADS[1])
    R_yaw = bed.reaction("yaw_bearing", YAW)
    # equilibrium checks: unit hub-centre Fy into the drivetrain comes out at its supports
    tot = R_mb[:3, 1] + R_aL[:3, 1] + R_aR[:3, 1]
    print("drivetrain unit Fy -> supports (N):", np.round(tot), "| hub unit 'A Frad' -> flange (N):", np.round(R_hub[:3, 2]))
    print("arm forces per MN m of torque (MN, push-pull):", np.round(R_aL[1, 3] / 1e6, 3), np.round(R_aR[1, 3] / 1e6, 3),
          "| main bearing moment for unit Mz (MN m):", np.round(R_mb[5, 5] / 1e6, 4), "| MB axial share of Fx:", np.round(R_mb[0, 0] / 1e6, 3))
    def about(R, frm, to):
        R = R.copy(); R[3:] += np.cross((frm - to)[:, None], R[:3], axis=0); return R
    tot = about(R_mb, np.array([X_MB, SY, 0]), HUB) + about(R_aL, PADS[0], HUB) + about(R_aR, PADS[1], HUB)
    err = np.abs(tot[:, :6] - 1e6 * np.eye(6)).max() / 1e6
    print(f"drivetrain equilibrium: largest error over the six flange unit cases {err:.2e} (MN, MN m)")
    out = dict(R_hub=R_hub, R_mb=R_mb, R_armL=R_aL, R_armR=R_aR, R_yaw=R_yaw, hub_c=HUB, pads=PADS, x_mb=X_MB, yaw=YAW, h=H)
    for tag, (m, names, s) in res.items():
        out[f"{tag}_names"] = np.array(names)
        out[f"{tag}_origin"] = m.origin
        out[f"{tag}_grid"] = s["grid"].astype(np.uint16)
        out[f"{tag}_tris"] = s["tris"].astype(np.uint32)
        out[f"{tag}_stress"] = s["stress"].astype(np.float32)       # nv × cases × 6 (Pa per unit)
        out[f"{tag}_disp"] = s["disp"].astype(np.float32)           # nv × 3 × cases (m per unit)
        bcx = m.X[np.unique(np.concatenate(m.bc))]
        d, _ = cKDTree(bcx).query(m.X[s["nodes"]])
        out[f"{tag}_near_bc"] = d < 2.5 * H                         # supports and load points: local artefacts, not hot spots
        out[f"{tag}_mass"] = m.mass
        out[f"{tag}_cg"] = m.cg
    out["rest_mass"], out["rest_cg"] = REST_MASS, REST_CG
    sv = res["drivetrain"][2]["nodes"]
    out["drivetrain_is_shaft"] = np.isin(sv, dt.shaft_nodes)
    np.savez_compressed(OUT, **out)
    print(f"wrote {OUT} ({os.path.getsize(OUT) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
