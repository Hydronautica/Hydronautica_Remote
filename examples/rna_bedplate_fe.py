"""3D unit-load FE of a representative cast bedplate (geared, 3-point drivetrain mount).

Geometry: a cast bedplate on the yaw bearing with a front main-bearing housing and two
rear lugs for the gearbox torque arms. Voxel (8-node hexahedron) linear FE in ductile
iron, fixed on the yaw-bearing face.

Drivetrain coupling: the main shaft and gearbox are one rigid body (a master node at the
hub centre) tied to the bedplate by springs:
  * main bearing seat: radial and axial springs only (a bearing passes no torque),
  * gearbox torque-arm mounts: elastomer springs in all three directions.
Hub-centre loads Fx..Mz go on the master node, so the drivetrain shares them between the
bearing and the torque arms the way the stiffness dictates, while staying linear.

Unit cases: hub-centre Fx, Fy, Fz (1 MN), Mx, My, Mz (1 MN m) and bedplate body
acceleration ax, ay, az (1 m/s^2). Output: surface mesh, per-vertex stress tensor and
displacement per unit case, written compactly to demo/rna/bedplate.json for the viewer.
"""

import base64
import json
import os
import time

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "demo", "rna", "bedplate.json")
E, NU, RHO = 169e9, 0.275, 7100.0
H = 0.10                                    # voxel size (m)
X0, X1, Y0, Y1, Z0, Z1 = -3.6, 1.0, 0.0, 3.2, -2.6, 2.6
SHAFT_Y = 2.55                              # shaft axis height above the yaw bearing (m)
HUB = np.array([-6.2, SHAFT_Y, 0.0])        # hub centre (master node)
BEAR_X = (-3.5, -2.9)                       # main bearing seat, axial extent
BEAR_R = (0.62, 1.05)                       # seat bore radius, outer radius of the housing
ARMS_X = (0.1, 0.7)


def solid(p):
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    r = np.hypot(x + 0.2, z)
    ring = (y < 0.35) & (r > 1.85) & (r < 2.5)                       # yaw bearing flange
    # tapered body rising from the ring towards the bearing housing (cast shell, 0.25 m walls)
    u = np.clip((y - 0.35) / 1.85, 0, 1)
    cx = -0.2 - 1.0 * u
    ro = 2.5 - 0.75 * u
    rr = np.hypot(x - cx, z)
    body = (y >= 0.35) & (y < 2.2) & (rr < ro) & (rr > ro - 0.27)
    deck = (y >= 1.95) & (y < 2.2) & (rr < ro) & (np.abs(z) > 0.85)   # top flanges each side of the shaft opening
    # forward snout: floor and side walls carrying the bearing housing
    snout_x = (x > -3.5) & (x < -1.0)
    floor = snout_x & (y > 1.2) & (y < 1.45) & (np.abs(z) < 1.25)
    walls = snout_x & (y > 1.2) & (y < 2.9) & (np.abs(z) > 1.0) & (np.abs(z) < 1.25)
    rho = np.hypot(y - SHAFT_Y, z)
    housing = (x > BEAR_X[0]) & (x < BEAR_X[1]) & (rho > BEAR_R[0]) & (rho < BEAR_R[1] + 0.25) & (y > 1.2)
    web = (x > BEAR_X[0]) & (x < BEAR_X[1]) & (np.abs(z) < 1.25) & (y > 1.2) & (y < SHAFT_Y) & (rho > BEAR_R[0])
    # rear lugs for the gearbox torque arms, tied into the body
    lugs = (x > ARMS_X[0] - 0.1) & (x < ARMS_X[1] + 0.1) & (np.abs(z) > 1.15) & (np.abs(z) < 1.85) & (y > 1.3) & (y < 2.35)
    return ring | body | deck | floor | walls | housing | web | lugs


def hex_stiffness(h):
    D = E / ((1 + NU) * (1 - 2 * NU)) * np.array([
        [1 - NU, NU, NU, 0, 0, 0], [NU, 1 - NU, NU, 0, 0, 0], [NU, NU, 1 - NU, 0, 0, 0],
        [0, 0, 0, (1 - 2 * NU) / 2, 0, 0], [0, 0, 0, 0, (1 - 2 * NU) / 2, 0], [0, 0, 0, 0, 0, (1 - 2 * NU) / 2]])
    corners = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float)
    s = np.array([-1, 1, 1, -1, -1, 1, 1, -1]); t = np.array([-1, -1, 1, 1, -1, -1, 1, 1]); w = np.array([-1, -1, -1, -1, 1, 1, 1, 1])

    def B_at(xi, eta, zeta):
        dN = np.array([s * (1 + t * eta) * (1 + w * zeta), t * (1 + s * xi) * (1 + w * zeta), w * (1 + s * xi) * (1 + t * eta)]) / 8
        dN = dN * 2 / h                                  # d/dx = d/dxi * 2/h for a cube of side h
        B = np.zeros((6, 24))
        for a in range(8):
            dx, dy, dz = dN[:, a]
            B[:, 3 * a:3 * a + 3] = [[dx, 0, 0], [0, dy, 0], [0, 0, dz], [dy, dx, 0], [0, dz, dy], [dz, 0, dx]]
        return B
    g = 1 / np.sqrt(3)
    K = np.zeros((24, 24))
    for xi in (-g, g):
        for eta in (-g, g):
            for zeta in (-g, g):
                B = B_at(xi, eta, zeta)
                K += B.T @ D @ B * (h / 2) ** 3
    return K, D @ B_at(0, 0, 0), corners


def skew(r):
    return np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])


def main():
    t0 = time.time()
    nx, ny, nz = round((X1 - X0) / H), round((Y1 - Y0) / H), round((Z1 - Z0) / H)
    ii, jj, kk = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    cen = np.stack([X0 + (ii + 0.5) * H, Y0 + (jj + 0.5) * H, Z0 + (kk + 0.5) * H], -1)
    filled = solid(cen)
    el = np.argwhere(filled)
    print(f"{len(el)} voxels ({len(el) * H**3:.1f} m3, {len(el) * H**3 * RHO / 1e3:.0f} t)")
    Ke, DB0, corners = hex_stiffness(H)
    # node numbering on the grid
    nid = lambda i, j, k: (i * (ny + 1) + j) * (nz + 1) + k
    conn = np.stack([nid(el[:, 0] + int(c[0]), el[:, 1] + int(c[1]), el[:, 2] + int(c[2])) for c in corners], 1)
    used, conn = np.unique(conn, return_inverse=True)
    conn = conn.reshape(-1, 8)
    nn = len(used)
    gi = used // ((ny + 1) * (nz + 1)); gj = (used // (nz + 1)) % (ny + 1); gk = used % (nz + 1)
    XYZ = np.stack([X0 + gi * H, Y0 + gj * H, Z0 + gk * H], 1)
    ndof = 3 * nn + 6                                   # + master node (rigid drivetrain)
    dofs = (3 * conn[:, :, None] + np.arange(3)).reshape(-1, 24)
    rows = np.repeat(dofs, 24, axis=1).ravel(); cols = np.tile(dofs, (1, 24)).ravel()
    vals = np.tile(Ke.ravel(), len(el))
    R, C, V = [rows], [cols], [vals]
    m = 3 * nn

    def tie(nodes, k, proj):
        """Penalty springs u_j - (u_m - skew(r_j) theta_m), projected per node."""
        for j, P in zip(nodes, proj):
            r = XYZ[j] - HUB
            T = np.hstack([np.eye(3), -skew(r)])          # 3 x 6
            kP = k * P
            dj = 3 * j + np.arange(3); dm = m + np.arange(6)
            for a, b, blk in ((dj, dj, kP), (dj, dm, -kP @ T), (dm, dj, -(kP @ T).T), (dm, dm, T.T @ kP @ T)):
                R.append(np.repeat(a, len(b))); C.append(np.tile(b, len(a))); V.append(blk.ravel())
    # main bearing seat: nodes on the bore surface
    rho = np.hypot(XYZ[:, 1] - SHAFT_Y, XYZ[:, 2])
    seat = np.flatnonzero((XYZ[:, 0] >= BEAR_X[0] - 1e-6) & (XYZ[:, 0] <= BEAR_X[1] + 1e-6) & (np.abs(rho - BEAR_R[0]) < 0.75 * H))
    proj = []
    for j in seat:
        n = np.array([0, XYZ[j, 1] - SHAFT_Y, XYZ[j, 2]]); n /= np.linalg.norm(n)
        proj.append(np.outer(n, n) + np.outer([1, 0, 0], [1, 0, 0]) * 0.5)   # radial + some axial, no tangential
    tie(seat, 2.0e10 / len(seat), proj)
    # torque-arm mounts: top faces of the rear lugs
    arm = np.flatnonzero((XYZ[:, 0] >= ARMS_X[0]) & (XYZ[:, 0] <= ARMS_X[1]) & (np.abs(np.abs(XYZ[:, 2]) - 1.5) < 0.31) & (np.abs(XYZ[:, 1] - 2.3) < 0.051))
    tie(arm, 4.0e9 / len(arm), [np.diag([0.5, 1.0, 0.5])] * len(arm))
    K = coo_matrix((np.concatenate(V), (np.concatenate(R), np.concatenate(C))), shape=(ndof, ndof)).tocsc()
    fixed = np.flatnonzero(XYZ[:, 1] < 1e-6)            # yaw bearing face
    fdof = (3 * fixed[:, None] + np.arange(3)).ravel()
    free = np.setdiff1d(np.arange(ndof), fdof)
    print(f"{nn} nodes, {ndof} dof, seat nodes {len(seat)}, arm nodes {len(arm)}; assembly {time.time() - t0:.1f} s")
    # load cases
    F = np.zeros((ndof, 9))
    for c in range(6):
        F[m + c, c] = 1e6                                # hub-centre Fx, Fy, Fz, Mx, My, Mz
    nodal_mass = np.zeros(nn)
    np.add.at(nodal_mass, conn.ravel(), RHO * H**3 / 8)
    for c in range(3):
        F[3 * np.arange(nn) + c, 6 + c] = -nodal_mass    # body load for a unit acceleration of the support frame (d'Alembert)
    t1 = time.time()
    lu = splu(K[free][:, free], permc_spec="MMD_AT_PLUS_A")
    U = np.zeros((ndof, 9)); U[free] = lu.solve(F[free])
    print(f"factor + 9 solves {time.time() - t1:.1f} s")
    # element stresses (centre) per case
    ue = U[dofs]                                         # (nel, 24, 9)
    sig = np.einsum("sk,ekc->ecs", DB0, ue)              # (nel, 9, 6): sxx syy szz txy tyz tzx
    # surface: voxel faces with an empty neighbour
    occ = np.zeros((nx + 2, ny + 2, nz + 2), bool); occ[1:-1, 1:-1, 1:-1] = filled
    face_defs = [((-1, 0, 0), [0, 4, 7, 3]), ((1, 0, 0), [1, 2, 6, 5]), ((0, -1, 0), [0, 1, 5, 4]),
                 ((0, 1, 0), [3, 7, 6, 2]), ((0, 0, -1), [0, 3, 2, 1]), ((0, 0, 1), [4, 5, 6, 7])]
    quads, qel = [], []
    for (di, dj, dk), cidx in face_defs:
        nb = occ[el[:, 0] + 1 + di, el[:, 1] + 1 + dj, el[:, 2] + 1 + dk]
        sel = np.flatnonzero(~nb)
        quads.append(conn[sel][:, cidx]); qel.append(sel)
    quads = np.vstack(quads); qel = np.concatenate(qel)
    sv, quads = np.unique(quads, return_inverse=True); quads = quads.reshape(-1, 4)
    # per-vertex stress: average of the elements whose faces touch the vertex
    acc = np.zeros((len(sv), 9, 6)); cnt = np.zeros(len(sv))
    for q in range(4):
        np.add.at(acc, quads[:, q], sig[qel]); np.add.at(cnt, quads[:, q], 1)
    vs = acc / cnt[:, None, None]
    vu = U[(3 * sv[:, None] + np.arange(3))]             # (nv, 3, 9)
    vm = lambda s: np.sqrt(0.5 * ((s[..., 0] - s[..., 1]) ** 2 + (s[..., 1] - s[..., 2]) ** 2 + (s[..., 2] - s[..., 0]) ** 2) + 3 * (s[..., 3] ** 2 + s[..., 4] ** 2 + s[..., 5] ** 2))
    names = ["Fx", "Fy", "Fz", "Mx", "My", "Mz", "ax", "ay", "az"]
    for c in range(9):
        print(f"  unit {names[c]:2s}: max von Mises {vm(vs[:, c]).max() / 1e6:8.3f} MPa, max displacement {np.abs(vu[:, :, c]).max() * 1e3:.4f} mm")
    # reaction check for case Fy: sum of support reactions = applied load
    react = (K @ U[:, 1])[fdof].reshape(-1, 3).sum(0)
    print("  support reaction for unit Fy (N):", np.round(react))
    # compact export
    def b64(a):
        return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()
    S = vs.transpose(1, 0, 2)                            # (9, nv, 6)
    s_scale = np.abs(S).reshape(9, -1).max(1) / 32000
    D = vu.transpose(2, 0, 1)                            # (9, nv, 3)
    d_scale = np.abs(D).reshape(9, -1).max(1) / 32000
    gidx = np.stack([gi[sv], gj[sv], gk[sv]], 1).astype(np.uint8)
    tris = np.vstack([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]]).astype(np.uint32)
    out = dict(
        note="Representative cast bedplate, voxel hex FE, ductile iron. Stress in Pa per unit case, displacement in m per unit case.",
        h=H, origin=[X0, Y0, Z0], shaft_y=SHAFT_Y, hub=HUB.tolist(), bearing_x=BEAR_X, bearing_r=BEAR_R, arms_x=ARMS_X,
        cases=names, units=["MN", "MN", "MN", "MN·m", "MN·m", "MN·m", "m/s²", "m/s²", "m/s²"],
        nv=int(len(sv)), nt=int(len(tris)), mass_t=float(len(el) * H**3 * RHO / 1e3),
        grid=b64(gidx), tris=b64(tris),
        s_scale=s_scale.tolist(), stress=b64(np.round(S / s_scale[:, None, None]).astype(np.int16)),
        d_scale=d_scale.tolist(), disp=b64(np.round(D / d_scale[:, None, None]).astype(np.int16)),
    )
    with open(OUT, "w") as fh:
        json.dump(out, fh, separators=(",", ":"))
    print(f"wrote {OUT}: {len(sv)} vertices, {len(tris)} triangles, {os.path.getsize(OUT) / 1e6:.1f} MB; total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
