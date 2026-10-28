"""Unit-load superposition for a cast component (illustration).

A simplified main-bearing housing (2D plane stress, ductile cast iron) is solved
once for each unit interface load: Fx = 1 MN, Fy = 1 MN, Mz = 1 MN m at the
bearing seat. Because the model is linear elastic, the stress tensor anywhere at
any time is

    sigma(x, t) = Fx(t) * sigma_Fx(x) + Fy(t) * sigma_Fy(x) + Mz(t) * sigma_Mz(x),

so a whole load history becomes hot-spot stress histories with a few multiply-adds,
instead of one FE solve per time step. The script checks this against direct
solves, then does a critical-plane fatigue check at the hot spot.

Figures go to figures/rna/. Geometry, loads and the S-N curve are illustrative.
"""

import os
import time

import matplotlib
import numpy as np
from matplotlib.tri import Triangulation
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu
from scipy.spatial import Delaunay

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures", "rna")
os.makedirs(OUT, exist_ok=True)
E, NU, THK = 169e9, 0.275, 0.50          # EN-GJS-400 ductile iron; plane stress, 0.5 m housing depth
W, HGT = 2.0, 1.4
BORE = (1.0, 0.86, 0.33)                  # bearing seat: centre x, y and radius (m)
NOTCH = [(-0.05, 0.58, 0.45), (2.05, 0.58, 0.45)]   # side cut-outs that form the waist
INK, ACC, WARN, MUTED = "#12201f", "#1f6f8b", "#c0392b", "#6b7b7a"


# --- geometry and mesh -------------------------------------------------------------------
def inside(p):
    x, y = p[..., 0], p[..., 1]
    ok = (x > -1e-9) & (x < W + 1e-9) & (y > -1e-9) & (y < HGT + 1e-9)
    for cx, cy, r in [BORE] + NOTCH:
        ok &= (x - cx) ** 2 + (y - cy) ** 2 > r**2 - 1e-9
    return ok


def size_field(p):
    """Target element size: fine near the notch roots and the bore, coarse elsewhere."""
    d = np.full(len(p), 1.0)
    for cx, cy, r in [BORE] + NOTCH:
        d = np.minimum(d, np.abs(np.hypot(p[:, 0] - cx, p[:, 1] - cy) - r))
    d = np.minimum(d, p[:, 1])            # base
    return np.clip(0.012 + 0.10 * d, 0.012, 0.05)


def mesh():
    rng = np.random.default_rng(1)
    pts = []
    # boundary points
    for cx, cy, r in [BORE] + NOTCH:
        n = int(2 * np.pi * r / 0.008)
        a = np.linspace(0, 2 * np.pi, n, endpoint=False)
        c = np.c_[cx + r * np.cos(a), cy + r * np.sin(a)]
        keep = (c[:, 0] >= 0) & (c[:, 0] <= W) & (c[:, 1] >= 0) & (c[:, 1] <= HGT)
        for ox, oy, orr in [BORE] + NOTCH:
            if (ox, oy, orr) != (cx, cy, r):
                keep &= np.hypot(c[:, 0] - ox, c[:, 1] - oy) > orr
        pts.append(c[keep])
    for xs, ys in [(np.linspace(0, W, 300), np.zeros(300)), (np.linspace(0, W, 150), np.full(150, HGT)),
                   (np.zeros(150), np.linspace(0, HGT, 150)), (np.full(150, W), np.linspace(0, HGT, 150))]:
        c = np.c_[xs, ys]
        pts.append(c[inside(c)])
    # interior: dense jittered grid thinned to the size field
    g = np.mgrid[0:W:0.006, 0:HGT:0.006].reshape(2, -1).T + rng.uniform(-0.002, 0.002, (1, 2))
    g = g[inside(g)]
    h = size_field(g)
    keep = rng.random(len(g)) < (0.006 / h) ** 2
    pts.append(g[keep])
    P = np.unique(np.round(np.vstack(pts), 6), axis=0)
    tri = Delaunay(P).simplices
    cen = P[tri].mean(axis=1)
    tri = tri[inside(cen)]
    # drop slivers
    a = P[tri]
    area = 0.5 * np.abs((a[:, 1, 0] - a[:, 0, 0]) * (a[:, 2, 1] - a[:, 0, 1]) - (a[:, 2, 0] - a[:, 0, 0]) * (a[:, 1, 1] - a[:, 0, 1]))
    tri = tri[area > 1e-8]
    return P, tri


# --- plane-stress constant-strain-triangle FE ---------------------------------------------
def assemble(P, tri):
    Dm = E / (1 - NU**2) * np.array([[1, NU, 0], [NU, 1, 0], [0, 0, (1 - NU) / 2]])
    rows, cols, vals, Bs = [], [], [], []
    for t in tri:
        x, y = P[t, 0], P[t, 1]
        A = 0.5 * ((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]))
        if A < 0:
            t = t[[0, 2, 1]]; x, y = P[t, 0], P[t, 1]; A = -A
        b = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]])
        c = np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]])
        B = np.zeros((3, 6))
        B[0, 0::2] = b; B[1, 1::2] = c; B[2, 0::2] = c; B[2, 1::2] = b
        B /= 2 * A
        k = THK * A * B.T @ Dm @ B
        dof = np.array([2 * t[0], 2 * t[0] + 1, 2 * t[1], 2 * t[1] + 1, 2 * t[2], 2 * t[2] + 1])
        rows.append(np.repeat(dof, 6)); cols.append(np.tile(dof, 6)); vals.append(k.ravel())
        Bs.append((dof, Dm @ B))
    n = 2 * len(P)
    K = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsc()
    return K, Bs


def bore_loads(P):
    """Nodal forces for unit Fx, Fy (N) and Mz (N m) spread over the bearing seat."""
    cx, cy, r = BORE
    on = np.flatnonzero(np.abs(np.hypot(P[:, 0] - cx, P[:, 1] - cy) - r) < 1e-4)
    th = np.arctan2(P[on, 1] - cy, P[on, 0] - cx)
    n = 2 * len(P)
    F = np.zeros((n, 3))
    # bearing-type contact: cos-weighted over the half of the seat the shaft pushes on
    for j, (ux, uy) in enumerate([(1, 0), (0, 1)]):
        w = np.maximum(0.0, np.cos(th) * ux + np.sin(th) * uy)
        w = w / w.sum()                                   # forces parallel to the load, summing to 1 MN
        F[2 * on, j] = 1e6 * w * ux
        F[2 * on + 1, j] = 1e6 * w * uy
    # moment: tangential forces around the seat
    F[2 * on, 2] = -np.sin(th) * 1e6 / (r * len(on))
    F[2 * on + 1, 2] = np.cos(th) * 1e6 / (r * len(on))
    return F


def solve_all(P, tri):
    K, Bs = assemble(P, tri)
    fixed = np.flatnonzero(P[:, 1] < 1e-6)
    fdof = np.r_[2 * fixed, 2 * fixed + 1]
    free = np.setdiff1d(np.arange(2 * len(P)), fdof)
    lu = splu(K[free][:, free])
    F = bore_loads(P)

    def stresses(f):
        u = np.zeros(2 * len(P)); u[free] = lu.solve(f[free])
        return np.array([Db @ u[dof] for dof, Db in Bs])        # element sxx, syy, txy (Pa)
    return stresses, F, lu, free


def nodal(P, tri, s_el):
    acc = np.zeros((len(P), s_el.shape[1])); cnt = np.zeros(len(P))
    for t, s in zip(tri, s_el):
        acc[t] += s; cnt[t] += 1
    return acc / np.maximum(cnt, 1)[:, None]


def von_mises(s):
    return np.sqrt(s[..., 0] ** 2 - s[..., 0] * s[..., 1] + s[..., 1] ** 2 + 3 * s[..., 2] ** 2)


# --- example interface load history (synthetic: parked rotor in a storm, 20 min at 10 Hz) ----
def storm_loads(n=12000, dt=0.1, seed=4):
    rng = np.random.default_rng(seed)
    t = np.arange(n) * dt
    f = np.fft.rfftfreq(n, dt)

    def band(fc, bw, sd):
        S = np.exp(-0.5 * ((f - fc) / bw) ** 2) * rng.normal(size=len(f)) * np.exp(1j * rng.uniform(0, 2 * np.pi, len(f)))
        x = np.fft.irfft(S, n); return sd * x / x.std()
    f1 = 0.167                                         # tower first mode: nacelle motion drives inertia loads
    Fx = 0.6e6 + band(0.02, 0.02, 0.25e6) + band(f1, 0.01, 0.35e6)          # drag + fore-aft inertia
    Fy = -3.1e6 + band(f1, 0.01, 0.12e6) + band(0.5, 0.2, 0.05e6)           # rotor weight + vertical
    Mz = 1.0e6 + band(0.02, 0.02, 0.5e6) + band(f1, 0.01, 0.6e6)            # overhang and gust moments
    # a breaking-wave slam at t = 610 s: a short inertia shock through the tower
    s = (t > 610) * np.exp(-(t - 610) / 2.5) * np.sin(2 * np.pi * f1 * (t - 610))
    Fx += 1.1e6 * s; Mz += 2.0e6 * s
    return t, np.c_[Fx, Fy, Mz]


# --- fatigue: rainflow and an illustrative ductile-iron S-N curve ---------------------------
def rainflow(x):
    tp = [x[0]]
    for i in range(1, len(x) - 1):
        if (x[i] - x[i - 1]) * (x[i + 1] - x[i]) < 0:
            tp.append(x[i])
    tp.append(x[-1])
    st, cyc = [], []
    for p in tp:
        st.append(p)
        while len(st) >= 3:
            X, Y = abs(st[-1] - st[-2]), abs(st[-2] - st[-3])
            if X < Y:
                break
            if len(st) == 3:
                cyc.append((Y, 0.5)); st.pop(0)
            else:
                cyc.append((Y, 1.0)); del st[-3:-1]
    cyc += [(abs(st[i + 1] - st[i]), 0.5) for i in range(len(st) - 1)]
    return np.array(cyc)


SN = dict(ds_ref=90e6, N_ref=2e6, m1=5.0, m2=9.0)     # illustrative: knee at 2e6, shallower beyond


def n_allow(ds):
    ds = np.maximum(ds, 1.0)
    return np.where(ds >= SN["ds_ref"], SN["N_ref"] * (SN["ds_ref"] / ds) ** SN["m1"], SN["N_ref"] * (SN["ds_ref"] / ds) ** SN["m2"])


def main():
    P, tri = mesh()
    stresses, F, lu, free = solve_all(P, tri)
    tt = Triangulation(P[:, 0], P[:, 1], tri)
    t0 = time.time()
    unit_el = np.stack([stresses(F[:, j]) for j in range(3)])          # (3, n_el, 3)
    t_units = time.time() - t0
    unit_nd = np.stack([nodal(P, tri, unit_el[j]) for j in range(3)])  # (3, n_nodes, 3)
    t, L = storm_loads()
    Lu = L / 1e6                                                       # unit cases are per MN and per MN m
    # hot spot: node with the largest von Mises stress over the record (superposed tensors)
    sig_all = np.einsum("tj,jnk->tnk", Lu[::50], unit_nd)              # coarse scan, every 5 s
    hot = int(np.argmax(von_mises(sig_all).max(axis=0)))
    # full-rate history at the hot spot by superposition
    t0 = time.time()
    sig_hot = Lu @ unit_nd[:, hot, :]                                   # (n_t, 3): sxx, syy, txy
    t_super = time.time() - t0
    # check against direct FE solves at a few instants
    checks = np.linspace(0, len(t) - 1, 9).astype(int)[1:-1]
    t0 = time.time()
    direct = []
    for i in checks:
        s_el = stresses(F @ Lu[i])
        direct.append(nodal(P, tri, s_el)[hot])
    t_direct_each = (time.time() - t0) / len(checks)
    direct = np.array(direct)
    err = np.abs(direct - sig_hot[checks]).max() / np.abs(sig_hot).max()
    print(f"mesh: {len(P)} nodes, {len(tri)} triangles; 3 unit solves {t_units:.2f} s")
    print(f"hot spot at ({P[hot, 0]:.3f}, {P[hot, 1]:.3f}) m; superposition of {len(t)} steps {1e3 * t_super:.2f} ms; "
          f"direct solve {t_direct_each:.2f} s per step ({t_direct_each * len(t) / 60:.0f} min for the record); max relative difference {err:.1e}")

    # ---------- figure 1: unit load cases -------------------------------------------------
    fig, axs = plt.subplots(1, 4, figsize=(16.5, 4.1), gridspec_kw=dict(width_ratios=[1, 1, 1, 1], wspace=0.08))
    ax = axs[0]
    ax.triplot(tt, lw=0.15, color=MUTED)
    cx, cy, r = BORE
    ax.annotate("", xy=(cx + r + 0.05, cy), xytext=(cx, cy), arrowprops=dict(arrowstyle="-|>", color=ACC, lw=2))
    ax.annotate("", xy=(cx, cy - r - 0.05), xytext=(cx, cy), arrowprops=dict(arrowstyle="-|>", color=ACC, lw=2))
    ax.text(cx - 0.3, cy + 0.42, "Fx, Fy, Mz at the bearing seat", color=ACC, fontsize=8)
    ax.plot([0, W], [0, 0], color=INK, lw=4); ax.text(W / 2, -0.12, "bolted to the bedplate (fixed)", ha="center", fontsize=8, color=INK)
    ax.plot(P[hot, 0], P[hot, 1], "o", ms=9, mfc="none", mec=WARN, mew=2)
    ax.set_title("Model: bearing housing, %d elements" % len(tri), fontsize=10, loc="left")
    names = ["Unit Fx = 1 MN", "Unit Fy = 1 MN", "Unit Mz = 1 MN·m"]
    vmax = max(von_mises(unit_nd[j]).max() for j in range(3)) / 1e6
    for j in range(3):
        a = axs[j + 1]
        c = a.tripcolor(tt, von_mises(unit_nd[j]) / 1e6, shading="gouraud", cmap="magma_r", vmin=0, vmax=vmax)
        a.plot(P[hot, 0], P[hot, 1], "o", ms=9, mfc="none", mec=WARN, mew=2)
        a.set_title(names[j] + f"\nhot spot {von_mises(unit_nd[j, hot]) / 1e6:.1f} MPa", fontsize=10, loc="left")
    for a in axs:
        a.set_aspect("equal"); a.axis("off"); a.set_xlim(-0.1, W + 0.1); a.set_ylim(-0.2, HGT + 0.05)
    cb = fig.colorbar(c, ax=axs[1:], shrink=0.8, pad=0.03); cb.set_label("von Mises stress per unit load (MPa)")
    fig.suptitle("Step 1. Solve the FE model once per unit interface load and store the stress tensor everywhere", x=0.01, ha="left", fontsize=12)
    fig.savefig(os.path.join(OUT, "1_unit_load_cases.png"), dpi=140, bbox_inches="tight"); plt.close(fig)

    # ---------- figure 2: superposition through a load history ------------------------------
    fig = plt.figure(figsize=(14, 7.8))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.25, 1], hspace=0.45, wspace=0.22)
    lab = [("Fx (MN)", 1e6), ("Fy (MN)", 1e6), ("Mz (MN·m)", 1e6)]
    for j in range(3):
        a = fig.add_subplot(gs[j, 0])
        a.plot(t, L[:, j] / 1e6, color=ACC, lw=0.6)
        a.set_ylabel(lab[j][0]); a.set_xlim(0, t[-1]); a.grid(alpha=0.25)
        if j == 0:
            a.set_title("Interface loads from the global model (example: parked rotor in a storm, slam at 610 s)", fontsize=10, loc="left")
        if j == 2:
            a.set_xlabel("time (s)")
    a = fig.add_subplot(gs[0:2, 1])
    comp = ["σxx", "σyy", "τxy"]
    cols = [ACC, "#2f7d4f", "#8e5ea2"]
    w0, w1 = np.searchsorted(t, 560), np.searchsorted(t, 700)
    for k in range(3):
        a.plot(t[w0:w1], sig_hot[w0:w1, k] / 1e6, color=cols[k], lw=1.1, label=comp[k] + " by superposition")
    in_win = (checks >= w0) & (checks < w1)
    for k in range(3):
        a.plot(t[checks], direct[:, k] / 1e6, "s", ms=7, mfc="none", mec=cols[k], mew=1.6)
    a.plot([], [], "s", mfc="none", mec=INK, label="direct FE solve (check)")
    a.set_xlim(t[0], t[-1]); a.set_title("Stress tensor at the hot spot: σ(t) = Fx(t)·σFx + Fy(t)·σFy + Mz(t)·σMz", fontsize=10, loc="left")
    a.set_xlabel("time (s)"); a.set_ylabel("stress (MPa)"); a.grid(alpha=0.25); a.legend(fontsize=8, loc="lower left", ncol=2)
    a.plot(t, sig_hot[:, 0] / 1e6, color=cols[0], lw=0.3, alpha=0.25)
    a.plot(t, sig_hot[:, 1] / 1e6, color=cols[1], lw=0.3, alpha=0.25)
    a.plot(t, sig_hot[:, 2] / 1e6, color=cols[2], lw=0.3, alpha=0.25)
    a.text(0.99, 0.97, f"max difference to direct solves: {err:.0e}", transform=a.transAxes, ha="right", va="top", fontsize=8, color=MUTED)
    a2 = fig.add_subplot(gs[2, 1])
    bars = [t_direct_each * len(t), t_units + t_super]
    a2.barh([0, 1], bars, color=[MUTED, ACC], height=0.55)
    a2.set_yticks([0, 1]); a2.set_yticklabels(["direct FE", "superposition"])
    a2.set_title(f"{len(t)} time steps: one FE solve each, or 3 unit solves + multiply-adds", fontsize=10, loc="left")
    a2.set_xscale("log"); a2.set_xlabel("compute time for the 20-minute record (s, log scale)")
    for i, v in enumerate(bars):
        a2.text(v * 1.15, i, f"{v:.1f} s" if v < 120 else f"{v / 60:.0f} min", va="center", fontsize=9)
    a2.set_xlim(min(bars) / 3, max(bars) * 8)
    fig.suptitle("Step 2. Any load history becomes stress histories by superposing the stored unit tensors", x=0.01, ha="left", fontsize=12)
    fig.savefig(os.path.join(OUT, "2_superposition.png"), dpi=140, bbox_inches="tight"); plt.close(fig)

    # ---------- figure 3: fatigue at the hot spot (critical plane) ----------------------------
    angles = np.radians(np.arange(0, 180, 2))
    sxx, syy, txy = sig_hot.T
    dmg, cycles_at = [], None
    for th in angles:
        sn = sxx * np.cos(th) ** 2 + syy * np.sin(th) ** 2 + 2 * txy * np.sin(th) * np.cos(th)
        cyc = rainflow(sn)
        d = (cyc[:, 1] / n_allow(cyc[:, 0])).sum()
        dmg.append(d)
    dmg = np.array(dmg)
    crit = int(np.argmax(dmg))
    th = angles[crit]
    sn = sxx * np.cos(th) ** 2 + syy * np.sin(th) ** 2 + 2 * txy * np.sin(th) * np.cos(th)
    cyc = rainflow(sn)
    vm = von_mises(sig_hot)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.3))
    a = axs[0]
    a.plot(np.degrees(angles), dmg / dmg.max(), color=ACC, lw=2)
    a.axvline(np.degrees(th), color=WARN, ls="--"); a.text(np.degrees(th) + 2, 0.05, f"critical plane {np.degrees(th):.0f}°", color=WARN, fontsize=9)
    a.set_xlabel("plane angle (°)"); a.set_ylabel("damage (relative to the critical plane)"); a.grid(alpha=0.25)
    a.set_title("Normal stress on each plane, rainflow, Miner", fontsize=10, loc="left")
    a = axs[1]
    w0, w1 = np.searchsorted(t, 590), np.searchsorted(t, 650)
    a.plot(t[w0:w1], sn[w0:w1] / 1e6, color=ACC, lw=1.4, label="normal stress on the critical plane")
    a.plot(t[w0:w1], vm[w0:w1] / 1e6, color=MUTED, lw=1, ls="--", label="von Mises (always positive)")
    a.set_xlabel("time (s)"); a.set_ylabel("stress (MPa)"); a.grid(alpha=0.25); a.legend(fontsize=8)
    a.set_title("Why superpose tensors, not von Mises: the sign matters", fontsize=10, loc="left")
    a = axs[2]
    ds = np.logspace(6.5, 8.6, 200)
    a.loglog(n_allow(ds), ds / 1e6, color=INK, lw=1.5, label="illustrative ductile-iron S-N")
    big = cyc[cyc[:, 0] > 5e6]
    rng_bins = np.logspace(np.log10(5e6), np.log10(cyc[:, 0].max() * 1.01), 18)
    cnt, edges = np.histogram(big[:, 0], bins=rng_bins, weights=big[:, 1])
    mid = np.sqrt(edges[1:] * edges[:-1])
    ok = cnt > 0
    a.loglog(np.cumsum(cnt[::-1])[::-1][ok], mid[ok] / 1e6, "o-", color=ACC, ms=4, label="record: cycles exceeding each range")
    a.set_xlabel("cycles"); a.set_ylabel("stress range (MPa)"); a.grid(alpha=0.25, which="both"); a.legend(fontsize=8)
    a.set_title(f"Miner damage {dmg[crit]:.1e} per 20-min record ({dmg[crit] * 18:.1e} for a 6-hour storm)", fontsize=10, loc="left")
    fig.suptitle("Step 3. Fatigue at the hot spot from the superposed stress history", x=0.01, ha="left", fontsize=12)
    fig.savefig(os.path.join(OUT, "3_hotspot_fatigue.png"), dpi=140, bbox_inches="tight"); plt.close(fig)
    print("critical plane", np.degrees(th), "damage", dmg[crit])
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
