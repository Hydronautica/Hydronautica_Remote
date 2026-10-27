"""How do NLS, stream function and NewWave kinematics diverge as H -> breaking?

For a fixed period and depth, the wave height is swept from small up to 97%
of the limiting (breaking) height of a steady wave, H_b. At each height the
three waves have the same crest-to-trough height at the focal point. The
linear-type waves (NewWave and the NLS reconstruction) are shown with Wheeler
stretching (solid) and linear extrapolation (dashed), since in steep crests
the stretching choice matters as much as the wave theory.
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np

import nls
from nls.stream_function import breaking_height

os.makedirs("figures", exist_ok=True)
warnings.filterwarnings("ignore", "All-NaN slice")

T, d = 10.0, 100.0
FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97]
PROFILE_FRACTIONS = [0.3, 0.6, 0.9, 0.97]
STRETCHINGS = ("wheeler", "linear")

Hb = breaking_height(T, d)
omega = 2 * np.pi / T
print(f"T = {T} s, d = {d} m, breaking height H_b = {Hb:.2f} m "
      f"(H_b / gT^2 = {Hb / (9.81 * T**2):.4f})")


def column_stats(wave, t, z, stretching):
    """Kinematics on the focal vertical x = 0 over times t."""
    kin = [wave.kinematics(0.0, z, ti, stretching=stretching) for ti in t]
    ax = np.array([k["ax"] for k in kin])
    w = np.array([k["w"] for k in kin])
    i0 = int(np.argmin(np.abs(t)))
    u_crest = kin[i0]["u"]
    return dict(
        u_profile=u_crest,
        u_top=u_crest[np.isfinite(u_crest)][-1],
        ax_profile=np.nanmax(np.abs(ax), axis=0),
        w_profile=np.nanmax(np.abs(w), axis=0),
    )


def nls_wave(H, t_half):
    dw = nls.peregrine_for_height(H, period=T)
    solver = nls.NLSSolver(L=200, N=4096)
    Tw = dw.to_normalised_t(t_half)
    pre = solver.run(lambda X: nls.peregrine(X, -2.0), -2.0, -Tw, dt=0.001, n_out=2)
    res = solver.run(pre.psi[-1], -Tw, Tw, dt=0.001, n_out=121)
    return nls.NLSWave(res, dw, depth=d, phase=np.pi), dw


rows = []
for f in FRACTIONS:
    H = f * Hb
    sf = nls.StreamFunctionWave(H, T, d)
    nw = nls.NewWave(Tp=T, depth=d, H=H)
    nl, dw = nls_wave(H, 0.6 * T)
    t = nl.physical_times()           # +-0.6 T around the focus
    z = np.linspace(-0.6 * H - 30.0, 0.75 * H, 140)

    row = dict(f=f, H=H, z=z, eps=dw.eps,
               leak=nl.negative_wavenumber_fraction(0.0), N_sf=sf.N, res_sf=sf.residual)
    row["Stream function"] = dict(crest=sf.crest, c=sf.c, **column_stats(sf, t, z, None))
    for name, wave, c in (("NewWave", nw, omega / nls.wavenumber(omega, d)),
                          ("NLS Peregrine", nl, dw.w0 / dw.k0)):
        crest = float(wave.eta(0.0, 0.0))
        row[name] = {s: dict(crest=crest, c=c, **column_stats(wave, t, z, s))
                     for s in STRETCHINGS}
    rows.append(row)
    print(f"H/H_b = {f:4.2f}  H = {H:5.2f} m  SF N={sf.N:2d} res={sf.residual:.0e}  "
          f"NLS k0a0={dw.eps:.3f} k<0 energy={row['leak']:.0e}")

# --- table ------------------------------------------------------------------
print(f"\n{'H/Hb':>5s} {'':16s} {'crest/H':>7s} {'u_crest':>8s} {'u/c':>5s} "
      f"{'|ax|max':>8s} {'|w|max':>7s}   (Wheeler; linear-extrapolation in brackets)")
for r in rows:
    s = r["Stream function"]
    print(f"{r['f']:5.2f} {'Stream function':16s} {s['crest'] / r['H']:7.3f} "
          f"{s['u_top']:8.2f} {s['u_top'] / s['c']:5.2f} "
          f"{np.nanmax(s['ax_profile']):8.2f} {np.nanmax(s['w_profile']):7.2f}")
    for name in ("NewWave", "NLS Peregrine"):
        a, b = r[name]["wheeler"], r[name]["linear"]
        print(f"{'':5s} {name:16s} {a['crest'] / r['H']:7.3f} "
              f"{a['u_top']:5.2f}({b['u_top']:5.2f}) {a['u_top'] / a['c']:4.2f}"
              f"({b['u_top'] / b['c']:4.2f}) {np.nanmax(a['ax_profile']):5.2f}"
              f"({np.nanmax(b['ax_profile']):5.2f}) {np.nanmax(a['w_profile']):5.2f}"
              f"({np.nanmax(b['w_profile']):5.2f})")

# --- trend plots --------------------------------------------------------------
fr = np.array([r["f"] for r in rows])
Hs = np.array([r["H"] for r in rows])
colors = {"Stream function": "C0", "NewWave": "C1", "NLS Peregrine": "C2"}


def series(name, fn, stretching="wheeler"):
    if name == "Stream function":
        return np.array([fn(r[name], r) for r in rows])
    return np.array([fn(r[name][stretching], r) for r in rows])


panels = [
    ("Crest elevation / H", lambda s, r: s["crest"] / r["H"], None),
    ("u at crest / (ωH/2)", lambda s, r: s["u_top"] / (omega * r["H"] / 2), None),
    ("u at crest / phase speed c", lambda s, r: s["u_top"] / s["c"], 1.0),
    ("max |aₓ| / (ω²H/2)", lambda s, r: np.nanmax(s["ax_profile"]) / (omega**2 * r["H"] / 2), None),
    ("max |w| / (ωH/2)", lambda s, r: np.nanmax(s["w_profile"]) / (omega * r["H"] / 2), None),
]
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax, (title, fn, ref) in zip(axes.flat, panels):
    for name, col in colors.items():
        ax.plot(fr, series(name, fn), "o-", color=col, ms=4, label=name)
        if name != "Stream function":
            ax.plot(fr, series(name, fn, "linear"), "s--", color=col, ms=3, lw=1,
                    label=f"{name} (linear extrap.)")
    if ref is not None:
        ax.axhline(ref, c="k", ls=":", lw=1)
    ax.set_title(title)
    ax.set_xlabel("H / H_b")
    ax.grid(alpha=0.3)
axes[0, 0].legend(fontsize=8)
ax = axes[1, 2]
ax.plot(fr, [r["eps"] for r in rows], "o-", color="C2", label="NLS background k₀a₀")
ax.set_xlabel("H / H_b")
ax.set_ylabel("k₀a₀", color="C2")
ax2 = ax.twinx()
ax2.semilogy(fr, [max(r["leak"], 1e-16) for r in rows], "s--", color="C3")
ax2.set_ylabel("NLS energy at k < 0 (narrow-band violation)", color="C3")
ax.set_title("NLS validity")
ax.grid(alpha=0.3)
fig.suptitle(f"Kinematics vs wave height, T = {T} s, d = {d} m, H_b = {Hb:.1f} m "
             "(solid: Wheeler, dashed: linear extrapolation)")
fig.tight_layout()
fig.savefig("figures/breaking_sweep_trends.png", dpi=150)

# --- profiles at selected heights ----------------------------------------------
sel = [r for r in rows if r["f"] in PROFILE_FRACTIONS]
fig, axes = plt.subplots(2, len(sel), figsize=(4 * len(sel), 9), sharey="row")
for j, r in enumerate(sel):
    for i, (key, xlabel) in enumerate((("u_profile", "u under crest (m/s)"),
                                       ("ax_profile", "max |aₓ| (m/s²)"))):
        ax = axes[i, j]
        ax.plot(r["Stream function"][key], r["z"], color=colors["Stream function"],
                label="Stream function")
        for name in ("NewWave", "NLS Peregrine"):
            ax.plot(r[name]["wheeler"][key], r["z"], color=colors[name], label=name)
            ax.plot(r[name]["linear"][key], r["z"], "--", color=colors[name], lw=1)
        ax.axhline(0, c="k", lw=0.5)
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.set_title(f"H/H_b = {r['f']}  (H = {r['H']:.1f} m)")
axes[0, 0].set_ylabel("z (m)")
axes[1, 0].set_ylabel("z (m)")
axes[0, 0].legend(fontsize=8)
fig.suptitle("Profiles on the focal vertical (solid: Wheeler, dashed: linear extrapolation)")
fig.tight_layout()
fig.savefig("figures/breaking_sweep_profiles.png", dpi=150)
