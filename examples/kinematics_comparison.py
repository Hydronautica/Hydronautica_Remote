"""Compare wave kinematics: NLS Peregrine breather vs stream function vs NewWave.

All three have the same crest-to-trough height H (measured at the focal
point), period T and water depth d. Kinematics are evaluated on the vertical
through the focus x = 0:

* surface elevation time series,
* horizontal velocity profile under the crest,
* maximum (over time) horizontal acceleration and vertical velocity profiles.
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np

import nls

os.makedirs("figures", exist_ok=True)
warnings.filterwarnings("ignore", "All-NaN slice")

H, T, d = 12.0, 10.0, 100.0   # wave height (m), period (s), depth (m)
STRETCH = "wheeler"            # for the linear-type waves: wheeler | linear | none

# --- the three waves ---------------------------------------------------------
stream = nls.StreamFunctionWave(H, T, d, N=24)
newwave = nls.NewWave(Tp=T, depth=d, H=H, gamma=3.3)

dw = nls.peregrine_for_height(H, period=T)
solver = nls.NLSSolver(L=200, N=4096)
T_window = dw.to_normalised_t(1.5 * T)
pre = solver.run(lambda X: nls.peregrine(X, -2.0), -2.0, -T_window, dt=0.001, n_out=2)
res = solver.run(pre.psi[-1], -T_window, T_window, dt=0.001, n_out=301)
nlsw = nls.NLSWave(res, dw, depth=d, phase=np.pi)

print(f"H = {H} m, T = {T} s, d = {d} m")
print(f"stream function: L = {stream.wavelength:.1f} m, c = {stream.c:.2f} m/s")
print(f"NewWave: crest = {newwave.crest:.2f} m (JONSWAP gamma = 3.3)")
print(f"NLS Peregrine: background a0 = {dw.a0:.2f} m, k0*a0 = {dw.eps:.3f}, "
      f"k<0 energy fraction = {nlsw.negative_wavenumber_fraction(0.0):.1e}")

waves = {
    "Stream function": (stream, dict(color="C0")),
    "NewWave (linear)": (newwave, dict(color="C1")),
    "NLS Peregrine": (nlsw, dict(color="C2")),
}

# --- evaluate on the focal vertical over +-1.5 periods ----------------------
t = nlsw.physical_times()  # NLS output times; reuse for the others
z = np.linspace(-60.0, 8.0, 273)
data = {}
for name, (wave, _) in waves.items():
    eta = np.array([wave.eta(0.0, ti) for ti in t]).ravel()
    kin = [wave.kinematics(0.0, z, ti, stretching=STRETCH) for ti in t]
    stack = {key: np.array([k[key] for k in kin]) for key in ("u", "w", "ax", "az")}
    i0 = int(np.argmin(np.abs(t)))
    data[name] = dict(eta=eta, crest=wave.kinematics(0.0, z, t[i0], stretching=STRETCH),
                      ax_max=np.nanmax(np.abs(stack["ax"]), axis=0),
                      w_max=np.nanmax(np.abs(stack["w"]), axis=0),
                      u_max=np.nanmax(stack["u"], axis=0))

# --- summary table -----------------------------------------------------------
print(f"\n{'':18s} {'crest':>7s} {'trough':>7s} {'H':>6s} {'u_crest':>8s} "
      f"{'u(z=0)':>7s} {'u(-20)':>7s} {'|ax|max':>8s} {'|w|max':>7s}")
for name, dd in data.items():
    c = dd["crest"]
    eta = dd["eta"]
    crest = eta.max()
    u_top = c["u"][np.isfinite(c["u"])][-1]
    iz0 = np.argmin(np.abs(z - 0.0))
    iz20 = np.argmin(np.abs(z + 20.0))
    print(f"{name:18s} {crest:7.2f} {crest - nls.wave_height(t, eta):7.2f} "
          f"{nls.wave_height(t, eta):6.2f} {u_top:8.2f} {c['u'][iz0]:7.2f} "
          f"{c['u'][iz20]:7.2f} {np.nanmax(dd['ax_max']):8.2f} {np.nanmax(dd['w_max']):7.2f}")
print("(m, m/s, m/s^2; u_crest is at the crest surface, maxima over time on x = 0)")

# --- plots -------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
ax = axes[0, 0]
for name, (_, sty) in waves.items():
    ax.plot(t, data[name]["eta"], label=name, **sty)
ax.axhline(0, c="k", lw=0.5)
ax.set_xlabel("t (s)")
ax.set_ylabel("η (m)")
ax.set_title("Surface elevation at focal point")
ax.legend()

def profile(ax, key, xlabel, title):
    for name, (_, sty) in waves.items():
        v = data[name][key] if key != "u_crest" else data[name]["crest"]["u"]
        ax.plot(v, z, label=name, **sty)
    ax.axhline(0, c="k", lw=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("z (m)")
    ax.set_title(title)
    ax.grid(alpha=0.3)

profile(axes[0, 1], "u_crest", "u (m/s)", "Horizontal velocity under crest")
profile(axes[1, 0], "ax_max", "max |aₓ| (m/s²)", "Max horizontal acceleration (total)")
profile(axes[1, 1], "w_max", "max |w| (m/s)", "Max vertical velocity")
axes[0, 1].legend()
fig.suptitle(f"H = {H} m, T = {T} s, d = {d} m  ({STRETCH} stretching for NewWave / NLS)")
fig.tight_layout()
fig.savefig("figures/kinematics_comparison.png", dpi=150)

# --- velocity field snapshots at the focus -----------------------------------
x = np.linspace(-120, 120, 241)
zz = np.linspace(-40, 8, 97)
Xg, Zg = np.meshgrid(x, zz)
fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
vmax = max(np.nanmax(np.abs(w.kinematics(Xg, Zg, 0.0, STRETCH)["u"])) for w, _ in waves.values())
for ax, (name, (wave, _)) in zip(axes, waves.items()):
    kin = wave.kinematics(Xg, Zg, 0.0, stretching=STRETCH)
    im = ax.pcolormesh(x, zz, kin["u"], cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
    s = 8
    ax.quiver(Xg[::6, ::s], Zg[::6, ::s], kin["u"][::6, ::s], kin["w"][::6, ::s],
              scale=vmax * 25, width=0.002)
    ax.plot(x, wave.eta(x, 0.0), "k", lw=1.5)
    ax.set_ylabel("z (m)")
    ax.set_title(name)
    ax.set_ylim(zz[0], zz[-1])
axes[-1].set_xlabel("x (m)")
fig.colorbar(im, ax=axes, label="u (m/s)", shrink=0.6)
fig.savefig("figures/kinematics_fields.png", dpi=150)
