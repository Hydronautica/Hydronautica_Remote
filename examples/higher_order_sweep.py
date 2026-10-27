"""Kinematics vs wave height for linear, second-order and Dysthe-type models.

For a fixed period and depth, H is swept towards the steady-wave breaking
height H_b. Every model is calibrated to the same crest-to-trough height H at
its own focal point:

* Stream function             -- fully nonlinear steady wave (reference)
* NewWave, linear             -- Wheeler stretching
* NewWave, 2nd order          -- Sharma-Dean bound waves, Taylor (linear) extrapolation
* NLS Peregrine, linear       -- Wheeler stretching
* NLS Peregrine, 2nd order    -- bound harmonic + set-down + mean flow
* Dysthe, 2nd order           -- Peregrine initial condition evolved with Dysthe
* Dysthe exact dispersion     -- same with the exact linear dispersion relation

Points where a model's crest particle velocity exceeds the phase speed (the
kinematic breaking criterion: the model itself says the wave breaks) or an
envelope model leaks more than 1e-3 of its energy to k < 0 (narrow-band
assumption violated) are drawn as open symbols: they are outside the model's
range and should not be used quantitatively.

Runs the heights in parallel; takes 10-20 minutes. Results are cached per
height in figures/sweep_cache/, so a rerun only recomputes models that are
missing or failed (delete the cache to start afresh).
"""

import os
import pickle
import warnings
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np

T, d = 10.0, 100.0
FRACTIONS = [0.1, 0.25, 0.4, 0.55, 0.7, 0.8, 0.9, 0.95]
PROFILE_FRACTIONS = [0.4, 0.7, 0.9]
WINDOW = 0.6          # +- periods around the focus for the time maxima
DT_OUT = 0.2          # s
MODELS = {
    # name: (colour, linestyle)
    "Stream function": ("C0", "-"),
    "NewWave linear": ("C1", "--"),
    "NewWave 2nd order": ("C1", "-"),
    "NLS linear": ("C2", "--"),
    "NLS 2nd order": ("C2", "-"),
    "Dysthe 2nd order": ("C4", "-"),
    "Dysthe exact-disp. 2nd order": ("C3", "-"),
}


def nls_wave(H, order):
    import nls
    dw = nls.peregrine_for_height(H, period=T, order=order)
    solver = nls.NLSSolver(L=200, N=4096)
    Tw = dw.to_normalised_t(WINDOW * T)
    pre = solver.run(lambda X: nls.peregrine(X, -2.0), -2.0, -Tw, dt=0.001, n_out=2)
    n = int(round(2 * WINDOW * T / DT_OUT)) + 1
    res = solver.run(pre.psi[-1], -Tw, Tw, dt=0.001, n_out=n)
    wave = nls.NLSWave(res, dw, depth=d, phase=np.pi, order=order)
    return wave, 0.0, 0.0, wave.physical_times(), dw.w0 / dw.k0, dw.eps


def build(name, H):
    """Return (wave, x_focus, t_focus, times, phase_speed, steepness)."""
    import nls
    from nls.dysthe import dysthe_for_height
    om = 2 * np.pi / T
    c_lin = om / float(nls.wavenumber(om, d))
    t_lin = np.arange(-WINDOW * T, WINDOW * T + 1e-9, DT_OUT)
    if name == "Stream function":
        sf = nls.StreamFunctionWave(H, T, d)
        return sf, 0.0, 0.0, t_lin, sf.c, np.nan
    if name.startswith("NewWave"):
        order = 1 if "linear" in name else 2
        nw = nls.NewWave(Tp=T, depth=d, H=H, order=order, n_modes=256)
        return nw, 0.0, 0.0, t_lin, c_lin, np.nan
    if name.startswith("NLS"):
        return nls_wave(H, 1 if "linear" in name else 2)
    disp = "exact" if "exact" in name else "dysthe"
    wave, t_f, x_f, eps = dysthe_for_height(H, T, depth=d, dispersion=disp)
    t = wave.physical_times()
    t = t[np.abs(t - t_f) <= WINDOW * T + 1e-9][::int(round(DT_OUT / (t[1] - t[0])))]
    return wave, x_f, t_f, t, wave.w0 / wave.k0, eps


CACHE = os.path.join("figures", "sweep_cache")


def valid(m):
    """Inside the model's range: no predicted breaking, narrow band respected."""
    return m is not None and m["u_top"] < m["c"] and m.get("leak", 0.0) < 1e-3


def evaluate(args):
    f, Hb = args
    warnings.filterwarnings("ignore")
    import nls
    H = f * Hb
    z = np.linspace(-0.6 * H - 30.0, 0.8 * H, 100)
    path = os.path.join(CACHE, f"H{f:.3f}.pkl")
    row = dict(f=f, H=H, z=z)
    if os.path.exists(path):
        with open(path, "rb") as fh:
            row.update(pickle.load(fh))
    for name in MODELS:
        if row.get(name) is not None:
            continue
        try:
            wave, x_f, t_f, times, c, eps = build(name, H)
            kin = [wave.kinematics(x_f, z, ti) for ti in times]
            i0 = int(np.argmin(np.abs(times - t_f)))
            eta = np.array([wave.eta(x_f, ti) for ti in np.linspace(t_f - 1.5 * T, t_f + 1.5 * T, 301)]
                           ).ravel() if name in ("Stream function",) or name.startswith("NewWave") \
                else np.array([wave.eta(x_f, ti) for ti in wave.physical_times()]).ravel()
            u0 = kin[i0]["u"]
            leak = wave.negative_wavenumber_fraction(t_f) \
                if hasattr(wave, "negative_wavenumber_fraction") else 0.0
            row[name] = dict(
                crest=float(np.max(eta)), c=c, eps=eps, leak=leak,
                u_profile=u0, u_top=u0[np.isfinite(u0)][-1],
                ax_profile=np.nanmax(np.abs([k["ax"] for k in kin]), axis=0),
                w_profile=np.nanmax(np.abs([k["w"] for k in kin]), axis=0))
        except Exception as e:      # e.g. beyond a model's range of validity
            print(f"  H/H_b={f}: {name} failed: {e}", flush=True)
            row[name] = None
        with open(path, "wb") as fh:
            pickle.dump(row, fh)
    print(f"done H/H_b = {f}", flush=True)
    return row


if __name__ == "__main__":
    import nls
    os.makedirs(CACHE, exist_ok=True)
    Hb = nls.breaking_height(T, d)
    om = 2 * np.pi / T
    print(f"T = {T} s, d = {d} m, H_b = {Hb:.2f} m")
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(evaluate, [(f, Hb) for f in FRACTIONS]))

    # --- table ----------------------------------------------------------------
    print(f"\n{'H/Hb':>5s} {'model':29s} {'crest/H':>7s} {'u_crest':>7s} {'u/c':>5s} "
          f"{'|ax|max':>7s} {'|w|max':>6s} {'k0a0':>6s} {'k<0':>7s}")
    for r in rows:
        for name in MODELS:
            m = r[name]
            if m is None:
                print(f"{r['f']:5.2f} {name:29s}  (failed)")
                continue
            print(f"{r['f']:5.2f} {name:29s} {m['crest'] / r['H']:7.3f} {m['u_top']:7.2f} "
                  f"{m['u_top'] / m['c']:5.2f} {np.nanmax(m['ax_profile']):7.2f} "
                  f"{np.nanmax(m['w_profile']):6.2f} {m['eps']:6.3f} {m.get('leak', 0):7.0e}"
                  f"{'' if valid(m) else '  *outside range'}")

    # --- trend plots --------------------------------------------------------------
    fr = np.array([r["f"] for r in rows])

    def series(name, fn):
        return np.array([fn(r[name], r) if r[name] is not None else np.nan for r in rows])

    panels = [
        ("Crest elevation / H", lambda m, r: m["crest"] / r["H"], None),
        ("u at crest / (ωH/2)", lambda m, r: m["u_top"] / (om * r["H"] / 2), None),
        ("u at crest / phase speed c", lambda m, r: m["u_top"] / m["c"], 1.0),
        ("max |aₓ| / (ω²H/2)", lambda m, r: np.nanmax(m["ax_profile"]) / (om**2 * r["H"] / 2), None),
        ("max |w| / (ωH/2)", lambda m, r: np.nanmax(m["w_profile"]) / (om * r["H"] / 2), None),
        ("Envelope-model background k₀a₀", lambda m, r: m["eps"], None),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    for ax, (title, fn, ref) in zip(axes.flat, panels):
        for name, (col, ls) in MODELS.items():
            y = series(name, fn)
            if np.all(np.isnan(y)):
                continue
            ok = np.array([valid(r[name]) for r in rows])
            ax.plot(fr, y, ls, color=col, lw=1, alpha=0.35)
            ax.plot(np.where(ok, fr, np.nan), np.where(ok, y, np.nan), ls, color=col,
                    marker="o", ms=4, label=name)
            ax.plot(fr[~ok], y[~ok], "o", mfc="none", color=col, ms=5)
        if ref is not None:
            ax.axhline(ref, c="k", ls=":", lw=1)
        ax.set_title(title)
        ax.set_xlabel("H / H_b")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"T = {T} s, d = {d} m, H_b = {Hb:.1f} m. Linear models: Wheeler; 2nd order: "
                 "Taylor extrapolation. Open symbols: model predicts breaking (u > c) or "
                 "leaves its narrow-band range")
    fig.tight_layout()
    fig.savefig("figures/higher_order_trends.png", dpi=150)

    # --- profiles ----------------------------------------------------------------
    sel = [r for r in rows if r["f"] in PROFILE_FRACTIONS]
    fig, axes = plt.subplots(2, len(sel), figsize=(5 * len(sel), 10), sharey="col")
    for j, r in enumerate(sel):
        for i, (key, xlabel) in enumerate((("u_profile", "u under crest (m/s)"),
                                           ("ax_profile", "max |aₓ| (m/s²)"))):
            ax = axes[i, j]
            for name, (col, ls) in MODELS.items():
                if r[name] is not None:
                    ax.plot(r[name][key], r["z"], ls, color=col, label=name)
            ax.axhline(0, c="k", lw=0.5)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("z (m)")
            ax.grid(alpha=0.3)
            if i == 0:
                ax.set_title(f"H/H_b = {r['f']}  (H = {r['H']:.1f} m)")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("figures/higher_order_profiles.png", dpi=150)
