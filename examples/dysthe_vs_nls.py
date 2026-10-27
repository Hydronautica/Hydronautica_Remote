"""What the Dysthe equation changes: the same Peregrine initial condition
evolved with the cubic NLS and with the Dysthe (fourth-order) equation.

Background steepness k0 a0 = 0.1. Shows the space-time envelope, the peak
amplification history and the envelope / surface at the moment of focus.

The Dysthe runs radiate short envelope ripples ahead of the focus (envelope
wavenumber ~2 k0). They are converged in dt and N, so they belong to the model:
at this steepness the focused breather is only about one wavelength long, at
the edge of the narrow-band assumption. Using the exact linear dispersion
relation (Trulsen et al. 2000) halves them and changes the peak noticeably.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from nls.dysthe import DystheSolver, DystheWave, peregrine_domain, peregrine_initial

os.makedirs("figures", exist_ok=True)

eps, T = 0.1, 10.0          # background steepness, carrier period (s)
T0 = -2.0                   # start 2 NLS time units before the NLS focus
L, N = peregrine_domain(eps)
t_nls = -2 * T0 / eps**2    # NLS focus time (units of 1/w0)

runs = {}
for name, kw in (("NLS", dict(higher_order=False)),
                 ("Dysthe", dict()),
                 ("Dysthe, exact dispersion", dict(dispersion="exact")),
                 ("Dysthe, no mean flow", dict(mean_flow=False))):
    solver = DystheSolver(L, N, **kw)
    res = solver.run(peregrine_initial(solver.x, eps, T0), 0, 1.6 * t_nls, dt=1.0, n_out=641)
    runs[name] = (solver, res)
    amp = np.abs(res.psi).max(axis=1) / eps
    i = int(np.argmax(amp))
    print(f"{name:26s} peak |B|/eps = {amp[i]:.3f} at t = {res.t[i] / t_nls:.3f} t_NLS")

fig = plt.figure(figsize=(13, 9))
gs = fig.add_gridspec(2, 3)
for col, name in enumerate(("NLS", "Dysthe")):
    solver, res = runs[name]
    ax = fig.add_subplot(gs[0, col])
    win = np.abs(solver.x) < 60
    # periods of the carrier: t' / 2 pi ; wavelengths: xi / 2 pi
    im = ax.pcolormesh(solver.x[win] / (2 * np.pi), res.t / (2 * np.pi),
                       np.abs(res.psi[:, win]) / eps, shading="auto", vmin=0, vmax=3,
                       cmap="viridis", rasterized=True)
    ax.set_xlabel("x - c_g t (wavelengths)")
    ax.set_ylabel("t (carrier periods)")
    ax.set_title(f"{name}: |A| / a₀")
    if col == 1:
        fig.colorbar(im, ax=ax)

ax = fig.add_subplot(gs[0, 2])
for name, (solver, res) in runs.items():
    ax.plot(res.t / (2 * np.pi), np.abs(res.psi).max(axis=1) / eps, label=name)
ax.set_xlabel("t (carrier periods)")
ax.set_ylabel("max |A| / a₀")
ax.set_title("Peak amplification")
ax.legend()
ax.grid(alpha=0.3)

ax = fig.add_subplot(gs[1, 0])
for name, (solver, res) in runs.items():
    i = int(np.argmax(np.abs(res.psi).max(axis=1)))
    j = int(np.argmax(np.abs(res.psi[i])))
    xi = solver.x - solver.x[j]
    win = np.abs(xi) < 40
    ax.plot(xi[win] / (2 * np.pi), np.abs(res.psi[i, win]) / eps, label=name)
ax.set_xlabel("distance from peak (wavelengths)")
ax.set_ylabel("|A| / a₀")
ax.set_title("Envelope at focus (wave travels → )")
ax.legend()
ax.grid(alpha=0.3)

ax = fig.add_subplot(gs[1, 1:])
for name in ("NLS", "Dysthe", "Dysthe, exact dispersion"):
    solver, res = runs[name]
    i = int(np.argmax(np.abs(res.psi).max(axis=1)))
    sub = type(res)(res.x, res.t[i:i + 1], res.psi[i:i + 1], res.params)
    wave = DystheWave(sub, solver, period=T, order=2)
    j = int(np.argmax(np.abs(res.psi[i])))
    k0 = wave.k0
    x_peak = solver.x[j] / k0 + wave.cg * wave.times[0]
    wave.phase = -np.angle(res.psi[i, j]) - k0 * x_peak + wave.w0 * wave.times[0]
    x = x_peak + np.linspace(-600, 600, 2400)
    ax.plot(x - x_peak, wave.eta(x, wave.times[0]), label=f"{name} (2nd-order surface)")
ax.set_xlabel("distance from focus (m)")
ax.set_ylabel("η (m)")
ax.set_title(f"Surface at focus, T = {T} s, a₀ = {eps / k0:.2f} m")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("figures/dysthe_vs_nls.png", dpi=150)
