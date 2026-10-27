"""A Peregrine rogue wave in physical units: 10 s swell, 1.5 m background amplitude."""

import os

import matplotlib.pyplot as plt
import numpy as np

import nls

os.makedirs("figures", exist_ok=True)

dw = nls.DeepWater(a0=1.5, period=10.0)
print(dw.describe())

T0 = -3.0
solver = nls.NLSSolver(L=200, N=4096)
res = solver.run(lambda X: nls.peregrine(X, T0), T0, -T0, dt=0.002, n_out=301)

# Surface elevation around the focus, in the frame moving with the group velocity.
i0 = np.argmin(np.abs(res.t))
# The NLS grid is coarser than the carrier wave, so interpolate the (smooth)
# envelope onto a fine grid with ~40 points per carrier wavelength first.
X_fine = np.linspace(-6, 6, int(12 / dw.x_scale / dw.wavelength * 40))
psi_fine = np.interp(X_fine, res.x, res.psi[i0].real) + 1j * np.interp(X_fine, res.x, res.psi[i0].imag)
# psi(0, 0) = -3, so phase = pi puts a crest (rather than a trough) at the focus.
phase = np.pi
x, t, eta = dw.surface(psi_fine, X_fine, res.t[i0], phase=phase)
print(f"max crest at focus: {eta.max():.2f} m (envelope 3 x a0 = {3 * dw.a0:.2f} m)")
print(f"crest-to-trough height: {eta.max() - eta.min():.2f} m vs background {2 * dw.a0:.2f} m")

# Time series at a fixed wave probe placed where the wave focuses.
x_probe = dw.to_physical_x(0.0, dw.to_physical_t(0.0))
t_phys = np.linspace(-150.0, 150.0, 3000)  # +-15 carrier periods around the focus
X_probe = dw.to_normalised_x(x_probe, t_phys)
T_probe = dw.to_normalised_t(t_phys)
psi_probe = nls.peregrine(X_probe, T_probe)  # exact; matches the run to plotting accuracy
eta_probe = np.real(dw.envelope(psi_probe) * np.exp(1j * (dw.k0 * x_probe - dw.w0 * t_phys + phase)))

fig, axes = plt.subplots(2, 1, figsize=(11, 7))
axes[0].plot(x - x_probe, eta)
axes[0].plot(x - x_probe, np.abs(dw.envelope(psi_fine)), "k--", lw=0.8)
axes[0].set_xlabel("distance from focus (m)")
axes[0].set_ylabel("η (m)")
axes[0].set_title(f"Surface at focus, t = {t:.0f} s")
axes[1].plot(t_phys, eta_probe)
axes[1].set_xlabel("time relative to focus (s)")
axes[1].set_ylabel("η (m)")
axes[1].set_title("Wave probe at focal point")
fig.tight_layout()
fig.savefig("figures/ocean_peregrine.png", dpi=150)
