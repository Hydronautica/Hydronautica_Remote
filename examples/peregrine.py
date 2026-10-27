"""Peregrine soliton: simulate from t=-4 to t=4 and compare with the exact solution."""

import os

import matplotlib.pyplot as plt
import numpy as np

import nls
from nls import plotting

os.makedirs("figures", exist_ok=True)

solver = nls.NLSSolver(L=200, N=4096)  # wide box: Peregrine decays only as 1/x^2
res = solver.run(lambda x: nls.peregrine(x, -4.0), -4.0, 4.0, dt=0.002, n_out=401)

i0 = np.argmin(np.abs(res.t))
print(f"peak |psi| = {res.max_amplitude().max():.4f} (exact 3)")
print(f"max error at t=0: {np.abs(res.psi[i0] - nls.peregrine(res.x, 0.0)).max():.2e}")
print("invariant drift:", nls.invariant_drift(res))

window = np.abs(res.x) < 10
sub = nls.Result(res.x[window], res.t, res.psi[:, window], res.params)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
plotting.plot_evolution(sub, ax=axes[0], title="Peregrine soliton |ψ|")
plotting.plot_snapshot(sub, 0.0, ax=axes[1], exact=nls.peregrine)
fig.tight_layout()
fig.savefig("figures/peregrine.png", dpi=150)
