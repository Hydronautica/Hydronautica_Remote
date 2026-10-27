"""Rogue waves emerging from a random narrow-banded sea state."""

import os

import matplotlib.pyplot as plt
import numpy as np

import nls
from nls import plotting

os.makedirs("figures", exist_ok=True)

solver = nls.NLSSolver(L=400, N=4096)
psi0 = nls.random_wave_field(solver.x, k_width=1.0, mean_amplitude=1.0, seed=1)
res = solver.run(psi0, 0, 30, dt=0.002, n_out=601)

print("invariant drift:", nls.invariant_drift(res))
events = nls.find_rogue_events(res)  # default threshold: H > 2 Hs
print(f"{len(events)} rogue events (|psi| > 2.83 x RMS); largest:")
for e in events[:5]:
    print(f"  t={e['t']:6.2f}  x={e['x']:7.2f}  amplification={e['amplification']:.2f}")

fig, axes = plt.subplots(2, 1, figsize=(10, 8))
plotting.plot_evolution(res, ax=axes[0], cmap="magma", title="Random sea |ψ|")
if events:
    axes[0].plot([e["x"] for e in events[:5]], [e["t"] for e in events[:5]], "c+", ms=12)
axes[1].plot(res.t, nls.amplification(res))
axes[1].axhline(nls.ROGUE_THRESHOLD, ls=":", c="k", label="H = 2 Hs")
axes[1].legend()
axes[1].set_xlabel("t")
axes[1].set_ylabel("max |ψ| / RMS")
fig.tight_layout()
fig.savefig("figures/random_sea.png", dpi=150)
