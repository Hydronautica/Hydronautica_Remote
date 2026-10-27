"""Modulational instability of a seeded background -> Akhmediev breathers.

A uniform wave with a tiny harmonic perturbation grows into an Akhmediev
breather, then returns close to the background (Fermi-Pasta-Ulam-Tsingou
recurrence) and repeats. Compare with the exact breather of the same period.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

import nls
from nls import plotting

os.makedirs("figures", exist_ok=True)

K = 1.0  # modulation wavenumber, inside the unstable band 0 < K < 2
a = nls.akhmediev_a_for_period(2 * np.pi / K)
_, _, peak = nls.breather_params(a)
print(f"a = {a:.4f}, exact AB peak = {peak:.4f}")

solver = nls.NLSSolver(L=2 * 2 * np.pi / K, N=256)  # two modulation periods
res = solver.run(lambda x: nls.modulated_background(x, eps=1e-3, K=K), 0, 40, dt=0.005, n_out=801)

events = nls.find_rogue_events(res, threshold=2.0, reference=1.0)
for e in events[:6]:
    print(f"  t={e['t']:6.2f}  x={e['x']:6.2f}  |psi|={e['amplitude']:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
plotting.plot_evolution(res, ax=axes[0], title=f"MI from seeded background, K={K}")
plotting.plot_peak(res, ax=axes[1])
axes[1].axhline(peak, ls=":", c="k", label="exact AB peak")
axes[1].legend()
fig.tight_layout()
fig.savefig("figures/akhmediev_mi.png", dpi=150)
