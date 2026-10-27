"""Kuznetsov-Ma breather: localised in x, periodic in t."""

import os

import matplotlib.pyplot as plt
import numpy as np

import nls
from nls import plotting

os.makedirs("figures", exist_ok=True)

a = 1.0
_, b, peak = nls.breather_params(a)
period = 2 * np.pi / b
f = lambda x, t: nls.kuznetsov_ma(x, t, a)

solver = nls.NLSSolver(L=40, N=1024)
res = solver.run(lambda x: f(x, 0.0), 0.0, 3 * period, dt=0.001, n_out=601)
print(f"period = {period:.4f}, peak = {peak:.4f}")
print(f"error after 3 periods: {np.abs(res.psi[-1] - f(res.x, res.t[-1])).max():.2e}")

window = np.abs(res.x) < 8
sub = nls.Result(res.x[window], res.t, res.psi[:, window], res.params)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
plotting.plot_evolution(sub, ax=axes[0], title=f"Kuznetsov-Ma breather, a={a}")
plotting.plot_peak(res, ax=axes[1], exact=f)
fig.tight_layout()
fig.savefig("figures/kuznetsov_ma.png", dpi=150)
