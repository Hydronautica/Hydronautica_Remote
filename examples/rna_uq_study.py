"""Uncertainty from sensors to fatigue: latent-force Kalman smoother vs the truth.

For each record (operating; parked storm with a wave slam the model does not know about) and
each sensor set S0-S3: simulate the measurements from a truth whose tower responds to the rotor
and the waves, run the latent-force smoother, draw 200 load histories, push each through the FE
chain to stress and damage, then add stress-model and S-N scatter. Checks that the bands are
honest (95 % bands should hold the truth about 95 % of the time) and where the true damage falls
in the estimated distribution. Writes figures/rna/uq_*.png and demo/rna/uq_study.json.
"""
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from rna.chain import assess_chain, load, turbine_for  # noqa: E402
from rna.lfm import LFM, aero_prior, damages, outputs, shaft_series, spots_from, truth_record, with_model_uq  # noqa: E402
from rna.virtual import SENSOR_SETS, estimate, prior_cov, r2  # noqa: E402

fe = load(os.path.join(ROOT, "demo", "rna", "fe_chain.npz"))
tb = turbine_for(fe)
FIG = os.path.join(ROOT, "figures", "rna"); os.makedirs(FIG, exist_ok=True)
CASES = {"op": dict(Hs=2.6, Tp=8.0, slam=None, title="Operating at 12 m/s, Hs 2.6 m"),
         "storm": dict(Hs=7.5, Tp=11.5, slam=(330.0, 3.0e6), title="Parked storm, Hs 7.5 m, wave slam at 330 s")}
COMPS = ["hub", "main_shaft", "main_bearing", "housing", "gears", "gear_bearings", "bedplate", "bolts"]
NAMES = {"hub": "Hub", "main_shaft": "Main shaft", "main_bearing": "Main bearing", "housing": "Gearbox housing",
         "gears": "Gears", "gear_bearings": "Gearbox bearings", "bedplate": "Bedplate", "bolts": "Blade-root bolts"}
SIG = {"flap": ("Blade A flapwise root moment", "MN·m", lambda x, o, s: x[:, 3]),
       "tilt": ("Shaft-flange tilt moment", "MN·m", lambda x, o, s: o["fl"][:, 5]),
       "mb": ("Main-bearing radial load", "MN", lambda x, o, s: np.hypot(o["mb"][:, 1], o["mb"][:, 2])),
       "hub": ("Hub hot-spot stress", "MPa", lambda x, o, s: o["hub"]),
       "tt": ("Tower-top tilt moment", "MN·m", lambda x, o, s: o["yaw"][:, 5])}
N_S = 200
out = {"sets": SENSOR_SETS, "cases": {}}
keep = {}
for kind, cs in CASES.items():
    t0 = time.time()
    rec = truth_record(kind, tb, cs["Hs"], cs["Tp"], slam=cs["slam"])
    comps, _ = assess_chain(rec, fe, tb)
    spots = spots_from(comps)
    ap = aero_prior(tb, kind)
    lf = LFM(tb, kind, ap, cs["Hs"] * 1.08, cs["Tp"])            # the radar reads Hs 8 % high
    P = prior_cov(tb, kind)
    res_k = {"title": cs["title"], "sets": {}}
    for which, s in enumerate(SENSOR_SETS):
        r = lf.smooth(rec, fe, which, n_samples=N_S)
        maps, xt = r["maps"], r["x_true"]
        d_true, o_true = damages(xt, maps, fe, spots, rec, tb)
        samp = {k: [] for k in COMPS}
        sig_s = {k: [] for k in SIG}
        for x in r["X"]:
            d, o = damages(x, maps, fe, spots, rec, tb)
            for k in COMPS: samp[k].append(d[k])
            for k, (_, _, f) in SIG.items(): sig_s[k].append(f(x, o, None))
        samp = {k: np.array(v) for k, v in samp.items()}
        tot = with_model_uq(samp, np.random.default_rng(11))
        # static BLUE from the earlier study, for comparison
        _, xb, _, _ = estimate(rec, fe, P, which, tb)
        d_blue, _ = damages(xb.T, maps, fe, spots, rec, tb)
        o_mean = outputs(r["Xmean"], maps, fe, spots)
        sigs = {}
        for k, (nm, u, f) in SIG.items():
            S = np.array(sig_s[k]); lo, hi = np.percentile(S, [2.5, 97.5], 0); tr = f(xt, o_true, None)
            sigs[k] = dict(r2=r2(tr, f(r["Xmean"], o_mean, None)), coverage=float(np.mean((tr >= lo) & (tr <= hi))),
                           band=float(np.mean(hi - lo)))
            if kind == "op" and s in ("S0", "S2"):
                keep[(s, k)] = (rec["t"], tr, f(r["Xmean"], o_mean, None), lo, hi)
        q = lambda v: [float(np.percentile(v, p)) for p in (5, 50, 95)]
        res_k["sets"][s] = dict(
            sensors=r["sensors"], signals=sigs,
            damage={k: dict(true=float(d_true[k]), est=q(samp[k]), total=q(tot[k]), blue=float(d_blue[k]),
                            pct_true=float(np.mean(samp[k] <= d_true[k]) * 100), pct_true_total=float(np.mean(tot[k] <= d_true[k]) * 100))
                    for k in COMPS})
        cv = " ".join(f"{k} R²{v['r2']:.2f}/cov{v['coverage']:.2f}" for k, v in sigs.items())
        print(f"{kind} {s} ({time.time() - t0:.0f}s) {cv}")
        print("    damage true | P5-P50-P95 est | +model:", " ; ".join(f"{k} {d_true[k]:.3g} | {q(samp[k])[0]:.3g}-{q(samp[k])[1]:.3g}-{q(samp[k])[2]:.3g} | {q(tot[k])[0]:.3g}-{q(tot[k])[2]:.3g}" for k in COMPS))
    out["cases"][kind] = res_k
with open(os.path.join(ROOT, "demo", "rna", "uq_study.json"), "w") as fh:
    json.dump(out, fh, indent=1)

# figure 1: histories with 95 % bands, S0 vs S2 (operating), a 120 s window
fig, ax = plt.subplots(4, 2, figsize=(12, 10), sharex=True)
for j, s in enumerate(("S0", "S2")):
    for i, k in enumerate(("flap", "tilt", "mb", "hub")):
        t, tr, m, lo, hi = keep[(s, k)]
        w = (t >= 200) & (t < 320)
        a = ax[i, j]
        a.fill_between(t[w], lo[w], hi[w], color="#0a6a8a", alpha=0.25, lw=0, label="95 % band")
        a.plot(t[w], m[w], color="#0a6a8a", lw=1.4, label="estimate (posterior mean)")
        a.plot(t[w], tr[w], color="#bb3328", lw=1.0, ls="--", label="truth")
        sg = out["cases"]["op"]["sets"][s]["signals"][k]
        a.set_title(f"{SIG[k][0]} · R² {sg['r2']:.2f} · truth inside band {100 * sg['coverage']:.0f} %", fontsize=9.5, loc="left")
        a.set_ylabel(SIG[k][1]); a.grid(alpha=0.3)
    ax[0, j].text(0.0, 1.28, f"{s}: {SENSOR_SETS[s]}", transform=ax[0, j].transAxes, fontsize=12, fontweight="bold")
ax[0, 0].legend(fontsize=8, loc="upper right"); ax[-1, 0].set_xlabel("time (s)"); ax[-1, 1].set_xlabel("time (s)")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "uq_histories.png"), dpi=130); plt.close(fig)

# figure 2: damage distributions per component and sensor set
fig, axs = plt.subplots(2, 4, figsize=(16, 8.6), sharey="row")
for i, kind in enumerate(CASES):
    for j, s in enumerate(SENSOR_SETS):
        a = axs[i, j]; D = out["cases"][kind]["sets"][s]["damage"]
        for c, k in enumerate(COMPS):
            d = D[k]
            if d["true"] < 1e-5 and d["est"][2] < 1e-5:
                continue
            a.plot([c, c], [max(d["total"][0], 1e-6), d["total"][2]], color="#9fc3d1", lw=9, solid_capstyle="butt")
            a.plot([c, c], [max(d["est"][0], 1e-6), d["est"][2]], color="#0a6a8a", lw=4, solid_capstyle="butt")
            a.plot(c, max(d["est"][1], 1e-6), "o", color="white", mec="#0a6a8a", ms=5)
            a.plot(c, max(d["true"], 1e-6), "D", color="#bb3328", ms=6)
            a.plot(c, max(d["blue"], 1e-6), "x", color="#444", ms=7, mew=1.6)
        a.set_yscale("log"); a.axhline(1, color="k", lw=0.8, ls=":")
        a.set_xticks(range(len(COMPS))); a.set_xticklabels([NAMES[k] for k in COMPS], rotation=60, ha="right", fontsize=8)
        a.set_title(f"{s} · {'operating' if kind == 'op' else 'parked storm'}", fontsize=10, loc="left"); a.grid(alpha=0.3, axis="y")
    axs[i, 0].set_ylabel("ageing (design hours per hour)")
h = [plt.Line2D([], [], color="#0a6a8a", lw=4), plt.Line2D([], [], color="#9fc3d1", lw=9), plt.Line2D([], [], marker="o", color="#0a6a8a", mfc="white", ls=""),
     plt.Line2D([], [], marker="D", color="#bb3328", ls=""), plt.Line2D([], [], marker="x", color="#444", ls="", mew=1.6)]
fig.legend(h, ["P5–P95 from the sensors (estimation)", "+ stress-model and S-N scatter", "median", "truth", "earlier static estimate"], loc="upper center", ncol=5, fontsize=9, frameon=False)
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(os.path.join(FIG, "uq_damage.png"), dpi=130); plt.close(fig)
print("figures written")
