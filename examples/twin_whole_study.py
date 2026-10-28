"""Whole-turbine state estimation with uncertainty: blades, drivetrain, tower, monopile, pile.

Truth: nonlinear beam (p-y soil) in both planes driven by blade-by-blade rotor loads and Morison
waves with drag. Estimator: rna/twin latent-force Kalman smoother on 4 linearised modes per plane,
with the rotor and drivetrain states. For each sensor set W0-W3, 100 load histories are drawn
and pushed to every hot spot. Writes figures/twin/whole_*.png and demo/stormtwin/whole_study.json.
"""
import copy
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "demo", "stormtwin"))
import structure as st  # noqa: E402
from rna.chain import assess_chain, load, turbine_for  # noqa: E402
from rna.lfm import MODEL_UQ, aero_prior, damages, spots_from  # noqa: E402
from rna.virtual import r2  # noqa: E402
from twin.whole import Beam, WholeLFM, blade_del, hot_spots, simulate, spot_damage  # noqa: E402

SETS = {"W0": "SCADA + nacelle accelerometer", "W1": "W0 + tower-base strain gauges",
        "W2": "W1 + tower-top gauges + transition-piece accelerometer", "W3": "W2 + blade-root strain gauges"}
CASES = {"op": dict(Hs=2.6, Tp=8.0, title="Operating at 12 m/s, Hs 2.6 m"), "storm": dict(Hs=7.5, Tp=11.5, title="Parked storm, Hs 7.5 m")}
N_S = 100
fe = load(os.path.join(ROOT, "demo", "rna", "fe_chain.npz"))
tb = turbine_for(fe)
tbs = st.Turbine(m_rna=3 * tb.blade_mass + tb.hub_mass + tb.nacelle_mass, mean_thrust=0.0)
FIG = os.path.join(ROOT, "figures", "twin"); os.makedirs(FIG, exist_ok=True)
UQ = dict(MODEL_UQ, tower_top=(0.05, 3, 0.2), tower_mid=(0.05, 3, 0.2), tower_base=(0.05, 3, 0.2), door=(0.08, 3, 0.2),
          flange=(0.05, 3, 0.2), mudline=(0.06, 3, 0.2), pile=(0.10, 3, 0.2))
DT_KEYS = ["hub", "main_shaft", "main_bearing", "housing", "bedplate", "bolts"]
NAMES = {"hub": "Hub", "main_shaft": "Main shaft", "main_bearing": "Main bearing", "housing": "Gearbox housing", "bedplate": "Bedplate", "bolts": "Blade-root bolts"}


def evaluate(U, x, maps, spots_rna, rec, models, beams, spots):
    dur = rec["t"].size * rec["dt"]
    d = {}
    for k, sp in spots.items():
        Mfa = U["fa"] @ beams["fa"].moment_row(sp["z"]); Mss = U["ss"] @ beams["ss"].moment_row(sp["z"])
        d[k] = spot_damage(Mfa, Mss, sp, models["fa"], dur)
    dd, _ = damages(x, maps, fe, spots_rna, rec, tb)
    d.update({k: dd[k] for k in DT_KEYS})
    bd = blade_del(x, dur)
    d["blade_flap_DEL"], d["blade_edge_DEL"] = bd["flap"], bd["edge"]
    mud = 2 * models["fa"].node(-tbs.depth) + 1
    d["mudline_rotation"] = float(np.degrees(np.max(np.hypot(U["fa"][:, mud], U["ss"][:, mud]))))
    return d


result = {"sets": SETS, "cases": {}}
store = {}
for kind, cs in CASES.items():
    t0 = time.time()
    models = {p: st.Model(copy.deepcopy(tbs)) for p in ("fa", "ss")}
    rec = simulate(kind, tb, models, cs["Hs"], cs["Tp"])
    comps, _ = assess_chain(rec, fe, tb)
    spots_rna = spots_from(comps)
    em = {p: st.Model(copy.deepcopy(tbs)) for p in ("fa", "ss")}
    beams = {p: Beam(em[p]) for p in em}
    spots = hot_spots(em)
    lf = WholeLFM(tb, kind, aero_prior(tb, kind), em, cs["Hs"] * 1.08, cs["Tp"])
    zc, Mop = beams["fa"].moment_op()
    res_k = {"title": cs["title"], "spot_names": {k: v["name"] for k, v in spots.items()}, "sets": {}}
    for which, s in enumerate(SETS):
        r = lf.smooth(rec, fe, which, n_samples=N_S)
        maps = r["maps"]
        truth = evaluate({"fa": rec["U_fa"], "ss": rec["U_ss"]}, r["x_true"], maps, spots_rna, rec, models, beams, spots)
        X = r["mx"].T[None] + np.einsum("nij,snj->sni", r["Gx"], r["Z"])
        Ufa, Uss = lf.displacement("fa", r["Z"]), lf.displacement("ss", r["Z"])
        samp = {k: [] for k in truth}
        for i in range(N_S):
            d = evaluate({"fa": Ufa[i], "ss": Uss[i]}, X[i], maps, spots_rna, rec, models, beams, spots)
            for k in samp: samp[k].append(d[k])
        samp = {k: np.array(v) for k, v in samp.items()}
        rng = np.random.default_rng(13)
        tot = {}
        for k, v in samp.items():
            if k in UQ:
                cov, m, s10 = UQ[k]
                tot[k] = v * np.exp(rng.normal(0, cov, v.size)) ** m * 10 ** rng.normal(0, s10, v.size)
            else:
                tot[k] = v
        q = lambda v: [float(np.percentile(v, p)) for p in (5, 50, 95)]
        # dynamic moment profile (std over the record), fore-aft
        prof_t = (rec["U_fa"] @ Mop.T).std(0) / 1e6
        prof_s = np.array([(Ufa[i] @ Mop.T).std(0) / 1e6 for i in range(N_S)])
        sig = {}
        for zz, lab in ((tbs.z_interface + 3, "Tower-base moment"), (-tbs.depth, "Mudline moment"), (-tbs.depth - 6, "Pile moment, 6 m below the seabed")):
            for p, U, Us in (("fa", rec["U_fa"], Ufa), ("ss", rec["U_ss"], Uss)):
                row = beams[p].moment_row(zz)
                tr = U @ row / 1e6; S = Us @ row / 1e6
                lo, hi = np.percentile(S, [2.5, 97.5], 0); mean = lf.displacement(p, r["ms"]) @ row / 1e6
                sig[f"{lab} ({p})"] = dict(r2=r2(tr, mean), coverage=float(np.mean((tr >= lo) & (tr <= hi))))
                if p == "fa": store[(kind, s, lab)] = (rec["t"], tr, mean, lo, hi)
        xt = r["x_true"]; xm = r["mx"].T + np.einsum("nij,nj->ni", r["Gx"], r["ms"])
        sig["Blade A flapwise root moment"] = dict(r2=r2(xt[:, 3], xm[:, 3]), coverage=float(np.mean((xt[:, 3] >= np.percentile(X[:, :, 3], 2.5, 0)) & (xt[:, 3] <= np.percentile(X[:, :, 3], 97.5, 0)))))
        store[(kind, s, "flap")] = (rec["t"], xt[:, 3], xm[:, 3], *np.percentile(X[:, :, 3], [2.5, 97.5], 0))
        store[(kind, s, "profile")] = (zc, prof_t, np.percentile(prof_s, [5, 50, 95], 0))
        res_k["sets"][s] = dict(sensors=r["sensors"], signals=sig,
                                result={k: dict(true=float(truth[k]), est=q(samp[k]), total=q(tot[k])) for k in truth})
        print(f"{kind} {s} ({time.time() - t0:.0f}s) " + " | ".join(f"{k} R²{v['r2']:.2f} cov{v['coverage']:.2f}" for k, v in sig.items() if "(fa)" in k or "Blade" in k))
        print("    " + " ; ".join(f"{k} {truth[k]:.3g} [{q(samp[k])[0]:.3g}–{q(samp[k])[2]:.3g}]" for k in truth))
    result["cases"][kind] = res_k
with open(os.path.join(ROOT, "demo", "stormtwin", "whole_study.json"), "w") as fh:
    json.dump(result, fh, indent=1)

# figure 1: dynamic moment profile from pile toe to hub
fig, axs = plt.subplots(2, 4, figsize=(15, 9.5), sharey=True)
for i, kind in enumerate(CASES):
    for j, s in enumerate(SETS):
        a = axs[i, j]; zc, pt, ps = store[(kind, s, "profile")]
        a.fill_betweenx(zc, ps[0], ps[2], color="#0a6a8a", alpha=0.3, lw=0, label="estimate P5–P95")
        a.plot(ps[1], zc, color="#0a6a8a", lw=1.5, label="estimate median"); a.plot(pt, zc, "--", color="#bb3328", lw=1.2, label="truth")
        a.axhline(0, color="#2d7fa3", lw=0.8); a.axhline(-tbs.depth, color="#8a6d3b", lw=1.2)
        a.axhspan(-tbs.depth - tbs.embedment, -tbs.depth, color="#8a6d3b", alpha=0.08)
        sensors = {"W0": [tbs.z_hub], "W1": [tbs.z_hub, tbs.z_interface + 3], "W2": [tbs.z_hub, tbs.z_interface + 3, tbs.z_hub - 6, tbs.z_interface], "W3": [tbs.z_hub, tbs.z_interface + 3, tbs.z_hub - 6, tbs.z_interface]}[s]
        for zz in sensors: a.plot(0, zz, ">", color="#1c7f47", ms=8, clip_on=False)
        a.set_title(f"{s} · {'operating' if kind == 'op' else 'parked storm'}", loc="left", fontsize=10); a.grid(alpha=0.3); a.set_xlim(left=0)
        if i == 1: a.set_xlabel("dynamic fore-aft moment, std (MN·m)")
    axs[i, 0].set_ylabel("elevation (m)")
axs[0, 0].text(0.02, 0.5, "sea", color="#2d7fa3", transform=axs[0, 0].get_yaxis_transform(), va="bottom")
axs[0, 0].legend(fontsize=8, loc="upper right")
fig.text(0.5, 0.005, "▶ sensor elevations (green) · brown band: pile in the soil", ha="center", fontsize=9)
fig.tight_layout(rect=(0, 0.02, 1, 1)); fig.savefig(os.path.join(FIG, "whole_profile.png"), dpi=130); plt.close(fig)

# figure 2: histories with bands (operating, W1 = tower-base gauges only)
fig, axs = plt.subplots(4, 2, figsize=(13, 10), sharex=True)
for j, s in enumerate(("W0", "W1")):
    for i, lab in enumerate(("Tower-base moment", "Mudline moment", "Pile moment, 6 m below the seabed", "flap")):
        t, tr, m, lo, hi = store[("op", s, lab)]
        w = (t >= 200) & (t < 320); a = axs[i, j]
        a.fill_between(t[w], lo[w], hi[w], color="#0a6a8a", alpha=0.25, lw=0); a.plot(t[w], m[w], color="#0a6a8a", lw=1.3); a.plot(t[w], tr[w], "--", color="#bb3328", lw=1)
        key = f"{lab} (fa)" if lab != "flap" else "Blade A flapwise root moment"
        sg = result["cases"]["op"]["sets"][s]["signals"][key]
        a.set_title(f"{key.replace(' (fa)', ', fore-aft')} · R² {sg['r2']:.2f} · inside band {100 * sg['coverage']:.0f} %", fontsize=9.5, loc="left")
        a.set_ylabel("MN·m"); a.grid(alpha=0.3)
    axs[0, j].text(0, 1.25, f"{s}: {SETS[s]}", transform=axs[0, j].transAxes, fontsize=12, fontweight="bold")
axs[-1, 0].set_xlabel("time (s)"); axs[-1, 1].set_xlabel("time (s)")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "whole_histories.png"), dpi=130); plt.close(fig)

# figure 3: damage at every hot spot, as estimate ÷ truth
keys = ["tower_top", "tower_mid", "door", "tower_base", "flange", "mudline", "pile"] + DT_KEYS + ["blade_flap_DEL", "blade_edge_DEL"]
extra = {"blade_flap_DEL": "Blade root flap DEL", "blade_edge_DEL": "Blade root edge DEL"}
labels = [result["cases"]["op"]["spot_names"].get(k, NAMES.get(k, extra.get(k, k))) for k in keys]
fig, axs = plt.subplots(2, 4, figsize=(17, 9.6), sharey=True)
for i, kind in enumerate(CASES):
    for j, s in enumerate(SETS):
        a = axs[i, j]; R = result["cases"][kind]["sets"][s]["result"]
        for c, k in enumerate(keys):
            d = R[k]; tr = d["true"]
            if tr < 1e-6: continue
            a.plot([c, c], [d["total"][0] / tr, d["total"][2] / tr], color="#9fc3d1", lw=8, solid_capstyle="butt")
            a.plot([c, c], [d["est"][0] / tr, d["est"][2] / tr], color="#0a6a8a", lw=3.5, solid_capstyle="butt")
            a.plot(c, d["est"][1] / tr, "o", color="white", mec="#0a6a8a", ms=4.5)
        a.axvspan(-0.5, 6.5, color="#eef2f5", zorder=0); a.axvspan(12.5, 14.5, color="#f4efe6", zorder=0)
        a.set_yscale("log"); a.set_ylim(0.1, 100); a.axhline(1, color="#bb3328", lw=1.2)
        a.set_xticks(range(len(keys))); a.set_xticklabels(labels, rotation=70, ha="right", fontsize=7.5)
        a.set_title(f"{s} · {'operating' if kind == 'op' else 'parked storm'}", fontsize=10, loc="left"); a.grid(alpha=0.3, axis="y")
    axs[i, 0].set_ylabel("estimate ÷ truth (damage; DEL for blades)")
h = [plt.Line2D([], [], color="#0a6a8a", lw=3.5), plt.Line2D([], [], color="#9fc3d1", lw=8), plt.Line2D([], [], color="#bb3328", lw=1.2)]
fig.legend(h, ["P5–P95 from the sensors", "+ stress-model and S-N scatter", "truth (ratio 1)"], loc="upper center", ncol=3, fontsize=9, frameon=False)
fig.text(0.5, 0.005, "grey: tower and foundation · white: drivetrain (FE chain) · beige: blades", ha="center", fontsize=9)
fig.tight_layout(rect=(0, 0.02, 1, 0.95)); fig.savefig(os.path.join(FIG, "whole_damage.png"), dpi=130); plt.close(fig)
print("figures written")
