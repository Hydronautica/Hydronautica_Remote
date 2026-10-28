"""Pack the whole-turbine uncertainty replay for StormTwin: whole_replay.json, loaded on demand.

For each record (operating, parked storm) and sensor set (W0-W3): the smoother's modal means and
covariances in both planes at 2 Hz, from which the page rebuilds the bending moment and its 95 %
band at any elevation (moments are linear in the modes); the truth moments for reference; blade
and main-bearing signals with bands; and the study's damage distributions and checks
(whole_study.json, from examples/twin_whole_study.py).
"""
import base64
import copy
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import structure as st  # noqa: E402
from rna.chain import load, turbine_for  # noqa: E402
from rna.lfm import aero_prior  # noqa: E402
from twin.whole import Beam, WholeLFM, hot_spots, section_props, simulate  # noqa: E402

STEP = 2                                     # 4 Hz smoother -> 2 Hz replay
CASES = {"op": (2.6, 8.0), "storm": (7.5, 11.5)}
SETS = ["W0", "W1", "W2", "W3"]


def b64(a, dtype=np.float32):
    return base64.b64encode(np.ascontiguousarray(np.asarray(a, dtype)).tobytes()).decode()


fe = load(os.path.join(ROOT, "demo", "rna", "fe_chain.npz"))
tb = turbine_for(fe)
tbs = st.Turbine(m_rna=3 * tb.blade_mass + tb.hub_mass + tb.nacelle_mass, mean_thrust=0.0)
with open(os.path.join(HERE, "whole_study.json")) as fh:
    study = json.load(fh)

out = dict(study=study, sets={k: v for k, v in study["sets"].items()}, records={})
geom = None
for kind, (Hs, Tp) in CASES.items():
    models = {p: st.Model(copy.deepcopy(tbs)) for p in ("fa", "ss")}
    rec = simulate(kind, tb, models, Hs, Tp)
    em = {p: st.Model(copy.deepcopy(tbs)) for p in ("fa", "ss")}
    beams = {p: Beam(em[p]) for p in em}
    lf = WholeLFM(tb, kind, aero_prior(tb, kind), em, Hs * 1.08, Tp)
    zc, Mop = beams["fa"].moment_op()
    keep = np.arange(0, zc.size)                                   # every element midpoint
    if geom is None:
        spots = hot_spots(em)
        zs = em["fa"].z
        geom = dict(z_nodes=zs.tolist(), D=[float(tbs.diameter(z)) for z in zs], depth=tbs.depth, embedment=tbs.embedment,
                    z_interface=tbs.z_interface, z_hub=tbs.z_hub, z_door=tbs.z_door, zc=zc[keep].tolist(),
                    spots={k: dict(z=float(v["z"]), name=v["name"]) for k, v in spots.items()},
                    sensors={"W0": [["nacelle accelerometer", tbs.z_hub]],
                             "W1": [["nacelle accelerometer", tbs.z_hub], ["tower-base gauges", tbs.z_interface + 3]],
                             "W2": [["nacelle accelerometer", tbs.z_hub], ["tower-base gauges", tbs.z_interface + 3], ["tower-top gauges", tbs.z_hub - 6], ["TP accelerometer", tbs.z_interface]],
                             "W3": [["nacelle accelerometer", tbs.z_hub], ["tower-base gauges", tbs.z_interface + 3], ["tower-top gauges", tbs.z_hub - 6], ["TP accelerometer", tbs.z_interface], ["blade-root gauges", tbs.z_hub]]})
    n = rec["t"].size
    idx = np.arange(0, n, STEP)
    R = dict(n=int(idx.size), dt=float(rec["dt"] * STEP), title=study["cases"][kind]["title"], planes={}, sets={})
    for p in ("fa", "ss"):
        md = lf.md[p]
        R["planes"][p] = dict(rows=b64((Mop[keep] @ md["Phi"]) / 1e6), m0=b64(Mop[keep] @ md["u0"] / 1e6),
                              truth=b64(np.round((rec[f"U_{p}"][idx] @ Mop[keep].T) / 1e6 * 10), np.int16))     # 0.1 MN m
    xt = rec["blade"][0, 3] / 1e6
    R["truth"] = dict(flap=b64(xt[idx]), mb=None)
    for which, s in enumerate(SETS):
        r = lf.smooth(rec, fe, which, n_samples=60)
        ms, Ps = r["ms"][idx], r["Ps"][idx]
        S = {}
        for p in ("fa", "ss"):
            i0, nm = lf.i[p], lf.nm
            S[f"q_{p}"] = b64(ms[:, i0:i0 + nm])
            iu = np.triu_indices(nm)
            S[f"P_{p}"] = b64(Ps[:, i0:i0 + nm, i0:i0 + nm][:, iu[0], iu[1]])
        # blade A flap (linear) and main-bearing radial load (nonlinear: from the sampled histories)
        X = r["mx"].T[None] + np.einsum("nij,snj->sni", r["Gx"], r["Z"])
        flap = np.percentile(X[:, idx, 3], [2.5, 50, 97.5], 0)
        mbr = np.hypot(*np.moveaxis(np.einsum("nij,snj->sni", r["maps"]["mb"][:, 1:3], X), -1, 0))
        mbq = np.percentile(mbr[:, idx], [2.5, 50, 97.5], 0)
        if R["truth"]["mb"] is None:
            xtr = r["x_true"]
            mbt = np.hypot(*(np.einsum("nij,nj->ni", r["maps"]["mb"][:, 1:3], xtr)).T)
            R["truth"]["mb"] = b64(mbt[idx])
        S["flap"] = b64(flap); S["mb"] = b64(mbq)
        S["sensors"] = r["sensors"]
        R["sets"][s] = S
        print(kind, s, "packed")
    out["records"][kind] = R
out["geom"] = geom
out["nm"] = lf.nm
path = os.path.join(HERE, "whole_replay.json")
with open(path, "w") as fh:
    json.dump(out, fh, separators=(",", ":"))
print(f"wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB)")
