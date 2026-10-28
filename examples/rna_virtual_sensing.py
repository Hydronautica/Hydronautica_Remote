"""How much of the rotor-nacelle load chain can be recovered from realistic sensor sets?

Truth: a synthetic ten-minute record. Measurements: each sensor set's channels from the
truth plus noise. Estimate: the static BLUE of rna.virtual at every time step, then the whole
FE chain and every component's damage run on the estimate. Compared with the truth:
R² of key load and stress histories and the ratio of estimated to true damage.

Writes demo/rna/virtual_sensing.json for the StormTwin page.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from rna import synthetic  # noqa: E402
from rna.chain import assess_chain, load, run, turbine_for  # noqa: E402
from rna.fe import von_mises  # noqa: E402
from rna.virtual import SENSOR_SETS, as_record, estimate, prior_cov, r2  # noqa: E402

fe = load(os.path.join(ROOT, "demo", "rna", "fe_chain.npz"))
tb = turbine_for(fe)
PARTKEY = {"hub": "L_hub", "drivetrain": "L_dt", "bedplate": "L_bed"}


def vm_at(part, v, ch):
    return von_mises(np.einsum("ck,cn->nk", fe[f"{part}_stress"][v].astype(float), ch[PARTKEY[part]])) / 1e6


def signals(rec, ch, spots):
    mb = np.hypot(ch["mb"][1], ch["mb"][2]) / 1e6
    out = {"Blade A flapwise root moment": rec["blade"][0, 3] / 1e6,
           "Hub tilt moment (shaft flange)": ch["flange"][5] / 1e6,
           "Hub yaw moment (shaft flange)": ch["flange"][4] / 1e6,
           "Main-bearing radial load": mb,
           "Torque-arm force, left": ch["armL"][1] / 1e6,
           "Tower-top tilt moment": ch["yaw"][5] / 1e6}
    for name, (part, v) in spots.items():
        out[name] = vm_at(part, v, ch)
    return out


result = {"sets": SENSOR_SETS, "records": {}}
for kind in ("op", "storm"):
    rec = synthetic(kind, tb)
    P = prior_cov(tb, kind)
    comps_t, ch_t = assess_chain(rec, fe, tb)
    spots = {f"{c['name']} hot-spot stress": (c["part"], c["vertex"]) for c in comps_t if c.get("part") and c["key"] != "main_shaft"}
    sig_t = signals(rec, ch_t, spots)
    dmg_t = {c["key"]: c["ageing"] for c in comps_t if c["ageing"]}
    rows = {}
    for s, which in zip(SENSOR_SETS, range(4)):
        x_t, x_h, sd, names = estimate(rec, fe, P, which, tb)
        r_h = as_record(rec, x_h, fe)
        comps_h, ch_h = assess_chain(r_h, fe, tb)
        sig_h = signals(r_h, ch_h, spots)
        dmg_h = {c["key"]: c["ageing"] for c in comps_h if c["ageing"]}
        rows[s] = dict(sensors=names,
                       r2={k: r2(sig_t[k], sig_h[k]) for k in sig_t},
                       damage={k: dmg_h.get(k, 0) / dmg_t[k] for k in dmg_t if dmg_t[k] > 1e-4},
                       names={c["key"]: c["name"] for c in comps_t})
        print(kind, s, " | ".join(f"{k[:22]} {v:.2f}" for k, v in rows[s]["r2"].items()))
        print("      damage est/true:", " ".join(f"{k} {v:.2f}" for k, v in rows[s]["damage"].items()))
    result["records"][kind] = rows
with open(os.path.join(ROOT, "demo", "rna", "virtual_sensing.json"), "w") as fh:
    json.dump(result, fh, indent=1)
