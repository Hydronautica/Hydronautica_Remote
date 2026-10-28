"""Build index.html from the three FE models (fe_chain.npz, from examples/rna_fe_chain.py), the two
illustrative load records built from blade-root loads, and every component's damage (rna package)."""
import base64
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from rna import synthetic  # noqa: E402
from rna.chain import assess_chain, closure, load, turbine_for  # noqa: E402

ORDER = ["hub", "main_shaft", "main_bearing", "housing", "gears", "gear_bearings", "bedplate",
         "pitch_bearings", "bolts", "yaw_bearing", "equipment", "generator"]
PARTS = {"hub": "hub", "dt": "drivetrain", "bed": "bedplate"}
UNITS = {"hub": ["MN", "MN", "MN", "MN·m", "MN·m", "MN·m"] * 3 + ["m/s²"] * 3,
         "dt": ["MN"] * 3 + ["MN·m"] * 3 + ["m/s²"] * 3,
         "bed": ["MN"] * 3 + ["MN·m"] * 3 + ["MN"] * 6 + ["m/s²"] * 3}


def b64(a, dtype=np.float32):
    return base64.b64encode(np.ascontiguousarray(np.asarray(a, dtype)).tobytes()).decode()


def q8(a, axis_cases):
    """int8 per load case (max-abs scaling)."""
    a = np.moveaxis(a, axis_cases, 0)
    sc = np.abs(a).reshape(a.shape[0], -1).max(1) / 127 + 1e-30
    return b64(np.round(a / sc.reshape((-1,) + (1,) * (a.ndim - 1))), np.int8), sc.tolist()


def part_data(fe, key):
    p = PARTS[key]
    S = fe[f"{p}_stress"]                                     # nv × cases × 6
    D = fe[f"{p}_disp"]                                       # nv × 3 × cases
    tris = fe[f"{p}_tris"]
    out = dict(origin=fe[f"{p}_origin"].tolist(), h=float(fe["h"]), nv=int(len(S)), names=fe[f"{p}_names"].tolist(),
               units=UNITS[key], mass_t=float(fe[f"{p}_mass"]) / 1e3, cg=fe[f"{p}_cg"].tolist(),
               grid=b64(fe[f"{p}_grid"], np.uint8), nearbc=b64(fe[f"{p}_near_bc"], np.uint8))
    if key == "dt":
        sh = fe["drivetrain_is_shaft"][tris].all(1)
        tris = np.vstack([tris[sh], tris[~sh]])
        out["n_shaft_tris"] = int(sh.sum())
        out["is_shaft"] = b64(fe["drivetrain_is_shaft"], np.uint8)
    out["nt"] = int(len(tris))
    out["tris"] = b64(tris, np.uint16)
    out["S"], out["s_scale"] = q8(S, 1)
    out["D"], out["d_scale"] = q8(D, 2)
    return out


def pack(c):
    d = {k: c[k] for k in ("key", "name", "method", "standard", "ageing", "check", "metrics", "note")}
    d["signal"] = None if c["signal"] is None else dict(label=c["signal"][0], unit=c["signal"][1], data=b64(c["signal"][2]))
    d["cum"] = None if c["cum"] is None else b64(c["cum"] / 60.0)    # design-life minutes used
    if "part" in c:
        d["part"] = {v: k for k, v in PARTS.items()}[c["part"]]
        d["vertex"] = c["vertex"]
    return d


def main():
    fe = load(os.path.join(HERE, "fe_chain.npz"))
    tb = turbine_for(fe)
    data = dict(parts={k: part_data(fe, k) for k in PARTS}, records={}, comps={}, closure={},
                chain={k: fe[k].tolist() for k in ("R_hub", "R_mb", "R_armL", "R_armR", "R_yaw")},
                geo=dict(hub=fe["hub_c"].tolist(), x_mb=float(fe["x_mb"]), pads=fe["pads"].tolist(), yaw=fe["yaw"].tolist(),
                         r_root=tb.r_root, rest_mass_t=float(fe["rest_mass"]) / 1e3))
    for kind in ("op", "storm"):
        rec = synthetic(kind, tb)
        comps, ch = assess_chain(rec, fe, tb)
        comps = {c["key"]: c for c in comps}
        n = rec["t"].size
        data["records"][kind] = dict(dt=rec["dt"], n=n, blade=b64(rec["blade"].reshape(18, n) / 1e6), psi=b64(rec["psi"]),
                                     A=b64(rec["A"].T), rpm=b64(rec["rpm"]),
                                     hub=b64(np.vstack([rec["hub_F"].T, rec["hub_M"].T]) / 1e6),
                                     tt=b64(np.vstack([rec["tt_F"].T, rec["tt_M"].T]) / 1e6))
        data["comps"][kind] = [pack(comps[k]) for k in ORDER]
        data["closure"][kind] = closure(rec, ch)
        print(kind, ", ".join(f"{k} {comps[k]['ageing']:.3g}" for k in ORDER if comps[k]["ageing"] is not None), data["closure"][kind])
    with open(os.path.join(HERE, "template.html")) as fh:
        page = fh.read()
    page = page.replace("__RNADATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    with open(os.path.join(HERE, "index.html"), "w") as fh:
        fh.write(page)
    print("wrote index.html,", os.path.getsize(os.path.join(HERE, "index.html")) // 1024, "kB")


if __name__ == "__main__":
    main()
