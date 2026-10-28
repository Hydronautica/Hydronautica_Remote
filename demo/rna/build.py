"""Build index.html: bedplate FE fields (bedplate.json, from examples/rna_bedplate_fe.py), the two
illustrative load records built from blade-root loads, and every component's damage (rna package)."""
import base64
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from rna import SN, Turbine, assess, bedplate_inputs, synthetic, unit_load_hotspot  # noqa: E402

CAST_SN = SN(80e6, 2e6, 7, 13)       # representative ductile cast iron curve in stress range
ORDER = ["bedplate", "main_shaft", "main_bearing", "gears", "gear_bearings", "pitch_bearings",
         "bolts", "yaw_bearing", "equipment", "hub", "generator"]


def b64(a):
    return base64.b64encode(np.asarray(a, np.float32).tobytes()).decode()


def bedplate_fields(fe):
    raw = np.frombuffer(base64.b64decode(fe["stress"]), np.int16).astype(float)
    S = raw.reshape(9, fe["nv"], 6) * np.asarray(fe["s_scale"])[:, None, None]
    return S.transpose(1, 0, 2)                                    # nv × 9 × 6, Pa per unit


def candidates(U, L, k=60):
    S = np.einsum("pck,cn->pnk", U, L[:, ::10])
    a, b, c, xy, yz, zx = np.moveaxis(S, -1, 0)
    vm = np.sqrt(0.5 * ((a - b) ** 2 + (b - c) ** 2 + (c - a) ** 2) + 3 * (xy ** 2 + yz ** 2 + zx ** 2))
    return np.argsort(vm.max(1))[-k:]


def pack(c):
    d = {k: c[k] for k in ("key", "name", "method", "standard", "ageing", "check", "metrics", "note")}
    d["signal"] = None if c["signal"] is None else dict(label=c["signal"][0], unit=c["signal"][1], data=b64(c["signal"][2]))
    d["cum"] = None if c["cum"] is None else b64(c["cum"] / 60.0)    # design-life minutes used
    if "point" in c:
        d["vertex"] = c["point"]
    return d


def main():
    with open(os.path.join(HERE, "bedplate.json")) as fh:
        fe = json.load(fh)
    U = bedplate_fields(fe)
    tb = Turbine()
    data = dict(records={}, comps={})
    for kind in ("op", "storm"):
        rec = synthetic(kind, tb)
        L = bedplate_inputs(rec)
        cand = candidates(U, L)
        bp = unit_load_hotspot(U[cand], L, CAST_SN, rec=rec)
        bp["point"] = int(cand[bp["point"]])
        comps = {c["key"]: c for c in assess(rec, tb) + [bp]}
        data["records"][kind] = dict(dt=rec["dt"], n=int(rec["t"].size), L=[b64(x) for x in L])
        data["comps"][kind] = [pack(comps[k]) for k in ORDER]
        print(kind, ", ".join(f"{k} {comps[k]['ageing']:.3g}" for k in ORDER if comps[k]["ageing"] is not None))
    with open(os.path.join(HERE, "template.html")) as fh:
        page = fh.read()
    page = page.replace("__FEDATA__", json.dumps(fe).replace("</", "<\\/"))
    page = page.replace("__RNADATA__", json.dumps(data).replace("</", "<\\/"))
    with open(os.path.join(HERE, "index.html"), "w") as fh:
        fh.write(page)
    print("wrote index.html,", os.path.getsize(os.path.join(HERE, "index.html")) // 1024, "kB")


if __name__ == "__main__":
    main()
