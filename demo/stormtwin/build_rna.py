"""Pack the rotor-nacelle FE load chain for StormTwin: rna_chain.json, loaded by the page on demand.

Needs demo/rna/fe_chain.npz (examples/rna_fe_chain.py) and demo/rna/virtual_sensing.json
(examples/rna_virtual_sensing.py).
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "demo", "rna"))
from build import part_data  # noqa: E402  (demo/rna/build.py)
from rna import synthetic  # noqa: E402
from rna.chain import assess_chain, load, turbine_for  # noqa: E402
from rna.components import _design_rate  # noqa: E402
from rna.virtual import gauge_row, gauges, prior_cov  # noqa: E402

fe = load(os.path.join(ROOT, "demo", "rna", "fe_chain.npz"))
tb = turbine_for(fe)
comps, _ = assess_chain(synthetic("op", tb), fe, tb)
C = {c["key"]: c for c in comps}
PK = {"hub": "hub", "drivetrain": "dt", "bedplate": "bed"}
spots = {k: dict(part=PK[C[k]["part"]], vertex=C[k]["vertex"], normal=C[k]["normal"]) for k in ("hub", "housing", "bedplate")}
spots["main_shaft"] = dict(part="dt", ring=C["main_shaft"]["ring"], angles=C["main_shaft"]["ring_angles"], phi0=C["main_shaft"]["phi0"])
with open(os.path.join(ROOT, "demo", "rna", "virtual_sensing.json")) as fh:
    study = json.load(fh)
out = dict(
    parts={k: part_data(fe, k) for k in ("hub", "dt", "bed")},
    chain={k: fe[k].tolist() for k in ("R_hub", "R_mb", "R_armL", "R_armR", "R_yaw")},
    geo=dict(hub=fe["hub_c"].tolist(), x_mb=float(fe["x_mb"]), pads=fe["pads"].tolist(), yaw=fe["yaw"].tolist(), rest_mass_t=float(fe["rest_mass"]) / 1e3),
    turbine=dict(r_root=tb.r_root, blade_mass=tb.blade_mass, blade_cg=tb.blade_cg, r_flap=tb.r_flap, r_edge=tb.r_edge,
                 rated_torque=tb.rated_torque, rated_rpm=tb.rated_rpm),
    prior={k: np.round(prior_cov(tb, k), 6).tolist() for k in ("op", "storm")},
    gauges=[dict(part=PK[p], vertex=v, normal=n.tolist(), row=np.round(gauge_row(fe, p, v, n), 6).tolist()) for p, v, n in gauges(fe)],
    spots=spots, bolts_beta=C["bolts"]["beta"],
    design_rate={str(m): _design_rate(m, tb) for m in (8.7, 6.6, 10 / 3)},
    study=study,
)
path = os.path.join(HERE, "rna_chain.json")
with open(path, "w") as fh:
    json.dump(out, fh, separators=(",", ":"))
print(f"wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB)")
