"""Build index.html: the farm datasets (demo_data_A/B/C.json), the drivetrain FE chain (rna_chain.json,
from build_rna.py) and the live structural models (live_struct_*.json, from build_live.py), all inline
so the page needs no network requests for its data."""

import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

farms, live = [], {}
for path in sorted(glob.glob(os.path.join(HERE, "demo_data_*.json"))):
    with open(path) as fh:
        d = json.load(fh)
    if "history" not in d:
        continue
    fid = os.path.basename(path)[10]
    lp = os.path.join(HERE, f"live_struct_{fid}.json")
    if os.path.exists(lp):
        with open(lp) as fh:
            ls = json.load(fh)
        live[d["meta"]["farm"]["name"]] = ls
        # the tracked tower frequencies follow each turbine's true soil (see build_live.py)
        for t, lt in zip(d["turbines"], ls["turbines"]):
            for k in ("tower_fa", "tower_ss", "tower_2"):
                mo = t["modal"]["modes"][k]
                mo["f"] = [round(v * lt["freq_scale"], 5) for v in mo["f"]]
                mo["pred"] = [round(v * lt["freq_scale"], 5) for v in mo["pred"]]
    farms.append(d)
with open(os.path.join(HERE, "rna_chain.json")) as fh:
    rna = json.load(fh)
for p in rna["parts"].values():                       # displacements are not used here
    p.pop("D", None); p.pop("d_scale", None)
with open(os.path.join(HERE, "template.html")) as fh:
    page = fh.read()
enc = lambda o: json.dumps(o, separators=(",", ":")).replace("</", "<\\/")
page = page.replace("__DATA__", enc({"farms": farms})).replace("__RNADATA__", enc(rna)).replace("__LIVEDATA__", enc(live))
with open(os.path.join(HERE, "index.html"), "w") as fh:
    fh.write(page)
print("wrote index.html with", len(farms), "farms,", os.path.getsize(os.path.join(HERE, "index.html")) // 1024, "kB")
