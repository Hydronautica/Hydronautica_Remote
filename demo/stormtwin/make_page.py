"""Inject the farm datasets (demo_data_A/B/C.json) into template.html -> index.html."""

import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

farms = []
for path in sorted(glob.glob(os.path.join(HERE, "demo_data_*.json"))):
    with open(path) as fh:
        d = json.load(fh)
    if "history" in d:
        farms.append(d)
with open(os.path.join(HERE, "template.html")) as fh:
    page = fh.read()
data = json.dumps({"farms": farms}, separators=(",", ":")).replace("</", "<\\/")
with open(os.path.join(HERE, "index.html"), "w") as fh:
    fh.write(page.replace("__DATA__", data))
print("wrote index.html with", len(farms), "farms,", os.path.getsize(os.path.join(HERE, "index.html")) // 1024, "kB")
