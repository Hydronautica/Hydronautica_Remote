"""Inject the bedplate FE results (bedplate.json, from examples/rna_bedplate_fe.py) into template.html -> index.html."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "bedplate.json")) as fh:
    data = fh.read().replace("</", "<\\/")
with open(os.path.join(HERE, "template.html")) as fh:
    page = fh.read()
with open(os.path.join(HERE, "index.html"), "w") as fh:
    fh.write(page.replace("__FEDATA__", data))
print("wrote index.html,", os.path.getsize(os.path.join(HERE, "index.html")) // 1024, "kB")
