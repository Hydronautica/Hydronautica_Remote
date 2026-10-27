"""Inject demo_data.json into template.html and write index.html."""

import os

HERE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(HERE, "template.html")) as fh:
    page = fh.read()
with open(os.path.join(HERE, "demo_data.json")) as fh:
    data = fh.read().replace("</", "<\\/")
with open(os.path.join(HERE, "index.html"), "w") as fh:
    fh.write(page.replace("__DATA__", data))
print("wrote index.html", os.path.getsize(os.path.join(HERE, "index.html")) // 1024, "kB")
