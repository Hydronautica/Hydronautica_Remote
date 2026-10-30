"""Inline core.js and ui.js into one HTML file (for sharing as a single page): python3 make_single.py -> windsim_single.html"""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
page = open(os.path.join(HERE, "index.html")).read()
for name in ("core.js", "ui.js"):
    code = open(os.path.join(HERE, name)).read().replace("</", "<\\/")
    page = page.replace(f'<script src="{name}"></script>', f"<script>\n{code}\n</script>")
open(os.path.join(HERE, "windsim_single.html"), "w").write(page)
print("wrote windsim_single.html", os.path.getsize(os.path.join(HERE, "windsim_single.html")) // 1024, "kB")
