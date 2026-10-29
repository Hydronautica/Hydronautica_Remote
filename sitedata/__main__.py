"""python -m sitedata fetch --bbox S W N E [--name NAME] [-o site.json]
   python -m sitedata fetch --center LAT LON --size W_KM H_KM ...
   python -m sitedata fetch request.json          (the area request StormTwin downloads)
   python -m sitedata stations --near LAT LON      (nearest WIS stations)"""

import argparse
import json
import sys

from .geo import LocalGrid
from .site import build_site
from . import wis


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m sitedata", description="Site data for StormTwin from online databases")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="bathymetry, climate and tides for an area -> StormTwin map file")
    f.add_argument("request", nargs="?", help="area request JSON from StormTwin")
    f.add_argument("--bbox", nargs=4, type=float, metavar=("S", "W", "N", "E"))
    f.add_argument("--center", nargs=2, type=float, metavar=("LAT", "LON"))
    f.add_argument("--size", nargs=2, type=float, metavar=("W_KM", "H_KM"), default=(15.0, 12.0))
    f.add_argument("--cell", type=float, default=None, help="grid cell in km (default: about 100 cells across)")
    f.add_argument("--name")
    f.add_argument("--bathymetry", default="auto", choices=["auto", "ncei", "etopo", "emodnet"])
    f.add_argument("--stride", type=int, default=1, help="WIS: use every Nth hour (1 = hourly)")
    f.add_argument("--no-daily", action="store_true", help="leave out the daily wave/wind series")
    f.add_argument("-o", "--output")
    s = sub.add_parser("stations", help="nearest WIS stations")
    s.add_argument("--near", nargs=2, type=float, required=True, metavar=("LAT", "LON"))
    s.add_argument("-n", type=int, default=5)
    a = ap.parse_args(argv)

    if a.cmd == "stations":
        for st in wis.nearest(a.near[0], a.near[1], a.n):
            print(f"{st['id']:>9} {st['region']:<13} {st['lat']:8.4f} {st['lon']:9.4f}  {st['distance_km']:6.1f} km")
        return
    req = {}
    if a.request:
        with open(a.request) as fh:
            req = json.load(fh)
    if a.bbox:
        bbox = a.bbox
    elif "bbox" in req:
        b = req["bbox"]; bbox = [b["south"], b["west"], b["north"], b["east"]]
    elif a.center:
        g = LocalGrid.around(a.center[0], a.center[1], *a.size)
        bbox = [g.south, g.west, g.north, g.east]
    else:
        ap.error("give --bbox, --center or a request file")
    cell = a.cell or req.get("cell_km")
    g = LocalGrid(*bbox, cell_km=1.0)
    if not cell:
        cell = max(0.05, round(max(g.W, g.H) / 100, 3))
    g = LocalGrid(*bbox, cell_km=cell)
    name = a.name or req.get("name")
    log = lambda m: print(m, file=sys.stderr)
    site = build_site(g, name=name, bathy_source=a.bathymetry if a.bathymetry != "auto" else req.get("bathymetry", "auto"),
                      wis_stride=a.stride, include_daily=not a.no_daily, log=log)
    out = a.output or (site["name"].replace(" ", "_").replace(",", "") + ".json")
    with open(out, "w") as fh:
        json.dump(site, fh, separators=(",", ":"))
    m = site["meta"]
    print(f"Wrote {out}: {site['W']:.1f} x {site['H']:.1f} km, {site['nx']} x {site['ny']} cells; "
          f"Hs mean {m.get('hs', '-')} m, Hs50 {m.get('hs50', '-')} m, wind A {m.get('A', '-')} k {m.get('k', '-')}, "
          f"tide {m.get('tide', '-')} m, current {m.get('current', '-')} m/s. Import it in StormTwin: Maps > Import map.")


if __name__ == "__main__":
    main()
