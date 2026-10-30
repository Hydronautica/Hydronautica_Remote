"""A storm over a fetched site's real seabed in HOS-Ocean, with turbine probes and an optional flat-seabed control.

    python -m sitedata fetch --center 36.89 -75.49 --size 16 12 --name "Virginia" -o site_va.json
    python examples/hos_site_storm.py site_va.json --crop 4 14 5.5 10.5 --out runs/va --run --flat

The crop (x0 x1 y0 y1, km on the site map, y southwards) keeps the run affordable: a 10 x 5 km strip with a 14 s
sea is about 50 000 modes and takes roughly an hour on four cores. The storm defaults to the site's 50-year Hs, the
mean Tp of the record's Hs >= 5 m states and its prevailing wave direction. Turbines are laid out on a grid in the
crop (or taken from --turbines, a JSON list of {name, x, y} in km on the site map).

The flat control uses exactly the same input file with the seabed replaced by the crop's mean depth, so the two runs
share the grid and the incoming sea; the ratio of their Hs at each turbine is what the bathymetry does there. Keep in
mind that a 10-minute record gives Hs to about +/-10 % and the largest wave only loosely (for a linear sea of 600 s,
Hmax/Hs spans about 1.2 to 1.5): use longer runs or several seeds before reading extremes from it.
"""

import argparse
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nls.hos_ocean import HOSCase, check_bathymetry, read_probes, results_json, run, wave_stats  # noqa: E402


def storm_from_climate(site):
    c = site.get("climate") or {}
    sc = c.get("scatter")
    tp = 12.0
    if sc:
        C = np.array(sc["count"]); he = np.array(sc["hs_edges"]); te = np.array(sc["tp_edges"])
        w = C[he[:-1] >= 5].sum(0) if (he[:-1] >= 5).any() else C.sum(0)
        if w.sum():
            tp = float((w * (te[:-1] + te[1:]) / 2).sum() / w.sum())
    return {"Hs": round(site["meta"].get("hs50", 6.0), 2), "Tp": round(tp, 1), "from": site["meta"].get("waveDir", 270), "spread": 20, "gamma": 3.3, "duration": 100, "seed": 107}


def crop(site, x0, x1, y0, y1, turbines=None, spacing=2.0):
    nx, ny = site["nx"], site["ny"]; cx, cy = site["W"] / nx, site["H"] / ny
    D = np.array(site["depth"]).reshape(ny, nx)
    i0, i1, j0, j1 = int(x0 / cx), int(x1 / cx), int(y0 / cy), int(y1 / cy)
    if turbines is None:
        xs = np.arange(x0 + spacing / 2, x1, spacing); ys = np.arange(y0 + spacing / 2, y1, spacing)
        turbines = [{"name": f"T{k + 1:02d}", "x": float(x), "y": float(y)} for k, (y, x) in enumerate((y, x) for y in ys for x in xs)]
    tb = [{"name": t["name"], "x": t["x"] - i0 * cx, "y": t["y"] - j0 * cy} for t in turbines]
    return {"name": site["name"] + " (crop)", "W": (i1 - i0) * cx, "H": (j1 - j0) * cy, "nx": i1 - i0, "ny": j1 - j0,
            "depth": D[j0:j1, i0:i1].ravel().tolist(), "turbines": tb}


def flat_copy(src, dst):
    """Same case with a flat seabed at the mean depth of the bathymetry file (identical grid and incoming sea)."""
    os.makedirs(dst, exist_ok=True)
    for f in ("input_HOS-Ocean.yml", "prob.inp", "probes.json"):
        shutil.copy(os.path.join(src, f), os.path.join(dst, f))
    L = open(os.path.join(src, "bathy.dat")).read().splitlines()
    mean = np.mean([float(r[22:32]) for r in L[4:]])
    with open(os.path.join(dst, "bathy.dat"), "w") as fh:
        fh.write("\n".join(L[:4] + [r[:22] + "%10.4E" % mean for r in L[4:]]) + "\n")


def summary(folder, flat=None, skip=None):
    meta = json.load(open(os.path.join(folder, "probes.json")))
    A = read_probes(os.path.join(folder, "Results", "probes.dat"))
    skip = skip if skip is not None else 0.55 * A.t[-1]
    B = read_probes(os.path.join(flat, "Results", "probes.dat")) if flat else None
    print(f"{'probe':>6} {'Hs':>6} {'Hmax':>6} {'kurt':>5}" + (f" {'Hs flat':>8} {'ratio':>6}" if B else ""))
    for i, p in enumerate(meta["probes"]):
        k = A.t >= skip; a = wave_stats(A.t[k], A.eta[k, i])
        line = f"{p['name']:>6} {a['hs']:6.2f} {a['hmax']:6.2f} {a['kurtosis']:5.2f}"
        if B:
            kb = B.t >= skip; b = wave_stats(B.t[kb], B.eta[kb, i]); line += f" {b['hs']:8.2f} {a['hs'] / b['hs']:6.2f}"
        print(line)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("site"); ap.add_argument("--crop", nargs=4, type=float, required=True, metavar=("X0", "X1", "Y0", "Y1"))
    ap.add_argument("--turbines"); ap.add_argument("--out", default="hos_site_run")
    ap.add_argument("--hs", type=float); ap.add_argument("--tp", type=float); ap.add_argument("--dir", type=float); ap.add_argument("--spread", type=float)
    ap.add_argument("--ppw", type=float, default=6.0); ap.add_argument("--run", action="store_true"); ap.add_argument("--flat", action="store_true")
    a = ap.parse_args(argv)
    site = json.load(open(a.site))
    sea = storm_from_climate(site)
    for k, v in (("Hs", a.hs), ("Tp", a.tp), ("from", a.dir), ("spread", a.spread)):
        if v is not None:
            sea[k] = v
    exp = crop(site, *a.crop, turbines=json.load(open(a.turbines)) if a.turbines else None)
    exp["sea"] = sea
    case = HOSCase.from_stormtwin(exp, points_per_wavelength=a.ppw, outputs_per_period=16)
    real = os.path.join(a.out, "real"); case.write(real)
    print(f"Case written to {real}: {case.resolution()} modes, peak wavelength {case.wavelength:.0f} m, sea {sea}")
    if a.flat:
        flat_copy(real, os.path.join(a.out, "flat"))
    if a.run:
        run(real)
        print("seabed check:", check_bathymetry(case, real))
        json.dump(results_json(real), open(os.path.join(a.out, "hos_results.json"), "w"))
        if a.flat:
            run(os.path.join(a.out, "flat"))
        summary(real, os.path.join(a.out, "flat") if a.flat else None)
        print(f"Import {os.path.join(a.out, 'hos_results.json')} in StormTwin (Simulate > Sea > Recorded surface).")


if __name__ == "__main__":
    main()
