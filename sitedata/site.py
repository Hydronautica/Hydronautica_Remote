"""Assemble a StormTwin map file for an area: bathymetry grid, metocean from WIS, tides from CO-OPS."""

import math
from datetime import datetime, timezone

import numpy as np

from . import bathymetry, tides, wis
from .geo import LocalGrid

FORMAT = "stormtwin-site/1"


def build_site(grid, name=None, bathy_source="auto", wis_stride=1, include_daily=True, max_wis_km=60.0, log=print):
    """Everything StormTwin's map builder needs for the area, as a JSON-ready dict.

    The result is a regular StormTwin map (``W``, ``H``, ``nx``, ``ny``, ``depth``, ``soil``, ``excl``, ``meta``) with
    extra ``geo``, ``climate``, ``tide``, ``current`` and ``sources`` fields. Land cells are marked as exclusion zones.
    Missing data (no WIS station nearby, no tide station) is reported in ``sources`` and leaves StormTwin's defaults.
    """
    lat0, lon0 = grid.lat0, 0.5 * (grid.west + grid.east)
    name = name or f"Site {lat0:.3f}, {lon0:.3f}"
    depth, land, binfo = bathymetry.fetch(grid, bathy_source, log=log)
    sources = {"bathymetry": binfo}
    meta = {"lat": round(lat0, 5), "lon": round(lon0, 5), "soil0": 0}
    out = {"format": FORMAT, "name": name, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "W": round(grid.W, 3), "H": round(grid.H, 3), "nx": grid.nx, "ny": grid.ny,
           "depth": np.round(np.maximum(depth, 1.0), 2).ravel().tolist(),
           "soil": [255] * (grid.nx * grid.ny), "excl": land.astype(int).ravel().tolist(),
           "land": land.astype(int).ravel().tolist(), "geo": grid.as_dict(), "meta": meta, "sources": sources}

    st = wis.nearest(lat0, lon0, 1)[0]
    if st["distance_km"] <= max_wis_km:
        log(f"WIS: nearest station {st['id']} ({st['region']}), {st['distance_km']:.1f} km away")
        ser = wis.fetch_series(st, stride=wis_stride, log=log)
        cl = wis.climate(ser)
        out["climate"] = cl
        if include_daily:
            out["daily"] = wis.daily_series(ser)
        sources["climate"] = {"source": "USACE Wave Information Studies hindcast", "station": st["id"], "region": st["region"],
                              "lat": st["lat"], "lon": st["lon"], "distance_km": round(st["distance_km"], 2),
                              "years": cl["years"], "stride_h": wis_stride}
        meta.update({"hs": round(cl["hs_mean"], 2), "tp": round(cl["tp_mean"], 2), "hs50": round(cl["hs50"], 2),
                     "A": round(cl["wind10_A"], 2), "k": round(cl["wind10_k"], 3),
                     "windDir": round(cl["wind_dir"]), "waveDir": round(cl["wave_dir"])})
    else:
        sources["climate"] = {"source": None, "note": f"no WIS station within {max_wis_km:.0f} km "
                              f"(nearest {st['id']} at {st['distance_km']:.0f} km); WIS covers US coasts and the Great Lakes"}
        log(sources["climate"]["note"])

    for key, fn, field, conv in (("tide", tides.tide, "tide", lambda d: d["range_m"]), ("current", tides.current, "current", lambda d: d["peak_m_s"])):
        try:
            d = fn(lat0, lon0, log=log)
            if d["distance_km"] > 150:
                raise ValueError(f"nearest station {d['name']} is {d['distance_km']:.0f} km away")
            out[key] = d
            v = conv(d)
            if v is not None:
                meta[field] = round(float(v), 2)
            sources[key] = {"source": "NOAA CO-OPS", "station": d["station"], "name": d["name"], "distance_km": round(d["distance_km"], 1)}
        except Exception as ex:
            sources[key] = {"source": None, "note": str(ex)}
            log(f"{key}: not available ({ex})")
    return out
