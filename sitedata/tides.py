"""Tides and tidal currents from NOAA CO-OPS (US coasts): datums and harmonic constituents at the nearest
harmonic tide station, and the peak predicted current at the nearest current station."""

from datetime import date, timedelta

import numpy as np

from .geo import haversine_km
from .net import get_json

MD = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi"
DATA = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
MAIN = ("M2", "S2", "N2", "K1", "O1")


def _nearest(kind, lat, lon):
    st = get_json(f"{MD}/stations.json?type={kind}")["stations"]
    la = np.array([float(s["lat"]) for s in st]); lo = np.array([float(s["lng"]) for s in st])
    d = haversine_km(lat, lon, la, lo)
    i = int(np.argmin(d))
    return st[i], float(d[i])


def tide(lat, lon, log=print):
    s, dist = _nearest("harcon", lat, lon)
    dat = {d["name"]: d["value"] for d in get_json(f"{MD}/stations/{s['id']}/datums.json?units=metric").get("datums", []) if d.get("value") is not None}
    hc = get_json(f"{MD}/stations/{s['id']}/harcon.json?units=metric").get("HarmonicConstituents", [])
    amp = {c["name"]: c["amplitude"] for c in hc}
    out = {"station": s["id"], "name": s["name"], "lat": float(s["lat"]), "lon": float(s["lng"]), "distance_km": dist,
           "datums_m": dat, "constituents_m": {k: amp[k] for k in MAIN if k in amp}}
    if "MHHW" in dat and "MLLW" in dat:
        out["range_m"] = dat["MHHW"] - dat["MLLW"]                     # great diurnal range
    elif "MHW" in dat and "MLW" in dat:
        out["range_m"] = dat["MHW"] - dat["MLW"]
    else:                                                               # spring range estimate from constituents
        out["range_m"] = 2 * sum(amp.get(k, 0) for k in ("M2", "S2", "K1", "O1"))
    if "HAT" in dat and "LAT" in dat:
        out["hat_lat_m"] = dat["HAT"] - dat["LAT"]
    log(f"tides: {s['name']} ({s['id']}), {dist:.0f} km away, range {out['range_m']:.2f} m")
    return out


def current(lat, lon, days=30, log=print):
    s, dist = _nearest("currentpredictions", lat, lon)
    sid = s["id"] + (f"_{s['currbin']}" if s.get("currbin") else "")
    b = date.today().replace(day=1)
    q = (f"{DATA}?product=currents_predictions&station={s['id']}&bin={s.get('currbin') or 1}&begin_date={b:%Y%m%d}"
         f"&end_date={(b + timedelta(days=days)):%Y%m%d}&interval=MAX_SLACK&units=metric&time_zone=gmt&format=json&application=Hydronautica")
    pred = get_json(q).get("current_predictions", {}).get("cp", [])
    v = np.array([abs(float(p.get("Velocity_Major", 0))) for p in pred])
    peak = float(v.max() / 100.0) if len(v) else None                   # metric velocities are cm/s
    log(f"tidal current: {s['name']} ({sid}), {dist:.0f} km away, peak {peak if peak is not None else 'n/a'} m/s")
    return {"station": sid, "name": s["name"], "lat": float(s["lat"]), "lon": float(s["lng"]), "distance_km": dist,
            "peak_m_s": peak, "mean_max_m_s": float(np.mean(np.sort(v)[-max(1, len(v) // 2):]) / 100.0) if len(v) else None}
