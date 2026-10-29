"""USACE Wave Information Studies (WIS) hindcast: the long-term wave and wind climate at the nearest station.

Each WIS station has an aggregated dataset (hourly, 1980 onward) on the ERDC THREDDS server, read here over
OPeNDAP's ASCII service. ``wis_stations.json`` lists the ~4000 stations (id, region, lat, lon).

The climate summary gives what the design tools need: mean sea state, the 50-year significant wave height
(Gumbel fit to annual maxima, cross-checked with a peaks-over-threshold exponential fit), the 10 m wind Weibull
fit, prevailing wind and wave directions (vector mean of the energy / power), an Hs-Tp scatter table, wave and
wind roses, and monthly means.
"""

import json
import math
import os
import re
from datetime import datetime, timezone

import numpy as np

from .geo import haversine_km
from .net import get_bytes

THREDDS = "https://chldata.erdc.dren.mil/thredds/dodsC/wis"
VARS = ["waveHs", "waveTpPeak", "waveMeanDirection", "windSpeed", "windDirection"]
_STATIONS = os.path.join(os.path.dirname(__file__), "wis_stations.json")


def stations():
    with open(_STATIONS) as f:
        return json.load(f)["stations"]


def nearest(lat, lon, n=1):
    st = stations()
    d = haversine_km(lat, lon, np.array([s[2] for s in st]), np.array([s[3] for s in st]))
    idx = np.argsort(d)[:n]
    return [dict(id=st[i][0], region=st[i][1], lat=st[i][2], lon=st[i][3], distance_km=float(d[i])) for i in idx]


def _ascii_array(txt, name):
    """Values of one variable from an OPeNDAP .ascii response."""
    m = re.search(rf"^{name}(?:\.{name})?\[\d+\]\s*\n(.*?)(?:\n\s*\n|\Z)", txt, re.S | re.M)
    if not m:
        raise ValueError(f"{name} not found in OPeNDAP response")
    return np.array([float(v) for v in m.group(1).replace("\n", ",").split(",") if v.strip()])


def fetch_series(station, stride=1, start=None, log=print):
    """Hourly (or every ``stride`` hours) time series at a station: dict of arrays with 'time' as datetime64[s]."""
    base = f"{THREDDS}/{station['region']}/{station['id']}/{station['id']}.ncml"
    dds = get_bytes(base + ".dds").decode()
    n = int(re.search(r"time\[time = (\d+)\]", dds).group(1))
    sel = f"%5B0:{stride}:{n - 1}%5D"
    out = {}
    for v in ["time"] + VARS:
        log(f"WIS {station['id']}: {v} ({n // stride} values)")
        out[v] = _ascii_array(get_bytes(base + ".ascii?" + v + sel, timeout=600).decode(), v)
    out["time"] = out["time"].astype("int64").astype("datetime64[s]")
    for v in VARS:
        out[v][(out[v] < -900) | (out[v] > 9e3)] = np.nan
    if start is not None:
        keep = out["time"] >= np.datetime64(start)
        out = {k: a[keep] for k, a in out.items()}
    return out


# ----- statistics -----
def weibull_fit(u):
    """Weibull A, k by matching the mean and the mean cube (energy pattern factor method, robust for wind)."""
    u = u[np.isfinite(u) & (u > 0)]
    m1, m3 = u.mean(), np.mean(u**3)
    epf = m3 / m1**3
    k = 1.0 + 3.69 / epf**2                     # Akdag & Dinler (2009)
    A = m1 / math.gamma(1 + 1 / k)
    return float(A), float(k)


def gumbel_return(annual_max, T=50.0):
    """Return value from a Gumbel fit (method of moments) to annual maxima."""
    x = np.asarray(annual_max, float)
    x = x[np.isfinite(x)]
    beta = x.std(ddof=1) * math.sqrt(6) / math.pi
    mu = x.mean() - 0.5772 * beta
    return float(mu - beta * math.log(-math.log(1 - 1 / T)))


def pot_return(t, hs, T=50.0, quantile=0.99, gap_h=48):
    """Peaks over threshold (independent storms at least ``gap_h`` apart) with an exponential excess model."""
    ok = np.isfinite(hs)
    t, hs = t[ok], hs[ok]
    u = np.quantile(hs, quantile)
    over = np.where(hs > u)[0]
    peaks, cur = [], [over[0]] if len(over) else []
    for i in over[1:]:
        if (t[i] - t[cur[-1]]) / np.timedelta64(1, "h") > gap_h:
            peaks.append(max(cur, key=lambda j: hs[j])); cur = [i]
        else:
            cur.append(i)
    if cur:
        peaks.append(max(cur, key=lambda j: hs[j]))
    years = (t[-1] - t[0]) / np.timedelta64(365 * 24 * 3600, "s")
    lam = len(peaks) / years
    sigma = np.mean(hs[peaks] - u)
    return float(u + sigma * math.log(lam * T)), len(peaks), float(u)


def vector_mean_dir(direction_deg, weight):
    ok = np.isfinite(direction_deg) & np.isfinite(weight)
    r = np.radians(direction_deg[ok])
    return float((math.degrees(math.atan2(np.sum(weight[ok] * np.sin(r)), np.sum(weight[ok] * np.cos(r)))) + 360) % 360)


def rose(direction_deg, value, sectors=12, bins=None):
    edges = np.arange(sectors + 1) * 360 / sectors - 180 / sectors
    d = (np.asarray(direction_deg) + 180 / sectors) % 360 - 180 / sectors
    ok = np.isfinite(d) & np.isfinite(value)
    bins = bins if bins is not None else np.quantile(value[ok], [0, 0.25, 0.5, 0.75, 0.95, 1.0])
    H, _, _ = np.histogram2d(d[ok], value[ok], bins=[edges, bins])
    return {"sector_deg": list(np.arange(sectors) * 360 / sectors), "bins": [float(b) for b in bins], "freq": (H / ok.sum()).round(5).tolist()}


def climate(series, hub_heights=None):
    t, hs, tp, wd = series["time"], series["waveHs"], series["waveTpPeak"], series["waveMeanDirection"]
    ws, wdir = series["windSpeed"], series["windDirection"]
    ok = np.isfinite(hs)
    years = t.astype("datetime64[Y]").astype(int) + 1970
    full = [y for y in np.unique(years) if np.sum((years == y) & ok) > 0.8 * np.sum(years == y) and np.sum(years == y) > 0.8 * len(t) / len(np.unique(years))]
    amax = [np.nanmax(hs[years == y]) for y in full]
    hs50_g = gumbel_return(amax)
    hs50_p, npk, thr = pot_return(t, hs)
    A, k = weibull_fit(ws)
    hs_edges = np.arange(0, max(2.0, np.ceil(np.nanmax(hs) * 2) / 2) + 0.5, 0.5)
    tp_edges = np.arange(0, max(4.0, np.ceil(np.nanmax(tp))) + 1, 1.0)
    okj = ok & np.isfinite(tp)
    S, _, _ = np.histogram2d(hs[okj], tp[okj], bins=[hs_edges, tp_edges])
    month = t.astype("datetime64[M]").astype(int) % 12
    out = {
        "years": [int(min(years)), int(max(years))],
        "n_records": int(ok.sum()),
        "hs_mean": float(np.nanmean(hs)), "tp_mean": float(np.nanmean(tp)),
        "hs_p50": float(np.nanpercentile(hs, 50)), "hs_p99": float(np.nanpercentile(hs, 99)), "hs_max": float(np.nanmax(hs)),
        "hs50_gumbel": hs50_g, "hs50_pot": hs50_p, "pot_storms": npk, "pot_threshold": thr,
        "hs50": float(max(hs50_g, hs50_p)),
        "tp_at_hsmax": float(tp[np.nanargmax(hs)]),
        "annual_max": [{"year": int(y), "hs": float(a)} for y, a in zip(full, amax)],
        "wind10_A": A, "wind10_k": k, "wind10_mean": float(np.nanmean(ws)),
        "wind_dir": vector_mean_dir(wdir, ws**3), "wave_dir": vector_mean_dir(wd, hs**2 * np.nan_to_num(tp)),
        "scatter": {"hs_edges": hs_edges.tolist(), "tp_edges": tp_edges.tolist(), "count": S.astype(int).tolist()},
        "wave_rose": rose(wd, hs, bins=[0, 1, 2, 3, 4, 6, 99]),
        "wind_rose": rose(wdir, ws, bins=[0, 4, 8, 12, 16, 20, 99]),
        "monthly": [{"month": m + 1, "hs": float(np.nanmean(hs[month == m])), "wind": float(np.nanmean(ws[month == m]))} for m in range(12)],
        "fatigue": fatigue_table(series),
    }
    return out


MIS_EDGES = [0, 15, 45, 75, 105, 180.0001]     # |wave direction - wind direction| classes
MIS_CENTRES = [0, 30, 60, 90, 135]


def fatigue_table(series, sector_deg=30.0, u_step=1.0, u_max=36.0):
    """Joint wind-wave climate for fatigue, lumped from the hourly record.

    Cells are (wind direction sector, 10 m wind speed bin, wind-wave misalignment class). Each keeps its share of
    the time, the damage-equivalent significant wave height (Hs averaged with exponent 2, i.e. wave energy, which is
    what drives the wave-load spectrum) and the mean peak period. Returned as rows
    [sector index, speed bin index, misalignment class, probability, Hs, Tp, signed mean misalignment] for non-empty cells.
    """
    ws, wd, hs, tp, md = (series[k] for k in ("windSpeed", "windDirection", "waveHs", "waveTpPeak", "waveMeanDirection"))
    ok = np.isfinite(ws) & np.isfinite(wd) & np.isfinite(hs) & np.isfinite(tp) & np.isfinite(md)
    ws, wd, hs, tp, md = ws[ok], wd[ok], hs[ok], tp[ok], md[ok]
    ns = int(round(360 / sector_deg))
    si = np.floor(((wd + sector_deg / 2) % 360) / sector_deg).astype(int) % ns
    ui = np.minimum(np.floor(ws / u_step).astype(int), int(u_max / u_step) - 1)
    dmis = (md - wd + 180) % 360 - 180                       # signed, wave relative to wind
    mi = np.searchsorted(MIS_EDGES, np.abs(dmis), side="right") - 1
    key = (si * 1000 + ui) * 10 + mi
    keys, inv = np.unique(key, return_inverse=True)
    n = len(ws)
    cnt = np.bincount(inv).astype(float)
    hs2 = np.bincount(inv, weights=hs**2) / cnt
    tpm = np.bincount(inv, weights=tp) / cnt
    dm = np.bincount(inv, weights=dmis) / cnt
    rows = []
    for k, c, h, t, d in zip(keys, cnt, hs2, tpm, dm):
        rows.append([int(k // 10000), int(k // 10 % 1000), int(k % 10), round(c / n, 7), round(float(np.sqrt(h)), 3), round(float(t), 2), round(float(d), 1)])
    return {"sector_deg": sector_deg, "u_step": u_step, "height_m": 10.0, "mis_centres": MIS_CENTRES, "mis_edges": MIS_EDGES[:-1] + [180],
            "fields": ["sector", "u_bin", "mis_class", "p", "hs", "tp", "mis_mean"], "rows": rows, "hours": int(n)}


def daily_series(series):
    """Compact daily means and maxima for the twin's long-term views (about 17k days for 46 years)."""
    day = series["time"].astype("datetime64[D]")
    days, inv = np.unique(day, return_inverse=True)
    hs, ws = series["waveHs"], series["windSpeed"]
    cnt = np.bincount(inv, weights=np.isfinite(hs)).astype(float)
    hs_mean = np.bincount(inv, weights=np.nan_to_num(hs)) / np.maximum(cnt, 1)
    hs_max = np.full(len(days), -np.inf); np.maximum.at(hs_max, inv, np.nan_to_num(hs, nan=-np.inf))
    cw = np.bincount(inv, weights=np.isfinite(ws)).astype(float)
    ws_mean = np.bincount(inv, weights=np.nan_to_num(ws)) / np.maximum(cw, 1)
    return {"start": str(days[0]), "hs_mean": np.round(hs_mean, 2).tolist(), "hs_max": np.round(np.where(np.isfinite(hs_max), hs_max, 0), 2).tolist(),
            "wind_mean": np.round(ws_mean, 2).tolist()}
