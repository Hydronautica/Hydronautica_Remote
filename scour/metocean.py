"""Metocean input for the scour estimate: load a hindcast or build a synthetic one.

A record is a dict of equal-length arrays: time (datetime64[s]), hs (m), tp (s)
and uc (depth-averaged current speed, m/s), on a constant time step.
"""

import csv
from datetime import datetime, timezone

import numpy as np

M2 = 2 * np.pi / (12.4206 * 3600)     # principal lunar semi-diurnal (rad/s)
S2 = 2 * np.pi / (12.0 * 3600)        # principal solar semi-diurnal


def load_csv(path):
    """Read columns time, hs, tp, uc (ISO time; any extra columns ignored)."""
    t, hs, tp, uc = [], [], [], []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            t.append(np.datetime64(row["time"], "s"))
            hs.append(float(row["hs"])); tp.append(float(row["tp"])); uc.append(float(row["uc"]))
    rec = dict(time=np.array(t), hs=np.array(hs), tp=np.array(tp), uc=np.array(uc))
    step = np.diff(rec["time"]).astype(float)
    if len(step) and np.ptp(step) > 1:
        raise ValueError("records must be on a constant time step")
    return rec


def synthetic(start, years, hs_median=1.4, seasonal=0.4, tp_coef=3.6, u_spring=0.7,
              surge_per_m=0.05, storms=(), storms_per_year=3.0, seed=7, dt_hours=1.0):
    """Synthetic hourly record.

    Hs: log-normal AR(1) around a seasonal median, plus Gaussian storm peaks
    (given as (datetime, peak Hs, half-width h) and/or random ones).
    Tp: wind-sea steepness, tp_coef * sqrt(Hs), at least 4 s.
    Current: M2 + S2 tide (spring peak u_spring, S2 one third of M2) plus a
    storm-surge residual proportional to Hs above the median, along the flood.
    """
    rng = np.random.default_rng(seed)
    n = int(years * 8766 / dt_hours)
    t0 = np.datetime64(start, "s")
    time = t0 + (np.arange(n) * dt_hours * 3600).astype("timedelta64[s]")
    hours = np.arange(n) * dt_hours
    doy = ((time - time.astype("datetime64[Y]")).astype("timedelta64[D]").astype(float))
    mean = np.log(hs_median) + seasonal * np.cos(2 * np.pi * (doy - 20) / 365.25)
    phi = 0.985 ** dt_hours
    e = rng.normal(0, 0.45 * np.sqrt(1 - phi**2), n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    hs = np.exp(mean + x)
    peaks = [((np.datetime64(d, "s") - t0).astype(float) / 3600, h, w) for d, h, w in storms]
    for _ in range(rng.poisson(storms_per_year * years)):
        h0 = rng.uniform(0, hours[-1])
        winter = np.cos(2 * np.pi * ((h0 / 24) % 365.25 - 20) / 365.25)
        if rng.random() < 0.5 + 0.4 * winter:
            peaks.append((h0, hs_median * rng.uniform(2.8, 4.8), rng.uniform(10, 24)))
    for h0, hp, w in peaks:
        hs = np.maximum(hs, hp * np.exp(-0.5 * ((hours - h0) / (w / 1.5)) ** 2))
    tp = np.maximum(4.0, tp_coef * np.sqrt(hs))
    a_m2 = u_spring / (1 + 1 / 3)
    ts = hours * 3600
    tide = a_m2 * np.cos(M2 * ts) + a_m2 / 3 * np.cos(S2 * ts + 0.7)
    surge = surge_per_m * np.maximum(hs - hs_median, 0)
    uc = np.abs(tide + surge)
    return dict(time=time, hs=hs, tp=tp, uc=uc)


def to_datetime(t):
    return datetime.fromtimestamp(int(t.astype("datetime64[s]").astype(int)), timezone.utc).replace(tzinfo=None)
