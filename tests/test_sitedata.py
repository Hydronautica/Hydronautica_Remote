import math

import numpy as np
import pytest

from sitedata import bathymetry, site, tides, wis
from sitedata.geo import LocalGrid, haversine_km


def test_local_grid_round_trip_and_size():
    g = LocalGrid.around(36.89, -75.49, 16, 12, 0.25)
    assert g.W == pytest.approx(16, rel=1e-6) and g.H == pytest.approx(12, rel=1e-6)
    assert (g.nx, g.ny) == (64, 48)
    lat, lon = g.to_latlon(4.0, 3.0)
    x, y = g.to_km(lat, lon)
    assert (x, y) == (pytest.approx(4.0), pytest.approx(3.0))
    assert haversine_km(g.north, g.west, g.north, g.east) == pytest.approx(16, rel=2e-3)


def test_resample_matches_a_plane():
    lat_s = np.linspace(37.2, 36.6, 61); lon_s = np.linspace(-75.8, -75.2, 61)
    Z = 10 * lat_s[:, None] - 3 * lon_s[None, :]
    g = LocalGrid(36.8, -75.6, 37.0, -75.4, 0.5)
    lat, lon = g.centres()
    out = bathymetry._resample(lat_s, lon_s, Z, lat, lon)
    assert np.allclose(out, 10 * lat[:, None] - 3 * lon[None, :], atol=1e-9)


def test_weibull_fit_recovers_parameters():
    rng = np.random.default_rng(0)
    u = 9.0 * rng.weibull(2.2, 200000)
    A, k = wis.weibull_fit(u)
    assert A == pytest.approx(9.0, rel=0.02) and k == pytest.approx(2.2, rel=0.05)


def test_extremes_on_a_gumbel_sample():
    rng = np.random.default_rng(1)
    mu, beta = 4.0, 0.6
    amax = mu - beta * np.log(-np.log(rng.random(2000)))
    true50 = mu - beta * math.log(-math.log(1 - 1 / 50))
    assert wis.gumbel_return(amax) == pytest.approx(true50, rel=0.03)


def test_pot_on_exponential_storms():
    # hourly series of 40 years with independent storm peaks every ~10 days, excess exponential (sigma 0.8 m)
    rng = np.random.default_rng(2)
    n = 40 * 365 * 24
    t = np.datetime64("1980-01-01T00") + np.arange(n).astype("timedelta64[h]")
    hs = np.full(n, 1.0)
    peaks = np.arange(100, n - 100, 240)
    hs[peaks] = 3.0 + rng.exponential(0.8, len(peaks))
    est, npk, u = wis.pot_return(t, hs, quantile=0.995)
    lam = len(peaks) / 40
    # the threshold sits among the peaks; check against the exponential model above it
    exc = hs[peaks][hs[peaks] > u]
    expect = u + np.mean(exc - u) * math.log(len(exc) / 40 * 50)
    assert npk == len(exc) and est == pytest.approx(expect, rel=1e-6)


def test_vector_mean_direction_wraps():
    d = wis.vector_mean_dir(np.array([350.0, 10.0]), np.array([1.0, 1.0]))
    assert min(d, 360 - d) == pytest.approx(0.0, abs=1e-9)


def test_ascii_parser():
    txt = "Dataset {\n} x;\n---------------------------------------------\nwaveHs[3]\n1.5, 2.25, -999.0\n\n"
    assert np.allclose(wis._ascii_array(txt, "waveHs"), [1.5, 2.25, -999.0])


def _series(years=6):
    n = years * 365 * 24
    t = np.datetime64("2000-01-01T00") + np.arange(n).astype("timedelta64[h]")
    ph = 2 * np.pi * np.arange(n) / (365.25 * 24)
    rng = np.random.default_rng(3)
    hs = 1.2 + 0.4 * np.cos(ph) + 0.3 * rng.random(n)
    return {"time": t.astype("datetime64[s]"), "waveHs": hs, "waveTpPeak": 6 + 2 * hs, "waveMeanDirection": np.full(n, 80.0),
            "windSpeed": 8 * rng.weibull(2.0, n), "windDirection": np.full(n, 230.0)}


def test_climate_summary_and_daily():
    s = _series()
    c = wis.climate(s)
    assert c["years"] == [2000, 2005] and c["hs_mean"] == pytest.approx(np.mean(s["waveHs"]))
    assert c["wave_dir"] == pytest.approx(80, abs=1e-6) and c["wind_dir"] == pytest.approx(230, abs=1e-6)
    assert sum(map(sum, c["scatter"]["count"])) == len(s["waveHs"])
    assert c["monthly"][0]["hs"] > c["monthly"][6]["hs"]                  # winter rougher than summer
    d = wis.daily_series(s)
    assert len(d["hs_mean"]) == 6 * 365 and d["hs_max"][0] >= d["hs_mean"][0]


def test_build_site_offline(monkeypatch):
    g = LocalGrid.around(36.89, -75.49, 4, 3, 0.5)
    D = np.full((g.ny, g.nx), 25.0); land = np.zeros_like(D, bool); land[0, 0] = True; D[land] = 0
    monkeypatch.setattr(bathymetry, "fetch", lambda grid, src="auto", log=print: (D, land, {"source": "test", "description": "test"}))
    monkeypatch.setattr(wis, "fetch_series", lambda st, stride=1, log=print: _series(3))
    monkeypatch.setattr(tides, "tide", lambda lat, lon, log=print: {"station": "1", "name": "T", "distance_km": 10.0, "range_m": 1.2})
    monkeypatch.setattr(tides, "current", lambda lat, lon, log=print: {"station": "2", "name": "C", "distance_km": 400.0, "peak_m_s": 0.5})
    out = site.build_site(g, name="Test", log=lambda *a: None)
    assert out["nx"] * out["ny"] == len(out["depth"]) and out["excl"][0] == 1 and out["depth"][0] == 1.0
    assert out["meta"]["tide"] == 1.2 and "current" not in out["meta"]       # current station too far away
    assert out["sources"]["climate"]["station"].startswith("ST") and out["meta"]["waveDir"] == 80


def test_nearest_station_index():
    st = wis.nearest(36.89, -75.49, 2)
    assert st[0]["distance_km"] < 10 and st[0]["distance_km"] <= st[1]["distance_km"]
