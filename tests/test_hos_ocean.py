import json

import numpy as np
import pytest

from nls.hos_ocean import HOSCase, SeaState, read_probes, read_surface, results_json, summary


def _case(direction=270.0, **kw):
    x = np.linspace(50, 9950, 100)
    y = np.linspace(50, 7950, 80)
    Z = 20 + 0.004 * x[None, :] + 0 * y[:, None]       # deepens towards the east, 20 -> 60 m
    probes = [("A01", 5000.0, 4000.0), ("A02", 8000.0, 2000.0)]
    return HOSCase(x, y, Z, SeaState(Hs=8, Tp=12, direction_from=direction), probes, **kw)


def test_frame_points_along_travel_direction():
    c = _case(direction=270.0)                         # from the west: travels east
    ex, ey = c.frame()
    assert np.allclose(ex, [1, 0], atol=1e-12) and np.allclose(ey, [0, 1], atol=1e-12)
    u, v = c.to_frame(5000.0, 4000.0)
    assert u == pytest.approx(5000 - 50) and v == pytest.approx(4000 - 50)
    X, Y = c.from_frame(u, v)
    assert (X, Y) == (pytest.approx(5000.0), pytest.approx(4000.0))


def test_bathy_file_fixed_format_and_values():
    c = _case(direction=225.0)                         # diagonal: sampling goes beyond the map corners
    lines = c.bathy_text().splitlines()
    nx, ny = int(lines[2][0:5]), int(lines[2][6:11])
    rows = lines[4:]
    assert len(rows) == nx * ny
    for r in rows[:50] + rows[-50:]:
        assert len(r) == 32 and r[10] == " " and r[21] == " "
        float(r[0:10]); float(r[11:21]); float(r[22:32])
    d = np.array([float(r[22:32]) for r in rows])
    assert d.min() >= 20 - 1e-6 and d.max() <= 60 + 1e-6
    u = np.array([float(r[0:10]) for r in rows[:nx]])
    assert np.all(np.diff(u) > 0) and u[0] == 0


def test_bathy_follows_the_map_when_rotated():
    c = _case(direction=270.0)
    u, v, Z = c.bathy_grid()
    X, _ = c.from_frame(u, np.zeros_like(u))
    assert np.allclose(Z[0], 20 + 0.004 * np.clip(X, 50, 9950), atol=1e-6)


def test_yaml_and_probes(tmp_path):
    yaml = pytest.importorskip("yaml")
    c = _case()
    c.write(tmp_path)
    d = yaml.safe_load((tmp_path / "input_HOS-Ocean.yml").read_text())
    assert d["bathymetry"]["type"] == "file" and d["bathymetry"]["total depth"] == pytest.approx(c.depth_total, abs=1e-3)
    assert d["computation case"]["Hs"] == 8 and d["computation case"]["spreading"] == "invbeta_cos2"
    n1, n2 = d["numerical parameters"]["discretization"]["x"], d["numerical parameters"]["discretization"]["y"]
    assert n1 % 2 == 0 and n2 % 2 == 0
    assert d["domain size"]["x"] * c.wavelength == pytest.approx(9900, rel=1e-3)
    pr = (tmp_path / "prob.inp").read_text().split()
    assert len(pr) == 4
    meta = json.loads((tmp_path / "probes.json").read_text())
    assert [p["name"] for p in meta["probes"]] == ["A01", "A02"]


def test_from_stormtwin_flips_south_to_north(tmp_path):
    exp = {"name": "t", "W": 2.0, "H": 1.0, "nx": 2, "ny": 2, "depth": [10, 11, 30, 31],   # north row first
           "turbines": [{"name": "T1", "x": 0.5, "y": 0.25}], "sea": {"Hs": 5, "Tp": 9, "from": 270, "spread": 0}}
    c = HOSCase.from_stormtwin(exp)
    assert c.depth[-1, 0] == 10 and c.depth[0, 0] == 30    # row 0 is the south edge
    assert c.probes[0][1:] == (500.0, 750.0)               # 0.25 km south of the north edge
    assert "beta: 0.017453" in c.yaml()                     # long-crested stands in as 1 degree


def _write_probes(path, t, eta):
    with open(path, "w") as f:
        f.write("# some header\n#xprobe=   1000.00   2000.00\n#yprobe=    100.00    200.00\n")
        f.write('TITLE="Probes records versus time"\nVARIABLES = "time" "p1" "p2" \n')
        for i in range(len(t)):
            f.write(" ".join(f"{v:13.5E}" for v in (t[i], *eta[i])) + "\n")


def test_read_probes_and_summary(tmp_path):
    t = np.arange(0, 600, 0.25)
    eta = np.stack([2 * np.cos(2 * np.pi * t / 10), 1 * np.cos(2 * np.pi * t / 8)], axis=1)
    _write_probes(tmp_path / "probes.dat", t, eta)
    rec = read_probes(tmp_path / "probes.dat")
    assert rec.eta.shape == (len(t), 2) and np.allclose(rec.x, [1000, 2000])
    s = summary(rec, ["A", "B"])
    assert s[0]["hs"] == pytest.approx(4 * 2 / np.sqrt(2), rel=1e-3) and s[0]["hmax"] == pytest.approx(4, rel=1e-3)
    assert s[1]["kurtosis"] == pytest.approx(1.5, rel=1e-2)


def test_results_json(tmp_path):
    c = _case()
    c.write(tmp_path)
    (tmp_path / "Results").mkdir()
    t = np.arange(0, 2000, 0.5)
    _write_probes(tmp_path / "Results" / "probes.dat", t, np.stack([np.sin(t), np.cos(t)], 1))
    out = results_json(tmp_path)
    assert [p["name"] for p in out["probes"]] == ["A01", "A02"] and out["probes"][0]["t"][0] >= 20 * 12


def test_read_surface(tmp_path):
    p = tmp_path / "3d.dat"
    with open(p, "w") as f:
        f.write('TITLE=" 3D free surface elevation "\nVARIABLES="x","y","eta","phis"\n')
        f.write("ZONE SOLUTIONTIME =  0.00000E+00, I=    3, J=    2\n")
        for j in range(2):
            for i in range(3):
                f.write(f"{i * 10.0:12.5E} {j * 5.0:12.5E} {i + j:12.5E} {0.0:12.5E}\n")
        f.write("ZONE SOLUTIONTIME =  1.00000E+00, D=(1,2), I=    3, J=    2\n")
        for j in range(2):
            for i in range(3):
                f.write(f"{10 * (i + j):12.5E} {0.0:12.5E}\n")
    x, y, eta, t = read_surface(p)
    assert np.allclose(x, [0, 10, 20]) and np.allclose(y, [0, 5]) and t == 1.0
    assert eta.shape == (2, 3) and eta[1, 2] == 30
