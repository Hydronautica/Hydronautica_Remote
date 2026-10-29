"""Coupling to HOS-Ocean (Ecole Centrale de Nantes, LHEEA) for directional seas over real bathymetry.

HOS-Ocean is a high-order spectral solver for nonlinear, directionally spread
wave fields. Its current version takes a varying sea bed from a file
(``bathymetry.type: file``), generates the sea in a relaxation zone at the
inlet and absorbs it at the outlet. Unlike the 1-D model in
:mod:`nls.variable_depth`, it captures refraction and focusing by the
bathymetry (shoals acting as lenses), directional spreading and crossing seas.

HOS-Ocean is GPL software and stays a separate program: this module only
writes its input files and reads its output files, and can launch the
executable. Build it with ``third_party/hos-ocean/build.sh``, which applies our
fix for sea beds read from a file (see third_party/hos-ocean/README.md);
unpatched builds misplace the bathymetry.

Workflow::

    case = HOSCase.from_stormtwin("hos_case.json")          # exported from StormTwin's map
    case.write("run1")                                        # input_HOS-Ocean.yml, bathy.dat, prob.inp
    run("run1")                                               # or run HOS-Ocean there yourself
    res = read_probes("run1/Results/probes.dat")
    summary(res, case.probe_names)                            # Hs, Hmax, crest, kurtosis per turbine

or from the command line::

    python -m nls.hos_ocean prepare hos_case.json run1
    python -m nls.hos_ocean run run1
    python -m nls.hos_ocean results run1 -o hos_results.json   # import this file in StormTwin

Frames. The map is north-up with x east. HOS-Ocean sends the mean sea along
+x, so the case is built in a frame rotated to the mean wave direction: x'
along the direction the waves travel, y' to its left. The map is sampled on a
rectangle covering it in that frame (outside the map the nearest edge depth
is used) and HOS-Ocean ramps the depth to the total depth in its relaxation
zones.

Units. Following HOS-Ocean for an analytical spectrum, domain and zone sizes
are given in peak wavelengths (computed with the total depth), the duration
in peak periods and the output frequency per peak period; the bathymetry file
and probe positions are in metres from the start of the physical zone.
"""

import argparse
import json
import math
import os
import shutil
import subprocess
from dataclasses import dataclass, field

import numpy as np

from .kinematics import G, wavenumber

FILES = ("input_HOS-Ocean.yml", "bathy.dat", "prob.inp")


def _fft_size(n, minimum=16):
    """Smallest 2^a 3^b 5^c >= n (and even), for FFTW."""
    n = max(int(math.ceil(n)), minimum)
    while True:
        m = n
        for p in (2, 3, 5):
            while m % p == 0:
                m //= p
        if m == 1 and n % 2 == 0:
            return n
        n += 1


def _bilinear(xg, yg, Z, x, y):
    """Bilinear interpolation on a regular grid, clamped at the edges."""
    fx = np.clip((np.asarray(x) - xg[0]) / (xg[1] - xg[0]), 0, len(xg) - 1)
    fy = np.clip((np.asarray(y) - yg[0]) / (yg[1] - yg[0]), 0, len(yg) - 1)
    i0 = np.minimum(np.floor(fx).astype(int), len(xg) - 2)
    j0 = np.minimum(np.floor(fy).astype(int), len(yg) - 2)
    tx, ty = fx - i0, fy - j0
    return ((1 - tx) * (1 - ty) * Z[j0, i0] + tx * (1 - ty) * Z[j0, i0 + 1]
            + (1 - tx) * ty * Z[j0 + 1, i0] + tx * ty * Z[j0 + 1, i0 + 1])


@dataclass
class SeaState:
    Hs: float
    Tp: float
    gamma: float = 3.3
    direction_from: float = 270.0   # degrees, nautical: where the waves come from
    spread: float = 40.0            # half-width of the cos^2 spreading (degrees); 0 = long-crested (1 degree)
    duration: float = 100.0         # peak periods
    seed: int = 12345


@dataclass
class HOSCase:
    """A HOS-Ocean run over a depth grid.

    Parameters
    ----------
    x, y : 1-D arrays
        Map grid (m): x east, y north, increasing.
    depth : 2-D array, shape (len(y), len(x))
        Positive depth (m).
    sea : SeaState
    probes : list of (name, x, y) in map metres
    """

    x: np.ndarray
    y: np.ndarray
    depth: np.ndarray
    sea: SeaState
    probes: list = field(default_factory=list)
    order: int = 3                   # HOS nonlinearity order M
    points_per_wavelength: float = 8.0
    generation: float = 4.0          # wavelengths
    absorption_in: float = 2.0
    absorption_out: float = 4.0
    side_ramp: float = 1.0           # wavelengths over which HOS ramps the depth at the sides
    margin: float = 0.0              # extra metres of physical domain around the map
    bathy_spacing: float = None      # m; default: a quarter of the map cell or a peak wavelength / 8
    breaking: bool = True
    name: str = "StormTwin map"

    # ----- construction from StormTwin -----
    @classmethod
    def from_stormtwin(cls, path_or_dict, **overrides):
        """Build from StormTwin's "Export for HOS-Ocean" JSON.

        The JSON has the map grid (``W``, ``H`` in km, ``nx``, ``ny``, ``depth`` row
        by row from the north edge, cell centres), ``turbines`` [{name, x, y}] in
        map km (y southwards) and ``sea`` {Hs, Tp, gamma, from, spread, duration}.
        """
        d = path_or_dict
        if not isinstance(d, dict):
            with open(d) as f:
                d = json.load(f)
        nx, ny, W, H = int(d["nx"]), int(d["ny"]), float(d["W"]), float(d["H"])
        Z = np.asarray(d["depth"], dtype=float).reshape(ny, nx)[::-1]     # flip to south -> north
        x = (np.arange(nx) + 0.5) * W * 1000 / nx
        y = (np.arange(ny) + 0.5) * H * 1000 / ny
        s = d.get("sea", {})
        sea = SeaState(Hs=float(s.get("Hs", 8.0)), Tp=float(s.get("Tp", 12.0)), gamma=float(s.get("gamma", 3.3)),
                       direction_from=float(s.get("from", 270.0)), spread=float(s.get("spread", 40.0)),
                       duration=float(s.get("duration", 100.0)), seed=int(s.get("seed", 12345)))
        probes = [(t["name"], float(t["x"]) * 1000, (H - float(t["y"])) * 1000) for t in d.get("turbines", [])]
        kw = dict(name=d.get("name", "StormTwin map"))
        kw.update(overrides)
        return cls(x, y, Z, sea, probes, **kw)

    # ----- derived quantities -----
    @property
    def depth_total(self):
        return float(np.max(self.depth))

    @property
    def depth_min(self):
        return float(np.min(self.depth))

    @property
    def kp(self):
        return float(wavenumber(2 * np.pi / self.sea.Tp, self.depth_total))

    @property
    def wavelength(self):
        return 2 * np.pi / self.kp

    def frame(self):
        """Unit vectors (east, north) of x' (travel direction) and y' (its left)."""
        b = math.radians(self.sea.direction_from + 180.0)
        ex = np.array([math.sin(b), math.cos(b)])
        ey = np.array([-ex[1], ex[0]])
        return ex, ey

    def extent(self):
        """Physical domain in the rotated frame: origin (map metres), lengths Lx, Ly."""
        ex, ey = self.frame()
        corners = np.array([[self.x[0], self.y[0]], [self.x[-1], self.y[0]], [self.x[0], self.y[-1]], [self.x[-1], self.y[-1]]])
        u, v = corners @ ex, corners @ ey
        # HOS-Ocean sizes its domain in whole peak wavelengths: pad the map evenly up to them
        lam = self.wavelength
        lx, ly = u.max() - u.min() + 2 * self.margin, v.max() - v.min() + 2 * self.margin
        Lx, Ly = math.ceil(lx / lam - 1e-9) * lam, math.ceil(ly / lam - 1e-9) * lam
        u0, v0 = u.min() - self.margin - (Lx - lx) / 2, v.min() - self.margin - (Ly - ly) / 2
        return u0 * ex + v0 * ey, Lx, Ly

    def to_frame(self, px, py):
        """Map metres -> metres from the start of the HOS physical zone."""
        ex, ey = self.frame()
        o, _, _ = self.extent()
        p = np.stack([np.asarray(px, float) - o[0], np.asarray(py, float) - o[1]], axis=-1)
        return p @ ex, p @ ey

    def from_frame(self, u, v):
        ex, ey = self.frame()
        o, _, _ = self.extent()
        u, v = np.asarray(u, float), np.asarray(v, float)
        return o[0] + u * ex[0] + v * ey[0], o[1] + u * ex[1] + v * ey[1]

    def bathy_grid(self):
        """Depth sampled on the rotated rectangle: (u, v, depth[v, u])."""
        _, Lx, Ly = self.extent()
        cell = min(self.x[1] - self.x[0], self.y[1] - self.y[0])
        ds = self.bathy_spacing or max(5.0, min(cell / 2, self.wavelength / 8))
        nu, nv = int(math.ceil(Lx / ds)) + 1, int(math.ceil(Ly / ds)) + 1
        u, v = np.linspace(0, Lx, nu), np.linspace(0, Ly, nv)
        U, V = np.meshgrid(u, v)
        X, Y = self.from_frame(U, V)
        return u, v, _bilinear(self.x, self.y, self.depth, X, Y)

    def resolution(self):
        """(n1, n2) for HOS: points per peak wavelength over the whole domain."""
        _, Lx, Ly = self.extent()
        lam = self.wavelength
        nlx = math.ceil(self.absorption_in + self.generation + self.absorption_out + round(Lx / lam))
        nly = math.ceil(2 * self.side_ramp + round(Ly / lam))
        return _fft_size(nlx * self.points_per_wavelength), _fft_size(nly * self.points_per_wavelength)

    @property
    def probe_names(self):
        return [p[0] for p in self.probes]

    # ----- files -----
    def yaml(self):
        _, Lx, Ly = self.extent()
        lam, s = self.wavelength, self.sea
        n1, n2 = self.resolution()
        # invbeta_cos2: D(theta) ~ cos^2(pi theta / (2 beta)) for |theta| < beta; ~1 degree stands in for long-crested
        spread = (f"  spreading: invbeta_cos2\n  beta: {math.radians(max(s.spread, 1.0)):.6f} # radians\n")
        # Chalikov's curvature-based dissipation (the Barthelemy-Tian onset criterion is 2-D only in HOS-Ocean)
        brk = ("wave breaking:\n  Chalikov:\n    diffusion strength: 0.001\n    curvature threshold: 2.0\n\n" if self.breaking else "")
        return (
            f"# HOS-Ocean input written by nls.hos_ocean for {self.name}\n"
            f"# Frame: x along the mean wave travel direction ({(s.direction_from + 180) % 360:.1f} deg true), y to its left.\n"
            f"# Physical zone {Lx:.0f} m x {Ly:.0f} m; peak wavelength {lam:.1f} m at the total depth {self.depth_total:.2f} m.\n"
            "restart: disable\n\n"
            f"gravity: {G}\n\n"
            "domain size:\n"
            f"  x: {round(Lx / lam)} # peak wavelengths\n"
            f"  y: {round(Ly / lam)} # peak wavelengths\n\n"
            "relaxation zones:\n"
            "  generation:\n"
            f"    length: {self.generation}\n"
            "    initialisation: false\n"
            "  absorption:\n"
            "    type: standard\n"
            "    length:\n"
            f"      inlet: {self.absorption_in}\n"
            f"      outlet: {self.absorption_out}\n\n"
            + brk +
            "computation case:\n"
            "  3D: true\n"
            f"  duration: {s.duration} # peak periods\n"
            "  type: analytical spectrum\n"
            f"  Tp: {s.Tp}\n"
            f"  Hs: {s.Hs}\n"
            f"  gamma: {s.gamma}\n"
            "  theta0: 0.0 # along x\n"
            + spread +
            "  random phases: seed\n"
            f"  randomisation seed: {s.seed}\n\n"
            "numerical parameters:\n"
            "  time:\n"
            "    integration tolerance: 1.e-7\n"
            "    Dommermuth initialisation:\n"
            "      n: 4\n"
            "      Ta: 10.0\n"
            "  discretization:\n"
            f"    x: {n1}\n"
            f"    y: {n2}\n"
            f"  surface nonlinearity order: {self.order}\n"
            "  varying bathymetry:\n"
            "    methodology: base\n"
            "  dealiasing:\n"
            "    x: 3\n"
            "    y: 3\n\n"
            "bathymetry:\n"
            f"  depth: {self.depth_total:.3f}\n"
            "  type: file\n"
            f"  total depth: {self.depth_total:.3f}\n"
            "  file: bathy.dat\n"
            f"  minimum depth: {max(0.5, self.depth_min):.3f}\n"
            "  relaxation zones:\n"
            f"    right: {self.side_ramp}\n"
            f"    left: {self.side_ramp}\n\n"
            "output:\n"
            "  frequency: 8.0 # per peak period\n"
            "  free surface:\n"
            "    physical space: true\n"
            "    modal space: false\n"
            "  probes:\n"
            f"    activate: {'true' if self.probes else 'false'}\n"
            f"  max number: {max(1, len(self.probes))}\n"
        )

    def bathy_text(self):
        """bathy.dat in the fixed format HOS-Ocean reads: 2 header lines, (2(I5,1X),I5), 1 header line,
        then x, y, depth as (2(ES10.4,1X),ES10.4) with x running fastest."""
        u, v, Z = self.bathy_grid()
        if len(u) > 99999 or len(v) > 99999:
            raise ValueError("bathymetry grid too large for the I5 header")
        lines = [f"# Bathymetry for HOS-Ocean from {self.name}", "# x (m), y (m) from the start of the physical zone; depth (m, positive)",
                 f"{len(u):5d} {len(v):5d}", "# x y depth"]
        for j in range(len(v)):
            for i in range(len(u)):
                lines.append(f"{u[i]:10.4E} {v[j]:10.4E} {Z[j, i]:10.4E}")
        return "\n".join(lines) + "\n"

    def probes_text(self):
        u, v = self.to_frame([p[1] for p in self.probes], [p[2] for p in self.probes]) if self.probes else ([], [])
        return "".join(f"{a:.3f} {b:.3f}\n" for a, b in zip(np.atleast_1d(u), np.atleast_1d(v)))

    def write(self, folder):
        """Write the HOS-Ocean case files and a probes.json that maps probe numbers to turbines."""
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "input_HOS-Ocean.yml"), "w") as f:
            f.write(self.yaml())
        with open(os.path.join(folder, "bathy.dat"), "w") as f:
            f.write(self.bathy_text())
        with open(os.path.join(folder, "prob.inp"), "w") as f:
            f.write(self.probes_text())
        u, v = self.to_frame([p[1] for p in self.probes], [p[2] for p in self.probes]) if self.probes else ([], [])
        o, Lx, Ly = self.extent()
        ex, _ = self.frame()
        meta = {"name": self.name, "probes": [{"name": p[0], "x_map_m": p[1], "y_map_m": p[2], "x_hos_m": float(a), "y_hos_m": float(b)}
                                              for p, a, b in zip(self.probes, np.atleast_1d(u), np.atleast_1d(v))],
                "origin_map_m": [float(o[0]), float(o[1])], "x_axis_east_north": [float(ex[0]), float(ex[1])],
                "Lx_m": Lx, "Ly_m": Ly, "wavelength_m": self.wavelength, "resolution": self.resolution(),
                "sea": self.sea.__dict__}
        with open(os.path.join(folder, "probes.json"), "w") as f:
            json.dump(meta, f, indent=1)
        return folder


# ----- running and reading -----
LOCAL_BUILD = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "hos-ocean", "src", "build", "sources", "HOS-Ocean")


def find_executable(name=None):
    """The given path, $HOS_OCEAN, our patched build (third_party/hos-ocean/build.sh), or HOS-Ocean on PATH."""
    for cand in [name, os.environ.get("HOS_OCEAN"), LOCAL_BUILD, "HOS-Ocean", "HOS-ocean"]:
        if cand and (shutil.which(cand) or os.path.isfile(cand)):
            return shutil.which(cand) or cand
    return None


def run(folder, exe=None, timeout=None):
    """Run HOS-Ocean in ``folder`` (needs the executable on PATH, in $HOS_OCEAN, or given)."""
    path = find_executable(exe)
    if not path:
        raise FileNotFoundError("HOS-Ocean executable not found; build it with third_party/hos-ocean/build.sh "
                                "or set HOS_OCEAN to its path")
    return subprocess.run([path, "input_HOS-Ocean.yml"], cwd=folder, check=True, timeout=timeout)


@dataclass
class ProbeRecord:
    t: np.ndarray
    eta: np.ndarray          # shape (len(t), nprobes)
    x: np.ndarray            # HOS frame, m (including relaxation zones, as HOS writes them)
    y: np.ndarray


def _floats(line):
    try:
        return [float(v) for v in line.split()]
    except ValueError:
        return None


def read_probes(path):
    """Read Results/probes.dat (Tecplot-style header, then time and one column per probe)."""
    xs, ys, rows = [], [], []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s.startswith("#xprobe="):
                xs = [float(v) for v in s.split("=", 1)[1].split()]
            elif s.startswith("#yprobe="):
                ys = [float(v) for v in s.split("=", 1)[1].split()]
            elif s and s[0] not in "#TVZ":
                r = _floats(s)
                if r:
                    rows.append(r)
    if not rows:
        raise ValueError(f"no probe records in {path}")
    n = min(len(r) for r in rows)
    a = np.array([r[:n] for r in rows])
    return ProbeRecord(a[:, 0], a[:, 1:], np.array(xs), np.array(ys))


def read_surface(path, frame=-1):
    """Read one zone of Results/3d.dat: returns x (n1), y (n2), eta (n2, n1) at that output time and the time."""
    zones, cur, t = [], None, None
    xy = None
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s.startswith("ZONE"):
                if cur is not None:
                    zones.append((t, cur))
                tok = s.replace(",", " ").replace("=", " = ").split()
                t = float(tok[tok.index("SOLUTIONTIME") + 2]) if "SOLUTIONTIME" in tok else float(s.split('"')[1])
                ni = int(tok[tok.index("I") + 2]); nj = int(tok[tok.index("J") + 2])
                cur = []
                continue
            if cur is None or not s or s[0] in "#TV":
                continue
            r = _floats(s)
            if r:
                cur.append(r)
        if cur is not None:
            zones.append((t, cur))
    t0, z0 = zones[0]
    a0 = np.array(z0)
    nx = len(np.unique(a0[:, 0]))
    ny = a0.shape[0] // nx
    x, y = a0[:nx, 0], a0[::nx, 1]
    t, z = zones[frame]
    a = np.array(z)
    col = 2 if a.shape[1] >= 4 else 0
    return x, y, a[:, col].reshape(ny, nx), t


def read_bathymetry_used(path):
    """Results/beta.dat: the sea bed HOS-Ocean actually used. Returns x (n1), y (n2), depth (n2, n1), all in metres."""
    with open(path) as f:
        lines = f.read().splitlines()
    zone = next(l for l in lines if l.startswith("ZONE"))
    tok = zone.replace(",", " ").replace("=", " = ").split()
    ni, nj = int(tok[tok.index("I") + 2]), int(tok[tok.index("J") + 2])
    rows = [r for r in (_floats(l) for l in lines if l.strip() and l.strip()[0] not in "#TVZ") if r and len(r) >= 3][:ni * nj]
    a = np.array([r[:3] for r in rows])
    return a[:ni, 0], a[::ni, 1][:nj], -a[:, 2].reshape(nj, ni)


def check_bathymetry(case, folder):
    """Compare HOS-Ocean's sea bed with the case's over the physical zone (away from the ramps).

    Returns the RMS and largest depth difference in metres; a misplaced bed gives errors of the order of the
    depth range."""
    x, y, D = read_bathymetry_used(os.path.join(folder, "Results", "beta.dat"))
    with open(os.path.join(folder, "probes.json")) as f:
        meta = json.load(f)
    # HOS's x includes the inlet absorption and generation zones, y the right-hand ramp
    lam = case.wavelength
    x0, y0 = (case.absorption_in + case.generation) * lam, case.side_ramp * lam
    _, Lx, Ly = case.extent()
    u, v = x - x0, y - y0
    iu = (u > 0.05 * Lx) & (u < 0.95 * Lx)
    jv = (v > 0.05 * Ly) & (v < 0.95 * Ly)
    U, V = np.meshgrid(u[iu], v[jv])
    X, Y = case.from_frame(U, V)
    want = np.clip(_bilinear(case.x, case.y, case.depth, X, Y), max(0.5, case.depth_min), case.depth_total)
    diff = D[np.ix_(jv, iu)] - want
    return {"rms": float(np.sqrt(np.mean(diff**2))), "max": float(np.abs(diff).max()), "depth_range": float(np.ptp(want))}


def wave_stats(t, eta):
    """Hs (4 sigma), largest zero-up-crossing height, highest crest, kurtosis."""
    e = np.asarray(eta, float) - np.mean(eta)
    up = np.where((e[:-1] < 0) & (e[1:] >= 0))[0]
    H = [e[a:b + 1].max() - e[a:b + 1].min() for a, b in zip(up[:-1], up[1:])]
    return {"hs": float(4 * e.std()), "hmax": float(max(H) if H else 0.0), "crest": float(e.max()),
            "kurtosis": float(np.mean(e**4) / np.mean(e**2) ** 2) if e.std() > 0 else 3.0}


def summary(rec, names=None, skip=0.0):
    """Statistics per probe, ignoring the first ``skip`` seconds (ramp-up)."""
    keep = rec.t >= skip
    out = []
    for i in range(rec.eta.shape[1]):
        st = wave_stats(rec.t[keep], rec.eta[keep, i])
        st["name"] = names[i] if names and i < len(names) else f"p{i + 1}"
        out.append(st)
    return out


def results_json(folder, skip_periods=20.0, max_points=4000):
    """StormTwin import file: per-turbine statistics and the surface time series."""
    with open(os.path.join(folder, "probes.json")) as f:
        meta = json.load(f)
    rec = read_probes(os.path.join(folder, "Results", "probes.dat"))
    names = [p["name"] for p in meta["probes"]]
    skip = skip_periods * meta["sea"]["Tp"]
    stats = summary(rec, names, skip)
    keep = rec.t >= skip
    step = max(1, int(keep.sum() // max_points))
    for i, st in enumerate(stats):
        st["t"] = [round(float(v), 3) for v in rec.t[keep][::step]]
        st["eta"] = [round(float(v), 4) for v in rec.eta[keep, i][::step]]
    return {"source": "HOS-Ocean", "name": meta["name"], "sea": meta["sea"], "skip_s": skip, "probes": stats}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m nls.hos_ocean", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="write a HOS-Ocean case from a StormTwin export")
    p.add_argument("export"); p.add_argument("folder")
    for k, t in (("hs", float), ("tp", float), ("gamma", float), ("direction", float), ("spread", float), ("duration", float),
                 ("order", int), ("ppw", float)):
        p.add_argument("--" + k, type=t)
    r = sub.add_parser("run", help="run HOS-Ocean in a case folder")
    r.add_argument("folder"); r.add_argument("--exe")
    q = sub.add_parser("results", help="turn HOS-Ocean probe output into a StormTwin import file")
    q.add_argument("folder"); q.add_argument("-o", "--output", default="hos_results.json"); q.add_argument("--skip", type=float, default=20.0)
    a = ap.parse_args(argv)
    if a.cmd == "prepare":
        kw = {}
        if a.order: kw["order"] = a.order
        if a.ppw: kw["points_per_wavelength"] = a.ppw
        c = HOSCase.from_stormtwin(a.export, **kw)
        for k, attr in (("hs", "Hs"), ("tp", "Tp"), ("gamma", "gamma"), ("direction", "direction_from"), ("spread", "spread"), ("duration", "duration")):
            if getattr(a, k) is not None:
                setattr(c.sea, attr, getattr(a, k))
        c.write(a.folder)
        n1, n2 = c.resolution()
        _, Lx, Ly = c.extent()
        print(f"Wrote {a.folder}: {Lx / 1000:.1f} x {Ly / 1000:.1f} km, peak wavelength {c.wavelength:.0f} m, grid {n1} x {n2}, "
              f"{len(c.probes)} probes. Run HOS-Ocean there, then: python -m nls.hos_ocean results {a.folder}")
    elif a.cmd == "run":
        run(a.folder, a.exe)
    else:
        out = results_json(a.folder, a.skip)
        with open(a.output, "w") as f:
            json.dump(out, f)
        for st in out["probes"]:
            print(f"{st['name']:>8}  Hs {st['hs']:.2f} m  Hmax {st['hmax']:.2f} m  crest {st['crest']:.2f} m  kurtosis {st['kurtosis']:.2f}")
        print(f"Wrote {a.output}")


if __name__ == "__main__":
    main()
