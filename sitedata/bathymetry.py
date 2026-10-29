"""Bathymetry for a lat/lon box from public services, resampled onto a LocalGrid.

Sources (elevation is positive up in all of them; we return depth positive down):

* ``ncei``    NOAA NCEI DEM_all ImageServer: the best available model at each place (CUDEM / coastal DEMs around the
              US, ETOPO / GEBCO elsewhere), exported directly on our grid.
* ``etopo``   ETOPO 2022 15 arc-second (~450 m) from NOAA's ERDDAP, global.
* ``emodnet`` EMODnet Bathymetry mean depth (1/16 arc-minute, ~115 m), European seas.
* ``auto``    EMODnet inside its coverage, otherwise NCEI, falling back to ETOPO.
"""

import io
import urllib.parse

import numpy as np

from .net import get_bytes

NCEI = "https://gis.ngdc.noaa.gov/arcgis/rest/services/DEM_mosaics/DEM_all/ImageServer/exportImage"
ERDDAP = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/ETOPO_2022_v1_15s.csv"
EMODNET = "https://ows.emodnet-bathymetry.eu/wcs"
EMODNET_BOX = (15.0, -36.0, 90.0, 43.0)        # south, west, north, east (approximate coverage)

SOURCES = {
    "ncei": "NOAA NCEI DEM global mosaic (best available elevation model)",
    "etopo": "ETOPO 2022 15 arc-second (NOAA NCEI)",
    "emodnet": "EMODnet Bathymetry Digital Terrain Model (mean depth)",
}


def _tiff(data):
    from PIL import Image
    return np.array(Image.open(io.BytesIO(data)), dtype=float)


def _resample(lat_src, lon_src, Z, lat, lon):
    """Bilinear from a regular lat/lon grid (any orientation) to the grid's cell centres."""
    if lat_src[0] > lat_src[-1]:
        lat_src, Z = lat_src[::-1], Z[::-1]
    if lon_src[0] > lon_src[-1]:
        lon_src, Z = lon_src[::-1], Z[:, ::-1]
    fi = np.interp(lat, lat_src, np.arange(len(lat_src)))
    fj = np.interp(lon, lon_src, np.arange(len(lon_src)))
    FI, FJ = np.meshgrid(fi, fj, indexing="ij")
    i0 = np.clip(np.floor(FI).astype(int), 0, len(lat_src) - 2)
    j0 = np.clip(np.floor(FJ).astype(int), 0, len(lon_src) - 2)
    ti, tj = FI - i0, FJ - j0
    return ((1 - ti) * (1 - tj) * Z[i0, j0] + (1 - ti) * tj * Z[i0, j0 + 1]
            + ti * (1 - tj) * Z[i0 + 1, j0] + ti * tj * Z[i0 + 1, j0 + 1])


def elevation_ncei(grid):
    q = {"bbox": f"{grid.west},{grid.south},{grid.east},{grid.north}", "bboxSR": 4326, "imageSR": 4326,
         "size": f"{grid.nx},{grid.ny}", "format": "tiff", "pixelType": "F32",
         "interpolation": "RSP_BilinearInterpolation", "f": "image"}
    Z = _tiff(get_bytes(NCEI + "?" + urllib.parse.urlencode(q)))
    if Z.shape != (grid.ny, grid.nx):
        raise ValueError(f"NCEI returned {Z.shape}, expected {(grid.ny, grid.nx)}")
    return Z


def elevation_etopo(grid):
    pad = 1 / 120
    q = f"z%5B({grid.south - pad}):1:({grid.north + pad})%5D%5B({grid.west - pad}):1:({grid.east + pad})%5D"
    txt = get_bytes(ERDDAP + "?" + q).decode()
    rows = [r.split(",") for r in txt.strip().splitlines()[2:]]
    a = np.array([[float(v) for v in r] for r in rows])
    lats, lons = np.unique(a[:, 0]), np.unique(a[:, 1])
    Z = np.full((len(lats), len(lons)), np.nan)
    Z[np.searchsorted(lats, a[:, 0]), np.searchsorted(lons, a[:, 1])] = a[:, 2]
    lat, lon = grid.centres()
    return _resample(lats, lons, Z, lat, lon)


def elevation_emodnet(grid):
    pad = 1 / 32
    q = [("service", "WCS"), ("version", "2.0.1"), ("request", "GetCoverage"), ("coverageId", "emodnet__mean"),
         ("subset", f"Lat({grid.south - pad},{grid.north + pad})"), ("subset", f"Long({grid.west - pad},{grid.east + pad})"),
         ("format", "image/tiff")]
    data = get_bytes(EMODNET + "?" + urllib.parse.urlencode(q))
    if not data.startswith((b"II", b"MM")):
        raise ValueError("EMODnet returned no coverage here: " + data[:200].decode(errors="replace"))
    D = _tiff(data)
    D[(D < -15000) | (D > 15000)] = np.nan                      # no-data
    ny, nx = D.shape
    s, n, w, e = grid.south - pad, grid.north + pad, grid.west - pad, grid.east + pad
    lats = n - (np.arange(ny) + 0.5) / ny * (n - s)
    lons = w + (np.arange(nx) + 0.5) / nx * (e - w)
    lat, lon = grid.centres()
    # EMODnet's mean layer is elevation (negative below sea level) like the others
    return _resample(lats, lons, D, lat, lon)


def in_emodnet(grid):
    s, w, n, e = EMODNET_BOX
    return grid.south >= s and grid.north <= n and grid.west >= w and grid.east <= e


def fetch(grid, source="auto", log=print):
    """Depth (ny, nx, positive down; land 0), land mask, and a note of the source used."""
    order = {"auto": (["emodnet"] if in_emodnet(grid) else []) + ["ncei", "etopo"]}.get(source, [source])
    errors = []
    for src in order:
        try:
            Z = {"ncei": elevation_ncei, "etopo": elevation_etopo, "emodnet": elevation_emodnet}[src](grid)
            if np.isnan(Z).mean() > 0.5:
                raise ValueError("mostly no data in this area")
            if np.isnan(Z).any():
                Z = np.where(np.isnan(Z), np.nanmedian(Z), Z)
            land = Z >= 0
            log(f"bathymetry: {SOURCES[src]}, {grid.nx} x {grid.ny} cells, depth {np.min(-Z[~land]) if (~land).any() else 0:.1f}"
                f"-{np.max(-Z):.1f} m, land {100 * land.mean():.0f}%")
            return np.where(land, 0.0, -Z), land, {"source": src, "description": SOURCES[src]}
        except Exception as ex:  # try the next source
            errors.append(f"{src}: {ex}")
            log(f"bathymetry: {src} failed ({ex})")
    raise RuntimeError("no bathymetry source worked: " + "; ".join(errors))
