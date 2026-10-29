"""Latitude/longitude boxes and the local kilometre grid StormTwin maps use (x east, y south from the north-west corner)."""

import math
from dataclasses import dataclass

import numpy as np

R_EARTH = 6371.0088  # km


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(a))


@dataclass
class LocalGrid:
    """A lat/lon box mapped to a local equirectangular grid of nx by ny cells.

    The box is south, west, north, east in degrees. x runs east and y south in km from the north-west corner,
    matching StormTwin maps (W by H km, cell centres, rows from the north edge).
    """

    south: float
    west: float
    north: float
    east: float
    cell_km: float = 0.25

    def __post_init__(self):
        if not (-90 <= self.south < self.north <= 90):
            raise ValueError("need south < north within +/-90 degrees")
        if self.east <= self.west:
            raise ValueError("need west < east (boxes across the antimeridian are not supported)")
        self.lat0 = 0.5 * (self.south + self.north)
        self.kx = R_EARTH * math.radians(1) * math.cos(math.radians(self.lat0))   # km per degree of longitude
        self.ky = R_EARTH * math.radians(1)                                          # km per degree of latitude
        self.W = (self.east - self.west) * self.kx
        self.H = (self.north - self.south) * self.ky
        self.nx = max(2, int(round(self.W / self.cell_km)))
        self.ny = max(2, int(round(self.H / self.cell_km)))

    @classmethod
    def around(cls, lat, lon, width_km, height_km, cell_km=0.25):
        ky = R_EARTH * math.radians(1)
        kx = ky * math.cos(math.radians(lat))
        dlat, dlon = height_km / 2 / ky, width_km / 2 / kx
        return cls(lat - dlat, lon - dlon, lat + dlat, lon + dlon, cell_km)

    def centres(self):
        """Latitudes (ny, north first) and longitudes (nx) of the cell centres."""
        lon = self.west + (np.arange(self.nx) + 0.5) / self.nx * (self.east - self.west)
        lat = self.north - (np.arange(self.ny) + 0.5) / self.ny * (self.north - self.south)
        return lat, lon

    def to_km(self, lat, lon):
        return (np.asarray(lon) - self.west) * self.kx, (self.north - np.asarray(lat)) * self.ky

    def to_latlon(self, x, y):
        return self.north - np.asarray(y) / self.ky, self.west + np.asarray(x) / self.kx

    def as_dict(self):
        return {"south": self.south, "west": self.west, "north": self.north, "east": self.east, "lat0": self.lat0,
                "W_km": self.W, "H_km": self.H, "nx": self.nx, "ny": self.ny}
