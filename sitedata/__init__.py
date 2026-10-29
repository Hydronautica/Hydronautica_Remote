"""Site data from online databases: pick an area, get its bathymetry, wave and wind climate and tides.

* Bathymetry: NOAA NCEI "DEM_all" mosaic (best available elevation model at each place, global; high-resolution
  coastal models around the US), ETOPO 2022 15 arc-second (global, ~450 m) or EMODnet (European waters, ~115 m).
* Wave and wind climate: USACE Wave Information Studies (WIS) hindcast, hourly since 1980 at ~4000 stations
  along the US coasts and Great Lakes (nearest station to the area).
* Tides: NOAA CO-OPS tidal datums and harmonic constituents at the nearest station (US).

``python -m sitedata fetch --bbox S W N E --name "My site" -o site.json`` writes a file StormTwin's map builder
imports as a new map with all of this filled in.
"""

from .geo import LocalGrid
from .site import build_site

__all__ = ["LocalGrid", "build_site"]
