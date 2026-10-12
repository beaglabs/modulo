"""Geospatial input, tile addressing, GeoJSON and spatial analysis.

Geometry operations require the optional geo extra (Shapely / PyProj).
No implicit network access or map service is performed.
"""
from dataclasses import dataclass, field
from math import atan, cos, degrees, floor, log, pi, radians, sinh, tan
from pathlib import Path
from urllib.parse import urlparse
import json
import sqlite3

@dataclass(frozen=True)
class Tile:
    z: int
    x: int
    y: int

    def __post_init__(self):
        if not isinstance(self.z, int) or not 0 <= self.z <= 30:
            raise ValueError("zoom must be integer in [0, 30]")
        n = 1 << self.z
        if not all(isinstance(v, int) and 0 <= v < n for v in (self.x, self.y)):
            raise ValueError("tile x/y outside zoom level")

    @property
    def quadkey(self):
        return "".join(str(((self.x >> i) & 1) + 2 * ((self.y >> i) & 1))
                       for i in range(self.z - 1, -1, -1))

    def bounds(self):
        n = 1 << self.z
        def lon(x): return x / n * 360 - 180
        def lat(y): return degrees(atan(sinh(pi * (1 - 2 * y / n))))
        return (lon(self.x), lat(self.y + 1), lon(self.x + 1), lat(self.y))

def lonlat_to_tile(lon, lat, zoom):
    if not -180 <= lon <= 180 or not -85.05112878 <= lat <= 85.05112878:
        raise ValueError("coordinate outside Web Mercator tile limits")
    if not isinstance(zoom, int) or not 0 <= zoom <= 30:
        raise ValueError("invalid zoom")
    n = 1 << zoom
    x = min(n - 1, max(0, floor((lon + 180) / 360 * n)))
    y = min(n - 1, max(0, floor((1 - log(tan(radians(lat)) + 1 / cos(radians(lat))) / pi) / 2 * n)))
    return Tile(zoom, x, y)

def xyz_url(template, tile):
    """Resolve a tile URL; a trusted caller must control source URLs and credentials."""
    if not isinstance(tile, Tile): raise TypeError("expected Tile")
    if not all(k in template for k in ("{z}", "{x}", "{y}")):
        raise ValueError("XYZ template requires {z}, {x}, {y}")
    p = urlparse(template)
    if p.scheme != "https" or not p.netloc or p.username or p.password:
        raise ValueError("HTTPS tile endpoint required")
    return template.replace("{z}", str(tile.z)).replace("{x}", str(tile.x)).replace("{y}", str(tile.y))

def read_mbtiles(path, tile):
    """Return bytes from a local MBTiles SQLite database (TMS rows)."""
    if not isinstance(tile, Tile): raise TypeError("expected Tile")
    path = Path(path)
    if not path.is_file(): raise FileNotFoundError(path)
    uri = "file:" + str(path.resolve()) + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as db:
        row = db.execute(
            "SELECT tile_data FROM tiles WHERE zoom_level=? AND tile_column=? AND tile_row=?",
            (tile.z, tile.x, (1 << tile.z) - 1 - tile.y)).fetchone()
    return row[0] if row else None

@dataclass
class GeoObservation:
    geometry: dict
    crs: str = "EPSG:4326"
    properties: dict = field(default_factory=dict)
    timestamp: str | None = None
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.crs: raise ValueError("explicit CRS required")
        geometry_shape(self.geometry)

    def to_feature(self):
        return {"type": "Feature", "geometry": self.geometry, "properties": dict(self.properties)}

def geometry_shape(value):
    """Parse GeoJSON geometry / Feature using Shapely and reject empty/invalid shapes."""
    from shapely.geometry import shape
    if value.get("type") == "Feature": value = value["geometry"]
    g = shape(value)
    if g.is_empty or not g.is_valid: raise ValueError("empty or invalid geometry")
    return g

def load_geojson(value, crs="EPSG:4326"):
    """Load geometry or a collection; coordinates retain the explicitly supplied CRS."""
    if isinstance(value, (str, Path)):
        value = json.loads(Path(value).read_text(encoding="utf-8"))
    if value.get("type") == "FeatureCollection":
        return [GeoObservation(f["geometry"], crs, f.get("properties") or {}) for f in value["features"]]
    if value.get("type") == "Feature":
        return [GeoObservation(value["geometry"], crs, value.get("properties") or {})]
    return [GeoObservation(value, crs)]

def reproject(observation, destination):
    from pyproj import Transformer
    from shapely.ops import transform
    from shapely.geometry import mapping
    convert = Transformer.from_crs(observation.crs, destination, always_xy=True).transform
    geometry = transform(convert, geometry_shape(observation.geometry))
    return GeoObservation(mapping(geometry), destination, dict(observation.properties),
                          observation.timestamp, dict(observation.provenance))

def _pair(a, b):
    if a.crs != b.crs: raise ValueError("CRS mismatch; reproject explicitly")
    return geometry_shape(a.geometry), geometry_shape(b.geometry)

def intersects(a, b):
    x, y = _pair(a, b)
    return x.intersects(y)

def contains(a, b):
    x, y = _pair(a, b)
    return x.contains(y)

def buffer(observation, distance):
    """Buffer in units of the supplied CRS. Reproject to a suitable metric CRS first."""
    from shapely.geometry import mapping
    if distance < 0: raise ValueError("distance must be nonnegative")
    if __import__("pyproj").CRS.from_user_input(observation.crs).is_geographic:
        raise ValueError("buffer requires a projected CRS (distance in CRS units)")
    return GeoObservation(mapping(geometry_shape(observation.geometry).buffer(distance)),
                          observation.crs, dict(observation.properties),
                          observation.timestamp, dict(observation.provenance))

def intersection(a, b):
    from shapely.geometry import mapping
    x, y = _pair(a, b)
    result = x.intersection(y)
    if result.is_empty: return None
    return GeoObservation(mapping(result), a.crs)

def spatial_join(left, right, predicate="intersects"):
    """Return (left_index, right_index) pairs; caller controls data volume."""
    if predicate not in ("intersects", "contains", "within"):
        raise ValueError("unsupported predicate")
    result = []
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            x, y = _pair(a, b)
            if getattr(x, predicate)(y): result.append((i, j))
    return result
