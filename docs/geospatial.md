# Geospatial & multimodal baseline

Install optional GIS dependencies: `pip install -e '.[geo]'`.
The core NumPy-only SDK remains functional without GIS libraries.

## Tiles

```python
from modulo.geo import Tile, lonlat_to_tile, xyz_url, read_mbtiles
tile = lonlat_to_tile(-83.92, 35.96, 12)
print(tile.bounds(), tile.quadkey)
data = read_mbtiles("local.mbtiles", tile)  # MBTiles stores TMS tile rows
url = xyz_url("https://tiles.example.org/{z}/{x}/{y}.png", tile)
```

`xyz_url` *constructs* an HTTPS address but never fetches it. Caller must validate the
source's identity, license, allowed domains and download restrictions; do not pass
untrusted URLs to a privileged network fetcher. `read_mbtiles` reads local
MBTiles raster **or** vector tile bytes; decoding and WMTS/PMTiles/ArcGIS/COG
connectors are not included in this initial slice.

## Geometry and spatial analysis

```python
from modulo.geo import load_geojson, reproject, buffer, intersects, spatial_join
parcels = load_geojson("parcels.geojson")
meters = [reproject(p, "EPSG:3857") for p in parcels]
buffered = buffer(meters[0], 100)  # projected CRS units; use a suitable local CRS for accuracy
matches = spatial_join(meters, [buffered])
```

GeoJSON defaults to EPSG:4326; use explicit CRS when input is different.
Shapely validates geometries. No implicit CRS conversion. Spatial join is a
simple O(N*M) baseline suitable for small collections, not a spatial database.
EPSG:3857 is just a demo; local UTM/equal-area CRS may be preferable for distance/area.

## Multimodal fusion

```python
from modulo.fusion import MultimodalFusion
from modulo.domains import GeospatialEncoder, LogsEncoder
fusion = MultimodalFusion(
    {"geospatial": GeospatialEncoder(), "logs": LogsEncoder()},
    {"geospatial": 2, "logs": 5}, normalize=True)
features = fusion.encode({"geospatial": {"lat": 35.9, "lon": -83.9}})
print(features.vector, features.masks)
```

This deterministic early fusion is compatible with `SystemOne` as an encoder.
It is **not** learned cross-attention or a trained multimodal classifier.
The optional PyTorch `FusionBackbone` is the trainable model-building route.
Modality masks distinguish missing inputs from actual zero vectors.

## Dataset augmentation

USGS library spectra are measured physical reference data. Do not replace
them with generated rows or assume a synthetic spectrum is a valid sample.
If adding data augmentation later, preserve source attribution and instrument,
wavelength and specimen grouping; train on the training split only.
SDV is best for synthetic *tabular metadata* (e.g., SLED ERP/permits records),
not geospatial topology or scientific spectral fidelity. Distilabel is better
for generating/evaluating *instruction, textual and label feedback* datasets.
Neither is required for the measured USGS spectral ingestion workflow.
