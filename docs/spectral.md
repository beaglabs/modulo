# Spectral encoder

Modulo's `spectral` domain accepts **1D wavelength-indexed spectra** and **3D spectral raster cubes**. It returns an aligned, fixed-length NumPy vector for the existing adapter/head and `SystemOne.from_config` workflow. It is a deterministic preprocessing encoder, not a pretrained geological model.

```python
from modulo.registry import defaults
encoder = defaults().create("encoders", "spectral", size=128, start_nm=400, stop_nm=2500)
features = encoder({
    "wavelengths": [400, 1000, 1600, 2200, 2500],
    "unit": "nm",
    "values": [0.1, 0.4, 0.3, 0.2, 0.5],
})
```

A cube may be supplied with `{"wavelengths": [...], "cube": array, "band_axis": 0}` for [bands, height, width], or `"band_axis": -1` for [height, width, bands]. By default it computes per-band means ignoring NaNs; supply `"pixel": [row, column]` to take one spectrum. For a GeoTIFF, install `pip install -e '.[raster]'` and supply `{"path": "scene.tif", "wavelengths": [...], "unit": "nm"}`. Wavelength metadata must be supplied in band order; no unsupported inference from GeoTIFF tags is performed.

Wavelength units are nm (default) or um/µm. Wavelengths must be strictly increasing and match band count. Samples are linearly interpolated to a configured wavelength grid; uncovered bins are zero-filled, never extrapolated. `normalize=True` z-scores covered bins only; `min_coverage` rejects inadequate coverage. **Zero-filled bins are not observations**: use a common supported grid and consistent sensor coverage for training, or add a custom adapter with explicit coverage-mask handling. Inputs should already be radiometrically calibrated when reflectance is intended. Band averaging loses spatial detail; use a plugin spatial encoder or per-pixel calls if you need spectral-spatial structure.

Train or supply a real decision head before interpreting scores as mineral classes or prospectivity probabilities. This does not implement PU learning, pretraining, atmospheric correction, spectral library matching, or georeferencing transformation.
