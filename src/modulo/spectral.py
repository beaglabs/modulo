"""Wavelength-aware spectral features for 1D spectra and spectral raster cubes.

No model weights are bundled.  Wavelengths are converted to nm, aligned onto
an explicit grid, and out-of-coverage bins are filled with zero (not extrapolated).
"""
from pathlib import Path
import numpy as np


class SpectralEncoder:
    """Encode reflectance/intensity spectra on a stable wavelength grid.

    Supported inputs:
      {"wavelengths": [...], "values": [...], "unit": "nm" | "um"}
      {"wavelengths": [...], "cube": array, "band_axis": 0 | -1,
       "pixel": [row, col] (optional), "unit": "nm" | "um"}
      {"path": "hyperspectral.tif", "wavelengths": [...], "unit": "nm"}
    Raster inputs require rasterio, and band order must match wavelengths.
    Only finite samples are used. No extrapolation is performed.
    """

    def __init__(self, size=128, start_nm=400., stop_nm=2500.,
                 normalize=True, min_coverage=0.0):
        if size < 2 or not np.isfinite([start_nm, stop_nm]).all() or start_nm >= stop_nm:
            raise ValueError("Invalid spectral wavelength grid")
        if not 0 <= min_coverage <= 1:
            raise ValueError("min_coverage must be in [0, 1]")
        self.size = int(size)
        self.grid_nm = np.linspace(start_nm, stop_nm, self.size)
        self.normalize = bool(normalize)
        self.min_coverage = float(min_coverage)

    def __call__(self, observation):
        if not isinstance(observation, dict) or "wavelengths" not in observation:
            raise ValueError("Spectral observation requires wavelengths and values/cube/path")
        unit = observation.get("unit", "nm").lower()
        if unit not in ("nm", "um", "µm"):
            raise ValueError("Spectral wavelength unit must be nm or um")
        wavelengths = np.asarray(observation["wavelengths"], dtype=np.float64).reshape(-1)
        if unit != "nm":
            wavelengths = wavelengths * 1000.0
        if wavelengths.size < 2 or not np.isfinite(wavelengths).all():
            raise ValueError("Requires at least two finite wavelengths")
        if np.any(np.diff(wavelengths) <= 0):
            raise ValueError("Wavelengths must be strictly increasing in band order")
        if "values" in observation:
            values = np.asarray(observation["values"], dtype=np.float64).reshape(-1)
        else:
            if "path" in observation:
                path = Path(observation["path"])
                if path.suffix.lower() not in (".tif", ".tiff"):
                    raise ValueError("Spectral path must be a GeoTIFF")
                import rasterio  # optional dependency
                with rasterio.open(path) as dataset:
                    cube = dataset.read(masked=True).filled(np.nan)
                band_axis = 0
            elif "cube" in observation:
                cube = np.asarray(observation["cube"], dtype=np.float64)
                band_axis = observation.get("band_axis", 0)
            else:
                raise ValueError("Provide values, cube, or path")
            cube = np.asarray(cube, dtype=np.float64)
            if cube.ndim != 3:
                raise ValueError("Spectral cube must be three-dimensional")
            cube = np.moveaxis(cube, band_axis, 0)
            if cube.shape[0] != wavelengths.size:
                raise ValueError("Wavelength count does not match cube bands")
            pixel = observation.get("pixel")
            if pixel is None:
                valid = np.isfinite(cube)
                counts = valid.sum(axis=(1, 2))
                totals = np.where(valid, cube, 0).sum(axis=(1, 2))
                values = np.divide(totals, counts, out=np.full(counts.shape, np.nan), where=counts > 0)
            else:
                if len(pixel) != 2 or any(not isinstance(i, (int, np.integer)) for i in pixel):
                    raise ValueError("pixel must be [row, column]")
                values = cube[:, pixel[0], pixel[1]]
        if len(values) != len(wavelengths):
            raise ValueError("Wavelength count does not match spectral values")
        valid = np.isfinite(values)
        if np.count_nonzero(valid) < 2:
            raise ValueError("Requires at least two finite spectral values")
        wl = wavelengths[valid]
        val = values[valid]
        inside = (self.grid_nm >= wl[0]) & (self.grid_nm <= wl[-1])
        coverage = float(inside.mean())
        if not inside.any() or coverage < self.min_coverage:
            raise ValueError("Insufficient wavelength coverage of configured grid")
        # No fabricated signal outside the observed wavelength range.
        out = np.zeros(self.size, dtype=np.float64)
        out[inside] = np.interp(self.grid_nm[inside], wl, val)
        if self.normalize:
            mean = out[inside].mean()
            std = out[inside].std()
            out[inside] = (out[inside] - mean) / (std + 1e-6)
        return out.astype(np.float32)
