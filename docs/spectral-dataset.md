# Modulo spectral dataset preparation

The `dataset` CLI prepares the official **USGS Spectral Library version 7 measured ASCII release**, `ASCIIdata_splib07a.zip`. It does not use derivative `splib07b` spectra. Source: https://www.sciencebase.gov/catalog/item/586e8c88e4b0f5ce109fccae

## Local workflow (Mac/Linux)

```bash
pip install -e '.[dataset]'
modulo dataset fetch usgs-v7
modulo dataset prepare usgs-v7 --wavelength-min 400 --wavelength-max 2500 --step-nm 5
modulo dataset split usgs-v7 --group-by specimen --train 0.70 --val 0.15 --test 0.15 --seed 42
modulo dataset validate usgs-v7
modulo dataset export usgs-v7 --format parquet --output datasets/spectral-v1
```

`fetch` discovers the archive through the USGS ScienceBase item metadata. If the API is unavailable, download `ASCIIdata_splib07a.zip` manually and pass `--archive` to prepare, or use `fetch --url <verified-official-archive-url>`. An existing archive is reused; optionally pass `--sha256` to verify source integrity. The preparation step requires an empty destination.

## Canonical record

`records.parquet` contains each accepted spectrum's stable sample hash, filename-derived mineral label and specimen proxy, complete source path and wavelength-file path, source checksum, 5 nm reflectance grid, and coverage mask. Out-of-range positions are NaN, **not zero reflectance**. Interpolation is limited to the observed wavelength range, and the parser rejects bad-band sentinel values. Original source archives remain separate.

`manifest.json` records the source archive hash, counts, wavelength grid, and rejection reasons. `taxonomy.json` is an *uncurated label histogram*, not a verified taxonomy. `splits.json` records the deterministic specimen-proxy split; `audit.json` reports the split integrity and per-split label counts. `checksums.sha256` verifies exported deliverables.

## Scientific limitations and release gates

**These records are not automatically training-ready.** The USGS naming convention and metadata can vary by spectrometer. The parser only accepts spectra when it can pair exactly one wavelength file by instrument label and channel count; it reports rejected records rather than inventing an axis. This conservative assumption must be checked on the full source archive.

The first label is derived from a filename token, and the specimen group is a *filename-derived proxy* flagged `specimen_verified=false`. Before publishing model metrics or making scientific claims, attach the authoritative USGS sample metadata, verify mineral taxonomy and physical specimen identity, remove shared-source duplicates across datasets, and establish held-out instrument and specimen groups. The current validator prevents overlap between derived grouping keys, **not all physical specimen leakage**. Some classes may have no examples in validation/test. Avoid claiming robust cross-class performance until audited.

Do not train a mineral classifier using class names inferred from filenames without manually validating the label set. The dataset can still be used to inspect source quality and develop preprocessing and baseline infrastructure.

Raw datasets and generated parquet output are ignored by Git; do not commit the original library into the repository. The download and parsing steps require network access and enough disk space, but no GPU.
