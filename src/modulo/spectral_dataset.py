"""Reproducible USGS splib07a ASCII mineral dataset preparation.

The original archive contains one-number-per-line spectrum files and separate
wavelength files keyed by spectrometer; bad bands use -1.23e34.
Never infer mineral purity, specimen identifiers or reliable class taxonomy.
"""
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
import json
import re
import urllib.request
from zipfile import ZipFile
import numpy as np

ITEM_ID = "586e8c88e4b0f5ce109fccae"
CATALOG = f"https://www.sciencebase.gov/catalog/item/{ITEM_ID}"
API = f"https://www.sciencebase.gov/catalog/item/{ITEM_ID}?format=json"
BAD = -1.23e34
SOURCE = "usgs-splib07a"
SPECTRAL_RE = re.compile(r"_(REF|AREF|RREF|REFF|REFL|TRANS|EMIS)\.txt$", re.I)
IGNORED = ("wavelength", "bandpass", "fwhm", "errorbar", "standarddeviation")

def digest(path):
    h = sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def fetch(dest, url=None, expected_sha256=None):
    """Discover exact official archive URL from ScienceBase; no guessed endpoint."""
    dest = Path(dest)
    if dest.exists():
        if expected_sha256 and digest(dest) != expected_sha256.lower():
            raise ValueError("Existing archive checksum mismatch")
        return dest
    if url is None:
        with urllib.request.urlopen(API, timeout=45) as response:
            item = json.load(response)
        files = item.get("files", [])
        choices = [f for f in files if f.get("name", "").lower() == "asciidata_splib07a.zip"]
        if len(choices) != 1:
            raise ValueError("Could not uniquely discover ASCIIdata_splib07a.zip; pass --url from official USGS catalog")
        url = choices[0].get("downloadUri") or choices[0].get("url")
        if not url:
            raise ValueError("Official metadata did not contain an archive download URL")
    if not url.lower().startswith("https://"):
        raise ValueError("Only HTTPS downloads are allowed")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, tmp)
        with ZipFile(tmp) as z:
            if z.testzip() is not None:
                raise ValueError("Corrupt source archive")
        if expected_sha256 and digest(tmp) != expected_sha256.lower():
            raise ValueError("Archive checksum mismatch")
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
    return dest

def numbers(text):
    lines = text.splitlines()
    values = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith(("#", ";")):
            continue
        # USGS ASCII spectra have a free-text header; numeric channels follow.
        try:
            values.append(float(line))
        except ValueError:
            continue
    return np.asarray(values, dtype=np.float64)

def _name(path):
    return Path(path).name.lower()

def _instrument_key(name):
    name = _name(name)
    if "wavelengths_" in name:
        return name.split("wavelengths_", 1)[1].rsplit(".txt", 1)[0]
    # AREF / REF etc are usually final tags, preceding spectrometer code.
    parts = name.rsplit(".", 1)[0].split("_")
    if len(parts) < 3:
        return ""
    return "_".join(parts[-3:-1]).lower()

def _wavelength_for(spectrum, wavefiles, count):
    """Match the actual USGS instrument family and original channel count."""
    name = _name(spectrum).upper()
    m = re.search(r"_(ASD[A-Z0-9]*|BECK[A-Z0-9]*|NIC4[A-Z0-9]*|AVIRIS[A-Z0-9]*)_(?:A?REF|RREF|REFF|REFL|TRANS|EMIS)\\.TXT$", name)
    if not m:
        return None, "unknown_instrument"
    code = m.group(1)
    family = next(tag for tag in ("ASD", "BECK", "NIC4", "AVIRIS") if code.startswith(tag))
    matches = [(path, values) for path, values in wavefiles.items()
               if re.search(r"_WAVELENGTHS_" + family + r"(?:_|\\.)", _name(path), re.I)
               and len(values) == count]
    if len(matches) != 1:
        return None, "missing_or_ambiguous_wavelength_axis"
    return matches[0], None

def _specimen_and_label(name):
    """Conservative filename-derived labels; not authoritative mineral taxonomy."""
    stem = Path(name).stem
    stem = re.sub(r"^splib07a_", "", stem, flags=re.I)
    tokens = stem.split("_")
    if not tokens:
        return "", "", False
    mineral = re.split(r"[.\\-]", tokens[0])[0].strip()
    # Keep entire sample-name portion prior to instrument suffix. This is a
    # proxy grouping key, flagged explicitly until metadata confirms specimen.
    marker = re.search(r"_(ASD|BECK|NICOLET|AVIRIS)", stem, flags=re.I)
    identity = stem[:marker.start()] if marker else stem.rsplit("_", 1)[0]
    return mineral.lower(), identity.lower(), False

def prepare(archive, output, start_nm=400, stop_nm=2500, step_nm=5, min_coverage=.6):
    if step_nm <= 0 or start_nm >= stop_nm or not 0 <= min_coverage <= 1:
        raise ValueError("Invalid grid or coverage")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty to prevent accidental mixing")
    grid = np.arange(start_nm, stop_nm + step_nm / 2, step_nm, dtype=np.float64)
    rejected = Counter()
    records = []
    seen = set()
    with ZipFile(archive) as z:
        paths = sorted(p for p in z.namelist() if p.lower().endswith(".txt"))
        wavefiles = {}
        for path in paths:
            if "wavelengths_" in _name(path):
                wavefiles[path] = numbers(z.read(path).decode("utf-8", errors="replace"))
        for path in paths:
            name = _name(path)
            if not any(c.lower().startswith("chapterm_minerals") for c in Path(path).parts):
                continue
            if "splib07a" not in name or any(token in name for token in IGNORED):
                continue
            if not SPECTRAL_RE.search(name):
                continue
            values = numbers(z.read(path).decode("utf-8", errors="replace"))
            pair, issue = _wavelength_for(path, wavefiles, len(values))
            if issue:
                rejected[issue] += 1
                continue
            wavepath, wavelengths = pair
            # Data release uses micrometers for many native wavelength files.
            wl = wavelengths * 1000 if np.nanmax(wavelengths) < 300 else wavelengths.copy()
            valid = np.isfinite(values) & np.isfinite(wl) & (values > -1e30)
            wl, vals = wl[valid], values[valid]
            if len(wl) < 2 or np.any(np.diff(wl) <= 0):
                rejected["insufficient_or_nonmonotonic_channels"] += 1
                continue
            coverage_mask = (grid >= wl[0]) & (grid <= wl[-1])
            if coverage_mask.mean() < min_coverage:
                rejected["low_coverage"] += 1
                continue
            result = np.full(len(grid), np.nan, dtype=np.float32)
            result[coverage_mask] = np.interp(grid[coverage_mask], wl, vals).astype(np.float32)
            # Preserve only valid (non-gap) spectral spans; physical band gaps
            # cannot safely be reconstructed without instrument specifications.
            label, specimen, verified = _specimen_and_label(name)
            if not label or not specimen:
                rejected["missing_label_or_specimen"] += 1
                continue
            fingerprint = sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()
            if fingerprint in seen:
                rejected["exact_duplicate_spectrum"] += 1
                continue
            seen.add(fingerprint)
            records.append({
                "sample_id": sha256(path.encode()).hexdigest()[:20],
                "specimen_id": specimen, "specimen_verified": verified,
                "mineral": label, "source_file": path, "wavelength_file": wavepath,
                "measurement": name.rsplit("_", 1)[-1].split(".")[0].upper(),
                "wavelengths_nm": grid.astype(np.float32).tolist(),
                "reflectance": result.tolist(),
                "valid_mask": coverage_mask.tolist(),
                "source": SOURCE,
                "source_checksum_sha256": digest(archive),
            })
    if not records:
        raise ValueError("No usable spectra parsed. Inspect archive format and wavelength pairing; no fabricated records.")
    output.mkdir(parents=True, exist_ok=True)
    _write(output / "records.parquet", records)
    manifest = {
        "source": SOURCE, "catalog": CATALOG, "source_sha256": digest(archive),
        "sample_count": len(records), "specimen_groups_unverified": True,
        "wavelength_grid_nm": grid.tolist(), "min_coverage": min_coverage,
        "rejections": dict(rejected), "schema_version": 1,
        "warning": "Filename-derived mineral labels and specimen groups require metadata curation before scientific claims.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (output / "taxonomy.json").write_text(json.dumps(dict(Counter(r["mineral"] for r in records)), indent=2))
    return manifest

def _write(path, rows):
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Install dataset extra: pip install -e '.[dataset]'") from exc
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")

def _read(path):
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Install dataset extra: pip install -e '.[dataset]'") from exc
    return pq.read_table(path).to_pylist()

def split(dataset, seed=42, train=.7, val=.15, test=.15):
    if any(v <= 0 for v in (train, val, test)) or abs(train + val + test - 1) > 1e-8:
        raise ValueError("Split fractions must be positive and sum to one")
    dataset = Path(dataset)
    records = _read(dataset / "records.parquet")
    groups = defaultdict(list)
    for row in records:
        groups[row["specimen_id"]].append(row)
    if len(groups) < 3:
        raise ValueError("At least three independent specimen groups required")
    keys = sorted(groups)
    np.random.default_rng(seed).shuffle(keys)
    a, b = round(len(keys) * train), round(len(keys) * (train + val))
    if a < 1 or b <= a or b >= len(keys):
        raise ValueError("Not enough specimen groups to populate each split")
    splits = {"train": keys[:a], "validation": keys[a:b], "test": keys[b:]}
    for name, ids in splits.items():
        _write(dataset / f"{name}.parquet", [r for k in ids for r in groups[k]])
    report = {"seed": seed, "group_by": "specimen_id", "counts": {name: sum(len(groups[k]) for k in ids) for name, ids in splits.items()},
              "groups": {name: len(ids) for name, ids in splits.items()}}
    (dataset / "splits.json").write_text(json.dumps(report, indent=2))
    return report

def validate(dataset):
    dataset = Path(dataset)
    manifest = json.loads((dataset / "manifest.json").read_text())
    splits = {}
    for name in ("train", "validation", "test"):
        path = dataset / f"{name}.parquet"
        if not path.is_file():
            raise ValueError(f"Missing split {name}; run modulo dataset split")
        splits[name] = _read(path)
    group_sets = {name: {r["specimen_id"] for r in rows} for name, rows in splits.items()}
    overlaps = {f"{a}-{b}": len(group_sets[a] & group_sets[b]) for a, b in
                (("train", "validation"), ("train", "test"), ("validation", "test"))}
    samples = [r["sample_id"] for rows in splits.values() for r in rows]
    passed = (not any(overlaps.values()) and len(samples) == len(set(samples))
              and len(samples) == manifest["sample_count"])
    report = {"passed": passed, "sample_count": len(samples), "specimen_overlap": overlaps,
              "class_counts": {name: dict(Counter(r["mineral"] for r in rows)) for name, rows in splits.items()},
              "unverified_specimen_groups": manifest["specimen_groups_unverified"],
              "warning": "No-leakage is only guaranteed for proxy specimen identifiers; manual metadata validation required."}
    (dataset / "audit.json").write_text(json.dumps(report, indent=2))
    if not passed:
        raise ValueError(f"Dataset validation failed: {report}")
    return report

def export(dataset, output):
    dataset, output = Path(dataset), Path(output)
    validate(dataset)
    output.mkdir(parents=True, exist_ok=True)
    from shutil import copy2
    for name in ("manifest.json", "taxonomy.json", "splits.json", "audit.json",
                 "train.parquet", "validation.parquet", "test.parquet"):
        if (dataset / name).resolve() != (output / name).resolve():
            copy2(dataset / name, output / name)
    lines = [f"{digest(output / name)}  {name}" for name in sorted(
        ("manifest.json", "taxonomy.json", "splits.json", "audit.json", "train.parquet", "validation.parquet", "test.parquet"))]
    (output / "checksums.sha256").write_text("\n".join(lines) + "\n")
    return output
