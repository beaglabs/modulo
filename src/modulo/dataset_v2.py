"""USGS V2 photo/spectrum pairing with explicit specimen-identity evidence.

This module never guesses a pairing from visual similarity or filename stem.
A curator supplies a CSV crosswalk sourced from official specimen metadata.
"""
import csv
import json
import shutil
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
import numpy as np
from .spectral_dataset import _read, _write, digest, fetch as fetch_spectra, CATALOG

REQUIRED = {"sample_id", "specimen_id", "mineral", "photo_path", "evidence_uri"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}

def prepare(spectral_dir, crosswalk=None, images_root=None, output="datasets/usgs-v2", archive="datasets/raw/usgs-v2/ASCIIdata_splib07a.zip"):
    """Prepare paired V2 records from V1, curated crosswalk and local real photos."""
    spectral_dir, output = Path(spectral_dir), Path(output)
    if not crosswalk or not images_root:
        return stage_unpaired(spectral_dir, output, archive)
    crosswalk, images_root = Path(crosswalk), Path(images_root)
    if output.exists() and any(output.iterdir()):
        raise ValueError("V2 output directory must be empty")
    spectra = _read(spectral_dir / "records.parquet")
    by_id = {r["sample_id"]: r for r in spectra}
    if len(by_id) != len(spectra): raise ValueError("duplicate spectral sample identifiers")
    with crosswalk.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not REQUIRED <= set(reader.fieldnames or []):
            raise ValueError(f"crosswalk missing {sorted(REQUIRED - set(reader.fieldnames or []))}")
        links = list(reader)
    rejected = Counter()
    accepted = []
    seen = set()
    for link in links:
        sid = link["sample_id"].strip()
        specimen = link["specimen_id"].strip()
        mineral = link["mineral"].strip()
        evidence = link["evidence_uri"].strip()
        rel = Path(link["photo_path"].strip())
        if not sid or sid not in by_id:
            rejected["unknown_spectrum"] += 1; continue
        if not specimen or not mineral or not evidence:
            rejected["missing_verified_identity_or_evidence"] += 1; continue
        # Prevent traversal, symlink escape and absolute paths.
        if rel.is_absolute() or ".." in rel.parts or rel.suffix.lower() not in IMAGE_EXT:
            rejected["invalid_photo_path"] += 1; continue
        root = images_root.resolve()
        photo = (root / rel).resolve()
        if not photo.is_relative_to(root) or not photo.is_file():
            rejected["missing_or_escaping_photo"] += 1; continue
        pair = (sid, specimen, str(rel))
        if pair in seen:
            rejected["duplicate_pair"] += 1; continue
        seen.add(pair)
        record = dict(by_id[sid])
        record.update({"specimen_id": specimen, "specimen_verified": True,
                       "mineral": mineral, "photo_path": str(rel),
                       "photo_sha256": digest(photo), "evidence_uri": evidence,
                       "pair_verified": True})
        accepted.append(record)
    # Ambiguous labels or specimen assignments invalidate all affected spectral samples.
    candidates = defaultdict(list)
    for row in accepted: candidates[row["sample_id"]].append(row)
    conflicting = {k for k,v in candidates.items()
                   if len({(x["specimen_id"], x["mineral"]) for x in v}) != 1}
    rejected["conflicting_identity"] += sum(len(candidates[k]) for k in conflicting)
    accepted = [x for x in accepted if x["sample_id"] not in conflicting]
    specimen_labels = defaultdict(set)
    for x in accepted: specimen_labels[x["specimen_id"]].add(x["mineral"])
    bad_specimens = {k for k,v in specimen_labels.items() if len(v)>1}
    rejected["conflicting_specimen_label"] += sum(x["specimen_id"] in bad_specimens for x in accepted)
    accepted = [x for x in accepted if x["specimen_id"] not in bad_specimens]
    if not accepted: raise ValueError(f"no verified pairs: {dict(rejected)}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "images").mkdir()
    for row in accepted:
        origin = images_root / row["photo_path"]
        dest = output / "images" / row["photo_sha256"][:20]
        dest = dest.with_suffix(origin.suffix.lower())
        if not dest.exists(): shutil.copyfile(origin, dest)
        row["photo_path"] = str(dest.relative_to(output))
    _write(output / "records.parquet", accepted)
    manifest = {"schema_version": 2, "source": "usgs-v1-plus-curated-crosswalk",
                "crosswalk_sha256": digest(crosswalk), "count_pairs": len(accepted),
                "unique_spectra": len({r["sample_id"] for r in accepted}),
                "specimens": len({r["specimen_id"] for r in accepted}),
                "rejected": dict(rejected),
                "note": "Crosswalk evidence is curator-provided and NOT authenticated automatically"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest

def split(dataset, seed=42, train=.7, val=.15, test=.15):
    dataset = Path(dataset)
    if any(v<=0 for v in (train,val,test)) or abs(train+val+test-1)>1e-8:
        raise ValueError("fractions must be positive and total 1")
    rows = _read(dataset/"records.parquet")
    grouped = defaultdict(list)
    for row in rows: grouped[row["specimen_id"]].append(row)
    keys = sorted(grouped)
    if len(keys)<3: raise ValueError("at least three verified specimens required")
    rng=np.random.default_rng(seed); rng.shuffle(keys)
    a,b = round(train*len(keys)), round((train+val)*len(keys))
    if not 0<a<b<len(keys): raise ValueError("not enough specimens for all splits")
    for name, ids in (("train",keys[:a]),("validation",keys[a:b]),("test",keys[b:])):
        _write(dataset/f"{name}.parquet", [r for k in ids for r in grouped[k]])
    return audit(dataset)

def audit(dataset):
    dataset=Path(dataset)
    if not (dataset/"records.parquet").is_file():
        staging = dataset/"staging-manifest.json"
        if staging.is_file():
            manifest = json.loads(staging.read_text())
            result = {"passed": False, "status": "awaiting_verified_photo_pairs",
                      "spectra": manifest.get("spectra", 0), "verified_pairs": 0,
                      "worklist": manifest.get("worklist"),
                      "next": "Verify specimen-photo identities, then prepare with --crosswalk and --images-root in a fresh directory."}
            (dataset/"audit.json").write_text(json.dumps(result,indent=2))
            return result
        raise FileNotFoundError(f"No prepared V2 dataset at {dataset}. Run modulo dataset prepare usgs-v2 first.")
    rows=_read(dataset/"records.parquet")
    by_sample=defaultdict(set)
    by_specimen=defaultdict(set)
    problems=[]
    for row in rows:
        by_sample[row["sample_id"]].add(row["specimen_id"])
        by_specimen[row["specimen_id"]].add(row["mineral"])
        photo=dataset/row["photo_path"]
        if not photo.is_file() or digest(photo)!=row["photo_sha256"]:
            problems.append("missing_or_changed_photo:"+row["sample_id"])
        if not row.get("evidence_uri") or not row.get("pair_verified"):
            problems.append("unverified:"+row["sample_id"])
    if any(len(s)!=1 for s in by_sample.values()): problems.append("specimen_identity_conflicts")
    if any(len(s)!=1 for s in by_specimen.values()): problems.append("specimen_label_conflicts")
    parts={}
    if all((dataset/f"{p}.parquet").exists() for p in ("train","validation","test")):
        for p in ("train","validation","test"):parts[p]=_read(dataset/f"{p}.parquet")
        groups={p:{r["specimen_id"] for r in data} for p,data in parts.items()}
        samples={p:{r["sample_id"] for r in data} for p,data in parts.items()}
        for a,b in (("train","validation"),("train","test"),("validation","test")):
            if groups[a]&groups[b]:problems.append("specimen_leakage:"+a+"-"+b)
            if samples[a]&samples[b]:problems.append("spectrum_leakage:"+a+"-"+b)
        if sum(len(v) for v in parts.values())!=len(rows): problems.append("split_incomplete")
    result={"passed":not problems, "errors":problems, "pairs":len(rows),
            "specimens":len(by_specimen),
            "classes":dict(Counter(r["mineral"] for r in rows)),
            "split_counts":{p:len(v) for p,v in parts.items()},
            "warning":"Evidence URIs and mineral identities must be checked against source metadata by a curator."}
    (dataset/"audit.json").write_text(json.dumps(result,indent=2))
    if problems: raise ValueError(f"V2 audit failed: {problems}")
    return result

def fetch(output="datasets/raw/usgs-v2", url=None, expected_sha256=None):
    """Download official measured spectral archive and stage a V2 metadata manifest.

    Photo acquisition cannot be inferred from the spectral archive; no specimen
    imagery or verified crosswalk is claimed by this command.
    """
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    archive = fetch_spectra(directory / "ASCIIdata_splib07a.zip",
                            url=url, expected_sha256=expected_sha256)
    manifest = {
        "source": "usgs-splib07a",
        "source_catalog": CATALOG,
        "archive": str(archive),
        "archive_sha256": digest(archive),
        "photos_downloaded": 0,
        "verified_pairs": 0,
        "status": "spectral_archive_fetched_photo_crosswalk_required",
        "next": "Prepare usgs-v7 spectral records, acquire official specimen photos and verify the crosswalk before preparing usgs-v2",
    }
    (directory / "source-manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest

def stage_unpaired(spectral_dir, output, archive):
    """Prepare measured spectra and generate an explicitly unverified pairing worklist."""
    from .spectral_dataset import prepare as prepare_spectra
    if output.exists() and any(output.iterdir()):
        raise ValueError("V2 staging output must be empty; choose a new --output")
    if not (spectral_dir/"records.parquet").is_file():
        source = Path(archive)
        if not source.is_file():
            raise FileNotFoundError("Missing spectral archive. Run 'modulo dataset fetch usgs-v2' first")
        prepare_spectra(source, spectral_dir)
    records = _read(spectral_dir/"records.parquet")
    output.mkdir(parents=True, exist_ok=True)
    worklist=output/"pairing-worklist.csv"
    with worklist.open("w",newline="",encoding="utf-8") as handle:
        columns=["sample_id","specimen_proxy_unverified","mineral_label_unverified",
                 "source_file","specimen_id","mineral","photo_path","evidence_uri"]
        writer=csv.DictWriter(handle,fieldnames=columns)
        writer.writeheader()
        for row in records:
            writer.writerow({"sample_id":row["sample_id"],
                "specimen_proxy_unverified":row["specimen_id"],
                "mineral_label_unverified":row["mineral"],
                "source_file":row["source_file"]})
    result={"status":"awaiting_verified_photo_pairs",
        "spectra":len(records),"verified_pairs":0,
        "worklist":str(worklist),
        "message":"Spectra prepared. No guessed specimen-photo links created."}
    (output/"staging-manifest.json").write_text(json.dumps(result,indent=2))
    return result
