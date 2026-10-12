"""Discover official USGS V7 image metadata and propose auditable photo/spectrum links.

The ScienceBase SPECPR and HTMLmetadata archives contain explicit metadata/photo
links. A candidate is never automatically asserted as a verified physical pair.
"""
from pathlib import Path
from urllib.parse import urlparse, unquote
from urllib.request import urlopen, urlretrieve
from html.parser import HTMLParser
from zipfile import ZipFile
from collections import defaultdict
import csv
import json
import re
from .spectral_dataset import _read, digest

SCIENCEBASE = "https://www.sciencebase.gov/catalog/item/"
CATALOG_IDS = ("586e8d4de4b0f5ce109fccbb", "586e8bf2e4b0f5ce109fcca2")
IMAGE_SUFFIXES = {".jpg",".jpeg",".png",".gif",".tif",".tiff",".webp"}
class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.links=[]
    def handle_starttag(self, tag, attrs):
        if tag.lower() != "a": return
        href=dict(attrs).get("href")
        if href: self.links.append(href)

def _official_files(item):
    with urlopen(SCIENCEBASE+item+"?format=json", timeout=45) as response:
        metadata=json.load(response)
    return [{"name":f.get("name",""),"url":f.get("downloadUri") or f.get("url")}
            for f in metadata.get("files",[])
            if f.get("name") and (f.get("downloadUri") or f.get("url"))]

def _safe_extract(z, root):
    for item in z.infolist():
        if item.is_dir(): continue
        p=Path(item.filename)
        if p.is_absolute() or ".." in p.parts: continue
        if p.suffix.lower() not in IMAGE_SUFFIXES|{".htm",".html"}: continue
        target=(root/p).resolve()
        if not target.is_relative_to(root.resolve()): continue
        target.parent.mkdir(parents=True,exist_ok=True)
        with z.open(item) as source, target.open("wb") as dst:
            import shutil
            shutil.copyfileobj(source,dst)

def download_sources(output, archive_urls=None):
    """Discover archive URLs from official catalog metadata; stage HTML and photos.

    Archive URLs can be overridden only explicitly. Source files are never interpreted
    as verified sample identity until joined against exact metadata references.
    """
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    items=[]
    if archive_urls:
        items=[{"name":Path(urlparse(u).path).name,"url":u} for u in archive_urls]
    else:
        for item in CATALOG_IDS:
            items.extend(_official_files(item))
    selected=[x for x in items if x["name"].lower().endswith(".zip")
              and any(k in x["name"].lower() for k in ("htmlmetadata","specprslib07"))]
    summaries=[]
    for entry in selected:
        url=entry["url"]
        host=(urlparse(url).hostname or "").lower()
        if urlparse(url).scheme!="https" or not (host=="sciencebase.gov" or host.endswith(".sciencebase.gov") or host=="sciencebase.usgs.gov"):
            if not archive_urls: raise ValueError("Untrusted archive location: "+url)
            if urlparse(url).scheme!="https": raise ValueError("HTTPS required")
        location=output/entry["name"]
        if not location.is_file():
            temp=location.with_suffix(location.suffix+".part")
            try:
                urlretrieve(url,temp)
                with ZipFile(temp) as z:
                    if z.testzip() is not None: raise ValueError("corrupt archive")
                temp.replace(location)
            finally: temp.unlink(missing_ok=True)
        with ZipFile(location) as z: _safe_extract(z,output/"sources")
        summaries.append({"archive":str(location),"sha256":digest(location),"url":url})
    report={"archives":summaries,"html_files":len(list((output/"sources").rglob("*.html")))+len(list((output/"sources").rglob("*.htm"))),
            "image_files":sum(1 for p in (output/"sources").rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES),
            "note":"Discovered assets only; no inferred specimen identity"}
    (output/"discovery-manifest.json").write_text(json.dumps(report,indent=2))
    return report

def discover_candidates(dataset="datasets/usgs-v7", sources="datasets/raw/usgs-v2/sources",
                        output="datasets/usgs-v2/photo-candidates.csv"):
    """Only emit candidate links when an HTML record contains an explicit image href.

    Association to a spectrum is provisional until the metadata specimen identity
    is independently reconciled; no row includes the verified crosswalk fields.
    """
    dataset,sources,output=Path(dataset),Path(sources),Path(output)
    records=_read(dataset/"records.parquet")
    by_key=defaultdict(list)
    for r in records:
        stem=Path(r["source_file"]).stem.lower()
        stem=re.sub(r"_(asd[a-z0-9]*|beck[a-z0-9]*|nic[a-z0-9]*|aviris[a-z0-9]*)_(aref|ref|rref|reff|refl|trans|emis)$","",stem)
        stem=stem.removeprefix("splib07a_")
        by_key[stem].append(r)
    photos={p.name.lower():p for p in sources.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES}
    rows=[]; issues=[]
    for page in sources.rglob("*"):
        if page.suffix.lower() not in (".html",".htm") or not page.is_file(): continue
        parser=Links()
        try: parser.feed(page.read_text(encoding="utf-8",errors="replace"))
        except Exception as ex:
            issues.append({"file":str(page),"error":str(ex)}); continue
        key=page.stem.lower().removeprefix("splib07a_")
        matches=by_key.get(key,[])
        for href in parser.links:
            name=Path(unquote(urlparse(href).path)).name.lower()
            photo=photos.get(name)
            if not photo:continue
            for r in matches:
                rows.append({"sample_id":r["sample_id"],"source_file":r["source_file"],
                    "metadata_file":str(page),"photo_path":str(photo),
                    "photo_sha256":digest(photo),"metadata_href":href,
                    "status":"candidate_needs_specimen_verification"})
    output.parent.mkdir(parents=True,exist_ok=True)
    fields=["sample_id","source_file","metadata_file","photo_path","photo_sha256","metadata_href","status"]
    with output.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    report={"candidate_links":len(rows),"distinct_spectra":len({r["sample_id"] for r in rows}),
            "candidate_csv":str(output),"metadata_errors":issues,
            "verified_pairs":0,"note":"Filename matching links metadata to spectra provisionally; human or authoritative ID verification still required."}
    (output.parent/"candidate-report.json").write_text(json.dumps(report,indent=2))
    return report
