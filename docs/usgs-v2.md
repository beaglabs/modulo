# USGS V2 and seven-mask experiment

V2 uses **curator-verified links only**. The original USGS release does NOT guarantee
a photograph for every spectrum. No URL or filename-similarity-based joins are
accepted as proof of physical specimen identity.

1. Prepare the existing V1 dataset through `modulo dataset prepare usgs-v7`.
2. Obtain official photos and specimen metadata directly from USGS release assets,
   then curate a CSV crosswalk with these columns:
   `sample_id,specimen_id,mineral,photo_path,evidence_uri`.
   The evidence URI must point to the source record establishing the *same
   physical specimen* as the spectrum. The pipeline records but **cannot
   authenticate** the URI or the curator's conclusion.
3. Run:

```bash
pip install -e '.[dataset,geo,image,train]'
modulo dataset prepare usgs-v2 --spectral-dataset datasets/usgs-v7 \
  --crosswalk verified-crosswalk.csv --images-root usgs-photos \
  --output datasets/usgs-v2
modulo dataset split usgs-v2 --dataset datasets/usgs-v2 --seed 42
modulo dataset audit usgs-v2 --dataset datasets/usgs-v2
python -m modulo.benchmark_v2 --dataset datasets/usgs-v2 --output results/masks --epochs 25
```

V2 checks photo content hashes, basic identity conflicts and grouping integrity;
it does NOT automatically validate official metadata. A curator must verify
each entry against the release. Do not publish benchmark metrics without
independent review, enough class coverage, and multiple random seeds.

The benchmark compares ungated, one-gate, seven independent gates, and
seven shared-depth-conditioned gates. It uses **very simple pooled image pixel
features**, not CLIP. This is a smoke-test / architecture-ablation benchmark.
A credible multimodal scientific evaluation should replace that input with
frozen pretrained image features and report parameter-matched comparisons,
instrument-group evaluation, confidence calibration, and held-out specimen
results. The script refuses held-out labels absent from training.

HF Jobs example, once curated data is available in the job environment:

```bash
hf jobs uv run --flavor h200 --timeout 2h30m --with torch --with numpy \
  --with pillow --with pyarrow benchmark_v2.py
```

Ensure dataset files are accessible to the remote job and artifacts are
persisted outside the ephemeral runner before launching. The command above
is illustrative; the actual bundled benchmark runs as a Python module.
