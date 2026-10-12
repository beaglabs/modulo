# Modulo

**A modular System 1 SDK.** Build lightweight decisions over seven domains: **raster**, **geospatial**, **scientific**, **image**, **video**, **audio**, and **logs and traces**.

## Quick start

```bash
pip install -e .
modulo domains
modulo components encoders
python -m unittest discover -s tests -v
```

```python
from modulo import SystemOne, Components
from modulo.core import IdentityAdapter, LinearHead
from modulo.domains import GeospatialEncoder

model = SystemOne(Components(
    GeospatialEncoder(), IdentityAdapter(),
    LinearHead([[1,0],[-1,0]], [0,0], ["north","south"])
), domain="geospatial")
print(model.decide({"lat":45,"lon":-80}))
```

## Extensible components

Register **encoders**, **adapters**, and **heads** with `Registry.register(kind, name, factory)`, or install Python distributions exposing entry points under `modulo.encoders`, `modulo.adapters`, and `modulo.heads`. External plugins are only loaded via `registry.discover(allow=["approved-plugin"])` and execute trusted Python code. No automatic arbitrary-code installation or untrusted loading.

```python
from modulo import Registry
registry = Registry()
registry.register("encoders", "custom", lambda: lambda sample: [1.0, 2.0])
```

Models can be composed in code or `SystemOne.from_config(config, registry=registry)`. Choice outputs use `Decision(choice, confidence, probabilities, domain, evidence)`.

## Domains

| Domain | Built-in initial behavior |
|---|---|
| raster | Numeric pixels/arrays or GeoTIFF via optional rasterio |
| geospatial | WGS84 coordinate features |
| scientific | Fixed-length 1D signal features and simple CSV/XDI text |
| image | Frozen pretrained CLIP (optional image dependencies) |
| video | Bounded frame pooling with CLIP/OpenCV (optional dependencies) |
| audio | Frozen pretrained Wav2Vec2 at 16 kHz (optional dependencies) |
| logs | Trace/error/timeout numeric summaries |

These are starting encoders, **not yet domain-trained perception models**. Preserve CRS, timestamps, frequency axes, units, sensor metadata, and provenance for scientifically valid models. The initial built-in adapters and heads are simple; provide trained heads for useful decisions.

## Distillation / training / serving

`ClefTeacher` calls Cloudflare Workers AI explicitly using account credentials; no remote calls occur automatically. To train a cheap supervised student head, supply JSONL rows like `{"features":[0.1,0.3],"target":[0.8,0.2],"classes":["a","b"]}`:

```bash
modulo train --input dataset.jsonl --output student-head.json
modulo decide --config pipeline.json --input observation.json
modulo serve --config pipeline.json
```

The head can be loaded with `LinearHead(**json.load(open("student-head.json")))`. Cloudflare responses must be converted into class-aligned probability rows by a dataset preparation step; raw scientific file paths are not valid teacher inputs. Teacher output usage must comply with applicable service terms.

A manually dispatched GitHub Actions workflow can train the head from a dataset artifact. It does **not** pretend to train a 60M transformer without a real dataset, and it does not invoke the paid teacher automatically.

## Current scope

The SDK and plugin architecture are operational scaffolding. Advanced raster/Zarr ingestion, large neural fusion, temporal training, general schema heads, checkpoint lifecycle, and validated distillation remain future work. No hardware-specific model performance is claimed.
