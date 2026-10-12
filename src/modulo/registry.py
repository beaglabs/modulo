"""Opt-in registry. External entry points only load on explicit discovery."""
from importlib.metadata import entry_points

KINDS = ("encoders", "adapters", "heads")
DOMAINS = ("raster", "geospatial", "scientific", "spectral", "image", "video", "audio", "logs")

class Registry:
    def __init__(self):
        self._components = {kind: {} for kind in KINDS}

    def register(self, kind, name, factory):
        if kind not in KINDS or not name or not callable(factory):
            raise ValueError("Expected valid component kind/name/factory")
        if name in self._components[kind]:
            raise ValueError(f"Already registered: {kind}/{name}")
        self._components[kind][name] = factory
        return self

    def create(self, kind, name, **config):
        return self._components[kind][name](**config)

    def names(self, kind):
        return sorted(self._components[kind])

    def discover(self, *, allow=None):
        """Explicitly load allowlisted installed Python entry points (trusted code)."""
        allowed = set(allow or ())
        for kind in KINDS:
            for ep in entry_points(group=f"modulo.{kind}"):
                if ep.name in allowed:
                    self.register(kind, ep.name, ep.load())
        return self

def defaults():
    from .domains import RasterEncoder, GeospatialEncoder, ScientificEncoder, ImageEncoder, VideoEncoder, AudioEncoder, LogsEncoder
    from .spectral import SpectralEncoder
    from .core import IdentityAdapter, LinearHead
    r=Registry()
    for name, cls in (("raster",RasterEncoder),("geospatial",GeospatialEncoder),("scientific",ScientificEncoder),("spectral",SpectralEncoder),("image",ImageEncoder),("video",VideoEncoder),("audio",AudioEncoder),("logs",LogsEncoder)):
        r.register("encoders",name,cls)
    r.register("adapters","identity",IdentityAdapter)
    r.register("heads","linear",LinearHead)
    return r
