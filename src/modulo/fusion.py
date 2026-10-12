"""Deterministic multimodal feature fusion and a compatible decision adapter."""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class FusedFeatures:
    vector: np.ndarray
    masks: dict
    slices: dict

class MultimodalFusion:
    """Run named encoders and concatenate fixed-size outputs.

    Missing modalities get zero vectors and false presence masks. Encoder outputs
    must be finite fixed-size vectors. This is a non-trainable baseline; for
    learned cross-modal attention use neural.FusionBackbone after preprocessing.
    """
    def __init__(self, encoders, widths, normalize=False):
        self.encoders = dict(encoders)
        self.widths = dict(widths)
        if not self.widths or set(self.encoders) != set(self.widths):
            raise ValueError("encoders and widths must have the same nonempty keys")
        if any(not isinstance(w, int) or w < 1 for w in self.widths.values()):
            raise ValueError("positive integer feature widths required")
        self.normalize = normalize

    def encode(self, observations):
        if not isinstance(observations, dict): raise TypeError("expected modality dictionary")
        unexpected = set(observations) - set(self.encoders)
        if unexpected: raise ValueError(f"unknown modalities: {sorted(unexpected)}")
        features, masks, slices = [], {}, {}
        offset = 0
        for name, encoder in self.encoders.items():
            present = name in observations and observations[name] is not None
            if present:
                vector = np.asarray(encoder(observations[name]), dtype=np.float32).reshape(-1)
                if len(vector) != self.widths[name] or not np.isfinite(vector).all():
                    raise ValueError(f"{name}: wrong feature width or nonfinite values")
                if self.normalize:
                    norm = np.linalg.norm(vector)
                    if norm > 0: vector = vector / norm
            else:
                vector = np.zeros(self.widths[name], dtype=np.float32)
            masks[name] = present
            slices[name] = (offset, offset + self.widths[name])
            offset += self.widths[name]
            features.append(vector)
        return FusedFeatures(np.concatenate(features), masks, slices)

    def __call__(self, observations):
        """Compatible with SystemOne encoder interface."""
        return self.encode(observations).vector
