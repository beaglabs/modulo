"""Stable minimal SDK independent from training backends."""
from dataclasses import dataclass
from typing import Any
import numpy as np

@dataclass
class Decision:
    choice: str
    confidence: float
    probabilities: dict[str,float]
    domain: str
    evidence: dict[str,Any]

class IdentityAdapter:
    def __call__(self, features):
        return np.asarray(features,dtype=np.float32).reshape(-1)

class LinearHead:
    """Simple inference head; use trained checkpoints for learned weights."""
    def __init__(self, weights, bias, classes):
        self.weights=np.asarray(weights,dtype=np.float32)
        self.bias=np.asarray(bias,dtype=np.float32)
        self.classes=list(classes)
        if self.weights.ndim!=2 or self.weights.shape[0]!=len(self.classes) or self.weights.shape[1]<1 or self.bias.shape!=(len(self.classes),):
            raise ValueError("Invalid head dimensions")

    def __call__(self, vector):
        vector=np.asarray(vector,dtype=np.float32).reshape(-1)
        if vector.shape[0]!=self.weights.shape[1] or not np.isfinite(vector).all():
            raise ValueError("Feature mismatch or nonfinite input")
        logits=self.weights@vector+self.bias
        e=np.exp(logits-np.max(logits)); probs=e/e.sum()
        return dict(zip(self.classes, map(float,probs)))

class Components:
    def __init__(self, encoder, adapter, head):
        self.encoder=encoder
        self.adapter=adapter
        self.head=head

class SystemOne:
    def __init__(self, components: Components, domain: str):
        self.components=components
        self.domain=domain

    def decide(self, observation):
        feature=self.components.encoder(observation)
        vector=self.components.adapter(feature)
        scores=self.components.head(vector)
        if not scores or any(not np.isfinite(p) or p<0 for p in scores.values()):
            raise ValueError("Invalid head output")
        total=sum(scores.values())
        if total<=0: raise ValueError("Probabilities have zero mass")
        scores={k:float(v/total) for k,v in scores.items()}
        choice=max(scores,key=scores.get)
        return Decision(choice=choice,confidence=scores[choice],probabilities=scores,domain=self.domain,evidence={})

    @classmethod
    def from_config(cls, spec, registry=None):
        from .registry import defaults
        r=registry or defaults()
        parts={}
        for kind in ("encoder","adapter","head"):
            item=spec[kind]
            parts[kind]=r.create(kind+"s",item["name"],**item.get("config",{}))
        return cls(Components(parts["encoder"],parts["adapter"],parts["head"]),spec["domain"])
