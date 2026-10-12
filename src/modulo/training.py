"""Train small student heads from cached feature vectors and teacher probabilities."""
import json
from pathlib import Path
import numpy as np

def train_head(rows,epochs=100,lr=.02,seed=42):
    rng=np.random.default_rng(seed)
    rows=list(rows)
    if not rows:raise ValueError("No training rows")
    classes=list(rows[0]["classes"])
    if any(r["classes"]!=classes for r in rows):raise ValueError("Mismatched classes")
    x=np.asarray([r["features"] for r in rows],dtype=np.float32)
    y=np.asarray([r["target"] for r in rows],dtype=np.float32)
    if x.ndim!=2 or y.shape!=(len(x),len(classes)) or not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(y<0) or np.any(y.sum(1)<=0):
        raise ValueError("Invalid features or probability targets")
    y=y/y.sum(1,keepdims=True)
    weights=rng.normal(0,.01,(len(classes),x.shape[1])).astype(np.float32)
    bias=np.zeros(len(classes),dtype=np.float32)
    for _ in range(epochs):
        scores=x@weights.T+bias
        p=np.exp(scores-scores.max(1,keepdims=True));p/=p.sum(1,keepdims=True)
        delta=(p-y)/len(x)
        weights-=lr*(delta.T@x)
        bias-=lr*delta.sum(0)
    return {"classes":classes,"weights":weights.tolist(),"bias":bias.tolist()}

def save_head(path,head):
    Path(path).write_text(json.dumps(head),encoding="utf-8")

def read_rows(path):
    with open(path,encoding="utf-8") as f:
        for line in f:
            if line.strip(): yield json.loads(line)
