# Optional trainable fusion backbone

The SDK's default path runs simple domain encoders, adapters and heads with NumPy. `modulo.neural.FusionBackbone` and `TorchDecisionHead` are opt-in PyTorch modules for training deeper representations.

```python
from modulo.neural import FusionBackbone, TorchDecisionHead
backbone = FusionBackbone({"scientific": None, "image": 512}, width=640, depth=12, heads=10)
head = TorchDecisionHead(640, ["normal", "anomaly"])
logits = head(backbone([{"scientific": [1,2,3], "image": [0.1]*512}]))
```

Scientific arrays are chunked; frozen image/audio/video encoder features can be fed as single embeddings. The trainer, loss function, dataset splits, and optimizer remain caller-controlled for this optional architecture. Freeze feature encoders separately. Do not interpret a larger parameter count as proven quality or scientific accuracy.
