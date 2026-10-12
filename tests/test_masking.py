import unittest
import tempfile
from pathlib import Path
from modulo.masking import MaskedStudent
class MaskTests(unittest.TestCase):
    def test_shapes_and_gradients(self):
        import torch
        for mode in ("none","single","independent","shared"):
            m=MaskedStudent(12,3,width=32,depth=14,checkpoints=7,mode=mode)
            x=torch.randn(4,12)
            logits,masks=m(x,return_masks=True)
            self.assertEqual(tuple(logits.shape),(4,3))
            self.assertEqual(len(masks),0 if mode=="none" else 7)
            logits.sum().backward()
            self.assertTrue(m.project.weight.grad is not None)
