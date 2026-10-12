import unittest
class MaskTests(unittest.TestCase):
    @unittest.skipUnless(__import__("importlib").util.find_spec("torch"), "torch not installed")
    def test_shapes_and_gradients(self):
        import torch
        from modulo.masking import MaskedStudent
        for mode in ("none","single","independent","shared"):
            m=MaskedStudent(12,3,width=32,depth=14,checkpoints=7,mode=mode)
            logits,masks=m(torch.randn(4,12),return_masks=True)
            self.assertEqual(tuple(logits.shape),(4,3))
            self.assertEqual(len(masks),0 if mode=="none" else 7)
            logits.sum().backward()
            self.assertIsNotNone(m.project.weight.grad)
