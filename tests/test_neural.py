import unittest
try:
    import torch
    from modulo.neural import FusionBackbone,TorchDecisionHead
except ImportError:
    torch=None

@unittest.skipUnless(torch is not None,"Install torch to test neural components")
class TestNeural(unittest.TestCase):
    def test_backward_and_partial_modalities(self):
        m=FusionBackbone({"scientific":None,"image":4},width=32,depth=1,heads=4,max_tokens=16,patch=4)
        h=TorchDecisionHead(32,["normal","anomaly"])
        out=h(m([{"scientific":[1,2,3,4,5,6],"image":[.1,.2,.3,.4]},{"scientific":[2,3,4,5]}]))
        self.assertEqual(tuple(out.shape),(2,2))
        out.sum().backward()
        self.assertIsNotNone(h.linear.weight.grad)
