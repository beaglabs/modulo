import unittest
from modulo import SystemOne,Components,Registry
from modulo.core import IdentityAdapter,LinearHead
from modulo.domains import ScientificEncoder,GeospatialEncoder,LogsEncoder
from modulo.registry import defaults,DOMAINS
from modulo.training import train_head

class TestModulo(unittest.TestCase):
    def test_all_domains_registered(self):
        self.assertEqual(set(DOMAINS),set(defaults().names("encoders")))
    def test_simple_sdk(self):
        model=SystemOne(Components(GeospatialEncoder(),IdentityAdapter(),LinearHead([[1,0],[-1,0]],[0,0],["north","south"])),"geospatial")
        result=model.decide({"lat":45,"lon":-10})
        self.assertEqual(result.choice,"north")
        self.assertAlmostEqual(sum(result.probabilities.values()),1)
    def test_train_distillation(self):
        rows=[{"classes":["a","b"],"features":[i,1.0],"target":[.9,.1] if i>0 else [.1,.9]} for i in (-2,-1,1,2)]
        head=train_head(rows,epochs=200,lr=.1)
        self.assertEqual(len(head["weights"]),2)
        self.assertEqual(len(head["weights"][0]),2)
    def test_spectrum(self):
        self.assertEqual(len(ScientificEncoder()([1,2,1])),128)
    def test_logs(self):
        self.assertEqual(len(LogsEncoder()("INFO ok\nERROR timeout")),5)
    def test_extensible(self):
        reg=Registry().register("encoders","custom",lambda:lambda v:[v])
        self.assertEqual(reg.create("encoders","custom")(2),[2])

if __name__=="__main__":unittest.main()
