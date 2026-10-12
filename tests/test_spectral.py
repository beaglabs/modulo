import unittest
import numpy as np
from modulo.domains import ScientificEncoder
from modulo.spectral import SpectralEncoder
from modulo.registry import defaults, DOMAINS
from modulo import SystemOne
from modulo.core import IdentityAdapter, LinearHead, Components


class SpectralTests(unittest.TestCase):
    def test_registered_and_scientific_unchanged(self):
        self.assertIn("spectral", DOMAINS)
        self.assertIn("spectral", defaults().names("encoders"))
        self.assertEqual(ScientificEncoder()([1, 2, 3]).shape, (128,))

    def test_units_and_alignment(self):
        encoder = SpectralEncoder(size=3, start_nm=400, stop_nm=600, normalize=False)
        nm = encoder({"wavelengths": [400, 500, 600], "values": [1, 4, 2]})
        um = encoder({"wavelengths": [.4, .5, .6], "unit": "um", "values": [1, 4, 2]})
        np.testing.assert_allclose(nm, [1, 4, 2])
        np.testing.assert_allclose(um, nm, atol=1e-5)

    def test_cube_band_first_and_last_and_pixel(self):
        cube = np.array([[[1., 3.]], [[5., 7.]], [[9., 11.]]])
        enc = SpectralEncoder(size=3, start_nm=400, stop_nm=600, normalize=False)
        inp = {"wavelengths": [400, 500, 600]}
        np.testing.assert_allclose(enc({**inp, "cube": cube}), [2, 6, 10])
        np.testing.assert_allclose(enc({**inp, "cube": np.moveaxis(cube, 0, -1), "band_axis": -1}), [2, 6, 10])
        np.testing.assert_allclose(enc({**inp, "cube": cube, "pixel": [0, 1]}), [3, 7, 11])

    def test_nan_nodata_and_no_extrapolation(self):
        enc = SpectralEncoder(size=5, start_nm=300, stop_nm=700, normalize=False)
        result = enc({"wavelengths": [400, 500, 600], "values": [1, np.nan, 3]})
        np.testing.assert_allclose(result, [0, 1, 2, 3, 0])

    def test_validation(self):
        enc = SpectralEncoder()
        bad = [
            {"wavelengths": [600, 400], "values": [1, 2]},
            {"wavelengths": [400, 500], "values": [1]},
            {"wavelengths": [400, 500], "values": [np.nan, 2]},
            {"wavelengths": [400, 500], "unit": "meters", "values": [1, 2]},
        ]
        for observation in bad:
            with self.subTest(observation=str(observation)), self.assertRaises(ValueError):
                enc(observation)

    def test_system_one_config(self):
        spec = {
            "domain": "spectral",
            "encoder": {"name": "spectral", "config": {"size": 3, "start_nm": 400, "stop_nm": 600, "normalize": False}},
            "adapter": {"name": "identity"},
            "head": {"name": "linear", "config": {"weights": [[1, 0, 0], [-1, 0, 0]], "bias": [0, 0], "classes": ["positive", "negative"]}},
        }
        model = SystemOne.from_config(spec)
        decision = model.decide({"wavelengths": [400, 500, 600], "values": [1, 2, 3]})
        self.assertEqual(decision.choice, "positive")


if __name__ == "__main__":
    unittest.main()
