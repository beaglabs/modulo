import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
import numpy as np
from modulo.fusion import MultimodalFusion
from modulo.geo import Tile, lonlat_to_tile, xyz_url, read_mbtiles

class TestFusion(unittest.TestCase):
    def test_masks_and_dimensions(self):
        f = MultimodalFusion({"raster": lambda x: x, "logs": lambda x: x},
                             {"raster": 2, "logs": 3})
        out = f.encode({"raster": [2, 4]})
        np.testing.assert_array_equal(out.vector, [2, 4, 0, 0, 0])
        self.assertEqual(out.masks, {"raster": True, "logs": False})
        self.assertEqual(out.slices["logs"], (2, 5))

    def test_invalid(self):
        f = MultimodalFusion({"x": lambda x: x}, {"x": 2})
        with self.assertRaises(ValueError): f({"x": [float("nan"), 2]})

class TestTiles(unittest.TestCase):
    def test_xyz(self):
        tile = lonlat_to_tile(0, 0, 2)
        self.assertEqual(tile, Tile(2, 2, 2))
        self.assertEqual(tile.quadkey, "30")
        self.assertEqual(xyz_url("https://example.org/{z}/{x}/{y}.png", tile),
                         "https://example.org/2/2/2.png")
        with self.assertRaises(ValueError):
            xyz_url("http://example.org/{z}/{x}/{y}", tile)

    def test_mbtiles_flips_tms_y(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.mbtiles"
            with sqlite3.connect(path) as db:
                db.execute("CREATE TABLE tiles (zoom_level INTEGER, tile_column INTEGER, tile_row INTEGER, tile_data BLOB)")
                db.execute("INSERT INTO tiles VALUES (2, 2, 1, ?)", (b"tile",))
            self.assertEqual(read_mbtiles(path, Tile(2, 2, 2)), b"tile")

class TestGeometries(unittest.TestCase):
    @unittest.skipUnless(__import__("importlib").util.find_spec("shapely"), "shapely not installed")
    def test_spatial(self):
        from modulo.geo import GeoObservation, intersects, contains, intersection, spatial_join, reproject, buffer
        square = GeoObservation({"type":"Polygon","coordinates":[[[0,0],[1,0],[1,1],[0,1],[0,0]]]})
        point = GeoObservation({"type":"Point","coordinates":[0.5,0.5]})
        self.assertTrue(intersects(square, point))
        self.assertTrue(contains(square, point))
        self.assertIsNotNone(intersection(square, point))
        self.assertEqual(spatial_join([square], [point]), [(0,0)])
        metric = reproject(point, "EPSG:3857")
        self.assertGreater(buffer(metric, 100).geometry["coordinates"].__len__(), 0)
        with self.assertRaises(ValueError): buffer(point, 100)

if __name__ == "__main__": unittest.main()
