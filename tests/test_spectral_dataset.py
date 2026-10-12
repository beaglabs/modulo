import json
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from modulo import spectral_dataset as sd


class DatasetTests(unittest.TestCase):
    def fixture(self, root):
        archive = root / "source.zip"
        wavelengths = "\n".join(str(.4 + .1 * i) for i in range(5))
        with ZipFile(archive, "w") as z:
            z.writestr("Wavelengths/splib07a_Wavelengths_ASD_demo.txt", wavelengths)
            for index, mineral in enumerate(("Kaolinite", "Quartz", "Hematite", "Olivine", "Calcite", "Gypsum")):
                name = f"ChapterM_Minerals/splib07a_{mineral}.SPEC{index}_ASD_demo_AREF.txt"
                z.writestr(name, "\n".join(str((i + 1) * (index + 1) / 100) for i in range(5)))
            z.writestr("ChapterV_Vegetation/splib07a_Leaf.S1_ASD_demo_AREF.txt", "\n".join([".1"] * 5))
            z.writestr("ChapterM_Minerals/splib07b_Quartz.X_ASD_demo_AREF.txt", "\n".join([".1"] * 5))
        return archive

    def test_prepare_split_validate_export(self):
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            self.skipTest("Install optional dataset extra")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = self.fixture(root)
            dataset = root / "prepared"
            manifest = sd.prepare(archive, dataset, start_nm=400, stop_nm=800, step_nm=100)
            self.assertEqual(manifest["sample_count"], 6)
            self.assertTrue(manifest["specimen_groups_unverified"])
            self.assertEqual(len(sd._read(dataset / "records.parquet")), 6)
            split = sd.split(dataset, seed=42, train=.5, val=.25, test=.25)
            self.assertEqual(sum(split["counts"].values()), 6)
            report = sd.validate(dataset)
            self.assertTrue(report["passed"])
            output = root / "export"
            sd.export(dataset, output)
            self.assertTrue((output / "checksums.sha256").exists())
            self.assertEqual(sd.digest(archive), manifest["source_sha256"])

    def test_unsafe_split_rejected(self):
        with self.assertRaises(ValueError):
            sd.split("/nonexistent", train=.8, val=.2, test=.2)

    def test_wavelength_pair_ambiguity_rejected(self):
        wl = {"dir/a_Wavelengths_ASD_demo.txt": sd.np.arange(5),
              "dir/b_Wavelengths_ASD_demo.txt": sd.np.arange(5)}
        match, reason = sd._wavelength_for("splib07a_Quartz_A_ASD_demo_AREF.txt", wl, 5)
        self.assertIsNone(match)
        self.assertEqual(reason, "missing_or_ambiguous_wavelength_axis")

    def test_fetch_respects_checksum(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "existing.zip"
            path.write_bytes(b"abc")
            self.assertEqual(sd.fetch(path), path)
            with self.assertRaises(ValueError):
                sd.fetch(path, expected_sha256="0" * 64)

    def test_no_synthetic_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp) / "empty.zip"
            with ZipFile(archive, "w") as z:
                z.writestr("README.txt", "empty")
            with self.assertRaises(ValueError):
                sd.prepare(archive, Path(temp) / "out")


if __name__ == "__main__":
    unittest.main()
