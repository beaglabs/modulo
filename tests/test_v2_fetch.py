import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from modulo.cli import main
class V2FetchTests(unittest.TestCase):
    def test_cli_fetch_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("modulo.dataset_v2.fetch", return_value={"status":"ok"}) as f:
                main(["dataset","fetch","usgs-v2","--output",tmp])
                f.assert_called_once_with(tmp,url=None,expected_sha256=None)
    def test_fetch_manifest_stages_spectra(self):
        with tempfile.TemporaryDirectory() as tmp:
            import modulo.dataset_v2 as v2
            archive=Path(tmp)/"ASCIIdata_splib07a.zip"
            archive.write_bytes(b"test")
            with patch.object(v2,"fetch_spectra",return_value=archive):
                result=v2.fetch(tmp)
                self.assertEqual(result["verified_pairs"],0)
                self.assertEqual(result["photos_downloaded"],0)
                self.assertTrue((Path(tmp)/"source-manifest.json").exists())
