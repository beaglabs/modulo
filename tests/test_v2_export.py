import json
import tempfile
import unittest
from pathlib import Path
from modulo.dataset_v2 import export
from modulo.cli import main

class V2ExportTests(unittest.TestCase):
    def test_staging_does_not_export(self):
        with tempfile.TemporaryDirectory() as temp:
            dataset=Path(temp)/"dataset"
            dataset.mkdir()
            (dataset/"staging-manifest.json").write_text(json.dumps({"spectra":889}))
            dest=Path(temp)/"export"
            result=export(dataset,dest)
            self.assertFalse(result["exported"])
            self.assertFalse(dest.exists())
    def test_cli_accepts_v2(self):
        from unittest.mock import patch
        with patch("modulo.dataset_v2.export",return_value={"exported":False}) as fn:
            main(["dataset","export","usgs-v2","--format","parquet","--output","datasets/usgs-v2"])
            fn.assert_called_once_with("datasets/usgs-v2","datasets/usgs-v2")
