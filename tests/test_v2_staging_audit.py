import json
import tempfile
import unittest
from pathlib import Path
from modulo.dataset_v2 import audit

class TestV2StagingAudit(unittest.TestCase):
    def test_unpaired_staging(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            (p/"staging-manifest.json").write_text(json.dumps({"spectra":889,"worklist":"pairing-worklist.csv"}))
            result=audit(p)
            self.assertFalse(result["passed"])
            self.assertEqual(result["spectra"],889)
            self.assertEqual(result["verified_pairs"],0)
            self.assertEqual(result["status"],"awaiting_verified_photo_pairs")
            self.assertTrue((p/"audit.json").exists())
