import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile
from modulo.photo_discovery import download_sources
class OfficialUSGSHostTests(unittest.TestCase):
    def test_official_sciencebase_manager_endpoint_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("modulo.photo_discovery._official_files",side_effect=[
                [{"name":"HTMLmetadata.zip","url":"https://sciencebase.usgs.gov/manager/download/cm5h20i5b006x0upicqclbbgu"}],[]]), patch("modulo.photo_discovery.urlretrieve") as download, patch("modulo.photo_discovery.ZipFile") as zipfile:
                # Avoid hitting actual network; failure must not be 'Untrusted archive location'
                download.side_effect=RuntimeError("download invoked")
                with self.assertRaisesRegex(RuntimeError,"download invoked"):
                    download_sources(tmp)
