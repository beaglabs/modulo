import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from modulo.cli import main
class DiscoveryDownloadFailures(unittest.TestCase):
    def test_failure_is_reported_not_hidden(self):
        with patch("modulo.photo_discovery.download_sources",return_value={
            "status":"source_download_failed","html_files":0,
            "failures":[{"reason":"ScienceBase returned non-ZIP content"}],"archives":[]}) as d:
            with patch("modulo.photo_discovery.discover_candidates") as scan:
                main(["dataset","discover","usgs-v2","--download"])
                scan.assert_not_called()
                d.assert_called_once()
