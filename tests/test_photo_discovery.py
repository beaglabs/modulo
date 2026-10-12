import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from modulo.photo_discovery import discover_candidates

class TestPhotoDiscovery(unittest.TestCase):
    def test_explicit_html_photo_link_yields_only_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); sources=root/"sources"; sources.mkdir()
            (sources/"quartz_sample01.html").write_text('<a href="photo_images/quartz.jpg">photo</a>')
            (sources/"quartz.jpg").write_bytes(b"photo")
            records=[{"sample_id":"sample1","source_file":"chapterM/splib07a_quartz_sample01_ASD_REF.txt"}]
            with patch("modulo.photo_discovery._read",return_value=records):
                result=discover_candidates(root,sources,root/"candidates.csv")
            self.assertEqual(result["candidate_links"],1)
            self.assertEqual(result["verified_pairs"],0)
    def test_no_link_never_guesses(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); sources=root/"sources"; sources.mkdir()
            (sources/"quartz_sample01.html").write_text("No image link")
            (sources/"quartz.jpg").write_bytes(b"photo")
            records=[{"sample_id":"sample1","source_file":"splib07a_quartz_sample01_ASD_REF.txt"}]
            with patch("modulo.photo_discovery._read",return_value=records):
                result=discover_candidates(root,sources,root/"candidates.csv")
            self.assertEqual(result["candidate_links"],0)
