import unittest
import tempfile
from pathlib import Path
from zipfile import ZipFile
from modulo.photo_discovery import import_local_archive

class LocalUSGSArchiveTests(unittest.TestCase):
    def test_local_import(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            archive=root/"usgs_splib07.zip"
            with ZipFile(archive,"w") as z:
                z.writestr("release/sample.html",'<a href="sample.jpg">Photo</a>')
                z.writestr("release/sample.jpg",b"sample")
            result=import_local_archive(archive,root/"sources")
            self.assertEqual(result["html_files"],1)
            self.assertEqual(result["image_files"],1)
    def test_invalid_archive(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"invalid.zip";path.write_text("<html>error</html>")
            with self.assertRaises(ValueError): import_local_archive(path,Path(temp)/"sources")
