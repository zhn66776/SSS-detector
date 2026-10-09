from __future__ import annotations

import io
import json
import sys
import unittest
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from serve import InspectorData  # noqa: E402


class InspectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = InspectorData(ROOT)

    def test_metadata_and_contact_counts(self):
        metadata = self.app.metadata
        self.assertEqual(metadata["crs"]["epsg"], 32630)
        self.assertEqual(metadata["datasets"]["costa"]["summary"]["afl"]["total"], 2871)
        self.assertEqual(metadata["datasets"]["moray"]["summary"]["owf"]["total"], 9558)
        self.assertEqual(metadata["datasets"]["moray"]["summary"]["ecr"]["total"], 7556)
        self.assertEqual(len(self.app.contacts), 20983)

    def test_every_contact_is_in_reported_dataset_extent(self):
        for dataset in self.app.metadata["coverage"].values():
            for area in dataset.values():
                self.assertEqual(area["outside"], 0)
                self.assertGreater(area["inside"], 0)

    def test_default_magnetic_filter(self):
        result = self.app.filtered_contacts(
            {"dataset": ["moray"], "area": ["owf"], "cluster_m": ["0"]}
        )
        self.assertEqual(result["matched"], 9558 - 311)

    def test_explicit_empty_category_filter(self):
        result = self.app.filtered_contacts(
            {
                "dataset": ["moray"],
                "area": ["owf"],
                "categories_active": ["1"],
                "cluster_m": ["100"],
            }
        )
        self.assertEqual(result["matched"], 0)

    def test_id_search_is_not_limited_to_current_viewport(self):
        result = self.app.filtered_contacts(
            {
                "dataset": ["moray"],
                "area": ["owf"],
                "bbox": ["0,0,1,1"],
                "q": ["OWF-03795"],
                "cluster_m": ["100"],
            }
        )
        self.assertEqual(result["matched"], 1)
        self.assertEqual(result["features"][0]["id"], "OWF-03795")

    def test_native_crop_has_source_pixel_dimensions(self):
        body, headers = self.app.crop("moray", 512878.0, 6455187.0, 50.0)
        image = Image.open(io.BytesIO(body))
        self.assertEqual(image.size, (50, 50))
        self.assertEqual(headers["X-Native-Resolution-M"], "1.0")

    def test_native_zoom_tile_is_png(self):
        config = self.app.public_config()["datasets"]["moray"]
        z = config["max_zoom"]
        body, content_type = self.app.tile("moray", z, 0, 0)
        self.assertEqual(content_type, "image/png")
        self.assertEqual(Image.open(io.BytesIO(body)).size, (256, 256))

    def test_costa_fails_explicitly_without_decoder(self):
        costa = self.app.public_config()["datasets"]["costa"]
        if costa["status"] == "ecw_decoder_missing":
            with self.assertRaisesRegex(RuntimeError, "ECW"):
                self.app.crop("costa", 480499.8, 6562547.9, 50.0)

    def test_costa_native_crop_when_decoder_is_ready(self):
        costa = self.app.public_config()["datasets"]["costa"]
        if costa["status"] == "ready":
            body, headers = self.app.crop("costa", 480499.8, 6562547.9, 50.0)
            image = Image.open(io.BytesIO(body))
            self.assertEqual(image.size, (500, 500))
            self.assertEqual(headers["X-Native-Resolution-M"], "0.1")
            self.assertGreater(image.getextrema()[0][1] - image.getextrema()[0][0], 100)

    def test_original_archives_match_preparation_signature(self):
        for source in self.app.metadata["sources"].values():
            path = Path(source["path"])
            self.assertTrue(path.is_file())
            self.assertEqual(path.stat().st_size, source["size"])
            self.assertEqual(path.stat().st_mtime_ns, source["mtime_ns"])


if __name__ == "__main__":
    unittest.main()
