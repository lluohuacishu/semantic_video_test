"""逐区域明细必须关联到所属图片，包括跨批次和最后不足一批的情况。"""
from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_small_objects


class SampleRegions(Dataset):
    records = [{"image": f"image_{i}.png"} for i in range(5)]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        target = torch.zeros(16, 16, dtype=torch.long)
        target[2:3+index, 2:4] = 1
        image = target.float().unsqueeze(0).repeat(3, 1, 1)
        return image, target


class ExactPrediction(torch.nn.Module):
    def forward(self, images):
        return torch.nn.functional.one_hot(images[:, 0].long(), 8).permute(0, 3, 1, 2).float()


class SmallObjectTests(unittest.TestCase):
    def test_csv_keeps_global_image_index_across_batches(self):
        dataset = SampleRegions()
        classes = [{"index": i, "symbol": str(i)} for i in range(8)]
        metadata = {"classes": classes, "splits": {"test": dataset.records}}
        state = {"manifest_sha256": "test", "size": 16, "model_name": "test"}
        with tempfile.TemporaryDirectory(prefix="small-object-test-") as folder:
            output = Path(folder) / "result"
            with patch.object(sys, "argv", ["evaluate_small_objects", "--checkpoint", "test.pt", "--device", "cpu",
                                           "--batch-size", "2", "--output-dir", str(output)]), \
                 patch.object(evaluate_small_objects, "manifest_info", return_value=(Path(folder), metadata, "test")), \
                 patch.object(evaluate_small_objects, "SegmentationDataset", return_value=dataset), \
                 patch.object(evaluate_small_objects, "load_segmentation_model", return_value=(ExactPrediction(), state)):
                self.assertEqual(evaluate_small_objects.main(), 0)
            with (output / "components.csv").open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual([row["image"] for row in rows], [r["image"] for r in dataset.records])
        self.assertEqual([int(row["area_pixels"]) for row in rows], [2, 4, 6, 8, 10])
        self.assertTrue(all(float(row["pixel_recall"]) == 1 for row in rows))


if __name__ == "__main__":
    unittest.main()
