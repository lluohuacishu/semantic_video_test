"""分割任务的关键回归检查：标签几何、忽略区域、指标和模型接口。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from segmentation import (
    FPN_TRANSFER_MODEL_NAME, IGNORE_INDEX, HR_TRANSFER_MODEL_NAME, TinyUNet, SegmentationDataset, TransferSegmentationModel,
    balanced_sampling_probabilities, color_mask, confusion_matrix, fit_image,
    metrics_from_confusion, predict_original, segmentation_loss,
)


class SegmentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.classes = [{"index": i, "symbol": str(i)} for i in range(8)]

    def test_padding_keeps_image_and_mask_aligned(self):
        image = np.zeros((40, 80, 3), dtype=np.uint8)
        image[:, :40, 0] = 255
        image[:, 40:, 2] = 255
        mask = np.full((40, 80), 4, dtype=np.uint8)
        mask[:, 40:] = 1
        x, y, info = fit_image(Image.fromarray(image), 64, Image.fromarray(mask))
        self.assertEqual((info["scaled_width"], info["scaled_height"], info["top"]), (64, 32, 16))
        self.assertTrue(torch.all(y[:16] == IGNORE_INDEX))
        self.assertTrue(torch.all(y[16:48, :32] == 4))
        self.assertTrue(torch.all(y[16:48, 32:] == 1))
        self.assertGreater(x[0, 32, 8], 240)
        self.assertGreater(x[2, 32, 56], 240)
        self.assertEqual(set(y.unique().tolist()), {1, 4, 255})

    def test_mismatched_or_rgb_labels_are_rejected(self):
        image = Image.new("RGB", (80, 40))
        for mask in (Image.new("L", (80, 41)), Image.new("RGB", (80, 40)), Image.new("L", (80, 40), 9)):
            with self.subTest(mode=mask.mode, size=mask.size):
                with self.assertRaises(ValueError):
                    fit_image(image, 64, mask)

    def test_known_confusion_matrix_and_recall(self):
        target = torch.tensor([[0, 0, 1, 255]])
        prediction = torch.tensor([[0, 1, 1, 7]])
        matrix = confusion_matrix(prediction, target)
        self.assertEqual(matrix[0, 1], 1)
        self.assertEqual(matrix[1, 0], 0)
        metrics = metrics_from_confusion(matrix.numpy(), self.classes)
        self.assertEqual(metrics["valid_pixels"], 3)
        self.assertAlmostEqual(metrics["miou"], 0.5)
        self.assertAlmostEqual(metrics["pixel_accuracy"], 2/3)
        self.assertAlmostEqual(metrics["per_class"][0]["recall"], 0.5)
        self.assertAlmostEqual(metrics["per_class"][1]["recall"], 1.)
        self.assertIsNone(metrics["per_class"][2]["iou"])

    def test_false_positive_absent_class_counts_against_iou(self):
        matrix = confusion_matrix(torch.tensor([[0, 2]]), torch.tensor([[0, 0]]))
        metrics = metrics_from_confusion(matrix.numpy(), self.classes)
        self.assertAlmostEqual(metrics["miou"], 0.25)
        self.assertEqual(metrics["per_class"][2]["iou"], 0.)

    def test_loss_ignores_padding_even_when_logits_change(self):
        torch.manual_seed(0)
        logits = torch.randn(1, 8, 16, 16)
        target = torch.zeros((1, 16, 16), dtype=torch.long)
        target[:, :4] = 255
        other = logits.clone()
        other[:, :, :4] = torch.randn((1, 8, 4, 16)) * 100
        a = segmentation_loss(logits, target, torch.ones(8))
        b = segmentation_loss(other, target, torch.ones(8))
        self.assertTrue(torch.allclose(a, b))
        with self.assertRaises(ValueError):
            segmentation_loss(logits, torch.full_like(target, 255), torch.ones(8))

    def test_model_backward_and_original_size_prediction(self):
        torch.manual_seed(0)
        model = TinyUNet(base_channels=8)
        output = model(torch.rand(2, 3, 32, 32))
        self.assertEqual(tuple(output.shape), (2, 8, 32, 32))
        target = torch.randint(0, 8, (2, 32, 32))
        loss = segmentation_loss(output, target, torch.ones(8))
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        model.eval()
        result = predict_original(model, Image.new("RGB", (123, 57), (0, 90, 150)), 64, torch.device("cpu"))
        self.assertEqual(result.shape, (57, 123))
        self.assertEqual(result.dtype, np.uint8)
        self.assertLess(result.max(), 8)

    def test_ignore_color_is_not_a_ninth_class(self):
        palette = np.arange(24, dtype=np.uint8).reshape(8, 3)
        colors = color_mask(np.array([[2, 255]], dtype=np.uint8), palette)
        np.testing.assert_array_equal(colors[0, 0], palette[2])
        np.testing.assert_array_equal(colors[0, 1], [80, 80, 80])

    def test_balanced_sampling_increases_rare_class_exposure(self):
        rows = [{"class_pixels": [1000, 0, 0, 0, 0, 0, 0, 0]} for _ in range(90)]
        rows += [{"class_pixels": [500, 0, 500, 0, 0, 0, 0, 0]} for _ in range(10)]
        probabilities = balanced_sampling_probabilities(rows)
        self.assertAlmostEqual(probabilities.sum().item(), 1.)
        self.assertTrue(torch.all(probabilities > 0))
        self.assertGreater(probabilities[-10:].sum().item(), 0.25)

    def test_strong_augmentation_keeps_labels_aligned_and_cache_unchanged(self):
        # 使用构造的缓存样本，避免测试依赖真实数据文件。
        dataset = object.__new__(SegmentationDataset)
        dataset.size, dataset.augment, dataset.augmentation = 64, True, "strong"
        image = torch.zeros(3, 64, 64, dtype=torch.uint8)
        image[0, :, :32] = 255
        image[2, :, 32:] = 255
        mask = torch.full((64, 64), 4, dtype=torch.uint8)
        mask[:, 32:] = 1
        dataset.cached = [(image.clone(), mask.clone())]
        torch.manual_seed(24)
        for _ in range(4):
            x, y = dataset[0]
            red = (x[0] > x[2] + 0.2) & (y != 255)
            blue = (x[2] > x[0] + 0.2) & (y != 255)
            self.assertTrue(torch.all(y[red] == 4))
            self.assertTrue(torch.all(y[blue] == 1))
            self.assertTrue(set(y.unique().tolist()) <= {1, 4, 255})
        self.assertTrue(torch.equal(dataset.cached[0][0], image))
        self.assertTrue(torch.equal(dataset.cached[0][1], mask))

    def test_transfer_model_has_eight_channels_without_download(self):
        from unittest.mock import patch
        with patch("torch.hub.download_url_to_file", side_effect=AssertionError("unexpected download")):
            model = TransferSegmentationModel().eval()
            with torch.inference_mode():
                prediction = model(torch.rand(1, 3, 32, 32))
        self.assertEqual(tuple(prediction.shape), (1, 8, 32, 32))
        self.assertTrue(torch.isfinite(prediction).all())

    def test_high_resolution_transfer_model_uses_quarter_scale_low_feature(self):
        from unittest.mock import patch
        with patch("torch.hub.download_url_to_file", side_effect=AssertionError("unexpected download")):
            model = TransferSegmentationModel(low_layer="3").eval()
            self.assertEqual(model.network.backbone.return_layers, {"3": "low", "16": "high"})
            self.assertEqual(model.network.classifier.low_classifier.in_channels, 24)
            with torch.inference_mode():
                prediction = model(torch.rand(1, 3, 64, 64))
        self.assertEqual(tuple(prediction.shape), (1, 8, 64, 64))
        self.assertTrue(torch.isfinite(prediction).all())

    def test_fpn_transfer_model_fuses_three_feature_scales(self):
        from unittest.mock import patch
        with patch("torch.hub.download_url_to_file", side_effect=AssertionError("unexpected download")):
            model = TransferSegmentationModel(fused=True).eval()
            self.assertEqual(model.network.backbone.return_layers, {"3": "fine", "4": "low", "16": "high"})
            with torch.inference_mode():
                prediction = model(torch.rand(1, 3, 64, 64))
        self.assertEqual(tuple(prediction.shape), (1, 8, 64, 64))
        self.assertTrue(torch.isfinite(prediction).all())


if __name__ == "__main__":
    unittest.main()
