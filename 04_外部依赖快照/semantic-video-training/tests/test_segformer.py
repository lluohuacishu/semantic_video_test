"""八类适配的梯度、预训练参数和离线权重加载检查。"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from segmentation import (
    PREPROCESSING, SEGFORMER_MODEL_NAME, create_segmentation_model,
    load_segmentation_model, predict_original, segmentation_loss,
)


@unittest.skipUnless(importlib.util.find_spec("transformers"), "需要 SegFormer 独立环境")
class SegFormerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_background_and_last_class_receive_gradients(self):
        torch.manual_seed(7)
        model = create_segmentation_model(SEGFORMER_MODEL_NAME)
        x = torch.rand(2, 3, 64, 96)
        labels = torch.zeros(2, 64, 96, dtype=torch.long)
        labels[:, :, 40:] = 7
        labels[:, :8] = 255
        output = model(x)
        self.assertEqual(output.shape, (2, 8, 64, 96))
        output.retain_grad()
        loss = segmentation_loss(output, labels, torch.ones(8))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.all(output.grad[:, :, :8] == 0))
        self.assertGreater(output.grad[:, 0, 8:, :40].abs().sum(), 0)
        self.assertGreater(output.grad[:, 7, 8:, 40:].abs().sum(), 0)
        for module in (model.network.segformer, model.network.decode_head):
            gradients = [p.grad for p in module.parameters() if p.requires_grad]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
            self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0)

    def test_saved_model_loads_offline_and_restores_original_size(self):
        model = create_segmentation_model(SEGFORMER_MODEL_NAME).eval()
        image = Image.new("RGB", (117, 53), (24, 130, 172))
        before = predict_original(model, image, 64, torch.device("cpu"))
        with tempfile.TemporaryDirectory(prefix="segformer-test-") as folder:
            path = Path(folder) / "test.pt"
            torch.save({"model_name": SEGFORMER_MODEL_NAME, "model": model.state_dict(),
                        "preprocessing": PREPROCESSING, "size": 64}, path)
            loaded, _ = load_segmentation_model(path, torch.device("cpu"))
            after = predict_original(loaded, image, 64, torch.device("cpu"))
        np.testing.assert_array_equal(before, after)
        self.assertEqual(after.shape, (53, 117))
        self.assertLess(after.max(), 8)

    def test_pretrained_backbone_and_decoder_are_preserved(self):
        directory = ROOT / "checkpoints/pretrained/segformer-b0-ade"
        if not (directory / "model.safetensors").is_file():
            self.skipTest("本地未准备预训练权重")
        from transformers import SegformerForSemanticSegmentation
        reference = SegformerForSemanticSegmentation.from_pretrained(
            directory, local_files_only=True, use_safetensors=True, trust_remote_code=False,
        )
        model = create_segmentation_model(SEGFORMER_MODEL_NAME, pretrained_path=directory)
        adapted = model.network.state_dict()
        for name, value in reference.state_dict().items():
            if not name.startswith("decode_head.classifier."):
                self.assertTrue(torch.equal(value, adapted[name]), name)
        self.assertEqual(model.network.decode_head.classifier.out_channels, 8)
        self.assertEqual(model.network.config.id2label[0], "BW")
        self.assertEqual(model.network.config.id2label[7], "SR")


if __name__ == "__main__":
    unittest.main()
