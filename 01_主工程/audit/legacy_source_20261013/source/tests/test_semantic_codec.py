"""验证语义监督路径，以及不加载分割模型的独立类别解码。"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from image_codec import ImageCodec, SEMANTIC_ARCHITECTURE, read_package
from prepare_image_codec import SOURCE_REVISION
from semantic_image_codec import JointSemanticHyperprior, SemanticImageCodec, condition_image
from segmentation import segmentation_loss


class SemanticCodecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_semantic_only_supervision_reaches_encoder_and_condition_channels(self):
        torch.manual_seed(42)
        model = JointSemanticHyperprior().train()
        image = torch.rand(1, 3, 64, 64)
        target = torch.randint(0, 8, (1, 64, 64))
        output = model(condition_image(image, target))
        segmentation_loss(output["semantic_logits"], target, torch.ones(8)).backward()
        gradient = model.g_a[0].weight.grad
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient[:, :3].abs().sum()), 0)
        self.assertGreater(float(gradient[:, 3:].abs().sum()), 0)

    def test_receiver_reconstructs_rgb_and_classes_from_only_checkpoint_and_file(self):
        torch.manual_seed(1)
        model = JointSemanticHyperprior().eval()
        state = {"model": model.state_dict(), "model_name": SEMANTIC_ARCHITECTURE,
                 "source_revision": SOURCE_REVISION, "quality": 3, "num_classes": 8,
                 "joint_training_completed": True,
                 "classes": [{"index": i, "rgb": [i*30]*3} for i in range(8)]}
        image = np.random.default_rng(42).integers(0, 256, (53, 117, 3), dtype=np.uint8)
        guidance = np.zeros((53, 117), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            checkpoint = folder / "joint.pt"
            torch.save(state, checkpoint)
            sender = SemanticImageCodec(checkpoint)
            sender.encode(image, guidance, folder / "one.svic")
            sender.encode(image, np.full_like(guidance, 7), folder / "two.svic")
            # 随机未训练网络可能把所有 latent 舍入为零；真实码流依赖检查用训练权重另跑。
            x = torch.zeros(1, 3, 64, 64)
            with torch.inference_mode():
                a = sender.model.g_a(condition_image(x, torch.zeros(1, 64, 64, dtype=torch.long)))
                b = sender.model.g_a(condition_image(x, torch.full((1, 64, 64), 7, dtype=torch.long)))
            self.assertFalse(torch.equal(a, b))
            receiver = SemanticImageCodec(checkpoint)
            rgb, classes, report = receiver.decode(folder / "one.svic", folder / "rgb.png", folder / "index.png")
            header, streams = read_package(folder / "one.svic")
            with torch.inference_mode():
                direct = sender.model.decompress([[s] for s in streams], tuple(header["shape"]))
            expected_rgb = direct["x_hat"][0, :, :53, :117].mul(255).round().byte().permute(1, 2, 0).numpy()
            expected_classes = direct["semantic_logits"][0, :, :53, :117].argmax(0).byte().numpy()
            np.testing.assert_array_equal(rgb, expected_rgb)
            np.testing.assert_array_equal(classes, expected_classes)
            self.assertEqual(report["semantic_source"], "direct_latent_decoder")
            self.assertEqual(classes.shape, (53, 117))
            with self.assertRaisesRegex(ValueError, "联合语义码流"):
                ImageCodec().decode(folder / "one.svic", folder / "wrong.png")
            with self.assertRaises(FileExistsError):
                receiver.decode(folder / "one.svic", folder / "rgb.png", folder / "another.png")

    def test_guidance_validation_does_not_accept_ignore_labels_or_wrong_geometry(self):
        image = torch.zeros(1, 3, 64, 64)
        for guidance in (torch.zeros(1, 32, 32, dtype=torch.long),
                         torch.full((1, 64, 64), 255, dtype=torch.long),
                         torch.zeros(1, 64, 64)):
            with self.assertRaises(ValueError):
                condition_image(image, guidance)


if __name__ == "__main__":
    unittest.main()
