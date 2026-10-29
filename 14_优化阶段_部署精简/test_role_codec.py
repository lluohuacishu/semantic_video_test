"""Checks for role boundaries, required state, and invariant CPU operations."""
from __future__ import annotations

import unittest

from role_codec import RemovedModule, RoleCodec, models_on_meta, prune, validate_state
from sender_common import load_settings
import torch
from torch import nn


class RoleCodecTests(unittest.TestCase):
    def test_sender_keeps_latent_decoders_and_only_drops_display_generators(self):
        models = models_on_meta(load_settings()['runtime']['repo'])
        removed = prune(models, 'sender')
        self.assertEqual(sum(item['parameters'] for item in removed), 148574470)
        self.assertIsInstance(models['image'].vqgan.generator, RemovedModule)
        self.assertIsInstance(models['video'].vqgan.generator, RemovedModule)
        self.assertNotIsInstance(models['image'].dec, RemovedModule)
        self.assertNotIsInstance(models['video'].decoder, RemovedModule)
        for model in models.values():
            self.assertIn('z_vq.embedding.weight', model.state_dict())
            self.assertNotIn('vqgan.quantize.embedding.weight', model.state_dict())

    def test_receiver_keeps_generators_and_removes_only_encoding_branches(self):
        models = models_on_meta(load_settings()['runtime']['repo'])
        removed = prune(models, 'receiver')
        self.assertEqual(sum(item['parameters'] for item in removed), 134706944)
        self.assertIsNone(models['image'].q_enc)
        self.assertIsNone(models['video'].y_q_enc)
        for model in models.values():
            self.assertIsInstance(model.vqgan.encoder, RemovedModule)
            self.assertNotIsInstance(model.vqgan.generator, RemovedModule)
        self.assertNotIsInstance(models['video'].weight_generator, RemovedModule)
        self.assertNotIsInstance(models['video'].hyper_inv_tokenizer, RemovedModule)

    def test_removed_module_fails_instead_of_producing_fake_tensors(self):
        with self.assertRaisesRegex(RuntimeError, 'excluded'):
            RemovedModule('test')(torch.ones(1))

    def test_role_guards_reject_wrong_work_before_using_state(self):
        codec = RoleCodec.__new__(RoleCodec)
        codec.role = 'sender'
        for kwargs in [dict(), dict(x=object()), dict(packet={}, render_rgb=False)]:
            with self.assertRaises(ValueError):
                codec.frame({}, 0, **kwargs)
        codec.role = 'receiver'
        for kwargs in [dict(), dict(x=object(), packet={}), dict(packet={}, render_rgb=False)]:
            with self.assertRaises(ValueError):
                codec.frame({}, 0, **kwargs)

    def test_asset_validation_rejects_missing_extra_shape_dtype_and_nonfinite(self):
        with torch.device('meta'):
            model = nn.Linear(2, 3)
        good = dict(weight=torch.ones(3, 2), bias=torch.zeros(3))
        validate_state(model, good)
        bad = [dict(weight=good['weight']), dict(good, extra=torch.zeros(1)),
               dict(good, weight=torch.zeros(2, 3)), dict(good, weight=torch.zeros(3, 2, dtype=torch.float64)),
               dict(good, bias=torch.full((3,), float('nan')))]
        for state in bad:
            with self.subTest(keys=list(state)), self.assertRaises(ValueError):
                validate_state(model, state)

    def test_cached_norm_expression_and_lookup_match_original_on_cpu(self):
        torch.manual_seed(29)
        weight = torch.randn(32, 8)
        value = torch.randn(7, 8)
        original = (value**2).sum(1, keepdim=True) + (weight**2).sum(1) - 2*torch.matmul(value, weight.t())
        cached = (weight**2).sum(1)
        alternative = (value**2).sum(1, keepdim=True) + cached - 2*torch.matmul(value, weight.t())
        self.assertTrue(torch.equal(original.view(torch.int32), alternative.view(torch.int32)))
        indexes = torch.argmin(original, 1)
        one_hot = torch.zeros(len(indexes), len(weight)).scatter_(1, indexes[:, None], 1)
        old = one_hot @ weight
        new = torch.nn.functional.embedding(indexes, weight)
        self.assertTrue(torch.equal(old.view(torch.int32), new.view(torch.int32)))


if __name__ == '__main__':
    unittest.main()
