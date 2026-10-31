"""CPU-only guard checks: bitstream/reference identity and engine cache separation."""
import copy
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from generator_backend import identity, state_sha, mixed_onnx
from decoder_common import PACKAGES
from run_decoder_experiment import verify_results


class DecoderGuards(unittest.TestCase):
    def test_reference_and_entropy_changes_are_rejected(self):
        original = dict(stream_sha256='stream', stream_bytes=10, reference_checksums={'latent_sha256':'ordered'},
                        entropy_symbols_and_cdf_sha256='entropy', records=[dict(index=0, kind=0, latent_sha256='ref')])
        verify_results(original, copy.deepcopy(original))
        for key in ('stream_sha256', 'stream_bytes', 'reference_checksums', 'entropy_symbols_and_cdf_sha256'):
            changed = copy.deepcopy(original)
            changed[key] = 'different'
            with self.assertRaises(ValueError):
                verify_results(original, changed)
        for change in (dict(index=1), dict(kind=1), dict(latent_sha256='different')):
            changed = copy.deepcopy(original)
            changed['records'][0].update(change)
            with self.assertRaises(ValueError):
                verify_results(original, changed)
        changed = copy.deepcopy(original)
        changed['records'] = []
        with self.assertRaises(ValueError):
            verify_results(original, changed)

    def test_rgb_may_change_while_reference_guard_remains_strict(self):
        original = dict(stream_sha256='s', stream_bytes=10, reference_checksums={},
                        entropy_symbols_and_cdf_sha256='e', records=[dict(index=0, kind=0, latent_sha256='r', rgb_sha256='a')])
        actual = copy.deepcopy(original)
        actual['records'][0]['rgb_sha256'] = 'b'
        verify_results(original, actual)

    def test_original_weight_hash_detects_tensor_and_dtype_changes(self):
        model = torch.nn.Conv2d(1, 1, 1)
        clone = copy.deepcopy(model)
        self.assertEqual(state_sha(model), state_sha(clone))
        with torch.no_grad():
            clone.weight.add_(1.)
        self.assertNotEqual(state_sha(model), state_sha(clone))
        self.assertNotEqual(state_sha(model), state_sha(copy.deepcopy(model).half()))

    def test_cache_identity_separates_roles_shapes_and_precision(self):
        model = torch.nn.Conv2d(1, 1, 1)
        example = SimpleNamespace(dtype=torch.float32, is_cuda=True, ndim=4, shape=(1, 256, 24, 40), device='cuda:0')
        with patch('torch.cuda.get_device_name', return_value='test GPU'), patch('torch.cuda.get_device_capability', return_value=(8, 9)):
            base = identity(model, example, 'trt_fp32', 'image', '10.5.0')
            self.assertNotEqual(base, identity(model, example, 'trt_fp32', 'video', '10.5.0'))
            self.assertNotEqual(base, identity(model, example, 'trt_fp16_safe', 'image', '10.5.0'))
            example.shape = (1, 256, 20, 32)
            self.assertNotEqual(base, identity(model, example, 'trt_fp32', 'image', '10.5.0'))

    def test_mixed_precision_preserves_fp32_sensitive_and_output_layers(self):
        import sys
        sys.path.insert(0,str(PACKAGES))
        import onnx
        from onnx import helper,TensorProto
        nodes=[helper.make_node('Conv',['x','w'],['hidden'],name='/blocks.23/conv1/Conv'),
               helper.make_node('Conv',['hidden','w'],['upsampled'],name='/blocks.22/conv/Conv'),
               helper.make_node('Conv',['upsampled','w'],['rgb'],name='/blocks.28/Conv')]
        graph=helper.make_graph(nodes,'test',[helper.make_tensor_value_info('x',TensorProto.FLOAT,[1,1,2,2])],
            [helper.make_tensor_value_info('rgb',TensorProto.FLOAT,[1,1,2,2])],
            [helper.make_tensor('w',TensorProto.FLOAT,[1,1,1,1],[1.])])
        model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',17)])
        with tempfile.TemporaryDirectory(prefix='glc_decoder_onnx_') as temp:
            root=Path(temp).resolve();self.assertTrue(root.is_relative_to(Path(tempfile.gettempdir()).resolve()))
            onnx.save(model,str(root/'original.onnx'))
            policy=mixed_onnx(root/'original.onnx',root/'mixed.onnx')
            mixed=onnx.load(str(root/'mixed.onnx'))
            self.assertEqual([x['compute'] for x in policy],['float16','float32','float32'])
            types={v.name:v.type.tensor_type.elem_type for v in mixed.graph.value_info}
            self.assertEqual(types['hidden'],TensorProto.FLOAT)
            self.assertEqual(types['upsampled'],TensorProto.FLOAT)
            self.assertEqual(mixed.graph.output[0].type.tensor_type.elem_type,TensorProto.FLOAT)


if __name__ == '__main__':
    unittest.main()
