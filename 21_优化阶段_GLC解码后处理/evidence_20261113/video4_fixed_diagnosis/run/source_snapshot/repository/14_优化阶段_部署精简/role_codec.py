"""Role-specific, exact-weight GLC loading and optional redundant-work removal.

The official checkout and the default codec remain unchanged. Constructing on
meta avoids allocating removed training/other-role networks before loading.
"""
from __future__ import annotations

import gc
import importlib
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / '13_优化阶段_发送端免RGB重建'))
from sender_common import file_sha, load_json, load_settings, save_json
import glc_video_codec as ref
import numpy as np
import torch
from torch import nn

COMMON_REMOVALS = {
    'image': ['code_pred_loss', 'code_pred_pix_loss', 'vqgan.quantize'],
    'video': ['vqgan.quantize'],
}
ROLE_REMOVALS = {
    'sender': {'image': ['vqgan.generator'], 'video': ['vqgan.generator']},
    'receiver': {
        'image': ['vqgan.encoder', 'enc', 'hyper_enc', 'q_enc'],
        'video': ['vqgan.encoder', 'encoder', 'temporal_cond', 'y_temporal_combine',
                  'hyper_tokenizer', 'y_q_enc'],
    },
}


class RemovedModule(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def forward(self, *_args, **_kwargs):
        raise RuntimeError('Module excluded from this deployment role: ' + self.name)


def models_on_meta(repo):
    sys.path.insert(0, str(Path(repo).resolve()))
    image_type = importlib.import_module('src.models.image_model').GLC_Image
    video_type = importlib.import_module('src.models.video_model').GLC_Video
    with torch.device('meta'):
        return {'image': image_type(), 'video': video_type(hyper_K=16)}


def prune(models, role):
    if role not in ROLE_REMOVALS:
        raise ValueError('Unknown deployment role')
    removed = []
    for model_name, model in models.items():
        for path in COMMON_REMOVALS[model_name] + ROLE_REMOVALS[role][model_name]:
            prefix, _, leaf = path.rpartition('.')
            parent = model.get_submodule(prefix) if prefix else model
            value = getattr(parent, leaf)
            parameters = sum(p.numel() for p in value.parameters()) if isinstance(value, nn.Module) else value.numel()
            removed.append(dict(model=model_name, path=path, parameters=parameters))
            if isinstance(value, nn.Module):
                setattr(parent, leaf, RemovedModule(model_name + '.' + path))
            else:
                parent.register_parameter(leaf, None)
    return removed


def validate_state(model, state):
    expected = model.state_dict()
    if set(expected) != set(state):
        raise ValueError('Deployment checkpoint keys do not match role')
    for name, value in state.items():
        if value.shape != expected[name].shape or value.dtype != expected[name].dtype:
            raise ValueError('Deployment tensor shape/dtype mismatch: ' + name)
        if value.device.type != 'cpu' or not torch.isfinite(value).all():
            raise ValueError('Invalid deployment tensor: ' + name)


def export_assets(output):
    settings = load_settings()['runtime']
    ref.check_source(settings['repo'])
    ref.check_weights(settings['weights'])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    models = models_on_meta(settings['repo'])
    inventory = {}
    for name, model in models.items():
        inventory[name] = dict(
            full_parameters=sum(p.numel() for p in model.parameters()),
            children={key: sum(p.numel() for p in child.parameters()) for key, child in model.named_children()},
        )
    del models
    manifest = dict(schema='glc-role-exact-weights-v1', revision=ref.REVISION,
                    original_weights={name: item['sha256'] for name, item in ref.ASSETS.items()},
                    precision='FP32; original q tables before interpolation', inventory=inventory, roles={})
    for role in ROLE_REMOVALS:
        models = models_on_meta(settings['repo'])
        removed = prune(models, role)
        files = {}
        for name, model in models.items():
            official = 'GLC_image.pth.tar' if name == 'image' else 'GLC_Video.pth.tar'
            state = torch.load(settings['weights'] / official, map_location='cpu', weights_only=True)
            for wrapper in ('state_dict', 'net', 'net_g'):
                if wrapper in state:
                    state = state[wrapper]
            state = {key.removeprefix('module.'): value for key, value in state.items()}
            expected = model.state_dict()
            selected = {key: state[key] for key in expected}
            validate_state(model, selected)
            path = output / f'{role}_{name}.pt'
            torch.save(selected, path)
            files[name] = dict(file=path.name, sha256=file_sha(path), bytes=path.stat().st_size,
                               parameters=sum(value.numel() for value in selected.values()), keys=len(selected))
            del selected, state
            gc.collect()
        manifest['roles'][role] = dict(files=files, removed=removed,
                                      parameters=sum(info['parameters'] for info in files.values()))
        del models
    save_json(output / 'manifest.json', manifest)
    return manifest


class RoleCodec(ref.NeuralCodec):
    def __init__(self, assets, role, device='cuda', optimize=True):
        if role not in ROLE_REMOVALS:
            raise ValueError('Unknown deployment role')
        runtime = load_settings()['runtime']
        ref.check_source(runtime['repo'])
        self.profile = ref.configure(device)
        assets = Path(assets)
        manifest = load_json(assets / 'manifest.json')
        if (manifest['schema'] != 'glc-role-exact-weights-v1' or manifest['revision'] != ref.REVISION or
                manifest['original_weights'] != {name: value['sha256'] for name, value in ref.ASSETS.items()}):
            raise ValueError('Deployment assets have different model provenance')
        models = models_on_meta(runtime['repo'])
        prune(models, role)
        for name, model in models.items():
            info = manifest['roles'][role]['files'][name]
            if Path(info['file']).name != info['file']:
                raise ValueError('Deployment asset must be a local filename')
            path = assets / info['file']
            if file_sha(path) != info['sha256']:
                raise ValueError('Deployment asset checksum failed')
            state = torch.load(path, map_location='cpu', weights_only=True)
            validate_state(model, state)
            model.load_state_dict(state, strict=True, assign=True)
            del state
        self.image, self.video = models['image'], models['video']
        if role == 'sender':
            self.image.interpolate_q()
        else:
            # Same CPU interpolation/order as the official adapter; encoder q is absent.
            self.image.q_dec = nn.Parameter(ref.F.interpolate(
                self.image.q_dec.view(1, 1, 4, 256), size=(64, 256),
                mode='bilinear', align_corners=True).view(64, 256, 1, 1))
        for model in models.values():
            model.requires_grad_(False).eval().to(device)
        from c2_native import NativeGaussianRans
        self.entropy = NativeGaussianRans(runtime['native_library'], ref.GaussianRans())
        self.device, self.role, self.optimize = device, role, optimize
        self.loading = dict(role=role, manifest_sha256=file_sha(assets / 'manifest.json'), files=manifest['roles'][role]['files'])
        self.dpb = dict(ref_latent=None, ref_y=None)
        self._norms, self._active_masks = {}, {}
        if optimize and role == 'sender':
            with torch.inference_mode():
                self._norms = {id(model): (model.z_vq.embedding.weight ** 2).sum(1) for model in models.values()}

    def frame(self, header, index, x=None, packet=None, *, render_rgb=True):
        if self.role == 'sender' and (x is None or packet is not None or render_rgb):
            raise ValueError('Sender role only encodes without RGB')
        if self.role == 'receiver' and (x is not None or packet is None or not render_rgb):
            raise ValueError('Receiver role only decodes with full RGB')
        return super().frame(header, index, x=x, packet=packet, render_rgb=render_rgb)

    def codebook(self, model, z_shape, z=None, stream=None):
        if not self.optimize:
            return super().codebook(model, z_shape, z=z, stream=stream)
        weight = model.z_vq.embedding.weight
        if z is not None:
            flat = z.permute(0, 2, 3, 1).contiguous().view(-1, model.z_vq.emb_dim)
            # Cache only an invariant term; preserve the original FP32 expression order.
            distance = ((flat ** 2).sum(dim=1, keepdim=True) + self._norms[id(model)] -
                        2 * torch.matmul(flat, weight.t()))
            indexes = torch.argmin(distance, dim=1).unsqueeze(1)
            stream = ref.pack_indices(indexes.cpu().numpy())
        else:
            values = ref.unpack_indices(stream, z_shape[1] * z_shape[2])
            indexes = torch.from_numpy(values).to(self.device).reshape(-1, 1)
        quantized = ref.F.embedding(indexes.reshape(-1), weight).view(z_shape).permute(0, 3, 1, 2).contiguous()
        return quantized, stream

    def prior(self, model, params, is_i, y=None, streams=None):
        if not self.optimize:
            return super().prior(model, params, is_i, y=y, streams=streams)
        q_enc, q_dec, scales, means = model.separate_prior(params)
        batch, channels, height, width = means.shape
        key = (id(model), batch, channels, height, width, means.dtype, means.device)
        if key not in self._active_masks:
            masks = (model.get_mask_four_parts if is_i else model.get_mask_dual)(
                batch, channels, height, width, means.dtype, means.device)
            self._active_masks[key] = [(mask, mask.bool()) for mask in masks]
        common = model.y_spatial_prior_reduction(params) if is_i else params
        scaled_y = y * q_enc if y is not None else None
        reconstructed, encoded = None, []
        for stage, (mask, active) in enumerate(self._active_masks[key]):
            if stage:
                context = torch.cat((reconstructed, common), dim=1)
                if is_i:
                    context = getattr(model, f'y_spatial_prior_adaptor_{stage}')(context)
                scales, means = model.y_spatial_prior(context).chunk(2, 1)
            if scaled_y is not None:
                symbols_tensor = torch.round((scaled_y - means * mask) * mask)[active]
                # One D2H transfer/synchronization instead of two; values stay FP32.
                values = torch.stack((scales[active], symbols_tensor)).cpu().numpy()
                table_indexes = self.entropy.indexes(values[0])
                symbols = values[1]
                stream = self.entropy.encode(symbols, table_indexes)
            else:
                table_indexes = self.entropy.indexes(scales[active].cpu().numpy())
                stream = streams[stage]
                symbols = self.entropy.decode(stream, table_indexes)
            encoded.append(stream)
            residual = torch.zeros_like(means)
            residual[active] = torch.from_numpy(symbols).to(device=self.device, dtype=means.dtype)
            part = residual + means * mask
            reconstructed = part if reconstructed is None else reconstructed + part
        return reconstructed * q_dec, encoded
