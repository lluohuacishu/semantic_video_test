"""Run the registered frozen D1 teachers on RGB-only development inputs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from transformers import SegformerConfig, SegformerForSemanticSegmentation


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')


class FrozenSegFormer(nn.Module):
    """Same architecture and normalization as the existing SUIM Teacher."""
    def __init__(self):
        super().__init__()
        labels = ('BW', 'HD', 'PF', 'WR', 'RO', 'RI', 'FV', 'SR')
        config = SegformerConfig(
            num_labels=8, id2label=dict(enumerate(labels)),
            label2id={name: idx for idx, name in enumerate(labels)},
            hidden_sizes=[32, 64, 160, 256], depths=[2, 2, 2, 2],
            num_attention_heads=[1, 2, 5, 8], decoder_hidden_size=256,
            semantic_loss_ignore_index=255,
        )
        self.network = SegformerForSemanticSegmentation(config)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, image):
        logits = self.network(pixel_values=(image - self.mean) / self.std).logits
        return F.interpolate(logits, size=image.shape[-2:], mode='bilinear', align_corners=False)


def load_model(path):
    state = torch.load(path, map_location='cpu', weights_only=True)
    model = FrozenSegFormer()
    model.load_state_dict(state['model'], strict=True)
    return model.to('cuda').eval().requires_grad_(False)


def tiled_fish(model, rgb):
    height, width = rgb.shape[:2]
    total = torch.zeros((1, 8, height, width), device='cuda')
    counts = torch.zeros((1, 1, height, width), device='cuda')
    xs = sorted(set(list(range(0, max(1, width - 512), 384)) + [max(0, width - 512)]))
    ys = sorted(set(list(range(0, max(1, height - 512), 384)) + [max(0, height - 512)]))
    for y in ys:
        for x in xs:
            crop = torch.from_numpy(rgb[y:y+512, x:x+512].copy()).permute(2, 0, 1)[None]
            crop = crop.float().to('cuda') / 255
            total[:, :, y:y+512, x:x+512] += model(crop)
            counts[:, :, y:y+512, x:x+512] += 1
    return (total / counts).softmax(1)[0, 6].cpu().numpy()


def letterbox_diver(model, image):
    width, height = image.size
    ratio = 256 / max(width, height)
    scaled_width, scaled_height = max(1, round(width*ratio)), max(1, round(height*ratio))
    left, top = (256-scaled_width)//2, (256-scaled_height)//2
    canvas = Image.new('RGB', (256, 256))
    canvas.paste(image.resize((scaled_width, scaled_height), Image.Resampling.BILINEAR), (left, top))
    data = torch.from_numpy(np.asarray(canvas).copy()).permute(2, 0, 1)[None].float().to('cuda')/255
    logits = model(data)[:, :, top:top+scaled_height, left:left+scaled_width]
    probabilities = F.interpolate(logits, size=(height, width), mode='bilinear', align_corners=False).softmax(1)
    confidence, labels = probabilities.max(1)
    return labels[0].cpu().numpy().astype(np.uint8), confidence[0].cpu().numpy()


def run(args):
    plan = read_json(args.plan)
    inputs = read_json(args.inputs)['inputs']
    # Enforce the experiment boundary, without rescanning or hashing the corpus.
    if len(inputs) != 216 or any(row['role'] not in ('development', 'development_regression') for row in inputs):
        raise ValueError('D1 accepts only the registered 216 development inputs')
    if any(row.get('video_id') not in (None, 1, 2, 3) for row in inputs):
        raise ValueError('Reserved video passed to D1 inference')
    args.output.mkdir(parents=True, exist_ok=False)
    args.output_created = True
    (args.output/'packed').mkdir()
    (args.output/'base_labels').mkdir()
    save_json(args.output/'plan.json', plan)
    save_json(args.output/'inputs.json', {'inputs': inputs})
    started = time.perf_counter()
    save_json(args.output/'progress.json', {'status': 'loading_models', 'completed': 0, 'total': len(inputs)})
    torch.set_num_threads(4)
    torch.manual_seed(20261013)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no silent CPU fallback')
    fish_model = load_model(plan['models']['fish']['checkpoint'])
    diver_model = load_model(plan['models']['diver']['checkpoint'])
    print('Loaded frozen fish and diver models on ' + torch.cuda.get_device_name(0), flush=True)
    rows = []
    with (args.output/'predictions.jsonl').open('x', encoding='utf-8') as records:
        with torch.inference_mode():
            for idx, source in enumerate(inputs):
                with Image.open(source['rgb_path']) as original:
                    image = original.convert('RGB')
                rgb = np.asarray(image).copy()
                if image.size != (source['width'], source['height']):
                    raise ValueError('Input dimensions differ from prepared manifest')
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                fish_probability = tiled_fish(fish_model, rgb)
                t1 = time.perf_counter()
                labels, confidence = letterbox_diver(diver_model, image)
                t2 = time.perf_counter()
                packed = np.zeros(rgb.shape[:2], dtype=np.uint8)
                for threshold in plan['models']['fish']['thresholds']:
                    bit = plan['packed_bits'][f'fish_{threshold}']
                    packed |= (fish_probability >= threshold).astype(np.uint8) << bit
                for threshold in plan['models']['diver']['thresholds']:
                    bit = plan['packed_bits'][f'diver_{threshold}']
                    packed |= ((labels == 1) & (confidence >= threshold)).astype(np.uint8) << bit
                name = f'{idx:04d}.png'
                Image.fromarray(packed).save(args.output/'packed'/name)
                if 'video_id' not in source:
                    Image.fromarray(labels).save(args.output/'base_labels'/name)
                row = {'input_index': idx, 'id': source['id'],
                       'packed_path': 'packed/'+name,
                       'base_labels_path': 'base_labels/'+name if 'video_id' not in source else None,
                       'fish_seconds': t1-t0, 'diver_seconds': t2-t1,
                       'total_inference_seconds': t2-t0}
                rows.append(row)
                records.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
                records.flush()
                if (idx+1) % 8 == 0 or idx == 0 or idx+1 == len(inputs):
                    elapsed = time.perf_counter()-started
                    save_json(args.output/'progress.json', {'status':'running', 'completed':idx+1,
                              'total':len(inputs), 'elapsed_seconds':elapsed, 'last_id':source['id']})
                    print(f'{idx+1}/{len(inputs)} completed, elapsed {elapsed:.1f}s', flush=True)
    result = {'status': 'completed', 'input_count': len(rows),
              'process_work_seconds': time.perf_counter()-started,
              'fish_inference_seconds': sum(r['fish_seconds'] for r in rows),
              'diver_inference_seconds': sum(r['diver_seconds'] for r in rows),
              'torch': torch.__version__, 'device': torch.cuda.get_device_name(0),
              'gt_loaded': False, 'training': False, 'codec_run': False,
              'integrity_rescan_performed': False,
              'timing_note': 'Per-frame timers include preprocessing and synchronized GPU-to-CPU output; first frame is cold. Process work includes model load, image IO and PNG saves, excluding Python import startup.'}
    save_json(args.output/'inference_summary.json', result)
    save_json(args.output/'progress.json', {'status':'completed','completed':len(rows),'total':len(rows)})
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=Path(__file__).with_name('d1_run_plan_v1.json'))
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    try:
        run(options)
    except Exception as exc:
        if getattr(options, 'output_created', False):
            save_json(options.output/'failure.json', {'type':type(exc).__name__, 'message':str(exc)})
        raise
