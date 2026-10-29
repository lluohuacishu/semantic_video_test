"""Reuse the exact previous measurement/audit loop with a selected loader."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from role_codec import RoleCodec, ref
from sender_common import load_json, new_codec, save_json
import trial_worker


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--variant', choices=['original', 'slim', 'slim_fast'], required=True)
    parser.add_argument('--assets', type=Path, required=True)
    args, remaining = parser.parse_known_args()
    output = Path(remaining[remaining.index('--output') + 1])
    stats = {}
    def factory(decoder_only=False):
        role = 'receiver' if decoder_only else 'sender'
        if args.variant == 'original':
            codec, module = new_codec(decoder_only=decoder_only)
        else:
            codec, module = RoleCodec(args.assets, role, optimize=args.variant == 'slim_fast'), ref
        stats.update(deployment_role=role,
                     live_parameters=sum(p.numel() for model in [codec.image, codec.video] for p in model.parameters()),
                     tensor_bytes=sum(t.numel()*t.element_size() for model in [codec.image, codec.video] for t in model.state_dict().values()))
        return codec, module
    trial_worker.new_codec = factory
    sys.argv = [sys.argv[0], *remaining]
    trial_worker.main()
    value = load_json(output / 'result.json')
    value.update(variant=args.variant, **stats)
    save_json(output / 'result.json', value)


if __name__ == '__main__':
    main()
