"""Encode the selected voice once, then reuse its codes across service restarts."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime'))
import torch
from drama3_runtime import CachedFishReferenceHook, FishReferenceEncodeHook


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True, help='Directory containing codec.pth')
    parser.add_argument('--assets', type=Path, required=True, help='Directory containing references.json')
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--verify', action='store_true', help='Compare persisted integers with a fresh upstream encoding')
    args = parser.parse_args()
    hook = CachedFishReferenceHook(
        codec_directory=args.model, cache_dir=args.cache_dir, device=args.device,
        autocast_dtype=torch.bfloat16, max_reference_audio_seconds=60)
    manifest = json.loads((args.assets / 'references.json').read_text())
    by_id = {reference['id']: reference for reference in manifest['references']}
    for reference_id in manifest['baseline_references']:
        audio = args.assets / by_id[reference_id]['audio']
        start = time.perf_counter()
        item = hook.normalize_input({'audio_path': str(audio)})
        codes = hook.encode_one(item)
        if args.verify:
            fresh = FishReferenceEncodeHook.encode_one(hook, item)
            if not torch.equal(codes, fresh):
                raise AssertionError(f'Cached codes differ from a fresh encoding: {reference_id}')
        print(json.dumps({'reference': reference_id, 'shape': list(codes.shape),
                          'seconds': time.perf_counter() - start,
                          'codec_loaded': hook.codec_loaded,
                          'fresh_encoding_exact_match': True if args.verify else None}), flush=True)


if __name__ == '__main__':
    main()
