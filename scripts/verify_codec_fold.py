"""Compare CPU/GPU weight-norm folding and decode the same cached reference codes."""
import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime'))
import torch
from drama3_runtime import CachedFishReferenceHook, FishReferenceEncodeHook, load_codec
from sglang_omni.models.fishaudio_drama3.channels_last_decoder import ChannelsLastDecoder

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model', required=True)
parser.add_argument('--assets', type=Path, required=True)
parser.add_argument('--cache-dir', required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
torch.set_num_threads(16)
torch.set_grad_enabled(False)
manifest = json.loads((args.assets / 'references.json').read_text())
by_id = {ref['id']: ref for ref in manifest['references']}
hook = CachedFishReferenceHook(codec_directory=args.model, cache_dir=args.cache_dir,
    device='cuda:0', autocast_dtype=torch.bfloat16, max_reference_audio_seconds=60)
result = {'reference_checks': []}
for reference_id in manifest['baseline_references']:
    item = hook.normalize_input({'audio_path': str(args.assets / by_id[reference_id]['audio'])})
    cached = hook.encode_one(item)
    fresh = FishReferenceEncodeHook.encode_one(hook, item)
    repeated = FishReferenceEncodeHook.encode_one(hook, item)
    exact = torch.equal(cached, fresh)
    result['reference_checks'].append({'id': reference_id, 'exact': exact, 'shape': list(cached.shape),
        'different_codes': int((cached != fresh).sum()), 'code_count': cached.numel(),
        'fresh_repeat_exact': torch.equal(fresh, repeated)})

# Use the already loaded native GPU codec for the original fold path.
gpu_codec = hook.codec
gpu_codec.fold_weight_norm()
cpu_codec = load_codec(args.model, 'cpu')
cpu_codec.fold_weight_norm()
cpu_codec = cpu_codec.to('cuda:0')
weight_max_abs = 0.0
weights_allclose = True
native_state = gpu_codec.state_dict()
for name, tensor in cpu_codec.state_dict().items():
    original = native_state[name]
    if original.is_floating_point():
        weight_max_abs = max(weight_max_abs, (original - tensor).abs().max().item())
        weights_allclose &= torch.allclose(original, tensor, rtol=2e-5, atol=1e-7)
codes = cached[:, :128].unsqueeze(0).to('cuda:0')
with torch.inference_mode():
    original_audio = ChannelsLastDecoder(gpu_codec, dtype=torch.float16)(codes).float()
    cpu_fold_audio = ChannelsLastDecoder(cpu_codec, dtype=torch.float16)(codes).float()
difference = original_audio - cpu_fold_audio
error_rms = difference.square().mean().sqrt().item()
signal_rms = original_audio.square().mean().sqrt().item()
result.update(weight_max_abs=weight_max_abs, weights_allclose=bool(weights_allclose),
              weight_tolerance={'rtol': 2e-5, 'atol': 1e-7},
              waveform_max_abs=difference.abs().max().item(), waveform_error_rms=error_rms,
              waveform_snr_db=20 * math.log10(signal_rms / error_rms) if error_rms else None,
              waveform_exact=torch.equal(original_audio, cpu_fold_audio),
              waveform_finite=bool(torch.isfinite(cpu_fold_audio).all()),
              note='One 128-frame reference-code window, eager ChannelsLast float16 decode; not a listening score.')
args.out.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2), flush=True)
assert weights_allclose and result['waveform_finite']
