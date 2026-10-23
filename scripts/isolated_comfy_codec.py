"""Persistent FP32 Comfy codec, isolated from the dynamic TTS model manager."""
import argparse
import contextlib
import importlib.util
import json
from pathlib import Path
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('--comfy', required=True)
p.add_argument('--codec', required=True)
p.add_argument('--device', required=True)
a = p.parse_args()
sys.argv = [sys.argv[0]]
sys.path.insert(0, a.comfy)

with contextlib.redirect_stdout(sys.stderr):
    import soundfile as sf
    import torch
    from safetensors import safe_open
    import comfy.model_management
    pack = Path(__file__).resolve().parents[1] / 'upstream/comfy-drama3'
    spec = importlib.util.spec_from_file_location('drama3_pack', pack / '__init__.py',
                                                 submodule_search_locations=[str(pack)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    nodes = sys.modules['drama3_pack.nodes']
    torch.set_num_threads(8)
    torch.set_grad_enabled(False)
    device = torch.device(a.device)
    torch.cuda.set_device(device)
    codec = sys.modules['drama3_pack.drama3.loader'].load_codec(a.codec)
    # Only decoding is served here; the encoder and quantizer's encode-only blocks are unused.
    del codec.model.encoder
    del codec.model.quantizer.downsample
    del codec.model.quantizer.pre_module
    codec.model.to(device)
    torch.cuda.synchronize(device)

print(json.dumps({'type': 'ready', 'device': a.device}), flush=True)
with torch.inference_mode():
    for line in sys.stdin:
        request = json.loads(line)
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(device)
        with safe_open(request['codes'], framework='pt') as f:
            codes = f.get_tensor('codes').long()
            counts = json.loads(f.metadata()['takes'])
        assert sum(counts) == codes.shape[-1]
        pieces = []
        for take in codes.split(counts, -1):
            if take.shape[-1]:
                pieces.append(codec.model.decode(take[None].to(device), torch.float32).float().cpu())
        gap = torch.zeros((1, round(request['gap'] * nodes.SAMPLE_RATE)))
        waveform = torch.cat([piece for audio in pieces for piece in (gap, audio)][1:], -1)
        sf.write(request['wav'], waveform.T.numpy(), nodes.SAMPLE_RATE, subtype='FLOAT')
        peak_allocated = torch.cuda.max_memory_allocated(device) / 2**30
        peak_reserved = torch.cuda.max_memory_reserved(device) / 2**30
        # Return activation cache to the colocated dynamic TTS process after decoding.
        torch.cuda.empty_cache()
        print(json.dumps({'type': 'decoded', 'seconds': time.perf_counter() - started,
                          'peak_allocated_gib': peak_allocated,
                          'peak_reserved_gib': peak_reserved}), flush=True)
