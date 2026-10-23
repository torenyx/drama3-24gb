"""Exercise the official Fish Speech request, reference encoder and WAV decoder."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument('--out', required=True)
parser.add_argument('--model', type=Path, required=True)
parser.add_argument('--assets', type=Path, required=True)
parser.add_argument('--prompt-file', type=Path, required=True)
parser.add_argument('--references', help='Defaults to the manifest baseline_references')
parser.add_argument('--reference-cache', type=Path)
parser.add_argument('--seeds', default='1,0,2,1')
parser.add_argument('--cached-references', action='store_true')
parser.add_argument('--tower-device', default='cuda:1')
parser.add_argument('--codec-device', default='cuda:1')
args = parser.parse_args()
project = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project / 'upstream/fish-speech'))
import numpy as np
import soundfile as sf
import torch
from safetensors import safe_open
from fish_speech.models.text2semantic import inference as native
from fish_speech.models.text2semantic.fish_omni.model import FishOmniModel

if importlib.util.find_spec('torchcodec') is None:
    # This environment has libsndfile; retain native resampling and reference encoding.
    import torchaudio

    def load_audio_with_soundfile(path):
        waveform, rate = sf.read(path, dtype='float32', always_2d=True)
        return torch.from_numpy(waveform.T.copy()), rate

    torchaudio.load = load_audio_with_soundfile

torch.set_num_threads(16)
torch.set_grad_enabled(False)
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
assets = args.assets
checkpoint = args.model
metrics = {'framework': 'official Fish Speech', 'torch': torch.__version__,
           'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
           'devices': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
           'cached_references': args.cached_references, 'takes': [], 'stages': {}}
metrics.update(tower_device=args.tower_device, codec_device=args.codec_device)
metrics['audio_reader'] = 'soundfile' if importlib.util.find_spec('torchcodec') is None else 'torchaudio'


def sync():
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(i)


def timed(name, function):
    sync()
    started = time.perf_counter()
    value = function()
    sync()
    metrics['stages'][name] = time.perf_counter() - started
    print('STAGE', name, metrics['stages'][name], flush=True)
    return value


def persist():
    (out / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2))


with torch.inference_mode():
    model = timed('load_model', lambda: FishOmniModel.from_pretrained(checkpoint,
        devices=['cuda:0', 'cuda:1'], tower_device=args.tower_device, dtype=torch.bfloat16))
    persist()
    codec = timed('load_codec', lambda: native.load_codec_model(checkpoint / 'codec.pth', args.codec_device, model.dtype))
    manifest = json.loads((assets / 'references.json').read_text())
    entries = {r['id']: r for r in manifest['references']}
    codes, texts = [], []
    for name in (args.references.split(',') if args.references else manifest['baseline_references']):
        texts.append('<|speaker:0|>' + entries[name]['transcript'])
        if args.cached_references:
            if args.reference_cache is None:
                raise ValueError('--cached-references requires --reference-cache')
            with safe_open(args.reference_cache / f'{name}.safetensors', framework='pt') as f:
                codes.append(f.get_tensor('codes').long())
        else:
            value = timed(f'encode_{name}', lambda: native.encode_audio(assets / entries[name]['audio'], codec, args.codec_device).cpu())
            codes.append(value)
        np.save(out / f'reference-{name}.npy', codes[-1].numpy())
    text = args.prompt_file.read_text()
    (out / 'prompt.txt').write_text(text)
    original = native.generate_drama3
    current = None

    def observe(model, prompt, **kwargs):
        sync()
        started = time.perf_counter()
        frames, finish = original(model, prompt, **kwargs)
        sync()
        current['chunks'].append({'seconds': time.perf_counter() - started,
            'prompt_rows': len(prompt), 'frames': len(frames), 'finish': finish})
        print('CHUNK', json.dumps(current['chunks'][-1]), flush=True)
        return frames, finish

    native.generate_drama3 = observe
    persist()
    for index, seed in enumerate(map(int, args.seeds.split(','))):
        torch.manual_seed(seed)
        current = {'index': index, 'seed': seed, 'chunks': []}
        sync()
        started = time.perf_counter()
        pieces = []
        for response in native.generate_long_drama3(model=model, text=text, num_samples=1,
                max_new_tokens=600, top_p=.9, top_k=30, temperature=1.,
                prompt_text=texts, prompt_tokens=codes):
            if response.action == 'sample':
                pieces.append(response.codes.cpu())
        sync()
        current['generation_seconds'] = time.perf_counter() - started
        merged = torch.cat(pieces, dim=1)
        decode_start = time.perf_counter()
        audio = native.decode_to_audio(merged.to(args.codec_device), codec).float().cpu().numpy()
        sync()
        current['decode_seconds'] = time.perf_counter() - decode_start
        current['request_seconds'] = time.perf_counter() - started
        current['audio_seconds'] = len(audio) / codec.sample_rate
        current['rms_dbfs'] = float(20 * np.log10(max(np.sqrt(np.mean(audio ** 2)), 1e-12)))
        current['gpu_peak_allocated_gib'] = [torch.cuda.max_memory_allocated(i) / 2**30 for i in range(torch.cuda.device_count())]
        sf.write(out / f'trial-{index}-seed-{seed}.wav', audio, codec.sample_rate, subtype='PCM_16')
        np.save(out / f'trial-{index}-seed-{seed}.npy', merged.numpy())
        metrics['takes'].append(current)
        persist()
        print('TAKE', json.dumps(current), flush=True)
