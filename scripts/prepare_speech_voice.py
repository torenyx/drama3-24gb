"""Encode both supplied references as speaker 0 through speech.cpp's voice command."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import time
import soundfile as sf

parser = argparse.ArgumentParser()
parser.add_argument('--speech', required=True)
parser.add_argument('--model', required=True)
parser.add_argument('--assets', required=True)
parser.add_argument('--out', required=True)
args = parser.parse_args()
assets = Path(args.assets)
manifest = json.loads((assets / 'references.json').read_text())
refs = {r['id']: r for r in manifest['references']}
chosen = [refs[i] for i in manifest['baseline_references']]
target = Path(args.out)
target.parent.mkdir(parents=True, exist_ok=True)
start = time.perf_counter()
with tempfile.TemporaryDirectory(prefix='drama3-voice-') as folder:
    audios = []
    for index, reference in enumerate(chosen):
        audio = assets / reference['audio']
        if audio.suffix.lower() != '.wav':
            samples, rate = sf.read(audio, dtype='float32', always_2d=True)
            audio = Path(folder) / f'reference-{index}.wav'
            sf.write(audio, samples, rate, subtype='PCM_16')
        audios.append(str(audio))
    command = [args.speech, 'voice', args.model, *audios,
               str(target), '--device', 'cpu', '--threads', '16']
    for r in chosen:
        command.extend(['--transcript', r['transcript'], '--speaker', '0'])
    subprocess.run(command, check=True)
record = {'references': [r['id'] for r in chosen], 'speaker': 0,
          'seconds': time.perf_counter()-start, 'device': 'CPU', 'threads': 16,
          'processing': 'non-WAV inputs decoded to PCM16 WAV without resampling; upstream speech voice normalization'}
target.with_suffix('.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))
print(json.dumps(record), flush=True)
