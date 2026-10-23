"""Measure the real speech.cpp worker protocol, including first PCM delivery."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import time
import wave

parser = argparse.ArgumentParser()
parser.add_argument('--speech', required=True)
parser.add_argument('--model', required=True)
parser.add_argument('--voice', required=True)
parser.add_argument('--prompt', required=True)
parser.add_argument('--out', required=True)
parser.add_argument('--devices', default='Vulkan0')
parser.add_argument('--seeds', default='0,1,2')
parser.add_argument('--threads', type=int, default=16)
args = parser.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
text = Path(args.prompt).read_text()
(out / 'prompt.txt').write_text(text)
env = {**os.environ, 'SPEECH_NO_UPDATE_CHECK': '1', 'NO_COLOR': '1'}
results = {'backend': 'speech.cpp', 'model': args.model, 'voice': args.voice,
           'process_started_epoch': time.time(),
           'devices': args.devices.split(','), 'threads': args.threads,
           'startup_warmup': False, 'sampling': {'temperature': 1.0, 'top_p': 0.9, 'top_k': 30},
           'workers': {}, 'takes': []}


def persist():
    results['updated_epoch'] = time.time()
    (out / 'metrics.json').write_text(json.dumps(results, ensure_ascii=False, indent=2))


def start_worker(device):
    log = open(out / f'worker-{device}.log', 'w')
    start = time.perf_counter()
    process = subprocess.Popen([args.speech, 'worker', args.model, '--device', device,
                                '--threads', str(args.threads), '--no-warmup',
                                '--add-voice', f'makima={args.voice}'],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                               text=True, encoding='utf-8', env=env, bufsize=1)
    line = process.stdout.readline()
    if not line:
        raise RuntimeError(f'{device}: worker exited; see its log')
    ready = json.loads(line)
    if ready['type'] != 'ready':
        raise RuntimeError(ready)
    results['workers'][device] = {'pid': process.pid, 'ready_seconds': time.perf_counter()-start,
                                  'ready': ready}
    return process, log, ready['model']['sample_rate']


def synthesize(device, worker, seed):
    process, _, rate = worker
    request = {'type': 'synthesize', 'id': str(seed), 'text': text, 'voice': 'makima',
               'seed': seed, **results['sampling'], 'max_seconds': 600/21.5,
               'experts_on_cpu': False}
    start = time.perf_counter()
    process.stdin.write(json.dumps(request, ensure_ascii=False)+'\n')
    process.stdin.flush()
    take = {'device': device, 'seed': seed, 'ttfa_seconds': None, 'chunks': []}
    count = 0
    with wave.open(str(out / f'{device}-seed-{seed}.wav'), 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        for line in process.stdout:
            message = json.loads(line)
            now = time.perf_counter()-start
            if message['type'] == 'chunk':
                pcm = base64.b64decode(message['pcm'])
                if not pcm:
                    continue
                if take['ttfa_seconds'] is None:
                    take['ttfa_seconds'] = now
                take['chunks'].append({'seconds': now, 'samples': len(pcm)//2})
                count += len(pcm)//2
                wav.writeframesraw(pcm)
            elif message['type'] == 'end':
                assert count == message['samples'], (count, message)
                take.update(request_seconds=now, audio_seconds=count/rate,
                            rtfx=(count/rate)/now, stop=message['stop'])
                print('TAKE', json.dumps(take), flush=True)
                return take
            elif message['type'] in ('error', 'fatal', 'cancelled'):
                raise RuntimeError(message)
    raise RuntimeError(f'{device}: worker stopped before end')


workers = {}
try:
    with ThreadPoolExecutor(max_workers=len(results['devices'])) as pool:
        pending = {device: pool.submit(start_worker, device) for device in results['devices']}
        for device, future in pending.items():
            workers[device] = future.result()
    persist()
    for seed in map(int, args.seeds.split(',')):
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=len(workers)) as pool:
            futures = [pool.submit(synthesize, device, worker, seed) for device, worker in workers.items()]
            takes = [f.result() for f in futures]
        elapsed = time.perf_counter()-start
        results['takes'].extend(takes)
        results.setdefault('rounds', []).append({'seed': seed, 'wall_seconds': elapsed,
                                                'aggregate_rtfx': sum(t['audio_seconds'] for t in takes)/elapsed})
        persist()
finally:
    for process, log, _ in workers.values():
        process.stdin.close()
        process.wait(timeout=120)
        log.close()
