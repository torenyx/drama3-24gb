#!/usr/bin/env python3
"""Prepare and generate an example through A, B or C."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime'))
from profiles import ROOT, PROFILES, paths_for, environment_python, converted_model


def run(command, env):
    print('+', ' '.join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), env=env, check=True)


def voice_key(assets, model):
    manifest = json.loads((assets / 'references.json').read_text(encoding='utf-8'))
    selected = {r['id']: r for r in manifest['references']}
    identity = [str(assets), manifest, str(model), model.stat().st_size, model.stat().st_mtime_ns]
    for name in manifest['baseline_references']:
        audio = assets / selected[name]['audio']
        identity.append([str(audio), audio.stat().st_size, audio.stat().st_mtime_ns])
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]


def wait_ready(url, service, timeout=1800):
    client = build_opener(ProxyHandler({}))
    start = time.perf_counter()
    next_update = 30
    while time.perf_counter() - start < timeout:
        if service.poll() is not None:
            raise RuntimeError('Service exited during startup; see service.log')
        try:
            with client.open(url + '/health', timeout=3) as response:
                if response.status == 200:
                    return time.perf_counter() - start
        except (URLError, TimeoutError):
            pass
        if time.perf_counter() - start >= next_update:
            print(f'Waiting for C startup: {next_update}s; compilation details in service.log', flush=True)
            next_update += 30
        time.sleep(2)
    raise TimeoutError('Service not ready after 1800 seconds; see service.log')


def stop_service(service):
    if service is None:
        return
    try:
        os.killpg(service.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        service.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(service.pid, signal.SIGKILL)
        service.wait(timeout=10)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', choices=PROFILES, default='c-int4')
    p.add_argument('--model-root', type=Path, default=Path('models'))
    p.add_argument('--source-dir', type=Path, help='Existing official checkpoint')
    p.add_argument('--converted-dir', type=Path, help='Existing converted checkpoint')
    p.add_argument('--assets', type=Path, default=ROOT / 'examples/librispeech')
    p.add_argument('--prompt-file', type=Path)
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--gpus', type=int, choices=[1, 2], default=2)
    p.add_argument('--seeds', default='1', help='A/B seeds; C has no request seed')
    p.add_argument('--takes', type=int, default=1, help='C repeated generations')
    p.add_argument('--python', type=Path, help='Existing profile environment')
    p.add_argument('--speech', type=Path, help='Existing speech.cpp binary')
    p.add_argument('--devices', help='B Vulkan devices, e.g. Vulkan2,Vulkan3; CUDA_VISIBLE_DEVICES does not select Vulkan')
    p.add_argument('--comfy-root', type=Path, help='Existing pinned ComfyUI checkout')
    p.add_argument('--url', help='C only: reuse a running local service')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if a.profile == 'a-fp8' and a.gpus != 2:
        p.error('The resident A profile uses two GPUs')
    if a.takes < 1:
        p.error('takes must be positive')
    paths = paths_for(a.profile, a.model_root)
    source = (a.source_dir or Path(paths['source'])).expanduser().resolve()
    converted = (a.converted_dir or Path(paths['converted'])).expanduser().resolve()
    cache = Path(paths['cache'])
    assets = a.assets.expanduser().resolve()
    prompt = (a.prompt_file or assets / 'prompt.txt').expanduser().resolve()
    out = (a.output_dir or ROOT / f'outputs/{a.profile}-{a.gpus}gpu').expanduser().resolve()
    default_python = environment_python(a.profile)
    python = str(a.python or (default_python if default_python.is_file() else Path(sys.executable)))
    speech = str(a.speech or os.environ.get('SPEECH_BIN', ROOT / 'upstream/speech-bin/speech'))
    comfy = str(a.comfy_root or os.environ.get('COMFYUI_ROOT', ROOT / 'upstream/comfyui'))
    plan = {'profile': a.profile, 'gpus': a.gpus, 'python': python, 'source': str(source),
            'converted': str(converted), 'assets': str(assets), 'prompt': str(prompt), 'output': str(out)}
    if a.dry_run:
        print(json.dumps(plan, indent=2))
        return
    if sys.platform != 'linux':
        p.error('Run inference on Linux; --dry-run also works on Windows')
    manifest = json.loads((assets / 'references.json').read_text(encoding='utf-8'))
    prompt.read_text(encoding='utf-8')
    by_id = {r['id']: r for r in manifest['references']}
    for name in manifest['baseline_references']:
        ref = by_id[name]
        if not ref['transcript'].strip() or not (assets / ref['audio']).is_file():
            p.error(f'Missing audio/transcript for {name}')
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, 'PYTHON': python, 'COMFYUI_ROOT': comfy, 'SPEECH_BIN': speech,
           'DRAMA3_ASSETS': str(assets), 'DRAMA3_MODEL': str(converted),
           'DRAMA3_REFERENCE_CACHE': str(cache / 'reference-vq'), 'DRAMA3_PROMPT': str(prompt),
           'PYTHONUNBUFFERED': '1'}
    env.setdefault('OMP_NUM_THREADS', '16')
    env.setdefault('MKL_NUM_THREADS', '16')
    env['PATH'] = str(Path(python).absolute().parent) + os.pathsep + env.get('PATH', '')
    devices = env.get('CUDA_VISIBLE_DEVICES', '0,1').split(',')
    if len(devices) < a.gpus:
        p.error('CUDA_VISIBLE_DEVICES has fewer cards than --gpus')
    env['CUDA_VISIBLE_DEVICES'] = ','.join(devices[:a.gpus])
    env['PYTHONPATH'] = str(ROOT / 'upstream/fish-speech') + os.pathsep + env.get('PYTHONPATH', '')
    start = time.perf_counter()
    record = {**plan, 'stages_seconds': {}, 'status': 'running',
              'timing': 'Wall time includes preparation, startup and client overhead; excludes prior model download.'}
    service = None
    service_log = None
    def timed(name, command):
        begin = time.perf_counter()
        run(command, env)
        record['stages_seconds'][name] = time.perf_counter() - begin
    try:
        model = converted_model(a.profile, converted)
        ready = model.is_file()
        if a.profile == 'a-fp8':
            ready = ready and (converted / 'drama3_codec.safetensors').is_file()
        if a.profile == 'c-int4':
            ready = ready and (converted / 'codec.pth').is_file()
        if not ready:
            if not source.is_dir():
                raise FileNotFoundError('Run scripts/download_models.py or pass --source-dir')
            converter = {'a-fp8': 'convert_comfy_fp8.sh', 'b-q4': 'convert_speech_gguf.sh',
                         'c-int4': 'convert_int4.sh'}[a.profile]
            timed('conversion', ['bash', ROOT / 'scripts' / converter, source, converted])
        else:
            record['conversion_reused'] = True
        if a.profile == 'a-fp8':
            env['A_MODEL'] = str(model)
            env['A_CODEC'] = str(converted / 'drama3_codec.safetensors')
            voice_cache = cache / voice_key(assets, Path(env['A_CODEC']))
            voice_cache.mkdir(exist_ok=True)
            env['DRAMA3_COMFY_VOICE_CACHE'] = str(voice_cache)
            if not all((voice_cache / f'{name}.safetensors').is_file() for name in manifest['baseline_references']):
                timed('reference_preparation', ['bash', ROOT / 'scripts/run_a.sh', 'prepare', out / 'prepare'])
            timed('runner', ['bash', ROOT / 'scripts/run_a.sh', 'run', out, a.seeds])
        elif a.profile == 'b-q4':
            voice = cache / f'{voice_key(assets, model)}.voice.gguf'
            if not voice.is_file():
                timed('reference_preparation', [python, ROOT / 'scripts/prepare_speech_voice.py',
                      '--speech', speech, '--model', model, '--assets', assets, '--out', voice])
            timed('runner', [python, ROOT / 'scripts/benchmark_speech.py', '--speech', speech,
                  '--model', model, '--voice', voice, '--prompt', prompt, '--out', out,
                  '--devices', a.devices or ','.join(f'Vulkan{i}' for i in range(a.gpus)), '--seeds', a.seeds])
        else:
            if a.url:
                url = a.url
                record['service_reused'] = True
            else:
                timed('reference_preparation', [python, ROOT / 'scripts/prepare_reference_cache.py',
                      '--model', converted, '--assets', assets, '--cache-dir', cache / 'reference-vq', '--device', 'cuda:0'])
                with socket.socket() as sock:
                    sock.bind(('127.0.0.1', 0))
                    port = sock.getsockname()[1]
                env['DRAMA3_PORT'] = str(port)
                url = f'http://127.0.0.1:{port}'
                service_log = (out / 'service.log').open('w', encoding='utf-8')
                service = subprocess.Popen(['bash', str(ROOT / 'scripts/run_c.sh'),
                    'single-cached' if a.gpus == 1 else 'dual'], env=env, stdout=service_log,
                    stderr=subprocess.STDOUT, start_new_session=True)
                record['stages_seconds']['service_ready'] = wait_ready(url, service)
            timed('runner', [python, ROOT / 'scripts/benchmark_c_chunked.py', '--assets', assets,
                  '--prompt-file', prompt, '--out', out, '--url', url, '--takes', a.takes])
        record['status'] = 'completed'
    except BaseException as exc:
        record.update(status='failed', error=str(exc))
        raise
    finally:
        record['wall_seconds'] = time.perf_counter() - start
        stop_service(service)
        if service_log is not None:
            service_log.close()
        (out / 'run.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
        print(json.dumps(record, indent=2), flush=True)
    print('WAV and metrics:', out)


if __name__ == '__main__':
    main()
