#!/usr/bin/env python3
"""Create one isolated Linux environment and install its pinned upstream stack."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime'))
from profiles import ROOT, PROFILES, environment_python


def run(command, **kwargs):
    print('+', ' '.join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), check=True, **kwargs)


def install_speech():
    existing = os.environ.get('SPEECH_BIN')
    if existing and Path(existing).is_file():
        return
    target = ROOT / 'upstream/speech-bin'
    if (target / 'speech').is_file():
        return
    target.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as folder:
        run(['aria2c', '--split=16', '--max-connection-per-server=16', '--connect-timeout=15',
             '--timeout=30', '--max-tries=3', '--dir', folder,
             '--out', 'speech.zip', 'https://github.com/nyosegawa/speech.cpp/releases/download/'
             'v0.8.11/speech-0.8.11-linux-x64-vulkan.zip'])
        with zipfile.ZipFile(Path(folder) / 'speech.zip') as archive:
            archive.extractall(target)
    # The release's layout is discovered once; libraries stay beside their executable.
    binary = next(target.rglob('speech'))
    binary.chmod(0o755)
    if binary != target / 'speech':
        (target / 'speech').symlink_to(binary.relative_to(target))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=PROFILES, default='c-int4')
    args = parser.parse_args()
    if sys.platform != 'linux' or sys.version_info[:2] != (3, 12):
        parser.error('Use Linux x86_64 and Python 3.12 for these pinned environments')
    if not shutil.which('aria2c'):
        parser.error('Install aria2 first (Debian/Ubuntu: sudo apt install aria2)')
    profile = args.profile
    if profile == 'c-int4':
        nvcc = shutil.which('nvcc') or str(Path(os.environ.get('CUDA_HOME', '/usr/local/cuda')) / 'bin/nvcc')
        if not Path(nvcc).is_file() or not shutil.which('c++'):
            parser.error('C needs a CUDA toolkit (nvcc) and a C++ compiler for the first Marlin JIT build')
    python = environment_python(profile)
    run([sys.executable, '-m', 'venv', python.parent.parent])
    run([python, '-m', 'pip', 'install', 'uv==0.13.5'])
    uv = [python, '-m', 'uv', 'pip', 'install', '--python', python]
    source_names = {'a-fp8': 'comfy-drama3,comfyui',
                    'b-q4': 'speech-cpp,fish-speech',
                    'c-int4': 'sglang-omni,int4-quantizer'}[profile]
    run([sys.executable, ROOT / 'scripts/fetch_upstream.py', '--only', source_names])
    if profile == 'a-fp8':
        run(uv + ['torch==2.11.0', 'torchvision==0.26.0', 'torchaudio==2.11.0',
                  '--index-url', 'https://download.pytorch.org/whl/cu130'])
        requirements = python.parent.parent / 'comfy-headless.txt'
        requirements.write_text('\n'.join(line for line in
            (ROOT / 'upstream/comfyui/requirements.txt').read_text().splitlines()
            if not line.startswith('comfyui-')) + '\n')
        run(uv + ['-r', requirements, '-c', ROOT / 'requirements/a-fp8.lock.txt',
                  '-c', ROOT / 'requirements/a-fp8.txt', 'soundfile'])
    elif profile == 'b-q4':
        install_speech()
        run(uv + ['torch==2.8.0', 'torchaudio==2.8.0',
                  '--index-url', 'https://download.pytorch.org/whl/cu128'])
        run(uv + ['-r', ROOT / 'requirements/b-q4.txt', '-c', ROOT / 'requirements/b-q4.lock.txt',
                  '--override', ROOT / 'requirements/audio-override.txt'])
    else:
        # Use the pinned project's complete dependency declaration, including eager imports.
        run(uv + ['-e', str(ROOT / 'upstream/sglang-omni') + '[fishaudio]',
                  'ninja', '-c', ROOT / 'requirements/c-int4.lock.txt',
                  '--override', ROOT / 'requirements/c-override.txt'])
    run([python, ROOT / 'scripts/check_environment.py', '--profile', profile])
    print(f'Ready. Activate with: source {python.parent}/activate')


if __name__ == '__main__':
    main()
