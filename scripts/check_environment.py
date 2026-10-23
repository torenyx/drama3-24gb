"""Import the modules used by each public runner before downloading model weights."""
import argparse
import importlib
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime'))
from profiles import ROOT, PROFILES

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--profile', choices=PROFILES, required=True)
a = p.parse_args()
if a.profile == 'a-fp8':
    sys.path.insert(0, os.environ.get('COMFYUI_ROOT', str(ROOT / 'upstream/comfyui')))
    modules = ['torch', 'torchaudio', 'comfy_aimdo', 'comfy_kitchen',
               'comfy.text_encoders.qwen35', 'soundfile']
elif a.profile == 'b-q4':
    sys.path.insert(0, str(ROOT / 'upstream/fish-speech'))
    modules = ['torch', 'gguf', 'soundfile',
               'fish_speech.models.text2semantic.fish_omni.generate',
               'fish_speech.models.text2semantic.inference']
    binary = os.environ.get('SPEECH_BIN', str(ROOT / 'upstream/speech-bin/speech'))
    subprocess.run([binary, '--version'], check=True)
else:
    modules = ['torch', 'torchcodec', 'dac', 'audiotools', 'drama3_runtime',
               'sglang_omni.models.fishaudio_drama3.stages']
for name in modules:
    importlib.import_module(name)
    print('OK', name, flush=True)
