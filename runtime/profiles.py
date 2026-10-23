"""Shared paths for the three public inference profiles; no ML imports."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ('a-fp8', 'b-q4', 'c-int4')


def paths_for(profile, model_root):
    root = Path(model_root).expanduser().resolve()
    return {
        'source': str(root / 'drama-3'),
        'converted': str(root / 'derived' / profile),
        'cache': str(root / 'cache' / profile),
    }


def environment_python(profile):
    return ROOT / f'.venv-{profile}' / 'bin/python'


def converted_model(profile, directory):
    directory = Path(directory)
    if profile == 'a-fp8':
        return directory / 'drama3_fp8experts_bf16.safetensors'
    if profile == 'b-q4':
        return directory / 'Drama-35B-A3B-v3-Q4_K.gguf'
    return directory / 'model.safetensors.index.json'
