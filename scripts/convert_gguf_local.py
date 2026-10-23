"""Run speech.cpp's unmodified converter against the model already on disk."""
import argparse
import ast
import json
import os
from pathlib import Path
import runpy
import sys
import types

from check_source import check_source


def local_pins(source, manifest, pins_path):
    module = types.ModuleType('pins')
    tree = ast.parse(pins_path.read_text(encoding='utf-8'))
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == 'MODEL' for target in node.targets))
    pinned = ast.literal_eval(assignment.value)
    entries = {item['path']: item for item in manifest['files']}
    # Compare existing manifest metadata, without hashing or downloading the checkpoint again.
    for name, (size, checksum) in pinned['files'].items():
        if entries[name]['size'] != size or entries[name]['sha256'] != checksum:
            raise RuntimeError('ModelScope manifest does not match speech.cpp pinned checkpoint: ' + name)
    module.MODEL = pinned
    module.snapshot = lambda: str(source)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--upstream', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = check_source(args.source, root / 'modelscope-manifest.json')
    source = args.source.resolve()
    output = args.output.resolve()
    converter = args.upstream or root / 'upstream/speech-cpp/reference/drama3/convert.py'
    output.mkdir(parents=True, exist_ok=True)
    # Import redirection replaces only checkpoint acquisition. All tensor operations stay upstream.
    sys.modules['pins'] = local_pins(source, manifest, converter.with_name('pins.py'))
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    sys.argv = [str(converter), str(output)]
    runpy.run_path(str(converter), run_name='__main__')
    (output / 'source-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
