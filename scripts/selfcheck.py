"""Small checks for local-source validation and offline GGUF converter routing; no ML dependencies."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from check_source import check_source
from convert_gguf_local import local_pins

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / 'modelscope-manifest.json').read_text(encoding='utf-8'))
pins_path = root / 'upstream/speech-cpp/reference/drama3/pins.py'
module = local_pins(Path('/existing/model'), manifest, pins_path)
assert module.snapshot() == str(Path('/existing/model'))
assert module.MODEL['revision'] == 'e0a9218aa48888270f06ef1ed9ea67622c0e17d2'

with tempfile.TemporaryDirectory() as folder:
    temp = Path(folder)
    sample = temp / 'model'
    sample.mkdir()
    sample.joinpath('weight.bin').write_bytes(b'abc')
    tiny = {'repo': 'example/model', 'revision': 'test', 'files': [{'path': 'weight.bin', 'size': 3}]}
    tiny_path = temp / 'tiny.json'
    tiny_path.write_text(json.dumps(tiny), encoding='utf-8')
    assert check_source(sample, tiny_path) == tiny
    sample.joinpath('weight.bin').write_bytes(b'a')
    try:
        check_source(sample, tiny_path)
    except RuntimeError:
        pass
    else:
        raise AssertionError('incomplete model accepted')

    # An import-free fake converter exercises the actual runpy route and pins injection.
    fake = temp / 'convert.py'
    fake.write_text('import pathlib, sys\nfrom pins import snapshot\n'
                    'assert pathlib.Path(snapshot()).is_dir()\n'
                    'pathlib.Path(sys.argv[1], "routed.ok").write_text("ok")\n', encoding='utf-8')
    fake.with_name('pins.py').write_text(pins_path.read_text(encoding='utf-8'), encoding='utf-8')
    output = temp / 'out'
    source = sys.argv[1] if len(sys.argv) > 1 else None
    if source:
        check_source(source, root / 'modelscope-manifest.json')
        subprocess.run([sys.executable, str(root / 'scripts/convert_gguf_local.py'), source,
                        str(output), '--upstream', str(fake)], check=True)
        assert (output / 'routed.ok').read_text() == 'ok'

print('PASS: source completeness, pinned manifest metadata, offline converter routing')
