"""Small CPU checks for shared downloads, reusable references and public CLI paths."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from download_models import download_plan, complete
from sample import voice_key

root = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as folder:
    temp = Path(folder)
    plan = download_plan(temp, 'modelscope')
    assert len([p for p in plan if p['path'].endswith('.safetensors')]) == 8
    assert all('/resolve/ad18408' in p['url'] for p in plan)
    hf = download_plan(temp, 'huggingface')
    assert [p['destination'] for p in plan] == [p['destination'] for p in hf]
    assert all('/resolve/e0a9218' in p['url'] for p in hf)
    entry = {'destination': str(temp / 'weight'), 'size': 3}
    Path(entry['destination']).write_bytes(b'abc')
    assert complete(entry)
    Path(entry['destination'] + '.aria2').touch()
    assert not complete(entry)
    audio = temp / 'voice.wav'
    audio.write_bytes(b'old')
    model = temp / 'codec'
    model.write_bytes(b'codec')
    (temp / 'references.json').write_text(json.dumps({'baseline_references': ['voice'],
        'references': [{'id': 'voice', 'audio': 'voice.wav', 'transcript': 'Hello.'}]}))
    before = voice_key(temp, model)
    audio.write_bytes(b'new audio')
    assert voice_key(temp, model) != before
    for profile in ('a-fp8', 'b-q4', 'c-int4'):
        result = subprocess.check_output([sys.executable, str(root / 'scripts/sample.py'),
            '--profile', profile, '--model-root', str(temp), '--dry-run'], text=True)
        data = json.loads(result)
        assert data['profile'] == profile
        assert Path(data['assets'], 'references.json').is_file()
print('PASS: shared-source plans, interrupted downloads, reference invalidation and all profile CLI paths')
