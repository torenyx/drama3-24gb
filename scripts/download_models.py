#!/usr/bin/env python3
"""Download the shared official checkpoint with aria2; conversions reuse it."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime'))
from profiles import ROOT, PROFILES, paths_for

HF_REVISION = 'e0a9218aa48888270f06ef1ed9ea67622c0e17d2'


def download_plan(root, source):
    manifest = json.loads((ROOT / 'modelscope-manifest.json').read_text(encoding='utf-8'))
    plan = []
    for entry in manifest['files']:
        name = entry['path']
        # ModelScope-specific metadata and repository documentation are not model inputs.
        if name in ('.gitattributes', 'README.md', 'configuration.json'):
            continue
        if source == 'modelscope':
            url = (f"https://modelscope.cn/models/{manifest['repo']}/resolve/"
                   f"{manifest['revision']}/{quote(name)}")
        else:
            url = f"https://huggingface.co/{manifest['repo']}/resolve/{HF_REVISION}/{quote(name)}"
        plan.append({**entry, 'destination': str(Path(root) / 'drama-3' / name), 'url': url})
    return plan


def complete(entry):
    path = Path(entry['destination'])
    return (path.is_file() and path.stat().st_size == entry['size']
            and not Path(str(path) + '.aria2').exists())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=PROFILES, default='c-int4')
    parser.add_argument('--model-root', type=Path, default=Path('models'))
    parser.add_argument('--source', choices=['modelscope', 'huggingface'], default='modelscope')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    root = args.model_root.expanduser().resolve()
    plan = download_plan(root, args.source)
    pending = [entry for entry in plan if not complete(entry)]
    print(f"Shared official weights: {sum(e['size'] for e in plan)/1e9:.2f} GB; "
          f"remaining {sum(e['size'] for e in pending)/1e9:.2f} GB")
    print('A/B/C share this download. Conversion adds separate files.')
    if args.dry_run:
        for entry in plan:
            print(f"{entry['size']:>13}  {'have' if complete(entry) else 'need'}  {entry['url']}")
        return
    if pending:
        if not shutil.which('aria2c'):
            parser.error('Install aria2 (aria2c) first')
        with tempfile.TemporaryDirectory(prefix='drama3-download-') as folder:
            listing = Path(folder) / 'files.txt'
            lines = []
            for entry in pending:
                target = Path(entry['destination'])
                target.parent.mkdir(parents=True, exist_ok=True)
                lines.append(f"{entry['url']}\n  dir={target.parent}\n  out={target.name}\n")
            listing.write_text('\n'.join(lines), encoding='utf-8')
            subprocess.run(['aria2c', '--input-file', str(listing), '--continue=true',
                            '--split=16', '--max-connection-per-server=16', '--min-split-size=1M',
                            '--max-concurrent-downloads=4', '--auto-file-renaming=false',
                            '--file-allocation=none', '--summary-interval=10'], check=True)
        if any(not complete(entry) for entry in plan):
            raise RuntimeError('Download incomplete; rerun to resume')
    root.mkdir(parents=True, exist_ok=True)
    config = {**paths_for(args.profile, root), 'download_source': args.source,
              'hf_revision': HF_REVISION}
    (root / f'{args.profile}.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print(f"Ready. Run: python scripts/sample.py --profile {args.profile} --model-root {root}")


if __name__ == '__main__':
    main()
