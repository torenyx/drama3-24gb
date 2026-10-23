"""Fetch pinned source snapshots with aria2; no model weights or pip installs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile


def fetch(name, record, directory):
    destination = directory / record.get('file', name)
    if destination.exists():
        print(f'Keep existing {destination}')
        return
    with tempfile.TemporaryDirectory(dir=directory) as temporary:
        staging = Path(temporary)
        if 'repository' in record:
            repository = record['repository'].removeprefix('https://github.com/')
            url = f"https://codeload.github.com/{repository}/tar.gz/{record['commit']}"
            archive = 'source.tar.gz'
        else:
            url, archive = record['url'], record['file']
        subprocess.run(['aria2c', '--split=16', '--max-connection-per-server=16',
            '--file-allocation=none', '--auto-file-renaming=false',
            f'--dir={staging}', f'--out={archive}', url], check=True)
        if 'repository' in record:
            with tarfile.open(staging / archive) as bundle:
                bundle.extractall(staging / 'unpacked', filter='data')
            roots = list((staging / 'unpacked').iterdir())
            assert len(roots) == 1 and roots[0].is_dir(), name
            roots[0].rename(destination)
        else:
            shutil.move(staging / archive, destination)
    print(f'Ready: {destination}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', default='comfy-drama3,speech-cpp,fish-speech,int4-quantizer')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    locked = json.loads((root / 'sources.lock.json').read_text())
    names = args.only.split(',')
    for name in names:
        if name not in locked:
            parser.error(f'Unknown source {name}; choose from {", ".join(locked)}')
    directory = root / 'upstream'
    directory.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(fetch, name, locked[name], directory) for name in names]
        for job in jobs:
            job.result()


if __name__ == '__main__':
    main()
