"""Check the existing transfer manifest by file size, without rereading 41 GB."""
import argparse
import json
from pathlib import Path


def check_source(source, manifest_path):
    source = Path(source)
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    problems = []
    for item in manifest['files']:
        if item['path'] in ('.gitattributes', 'README.md', 'configuration.json', 'LICENSE'):
            continue
        path = source / item['path']
        if not path.is_file() or path.stat().st_size != item['size']:
            problems.append(item['path'])
        elif Path(str(path) + '.aria2').exists():
            problems.append(item['path'] + ' (download unfinished)')
    if problems:
        raise RuntimeError('Official model transfer is not complete: ' + ', '.join(problems))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('manifest', type=Path)
    args = parser.parse_args()
    result = check_source(args.source, args.manifest)
    print(f"Source ready: {result['repo']} / {result['revision']} ({len(result['files'])} files)")
