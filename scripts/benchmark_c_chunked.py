"""Generate independent utterance groups and join their mono PCM16 outputs."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import wave


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--prompt-file', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--url', default='http://127.0.0.1:18080')
    parser.add_argument('--takes', type=int, default=1)
    parser.add_argument('--gap', type=float, default=0.15)
    args = parser.parse_args()
    if args.takes < 1 or args.gap < 0:
        parser.error('takes must be positive and gap must be nonnegative')
    args.out.mkdir(parents=True, exist_ok=True)
    parts = [part.strip() for part in re.split(r'\[pause\]', args.prompt_file.read_text()) if part.strip()]
    if not parts:
        parser.error('The prompt has no utterances')
    records = []
    for index, part in enumerate(parts):
        if not part.startswith('<|speaker:'):
            part = '<|speaker:0|>' + part
        prompt = args.out / f'part-{index}.txt'
        prompt.write_text(part)
        directory = args.out / f'part-{index}'
        subprocess.run([sys.executable, str(Path(__file__).with_name('benchmark_sglang.py')),
            '--assets', str(args.assets), '--out', str(directory), '--url', args.url,
            '--takes', str(args.takes), '--top-p', '1.0', '--temperature', '1.0',
            '--prompt-file', str(prompt)], check=True)
        records.append(json.loads((directory / 'metrics.json').read_text()))
    summary = {'parts': parts, 'top_p': 1.0, 'temperature': 1.0, 'takes': [],
        'note': 'Independent requests with identical references, no generated-audio history; PCM16 join with gap.'}
    for trial in range(args.takes):
        pieces, expected = [], None
        for index in range(len(parts)):
            with wave.open(str(args.out / f'part-{index}/trial-{trial}.wav'), 'rb') as wav:
                actual = (wav.getnchannels(), wav.getsampwidth(), wav.getframerate())
                assert actual[:2] == (1, 2), actual
                if expected is None:
                    expected = actual
                assert actual == expected, 'PCM formats must match before concatenation'
                pieces.append(wav.readframes(wav.getnframes()))
        rate = expected[2]
        pcm = (b'\0\0' * round(rate * args.gap)).join(pieces)
        with wave.open(str(args.out / f'trial-{trial}.wav'), 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(rate)
            wav.writeframes(pcm)
        summary['takes'].append({'trial': trial,
            'request_seconds': sum(record['takes'][trial]['request_seconds'] for record in records),
            'audio_seconds': len(pcm) / 2 / rate,
            'part_request_ids': [record['takes'][trial]['headers']['x-request-id'] for record in records]})
    (args.out / 'metrics.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
