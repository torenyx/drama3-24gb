"""Small CPU checks for persistence, source invalidation, precision and lazy loading."""
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime'))
import torch
from drama3_runtime import CachedFishReferenceHook, FishReferenceEncodeHook


with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    (root / 'codec.pth').write_bytes(b'codec fixture')
    audio = root / 'reference.wav'
    audio.write_bytes(b'original fixture')
    codes = torch.arange(40).reshape(10, 4)

    def hook(dtype=None, seconds=60):
        return CachedFishReferenceHook(codec_directory=root, cache_dir=root / 'cache',
            device='cpu', autocast_dtype=dtype, max_reference_audio_seconds=seconds)

    with patch.object(FishReferenceEncodeHook, 'encode_one', return_value=codes) as encode:
        first = hook()
        item = first.normalize_input({'audio_path': str(audio)})
        assert torch.equal(first.encode_one(item), codes)
        restarted = hook(seconds=60.0)
        assert torch.equal(restarted.encode_one(item), codes)
        assert encode.call_count == 1, 'Restart must reuse persisted codes'
        assert not restarted.codec_loaded, 'Cache hits must not load the codec'
        audio.write_bytes(b'edited audio fixture with a different size')
        assert torch.equal(restarted.encode_one(item), codes)
        assert encode.call_count == 2, 'Edited references must not reuse stale codes'
        assert torch.equal(hook(torch.bfloat16).encode_one(item), codes)
        assert encode.call_count == 3, 'Reference arithmetic must be part of cache identity'
        (root / 'codec.pth').write_bytes(b'changed codec fixture')
        assert torch.equal(hook().encode_one(item), codes)
        assert encode.call_count == 4, 'Changed codec must invalidate the cache'
    print('PASS: restart reuse, lazy codec, audio/precision/codec invalidation')
