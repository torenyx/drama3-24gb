"""Project-owned Drama3 reference cache; upstream inference and sampling stay intact."""
from __future__ import annotations

import hashlib
import inspect
import json
import logging
import os
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tokenizers import Tokenizer

from sglang_omni.models.fishaudio_drama3.hf_config import FishOmniConfig
from sglang_omni.models.fishaudio_drama3.speech_request import Drama3SpeechPreprocessor
from sglang_omni.models.fishaudio_drama3.stages import resolve_codec_directory
from sglang_omni.models.fishaudio_s2_pro.stages import FishReferenceEncodeHook, load_codec
from sglang_omni.scheduling.reference_encoder import ReferenceEncodeService
from sglang_omni.scheduling.threaded_simple_scheduler import ThreadedSimpleScheduler
from sglang_omni.utils.checkpoint import resolve_checkpoint
from sglang_omni.utils.device import resolve_concrete_device

logger = logging.getLogger(__name__)


def file_identity(path):
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return [str(resolved), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


class CachedFishReferenceHook(FishReferenceEncodeHook):
    """Keep the original reference arithmetic, persisting codes instead of the codec.

    A restart with disk hits never loads the preprocessing codec onto the GPU.
    The upstream in-memory service still handles repeated requests and concurrency.
    """

    def __init__(self, *, codec_directory, cache_dir, device, autocast_dtype,
                 max_reference_audio_seconds, cache_only=False):
        self.codec_directory = str(codec_directory)
        self.disk_dir = Path(cache_dir)
        self.disk_dir.mkdir(parents=True, exist_ok=True)
        self.codec_loaded = False
        self.cache_only = cache_only
        self.codec_load_lock = threading.Lock()
        # The pinned Fish DAC contract is 44.1 kHz; no weights are needed for a cache hit.
        super().__init__(codec=SimpleNamespace(sample_rate=44100),
                         checkpoint_id=str(Path(codec_directory) / 'codec.pth'),
                         device=device, autocast_dtype=autocast_dtype,
                         max_reference_audio_seconds=max_reference_audio_seconds)
        identity = {'format': 1, 'sample_rate': 44100, 'dtype': str(autocast_dtype),
                    'device_type': self.device.type,
                    'compute_capability': (torch.cuda.get_device_capability(self.device)
                                           if self.device.type == 'cuda' else None),
                    'torch': torch.__version__, 'codec': file_identity(self.model_revision),
                    'encoder_source': file_identity(inspect.getfile(FishReferenceEncodeHook)),
                    'max_reference_audio_seconds': float(max_reference_audio_seconds)}
        self.encoder_config_hash = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()).hexdigest()

    def encode_reference_waveform(self, audio, sr):
        if not self.codec_loaded:
            with self.codec_load_lock:
                if not self.codec_loaded:
                    start = time.perf_counter()
                    self.codec = load_codec(self.codec_directory, str(self.device))
                    self.codec_loaded = True
                    logger.info('DRAMA3_CORE codec_lazy_load seconds=%.6f', time.perf_counter() - start)
        return super().encode_reference_waveform(audio, sr)

    def encode_one(self, item):
        key = self.cache_key(item)
        if key is None:
            return super().encode_one(item)
        serialized = key.to_string()
        filename = hashlib.sha256(serialized.encode()).hexdigest() + '.safetensors'
        path = self.disk_dir / filename
        start = time.perf_counter()
        if path.exists():
            with safe_open(path, framework='pt', device='cpu') as stored:
                if stored.metadata().get('key') != serialized:
                    raise ValueError(f'Reference cache identity mismatch: {path}')
                codes = stored.get_tensor('codes').long()
            logger.info('DRAMA3_CORE reference_disk_hit frames=%d seconds=%.6f file=%s',
                        codes.shape[-1], time.perf_counter() - start, filename)
            return codes
        if self.cache_only:
            raise ValueError('Reference cache miss: prepare this voice with prepare_reference_cache.py first')
        codes = super().encode_one(item)
        # Revalidate after encoding so an edited source cannot be cached under its old key.
        if not self.revalidate(item, key):
            raise RuntimeError('Reference audio changed during encoding; retry with a stable file')
        fd, temporary = tempfile.mkstemp(prefix=filename + '.', dir=self.disk_dir)
        os.close(fd)
        try:
            save_file({'codes': codes.to(torch.int16).contiguous()}, temporary,
                      metadata={'key': serialized, 'format': 'drama3-reference-v1'})
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
        logger.info('DRAMA3_CORE reference_disk_miss frames=%d seconds=%.6f file=%s',
                    codes.shape[-1], time.perf_counter() - start, filename)
        return codes


def create_preprocessing_executor(model_path, *, cache_dir, device=None, gpu_id=None,
                                  max_concurrency, max_prompt_rows, codec_path,
                                  reference_encode_dtype, max_reference_audio_seconds,
                                  cache_only=False):
    resolved_device = str(resolve_concrete_device(device, gpu_id))
    if resolved_device == 'cpu' and reference_encode_dtype == 'bfloat16':
        raise ValueError('CPU reference encoding requires float32')
    checkpoint = resolve_checkpoint(model_path)
    hook = CachedFishReferenceHook(
        codec_directory=resolve_codec_directory(codec_path), cache_dir=cache_dir,
        device=resolved_device,
        autocast_dtype=torch.bfloat16 if reference_encode_dtype == 'bfloat16' else None,
        max_reference_audio_seconds=max_reference_audio_seconds, cache_only=cache_only)
    preprocessor = Drama3SpeechPreprocessor(
        tokenizer=Tokenizer.from_file(os.path.join(checkpoint, 'tokenizer.json')),
        config=FishOmniConfig.from_pretrained(checkpoint), max_prompt_rows=max_prompt_rows,
        reference_encoder=ReferenceEncodeService(hook, log_prefix='Drama3 persistent reference'))
    return ThreadedSimpleScheduler(preprocessor, max_concurrency=max_concurrency)
