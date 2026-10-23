# SPDX-License-Identifier: Apache-2.0
# Factory adapted from SGLang-Omni Drama 3 stages.py (Apache-2.0).
"""Size a streaming-only vocoder for its bounded windows, not a whole utterance."""
from sglang_omni.models.fishaudio_drama3 import stages
from sglang_omni.models.fishaudio_drama3.vocoder import Drama3VocoderScheduler


class StreamingOnlyVocoderScheduler(Drama3VocoderScheduler):
    def validate_non_streaming_payload(self, payload):
        raise ValueError('This low-memory vocoder requires streaming audio requests')


def create_streaming_vocoder_executor(
    *, device=None, gpu_id=None, codec_path, stream_stride,
    stream_followup_stride_ramp, stream_followup_stride, stream_overlap_tokens,
    stream_crossfade_samples, max_batch_size, max_batch_wait_ms,
    window_cuda_graphs, decode_dtype, max_new_tokens,
    total_gpu_memory_fraction=None,
):
    if max_batch_size != 1:
        raise ValueError('The low-memory streaming configuration requires max_batch_size=1')
    resolved_device = str(stages.resolve_concrete_device(device, gpu_id))
    torch_device = stages.torch.device(resolved_device)
    # Fold on the CPU before transfer; GPU folding briefly holds both weight forms.
    codec = stages.load_codec(stages.resolve_codec_directory(codec_path), 'cpu')
    codec.fold_weight_norm()
    codec = codec.to(torch_device)
    num_codebooks = codec.quantizer.semantic_quantizer.n_codebooks + codec.quantizer.quantizer.n_codebooks
    decode = (stages.ChannelsLastDecoder(codec, dtype=stages.torch.float16)
              if decode_dtype == 'float16' else codec.from_indices)
    window_lengths = stages.stream_window_lengths(
        stream_stride=stream_stride, stream_followup_stride_ramp=stream_followup_stride_ramp,
        stream_followup_stride=stream_followup_stride, stream_overlap_tokens=stream_overlap_tokens)
    decode_window = stages.WindowGraphDecoder(
        decode, num_codebooks=num_codebooks, frame_length=int(codec.frame_length),
        device=torch_device, window_lengths=window_lengths) if window_cuda_graphs else decode
    # This scheduler admits one window at a time. Probe that actual shape;
    # the upstream two-row full-utterance probe is outside this configuration.
    capacity = max(window_lengths)
    with stages.torch.inference_mode():
        decode_window(stages.torch.zeros((1, num_codebooks, capacity),
                                        dtype=stages.torch.long, device=torch_device))
    stages.torch.cuda.synchronize(torch_device)
    stages.logger.info('DRAMA3_CORE streaming_window_budget max_window=%d frame_capacity=%d generation_limit=%d',
                       max(window_lengths), capacity, max_new_tokens)
    return StreamingOnlyVocoderScheduler(
        codec, device=resolved_device, stream_stride=stream_stride,
        stream_followup_stride_ramp=stream_followup_stride_ramp,
        stream_followup_stride=stream_followup_stride, stream_overlap_tokens=stream_overlap_tokens,
        stream_crossfade_samples=stream_crossfade_samples, max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms, decode_window=decode_window,
        max_padded_frames_per_decode=capacity)
