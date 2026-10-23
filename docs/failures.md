# Failure diagnosis and current limits

## A: abnormal opening after model switching

The original bad seed-1 opening was reproduced by decoding the saved bad codes with the native Fish DAC; waveform correlation was 0.99999956. The fault was already in the generated codes, not just the waveform decoder.

Cold/ordinary Comfy runs and runs that generated TTS codes continuously before decoding agreed. After dynamic TTS↔Codec switching, the same first-chunk input produced a different pre-sampling distribution (KL about 4.62). Isolating Codec from the TTS model manager restored the corresponding codes, identifying model switching as the trigger.

The isolated process preserves the fast graph path. A residency floor prevents accounting-driven dynamic weight eviction while AIMDO's counter drift continues. [Measured evidence](../benchmarks/README.md#a-resident-execution-and-continuous-slowdown)

## C: repeated silence and an excluded EOS

The upstream two-GPU factory also reproduced the long silence without our single-card streaming factory. Native full offline decoding of saved generated codes retained the silence. The last 96 first-codebook values in one bad sample contained only one distinct code.

In that sample, the repeated candidate had mean probability 96.60%, and EOS about 2.98%, usually rank two. Request p=0.9 and RAS p=0.95 both excluded EOS in all 96 inspected steps. This explains why silence continued, but does not uniquely establish why the model entered it. A shorter ending alone would still leave unspoken text.

Independent short groups at T=1/p=1 avoided long silence and substantial omissions in one 8-request batch. Whole-script p=1 still omitted dialogue in 1/8. [Sampling comparison](../benchmarks/README.md#c-segmentationsampling-experiment--two-gpus)

## A: accept/retry policy for stalled chunks

The upstream path retries an incomplete opening, but accepts a later stall after enough opening speech. In the reproducible seed-2 case it samples 424 frames, detects repetition, cuts 86 tail frames and returns 338. The caller adds that incomplete audio together with the full chunk text to history and moves on, skipping “うん、いい子だね。じゃあ、聞かせて。”

`runtime/stalled_chunk.py` retries that same failed chunk once. Only the retry relaxes main-semantic and semantic-RAS sampling to k=0/p=1; acoustic codebook sampling stays unchanged. Another stall is rejected. Six tested seeds contained the two target phrases after this policy.

In the control-free prompt ablation, seed 2 omitted the phrases and emitted EOS early. This bypasses the stall detector; detecting such omissions requires comparing the transcript with the target text.

## Prompt syntax and reference conditioning

The [Fish Audio guide](https://docs.fish.audio/features/text-to-speech/drama-3) supports speaker markers, bracketed natural-language directions, directions in any language, and bracketed sounds/pauses. Our script fits that syntax. A valid input can nevertheless trigger different generation trajectories.

The [native Fish Speech full test](../benchmarks/README.md#published-weights-with-native-fish-speech) also omitted the same phrases. Other users report related failures with published weights on [H100](https://github.com/fishaudio/fish-speech/issues/1356) and [H200](https://github.com/sgl-project/sglang-omni/issues/2823); the [vLLM-Omni recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/fishaudio/Drama-3.md) documents silence after an opening tag/filler and sensitivity to pause-heavy references.

These observations do not isolate reference length, prosody and generated-history effects from each other. Our own one-reference ablation did not solve the C failure. [Nine-input-variant records](../benchmarks/records/prompt-ablation-summary.json)

## C: first-request compiler allocation

The release runner was also tested with an empty Triton cache. The single-card engine loaded, but its first request failed when the autotuner requested a 256 MiB benchmark buffer. Reducing the engine memory fraction either left too few tokens for the reference plus generation budget or still failed the allocation. Restoring 0.87 and enabling PyTorch expandable segments completed all three groups. This points to allocator pressure/fragmentation during compilation; it is separate from silent or omitted speech. The runner now enables expandable segments by default. [Cold-compiler timings](../benchmarks/release-smoke.md)

## Measurement conditions

- ASR is used for phrase/paragraph coverage. Kana/kanji/name differences and small words need review; CER is not voice similarity.
- A/B/C use different engines, sampling and reference processing. Their audible differences cannot be assigned only to quantization precision.
- C single/dual reliability was not compared with identical segmentation and paired draws. GPU count has not been established as the cause of omissions.
- Peak single-card residency is close to capacity. Non-streaming payloads are rejected by the low-memory factory; this is a one-request experimental configuration.
- RTX 3090 performance and an old-driver-compatible software stack have not been validated.
