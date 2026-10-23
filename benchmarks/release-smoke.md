# Public-runner smoke tests

2026-10-23, Linux, RTX 4090 24GB. The first batch uses prepared environments and checkpoints. A and B ran concurrently on separate cards; C ran afterward. The isolated-environment batch follows below.

| Public entry point | Placement | WAV produced | Request time | Output duration |
|---|---|---|---:|---:|
| `run_a.sh prepare` then `run_a.sh run` | Two GPUs; isolated codec | Yes | 25.294 s | 27.132 s |
| `benchmark_speech.py` | One Vulkan GPU | Yes | 13.568 s | 37.477 s |
| `run_c.sh single-cached` + grouped client | One GPU | Yes | 19.887 s | 25.981 s |
| `run_c.sh dual` + grouped client | Two GPUs | Yes | 19.698 s | 26.771 s |

C used an empty Triton cache for each placement, cached reference codes, three independent requests at T=1/p=1 and the unchanged 1200-token generation limit. The first request includes compilation. These totals are distinct from the earlier 4.77–5.10 s warm grouped measurements. Neither model startup nor prior reference encoding is included in request time.

The first single-card attempt failed at Triton's 256 MiB autotuning buffer with the default PyTorch allocator. Lowering the engine budget either reduced available context below the request budget or still failed allocation. The shipped default keeps the original 0.87 engine fraction and enables `PYTORCH_ALLOC_CONF=expandable_segments:True`; the empty-cache test then completed.

These release samples were not separately transcribed or scored for speaker similarity. They establish that the exported entry points produce WAV output. The existing measured quality/failure batches remain in the [benchmark report](README.md).

Records: [A](records/release-smoke-a.json), [B](records/release-smoke-b.json), [C single](records/release-smoke-c-single.json), [C dual](records/release-smoke-c-dual.json). CPU cache fixtures and source-conversion self-checks also passed; shell entry points passed Bash syntax checks.

## Isolated installation and public reference

The public installer created separate `.venv-a-fp8`, `.venv-b-q4` and `.venv-c-int4` environments on the same Linux/RTX 4090 host. All three passed their module-import checks and generated the included [LibriSpeech example](../examples/librispeech/README.md) through `scripts/sample.py`. Pinned source snapshots and converted model weights were reused. Installed dependency versions are recorded in [requirements](../requirements/).

| Profile | GPUs | Request | Audio | Sample-command wall time | Preparation state |
|---|---:|---:|---:|---:|---|
| A · FP8/BF16 | 2 | 19.042 s | 27.814 s | 38.492 s | Cached reference; model reloaded |
| B · Q4_K | 1 | 10.106 s | 26.703 s | 21.973 s | Raw FLAC converted to WAV and encoded on CPU; worker started |
| C · INT4 | 1 | 4.391 s | 23.148 s | 64.156 s | Cached reference/kernels; service restarted |

Each row produces one final WAV. Request time covers generation and waveform output; wall time includes reference preparation, imports, startup and runner overhead, and begins after installation/download/conversion. C's cached-reference preparation still spends 13.297 s on process imports; service startup takes 46.010 s and the grouped runner takes 4.840 s.

### C first compilation and restart

The new C environment hit the upstream 600 s startup timeout during CUDA kernel compilation. The public runner now defaults to `SGLANG_OMNI_STARTUP_TIMEOUT=1800` and the client waits up to 1800 s. The successful retry reused partial compiler output, took 330.055 s to start the service and 375.245 s overall for two WAVs. Request totals were 26.595 s and 4.814 s. The table above records the subsequent one-WAV run with those compiler caches available.

The B test initially failed because the native voice encoder accepts WAV input. Reference preparation now converts other supported audio formats to PCM16 WAV at their original sample rate.

### Transcript coverage

Qwen3-ASR covered every target sentence in A, B and all three C outputs. A/B and C's first compilation output matched the normalized target text. The later C outputs contain short ASR differences: “the reference” → “their reference” and “this script” → “the script”. The displayed C sample is the final one-WAV run, with normalized character error rate 0.5865%.

Records: [A timing](records/open-a-metrics.json), [A workflow](records/open-a-run.json), [B timing](records/open-b-metrics.json), [B workflow](records/open-b-run.json), [C timing](records/open-c-metrics.json), [C workflow](records/open-c-run.json), [C compilation timing](records/open-c-build-metrics.json), [C compilation workflow](records/open-c-build-run.json), [all transcripts](records/open-asr.json).

Validation passed: public CLI/download-plan checks, reference-cache restart/invalidation fixtures and Bash syntax for the installer and changed launch/conversion scripts.
