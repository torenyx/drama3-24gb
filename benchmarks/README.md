# Measurements

2026-10-23, Linux, RTX 4090 24GB cards, approximately 251 GiB system RAM, no NVLink. Target text: [the measured Japanese script](../examples/prompt-ja.txt). Two reference clips, 0005 + 0051, total about 40.9 s, refer to the same speaker. Original reference audio is private and excluded.

`request_seconds` includes generation and waveform output, excluding service startup and advance reference preparation unless stated otherwise. Runs use warm OS/driver/compiler caches unless specified. The [release tests](release-smoke.md) include empty Triton-cache runs and new isolated environments.

## C reference-cache restart comparison

Two alternating process restarts per variant, three requests per restart. Same INT4 checkpoint and two-GPU placement.

| Median | Upstream reference path | Persistent-code cache hit |
|---|---:|---:|
| Service ready | 41.119 s | 40.741 s |
| First request, first PCM after ready | 0.632 s | 0.336 s |
| Later requests, first PCM | 0.196 s | 0.196 s |
| Secondary GPU resident after request | 6209 MiB | 2629 MiB |

The cache removes repeated reference encoding and encoder residency. An uncached reference incurs codec loading and encoding. [Per-restart records](records/c-cache-restarts.json)

## C single-card capacity

Cached references, one running request, streaming-only vocoder, maximum actual decode window 96 frames. The codec folds weights on CPU before transferring to GPU. Sampled card peak and post-request residency: **24022 MiB**.

| Earlier cached experiment | Audio | Request |
|---|---:|---:|
| r1 / trial 1 | 36.27 s | 6.46 s |
| r1 / trial 2 | 32.97 s | 5.88 s |
| r2 / trial 2 | 30.65 s | 5.46 s |
| r2 / after-rejection | 28.70 s | 5.12 s |

Those four samples covered the dialogue in ASR. Three other samples in that experiment omitted later dialogue. The whole-script dual-GPU controls also omitted dialogue: 3/6 in the baseline restart group, 1/6 in the cached restart group. Seeds were not fixed; these small groups do not show that the cache changes failure probability.

A later raw-reference experiment used a new empty cache per run:

| Run | Reference process | Service startup | First request | Continuous raw-reference-to-WAV | Coverage |
|---|---:|---:|---:|---:|---|
| r2 | 22.15 s | 46.88 s | 5.53 s | 74.72 s | Later paragraphs missing |
| r3 | 21.47 s | 46.88 s | 5.08 s | **73.57 s** | Dialogue covered, name spelling differs |

Across those six raw-reference requests, five had substantial omissions. [Raw workflow and ASR records](records/)

## C segmentation/sampling experiment — two GPUs

Each row is eight requests; the API does not accept seed, so these are not paired draws. Long silence means a contiguous near-silent stretch >10 s, measured using 100 ms RMS <−55 dBFS.

| References / input / sampling | Long silence | Substantial omission |
|---|---:|---:|
| Two clips, whole script, T=1, p=0.9 | 3/8 | 4/8 |
| Only 0005, whole script, T=1, p=0.9 | 7/8 | 7/8 |
| Two clips, whole script, T=1, p=1 | 0/8 | 1/8 |
| Two clips, whole script, T=0.8, p=1 | 2/8 | 2/8 |
| Two clips, three independent groups, T=1, p=1 | **0/8** | **0/8** |

The last row generates groups split at the two original `[pause]` markers and joins them with 0.15 s gaps. Each group uses the same references and starts without generated-audio history. First trial total 7.38 s; hot totals **4.77–5.10 s** for about 27 s of audio. The single-card tests used a different generation protocol.

[Grouped timings](records/c-root-chunked-p1.json) · [Coverage analysis](records/root-diagnosis-quality.json) · [Silence/EOS logits](records/c-root-logit-summary.json)

## A resident execution and continuous slowdown

Same published FP8 expert/BF16 other weights. An isolated FP32 codec process shares the second GPU. The TTS model placement is prepared once and its dynamic weight range is held resident.

| Seed, request order | Before residency floor | With residency floor |
|---|---:|---:|
| 0, first | 21.17 s | 21.92 s |
| 1 | 12.45 s | 13.06 s |
| 2, stalled-chunk retry | 38.74 s | 19.80 s |
| 3 | 61.29 s | 15.06 s |
| 4 | 62.77 s | 16.59 s |
| 5 | 51.02 s | 12.99 s |

The right column includes diagnostic counting/cProfile. All six completed without VBAR allocation failure; residency stayed at 76/114 pages. Five final WAVs matched the previous repaired PCM exactly; seed 4 differed and contained the two target phrases.

Without the floor, repeating the same seed and producing identical audio still slowed from 12.99 to 66.91 s. AIMDO's accounting grew beyond the physical card capacity while measured CUDA free memory stayed stable; its budget then evicted dynamic weights and graph replay decreased. The residency floor prevents that eviction for this fitting configuration. It does not repair the underlying accounting drift.

Earlier three-request controlled runs compared ordinary VRAM against isolated/resident decoding: seed 1 **41.59→12.45 s**, seed 2 **44.88→15.76 s**, identical corresponding PCM. The later six-request experiment above was needed to reveal and mitigate longer-run degradation.

[Continuous-run summary](records/a-service-summary.json) · [Final six requests](records/a-service-resident-floor.json) · [Earlier PCM comparison](records/a-fast-comparison.json)

## B GGUF baseline

speech.cpp 0.8.11 Vulkan, Q4_K, complete model on one GPU. Native voice file prepared beforehand. T=1, p=0.9, k=30.

| Seed | Audio | Request | First PCM | Substantial omission |
|---|---:|---:|---:|---|
| 0 | 33.48 s | 12.05 s | 0.707 s | Not found |
| 1 | 37.48 s | 13.98 s | 0.115 s | Not found |
| 2 | 39.29 s | 13.63 s | 0.123 s | Not found |

Process ready took about 9.4–10.8 s. A separate cold first request took 17.25 s to first PCM and 34.84 s total. Two independent GPU workers achieved about 5.25–5.53 aggregate audio-seconds/s in the hot two-worker test; that is throughput, not half the latency for one request. [Native worker records](records/b-single.json) · [Parallel PCM check](records/parallel-output-check.json)

## Published weights with native Fish Speech

Two backbone GPUs plus a third GPU for tower/codec. At the locked source revision, the multi-device path does not use its single-device decode CUDA graphs. Native BF16 codec/reference processing differs from A's normalized/FP32 reference path, so this is not a bit-controlled engine-only comparison.

| Seed | Request | Audio | Result |
|---|---:|---:|---|
| 1 | 75.84 s | 35.48 s | Dialogue covered |
| 0 | 93.30 s | 46.58 s | Middle chunk reaches 600 frames; two phrases missing |
| 2 | 90.74 s | 45.42 s | Same missing phrases; middle chunk reaches cap |
| 1, repeat | 71.64 s | 35.48 s | Same codes and PCM as first seed 1 |

Seed 0/2 have about 17 s of long silence; the last 96 first-codebook values of the middle chunk are identical. This reproduces a generation failure outside INT4/ComfyUI. [Native timings](records/d-native-fish-three.json) · [Native analysis](records/d-native-summary.json) · [Transcripts](records/asr-d-native-fish.json)

## Prompt ablation

Original three text boundaries, identical references and history policy, A retry disabled. Seed 0/1 contained the target phrases in all three groups. For seed 2:

| Input | Two target phrases | Middle chunk |
|---|---|---|
| Original | Missing | Repetition stall; 338 returned frames |
| Delete only `[pause]` | Present | EOS after 361 frames |
| Delete all bracketed direction/event text | Missing | Premature EOS after 226 frames |

The target phrases, “うん、いい子だね。じゃあ、聞かせて。”, remain in every input variant. [All nine trajectories and ASR](records/prompt-ablation-summary.json)
