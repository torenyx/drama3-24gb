# Drama 3 on 24GB GPUs

[Simplified Chinese](README.zh-CN.md) · [Audio gallery](https://torenyx.github.io/drama3-24gb/) · [Measurements](benchmarks/README.md) · [Install & run](docs/setup.md)

Generate speech from text and a reference voice with **one or two 24GB GPUs**. Three profiles run published FP8/BF16 weights through ComfyUI, Q4_K through speech.cpp, or INT4 through SGLang-Omni.

**Tested on RTX 4090 / Linux.** A keeps its TTS model and isolated codec resident; C caches reference codes across restarts and fits streaming generation plus waveform decoding on one card.

## Choose a profile

| Your priority | Profile | GPUs | Engine |
|---|---|---:|---|
| Published expert precision | `a-fp8` (A) | 2 | ComfyUI-Drama3, FP8 experts + BF16 remainder |
| One-card native worker | `b-q4` (B) | 1; 2 for independent workers | speech.cpp 0.8.11, Q4_K GGUF |
| Fast grouped generation | `c-int4` (C) | 2; experimental 1 | SGLang-Omni, locally converted INT4 |

## Listen and compare

| Published FP8/BF16 · A | Q4_K GGUF · B | INT4 grouped · C |
|---|---|---|
| [Listen / download WAV](media/a-fp8-resident-seed1.wav) | [Listen / download WAV](media/b-q4k-seed1.wav) | [Listen / download WAV](media/c-int4-chunked-trial1.wav) |
| **28.49 s audio / 13.06 s request** | **37.48 s audio / 13.98 s request** | **26.63 s audio / 4.90 s requests** |
| 2 GPUs · seed 1 · continuous chunk history | 1 GPU · seed 1 · native worker | 2 GPUs · trial 1 · three independent requests |

RTX 4090 24GB, the same Japanese script and two reference clips (40.9 s). Timings include generation and waveform output with prepared references and warm services; they exclude startup. C's total sums the three requests before PCM concatenation. Sampling and reference processing differ. [Players, scripts and source records →](https://torenyx.github.io/drama3-24gb/)

## Run the included example

New isolated environments, the included LibriSpeech reference and converted weights:

| Profile | Output | Request | Sample-command wall time |
|---|---|---:|---:|
| A · 2 GPUs | [27.81 s WAV](media/open-a-fp8.wav) | 19.04 s | 38.49 s |
| B · 1 GPU | [26.70 s WAV](media/open-b-q4.wav) | 10.11 s | 21.97 s |
| C · 1 GPU | [23.15 s WAV](media/open-c-int4.wav) | 4.39 s | 64.16 s |

A reused encoded references; B prepared its voice from FLAC; C reused reference and compiler caches and restarted its service. C's earlier compilation run took 375.24 s for two outputs after a startup-timeout retry. [Installation records, first-build costs and transcripts →](benchmarks/release-smoke.md#isolated-installation-and-public-reference)

Linux x86_64, Python 3.12, `aria2c`, FFmpeg shared libraries, a CUDA toolkit/C++ compiler for C's first JIT build, and a CUDA-13-compatible NVIDIA driver. [Prerequisites →](docs/setup.md#prerequisites)

```bash
git clone https://github.com/torenyx/drama3-24gb.git
cd drama3-24gb
PYTHON=python3.12 bash scripts/setup.sh --profile c-int4
source .venv-c-int4/bin/activate

python scripts/download_models.py --profile c-int4 --model-root models
python scripts/sample.py --profile c-int4 --model-root models --gpus 2
```

The downloader uses the pinned ModelScope mirror by default, 16 connections per file and four concurrent files, with resume. Use `--source huggingface` for the official Hugging Face revision. **A/B/C share the same 41.5 GB official download**; each conversion adds files.

The sample command converts once, prepares the included [LibriSpeech reference and transcript](examples/librispeech/README.md), starts the selected engine, writes WAV + `metrics.json` + `run.json`, and stops its own service. Results go to `outputs/c-int4-2gpu/`. `run.json` separates reference preparation, startup and runner wall time; `metrics.json` records request timings.

Use `--gpus 1` for C's experimental one-card profile. To run A or B, install that profile and change `--profile` in the sample command; use two GPUs for A and one for B. [Existing weights, conversion sizes, your own voice and resident-service reuse →](docs/setup.md)

## Runtime changes

| Change | Measured effect |
|---|---|
| **C: persistent reference codes, lazy encoder** | Secondary GPU residency **6209 → 2629 MiB**; first-request first PCM **0.632 → 0.336 s** |
| **C: bounded streaming decode window, CPU weight-norm folding** | Engine and waveform decoder fit **one 24GB GPU**, sampled peak **24022 MiB** |
| **A: isolated codec, reuse model placement, dynamic residency floor** | Later requests in a six-request batch stay about **13–20 s**, instead of degrading to **39–63 s** |
| **A: retry a stalled chunk locally** | Restores the two omitted phrases in the reproduced seed-2 stall |

C's cache saves repeated reference work and encoder memory; measured startup remained 41.12 → 40.74 s. Quantizers and generation kernels come from upstream. [Before/after records →](benchmarks/README.md)

## Quality and experiment records

C whole-script p=0.9 had substantial omissions in 4/8 runs and long silence in 3/8. Three independent groups at p=1 had neither in the tested eight-run batch, with warm request totals 4.77–5.10 s. The grouped client implements that policy; it does not automatically transcribe or reject missing speech.

[Silence, abnormal opening, early EOS and prompt ablations](docs/failures.md) · [Native Fish Speech control](benchmarks/README.md#published-weights-with-native-fish-speech) · [CSV](benchmarks/measured-runs.csv) · [Diagnostic tools](experiments/README.md)

## Built on

[Fish Speech / Drama 3](https://github.com/fishaudio/fish-speech), [ComfyUI-Drama3](https://github.com/PedroMarinhoDev/ComfyUI-Drama3), [ComfyUI](https://github.com/Comfy-Org/ComfyUI), [speech.cpp](https://github.com/nyosegawa/speech.cpp), [SGLang-Omni](https://github.com/sgl-project/sglang-omni), and the [published INT4 converter](https://huggingface.co/fishaudio/drama-3-int4). [Pinned source revisions](sources.lock.json).

Project glue: GPL-3.0-or-later; marked adaptations retain Apache-2.0. Weights use the Fish Audio Community License. The included LibriSpeech reference uses CC BY 4.0. See [NOTICE](NOTICE).
