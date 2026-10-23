# Install and run

## Prerequisites

Linux x86_64, **Python 3.12**, Git, aria2, FFmpeg shared libraries and a GPU driver compatible with the chosen Torch build. A/C use CUDA 13; B uses a Vulkan binary and a separate Torch 2.8 conversion environment. Measurements use RTX 4090 24GB. RTX 3090 is not validated with this software stack.

On a Debian/Ubuntu system with Python 3.12:

```bash
sudo apt install aria2 ffmpeg libsndfile1 libvulkan1 python3.12-venv build-essential
```

The measured C environment uses FFmpeg 8.1. If its shared libraries are outside standard loader paths, set `LD_LIBRARY_PATH=/path/to/ffmpeg/lib:$LD_LIBRARY_PATH`. `check_environment.py` imports TorchCodec so a missing FFmpeg runtime is found before downloading model weights.

C also needs a CUDA toolkit with `nvcc` (in `PATH`, `$CUDA_HOME/bin`, or `/usr/local/cuda/bin`) and a C++ compiler. The installer supplies Ninja. The first Marlin kernel build can take several minutes on a fresh compiler cache; later launches reuse it. The approximately 41 s startup measurements use already compiled kernels.

The server and unified client allow 30 minutes for startup (`SGLANG_OMNI_STARTUP_TIMEOUT=1800`). A fresh isolated C environment exceeded the upstream 600 s limit during kernel compilation; the successful retry reused partial compiler output. See the [isolated installation results](../benchmarks/release-smoke.md#isolated-installation-and-public-reference) for measured startup and request costs.

## Install a profile

```bash
PYTHON=python3.12 bash scripts/setup.sh --profile c-int4
source .venv-c-int4/bin/activate
```

Each profile has its own isolated `.venv-<profile>`. The installer fetches fixed source commits from [sources.lock.json](../sources.lock.json), installs that profile's dependencies, and checks imports. It does not change an existing scientific-computing environment.

| Profile | Installed stack |
|---|---|
| `a-fp8` | Torch/TorchAudio 2.11.0 CUDA 13, TorchVision 0.26.0; pinned ComfyUI 0.42.0 headless requirements, AIMDO 0.5.6, Comfy Kitchen 0.2.40 |
| `b-q4` | speech.cpp 0.8.11 Linux Vulkan binary; Torch/TorchAudio 2.8.0 CUDA 12.8 and the Fish Speech conversion modules |
| `c-int4` | SGLang-Omni at `1e8dcd26` with its `fishaudio` extra: Torch 2.14.1, SGLang 0.5.22, Transformers 5.19.0 and audio dependencies |

A omits ComfyUI frontend/workflow/document bundles because these runners are headless. B can reuse a binary by setting `SPEECH_BIN=/path/to/speech` before installation. C follows the pinned upstream package declaration, including its eager-import dependencies. UV's protobuf override handles the old audiotools dependency pin. The `requirements/*.lock.txt` files pin the packages installed in the isolated verification environments; the installer applies them as constraints. See [requirements](../requirements/).

For another profile, run the same installer with `--profile a-fp8` or `--profile b-q4`, then activate the corresponding environment. A/C need network access to Python/CUDA wheel indexes; source/binary downloads also need GitHub access.

## Download once for all profiles

```bash
python scripts/download_models.py --profile c-int4 --model-root /models/drama3 --dry-run
python scripts/download_models.py --profile c-int4 --model-root /models/drama3
```

Default source: [ModelScope](https://modelscope.cn/models/fishaudio/drama-3), revision `ad18408cd49d5a6176b9ea109311648e67c993e0`. Use `--source huggingface` for [Fish Audio](https://huggingface.co/fishaudio/drama-3), revision `e0a9218aa48888270f06ef1ed9ea67622c0e17d2`. The downloaded files have pinned sizes in the included manifest. The downloader resumes with **16 connections per file, four files in parallel** and keeps existing complete files. It checks transfer size/completion without rereading all weights to hash them.

A/B/C all reuse `/models/drama3/drama-3`. `--profile` selects the saved paths file, not another official model download. Already have the original checkpoint? Skip downloading and pass `--source-dir /existing/drama-3` to the sample command.

## Run the included reference-to-WAV sample

```bash
python scripts/sample.py --profile c-int4 --model-root /models/drama3 --gpus 2
# One-card streaming:
python scripts/sample.py --profile c-int4 --model-root /models/drama3 --gpus 1
```

For A/B, activate their environment and run:

```bash
python scripts/sample.py --profile a-fp8 --model-root /models/drama3 --gpus 2
python scripts/sample.py --profile b-q4 --model-root /models/drama3 --gpus 1
```

The command reuses converted weights if present. On first use it runs the upstream conversion wrapper, encodes the reference, starts inference, saves WAV/metrics, and shuts down the service it started. A/B reuse prepared voices by reference/codec identity; C uses the content/configuration-aware persistent VQ cache.

Default input: [LibriSpeech reference, exact transcript and English target script](../examples/librispeech/README.md). The older Japanese benchmark uses separate private references.

| Output | Meaning |
|---|---|
| `outputs/<profile>-<gpus>gpu/*.wav` | Final waveform(s) |
| `metrics.json` | Engine request timing and generated duration |
| `run.json` | Preparation, startup and runner wall time; conversion reuse flag |
| `service.log` | C service startup/inference diagnostics |

`--output-dir` changes the output folder. A/B accept `--seeds 0,1,2`; C accepts `--takes 3` and has no upstream request seed. Repeating C starts a fresh service unless `--url` is supplied. A request time is not a cold reference-to-WAV total.

## Existing converted weights and devices

```bash
python scripts/sample.py --profile c-int4 --gpus 2 \
  --model-root /models/drama3 --converted-dir /existing/int4
```

The converted directory must contain the profile's standard filenames: A's `drama3_fp8experts_bf16.safetensors` plus `drama3_codec.safetensors`; B's `Drama-35B-A3B-v3-Q4_K.gguf`; or C's index, shards, tokenizer/config and codec.

Use `--python /existing/env/bin/python`, `--comfy-root /existing/ComfyUI` or `--speech /existing/speech` to reuse prepared software.

For A/C, select cards with `CUDA_VISIBLE_DEVICES=2,3`. B uses Vulkan enumeration: pass `--devices Vulkan2,Vulkan3 --gpus 2 --seeds 0,1` to run two independent workers. CUDA visibility does not select Vulkan cards. Use `speech devices` to inspect the native device list.

## Your own voice and text

Create a directory with `references.json`, the audio file(s) and `prompt.txt`, using [the reference manifest](../examples/references.example.json). Supply the exact transcript for each selected clip. Multiple entries in `baseline_references` condition the same speaker 0 in these runners.

```bash
python scripts/sample.py --profile c-int4 --gpus 2 --model-root /models/drama3 \
  --assets /path/to/your/assets --prompt-file /path/to/script.txt
```

C splits the script at literal `[pause]` markers, generates independent requests at T=1/p=1 and joins PCM with a 0.15 s gap. With no marker it sends one group. This trades generated-history continuity for the segmentation policy that worked better in the measured batch. It does not transcribe or automatically retry missing words.

## Keep C resident for repeated requests

```bash
export DRAMA3_MODEL=/models/drama3/derived/c-int4
export DRAMA3_ASSETS="$PWD/examples/librispeech"
export DRAMA3_REFERENCE_CACHE=/models/drama3/cache/c-int4/reference-vq
python scripts/prepare_reference_cache.py --model "$DRAMA3_MODEL" \
  --assets "$DRAMA3_ASSETS" --cache-dir "$DRAMA3_REFERENCE_CACHE" --device cuda:0
CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_c.sh dual
# Other terminal, same environment and files:
python scripts/sample.py --profile c-int4 --model-root /models/drama3 --gpus 2 \
  --url http://127.0.0.1:18080 --takes 3
```

The client leaves an existing service running. Single-card mode uses `run_c.sh single-cached`, prepared references, streaming only and one concurrent request. It has little memory headroom. `run_c.sh` keeps Triton/FlashInfer caches and defaults to expandable PyTorch allocation for cold autotuning. [Detailed runtime behavior](runtime.md)

## Conversion space and checks

Official source: about **41.5 GB**. A adds about **38.6 GB + 1.6 GB codec**. B retains a **143.1 GB F32 intermediate** plus a **20.8 GB Q4_K file**. C adds its INT4 checkpoint; retain the source if you want to convert the other profiles. Existing conversions can be reused without that temporary disk cost.

```bash
python scripts/test_public_cli.py
python scripts/check_environment.py --profile c-int4
python scripts/test_reference_cache.py  # C environment; CPU fixtures
```

[Published measurements](../benchmarks/README.md) · [Public-runner checks](../benchmarks/release-smoke.md) · [Diagnostic tools](../experiments/README.md)
