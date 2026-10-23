# Runtime details and low-level entry points

The GPU measurements were made on Linux, Python 3.12 and RTX 4090 24GB cards. The runners target an already prepared environment; they do not modify a system Python or install packages automatically. The public installer and unified example are described in [setup.md](setup.md).

## Tested software

| Route | Runtime |
|---|---|
| A | ComfyUI 0.42.0, Torch 2.11.0+cu130, comfy-aimdo 0.5.6, comfy-kitchen 0.2.40 |
| B | speech.cpp 0.8.11, Vulkan GPU backend |
| C | SGLang-Omni `1e8dcd26`, SGLang 0.5.22, Torch 2.14.1+cu130; torchaudio 2.11.0+cu130 in the measured environment |
| Native comparison | Fish Speech `54156693`, Torch 2.11.0+cu130 |

Use the matching upstream installation documentation: [ComfyUI](https://github.com/Comfy-Org/ComfyUI), [ComfyUI-Drama3](https://github.com/PedroMarinhoDev/ComfyUI-Drama3), [speech.cpp](https://github.com/nyosegawa/speech.cpp), [SGLang-Omni](https://github.com/sgl-project/sglang-omni). A and C were kept in separate Python environments. The recorded mixed Torch/torchaudio versions describe the test environment, not a general compatibility recommendation.

C also needs the upstream audio/FFmpeg dependencies. If your FFmpeg shared libraries are outside the loader search path, set `LD_LIBRARY_PATH=/path/to/ffmpeg/lib:$LD_LIBRARY_PATH`. CUDA 13 needs a compatible NVIDIA driver. The older driver on our 3090 node could not run this stack, so no 3090 speed is claimed.

## Fetch source, not weights

Install `aria2c` in the environment or OS, then:

```bash
python scripts/fetch_upstream.py
# Optionally fetch the matching framework checkouts too:
python scripts/fetch_upstream.py --only comfyui,sglang-omni
```

Source snapshots use 16 connections per transfer, with at most four transfers in parallel. Existing directories are kept. `sources.lock.json` pins every fetched revision. Upstream code and its license files remain in ignored `upstream/` directories.

The speech.cpp archive is sufficient for conversion, but does not include its ggml submodule for a source build; use the upstream installation instructions or the 0.8.11 binary release.

## Model preparation

Acquire [fishaudio/drama-3](https://huggingface.co/fishaudio/drama-3) under its model license. The measured source revision was `e0a9218aa48888270f06ef1ed9ea67622c0e17d2`. The matching file manifest is included. Conversion wrappers check sizes and recorded metadata without rereading all weights to compute hashes.

Our original files were downloaded directly from the [ModelScope mirror](https://modelscope.cn/models/fishaudio/drama-3), revision `ad18408cd49d5a6176b9ea109311648e67c993e0`. That mirror revision has a different identifier from the Hugging Face revision; the recorded weight metadata is checked against the converter pins.

```bash
# A: in the ComfyUI Python environment
PYTHON=python bash scripts/convert_comfy_fp8.sh \
  /models/drama-3 /models/drama-3-comfy /path/to/ComfyUI

# B: in the upstream speech.cpp converter Python environment
SPEECH_BIN=/path/to/speech PYTHON=python bash scripts/convert_speech_gguf.sh \
  /models/drama-3 /models/drama-3-q4

# C: environment supporting the published INT4 quantizer
PYTHON=python bash scripts/convert_int4.sh /models/drama-3 /models/drama-3-int4
```

The conversions are upstream implementations. Our GGUF adapter redirects checkpoint acquisition to the files already on disk. A preserves FP8 expert bytes and uses BF16 for other layers. C quantizes routed experts and attention/GDN dense layers using the published converter, with `--dense attn,gdn`.

The official source occupies about 41.5 GB. A produces about 38.6 GB of model weights plus 1.6 GB codec; B uses a 143.1 GB F32 intermediate and produces a 20.8 GB Q4_K file. Plan disk space accordingly. Existing converted checkpoints can be used directly; do not reconvert just to run a benchmark.

## Your reference assets

Create an asset directory containing:

```text
assets/
  references.json
  reference.wav
  prompt.txt
```

Copy `examples/references.example.json` to `references.json` and replace the transcript with the exact spoken words. For multiple clips of the same speaker, add entries and include their IDs in `baseline_references`. A explicitly binds the selected references to speaker 0. Use `<|speaker:0|>` for that speaker in the target script.

`examples/prompt-ja.txt` contains the original measured Japanese script. It has two `[pause]` markers. The C grouped client consumes those markers as split points; whole-script clients send them to the model. The original 0005/0051 recordings are not included, so reproducing the exact measured voice requires the original private recordings. A new reference runs the same procedure but is not the identical benchmark input.

## A: prepare references, then keep TTS and codec resident

Activate the prepared A environment:

```bash
export COMFYUI_ROOT=/path/to/ComfyUI
export A_MODEL=/models/drama-3-comfy/drama3_fp8experts_bf16.safetensors
export A_CODEC=/models/drama-3-comfy/drama3_codec.safetensors
export DRAMA3_ASSETS=/path/to/assets
export DRAMA3_PROMPT="$DRAMA3_ASSETS/prompt.txt"

CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_a.sh prepare outputs/a
CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_a.sh run outputs/a 0,1,2
```

Reference codes are stored beside the output directory in `voice-cache-comfy/`; this A cache uses reference IDs and transcript checks. When replacing audio under the same ID, rerun `prepare`. C has the more complete content/configuration cache identity described below.

The generation process preserves dynamic VRAM and graph execution. A separate resident process decodes on the second of the same two GPUs. The TTS placement is prepared once per process, and its dynamic weight range gets a residency floor. This is for the measured model that fits those GPUs, not a way to force an oversized model into memory.

The benchmark keeps the models resident across the requested seeds, writes WAV and codes, then exits. It is not an indefinitely running HTTP service. Per-chunk generation retains the first/latest generated-audio history. Default stalled-chunk retry is enabled; use the underlying `benchmark_comfy.py --help` for diagnostics and prompt ablations.

## B: native speech.cpp workers

```bash
python scripts/prepare_speech_voice.py --speech /path/to/speech \
  --model /models/drama-3-q4/Drama-35B-A3B-v3-Q4_K.gguf \
  --assets /path/to/assets --out cache/reference.voice.gguf

python scripts/benchmark_speech.py --speech /path/to/speech \
  --model /models/drama-3-q4/Drama-35B-A3B-v3-Q4_K.gguf \
  --voice cache/reference.voice.gguf --prompt /path/to/assets/prompt.txt \
  --out outputs/b --devices Vulkan0 --seeds 0,1,2
# Two independent workers: --devices Vulkan0,Vulkan1
```

The worker has a cold first-request cost. Benchmark output records both process-ready and first-PCM time; do not present a warmed worker's latency as cold startup.

## C: persistent reference codes and streaming

Activate the prepared C environment:

```bash
export DRAMA3_MODEL=/models/drama-3-int4
export DRAMA3_ASSETS=/path/to/assets
export DRAMA3_REFERENCE_CACHE="$PWD/cache/reference-vq"

python scripts/prepare_reference_cache.py --model "$DRAMA3_MODEL" \
  --assets "$DRAMA3_ASSETS" --cache-dir "$DRAMA3_REFERENCE_CACHE" --device cuda:0

CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_c.sh dual
# Or: CUDA_VISIBLE_DEVICES=0 bash scripts/run_c.sh single-cached
```

`run_c.sh` defaults to `PYTORCH_ALLOC_CONF=expandable_segments:True`. With an empty Triton cache, the default allocator failed to provide the 256 MiB autotuning buffer on our nearly full single card; expandable segments passed without reducing the 0.87 engine budget or 1200-token generation limit. Keep the persistent Triton cache: first-request compilation is outside the warm-request timings. [Release smoke tests](../benchmarks/release-smoke.md)

The single-card configuration requires all reference caches to be prepared before startup. A miss is rejected to avoid loading another codec into a nearly full GPU. The cache identity includes audio content, codec path/metadata, encoder source metadata, precision, Torch version, GPU architecture and reference length limit. Replacing an environment or moving the codec/source can require preparation again. Cached integers are stored losslessly; fresh GPU encodings need not always be bit-identical.

For the grouped configuration, in another terminal:

```bash
python scripts/benchmark_c_chunked.py --assets "$DRAMA3_ASSETS" \
  --prompt-file "$DRAMA3_ASSETS/prompt.txt" --out outputs/c-grouped --takes 8
```

For a whole-script control:

```bash
python scripts/benchmark_sglang.py --assets "$DRAMA3_ASSETS" \
  --prompt-file "$DRAMA3_ASSETS/prompt.txt" --out outputs/c-whole --takes 8 --top-p 0.9
```

These are distinct experiments. The grouped client does not automatically transcribe, reject or retry omissions. C's upstream speech API at the pinned commit does not accept a request seed; repeated trials are not matched-seed comparisons.

## Native control and checks

The measured full native control used two backbone GPUs and a third GPU for tower/codec. The straightforward two-GPU placement ran out of memory while loading the codec.

```bash
python experiments/benchmark_native_fish.py --model /models/drama-3 --assets /path/to/assets \
  --prompt-file /path/to/assets/prompt.txt --out outputs/native \
  --tower-device cuda:2 --codec-device cuda:2 --seeds 1,0,2,1

# In the prepared C environment; CPU fixtures, no model weights loaded:
python scripts/test_reference_cache.py
# After fetching speech.cpp source; standard-library-only conversion checks:
python scripts/selfcheck.py
```

`experiments/asr_check.py` supports a separately prepared Qwen3-ASR Comfy weight for script-coverage checks. ASR spelling errors around names should be reviewed; its CER is not a speaker-similarity measure. See [failure analysis](failures.md).
