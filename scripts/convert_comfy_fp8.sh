#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
SRC=${1:?Pass the official checkpoint directory}
OUT=${2:?Pass the output directory}
COMFYUI_ROOT=${3:-${COMFYUI_ROOT:?Pass ComfyUI root as argument 3 or set COMFYUI_ROOT}}
PYTHON=${PYTHON:-python}
export PYTHONPATH="$COMFYUI_ROOT${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" "$ROOT/scripts/check_source.py" "$SRC" "$ROOT/modelscope-manifest.json"
mkdir -p "$OUT"
PACK="$ROOT/upstream/comfy-drama3"
MODEL="$OUT/drama3_fp8experts_bf16.safetensors"
CODEC="$OUT/drama3_codec.safetensors"
# The upstream converter atomically renames its .part output on completion.
if [[ ! -f "$MODEL" ]]; then
    "$PYTHON" "$PACK/tools/convert.py" model "$SRC" "$MODEL" --format fp8 --dense bf16 --device cpu
fi
if [[ ! -f "$CODEC" ]]; then
    "$PYTHON" "$PACK/tools/convert.py" codec "$SRC/codec.pth" "$CODEC"
fi
cp "$ROOT/sources.lock.json" "$ROOT/modelscope-manifest.json" "$OUT/"
printf 'FP8 model: %s\nCodec: %s\n' "$MODEL" "$CODEC"
