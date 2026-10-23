#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
ACTION=${1:-run}
OUT=${2:-"$ROOT/outputs/a"}
SEEDS=${3:-0,1,2}
: "${COMFYUI_ROOT:?Set COMFYUI_ROOT to the pinned ComfyUI checkout}"
: "${A_MODEL:?Set A_MODEL to drama3_fp8experts_bf16.safetensors}"
: "${A_CODEC:?Set A_CODEC to drama3_codec.safetensors}"
: "${DRAMA3_ASSETS:?Set DRAMA3_ASSETS to the reference asset directory}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16} MKL_NUM_THREADS=${MKL_NUM_THREADS:-16}
ARGS=(--comfy "$COMFYUI_ROOT" --model "$A_MODEL" --codec "$A_CODEC"
  --assets "$DRAMA3_ASSETS" --out "$OUT" --seeds "$SEEDS")
if [[ -n "${DRAMA3_PROMPT:-}" ]]; then ARGS+=(--prompt-file "$DRAMA3_PROMPT"); fi
if [[ "$ACTION" == prepare ]]; then
  exec "${PYTHON:-python}" -u "$ROOT/scripts/benchmark_comfy.py" "${ARGS[@]}" --encode-only
elif [[ "$ACTION" != run ]]; then
  echo 'Usage: bash scripts/run_a.sh {prepare|run} [output-directory] [seeds]' >&2
  exit 2
fi
exec "${PYTHON:-python}" -u "$ROOT/scripts/benchmark_comfy.py" "${ARGS[@]}" \
  --reuse-voices --dynamic-vram --external-codec-device cuda:1 \
  --keep-tts-resident --retry-stalled 1 --resident-vbar-floor
