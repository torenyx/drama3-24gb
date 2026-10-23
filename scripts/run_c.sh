#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
MODE=${1:-dual}
: "${DRAMA3_MODEL:?Set DRAMA3_MODEL to the converted INT4 checkpoint directory}"
: "${DRAMA3_ASSETS:?Set DRAMA3_ASSETS to the reference asset directory}"
export DRAMA3_REFERENCE_CACHE=${DRAMA3_REFERENCE_CACHE:-"$ROOT/cache/reference-vq"}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16} MKL_NUM_THREADS=${MKL_NUM_THREADS:-16}
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}
export SGLANG_OMNI_STARTUP_TIMEOUT=${SGLANG_OMNI_STARTUP_TIMEOUT:-1800}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-"$ROOT/cache/triton"}
export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-"$ROOT/cache/flashinfer"}
export TMPDIR=${TMPDIR:-"$ROOT/cache/tmp"}
export PYTHONPATH="$ROOT/runtime${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$FLASHINFER_WORKSPACE_BASE"
ARGS=(--model-path "$DRAMA3_MODEL" --host 127.0.0.1 --port "${DRAMA3_PORT:-18080}"
  --allowed-local-media-path "$DRAMA3_ASSETS"
  --tts_engine.gpu 0 --tts_engine.factory.context_length 4096
  --tts_engine.factory.max_running_requests 1 --tts_engine.factory.max_prompt_rows 2300
  --tts_engine.factory.max_new_tokens 1200 --tts_engine.engine.cuda_graph_max_bs 1
  --tts_engine.engine.max_total_tokens 4096 --tts_engine.engine.cuda_graph_backend_prefill disabled
  --tts_engine.factory.first_frame_cuda_graphs false
  --vocoder.factory.codec_path "$DRAMA3_MODEL" --vocoder.factory.max_batch_size 1
  --preprocessing.factory.max_concurrency 1 --preprocessing.factory.cache_dir "$DRAMA3_REFERENCE_CACHE")
unset DRAMA3_STREAMING_ONLY
if [[ "$MODE" == dual ]]; then
  ARGS+=(--tts_engine.gpu_memory_fraction 0.96
    --preprocessing.gpu 1 --preprocessing.total_reserve_bytes 6GiB
    --vocoder.gpu 1 --vocoder.gpu_memory_fraction 0.35)
elif [[ "$MODE" == single-cached ]]; then
  export DRAMA3_STREAMING_ONLY=1
  ARGS+=(--tts_engine.gpu_memory_fraction "${DRAMA3_ENGINE_MEMORY_FRACTION:-0.87}"
    --preprocessing.gpu 0 --preprocessing.total_reserve_bytes 512MiB
    --preprocessing.factory.cache_only true
    --vocoder.gpu 0 --vocoder.gpu_memory_fraction 0.10)
else
  echo 'Usage: bash scripts/run_c.sh {dual|single-cached}' >&2
  exit 2
fi
exec "${PYTHON:-python}" -u "$ROOT/runtime/launch_cached_drama3.py" serve "${ARGS[@]}"
