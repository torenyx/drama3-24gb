#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
SRC=${1:?Pass the official checkpoint directory}
OUT=${2:?Pass the output directory}
PYTHON=${PYTHON:-python}
SPEECH_BIN=${SPEECH_BIN:-speech}
export PYTHONPATH="$ROOT/upstream/fish-speech${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" "$ROOT/scripts/check_source.py" "$SRC" "$ROOT/modelscope-manifest.json"
command -v "$SPEECH_BIN" >/dev/null || { printf 'Set SPEECH_BIN to a built speech.cpp executable\n' >&2; exit 1; }
mkdir -p "$OUT"
F32="$OUT/Drama-35B-A3B-v3-F32.gguf"
Q4="$OUT/Drama-35B-A3B-v3-Q4_K.gguf"
# Upstream writes F32 directly. The marker distinguishes a complete conversion after interruption.
if [[ ! -f "$OUT/f32.complete" ]]; then
    [[ ! -e "$F32" ]] || { printf 'Incomplete/unmarked F32 exists: %s; move it aside before rerunning\n' "$F32" >&2; exit 1; }
    "$PYTHON" "$ROOT/scripts/convert_gguf_local.py" "$SRC" "$OUT"
    touch "$OUT/f32.complete"
fi
if [[ ! -f "$OUT/q4.complete" ]]; then
    "$SPEECH_BIN" quantize "$F32" "$Q4.part" --type q4_k
    mv -- "$Q4.part" "$Q4"
    touch "$OUT/q4.complete"
fi
"$SPEECH_BIN" info "$Q4"
cp "$ROOT/sources.lock.json" "$ROOT/modelscope-manifest.json" "$OUT/"
printf 'Q4_K ready: %s\nF32 retained: %s\n' "$Q4" "$F32"
