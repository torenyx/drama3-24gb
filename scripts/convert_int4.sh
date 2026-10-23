#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
SRC=${1:?Pass the official checkpoint directory}
OUT=${2:?Pass the output directory}
export QUANT_STAGING="$SRC" QUANT_UPLOAD_REPO=''
mkdir -p "$OUT"
exec "${PYTHON:-python}" -u "$ROOT/upstream/drama3-int4-quantize.py" \
  "$SRC" "$OUT" --device cuda --dense attn,gdn
