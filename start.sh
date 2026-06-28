#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-.venv}"
if [ -f "$VENV_DIR/bin/activate" ]; then
  source "$VENV_DIR/bin/activate"
fi

HOST="${TURNSENSE_HOST:-0.0.0.0}"
PORT="${TURNSENSE_PORT:-8000}"
ONNX_PATH="${TURNSENSE_ONNX_PATH:-models/model_fp32.onnx}"
CMVN_FILE="${TURNSENSE_CMVN_FILE:-models/am.mvn}"
MAX_CONCURRENCY="${TURNSENSE_MAX_CONCURRENCY:-8}"
MAX_WORKERS="${TURNSENSE_MAX_WORKERS:-4}"
CLIP_MODE="${TURNSENSE_CLIP_MODE:-tail}"
CUDA_ARG=()

if [ "${TURNSENSE_USE_CUDA:-}" = "1" ] || [ "${TURNSENSE_USE_CUDA:-}" = "true" ]; then
  CUDA_ARG=(--use-cuda)
fi

exec python service.py \
  --host "$HOST" \
  --port "$PORT" \
  --onnx-path "$ONNX_PATH" \
  --cmvn-file "$CMVN_FILE" \
  --max-concurrency "$MAX_CONCURRENCY" \
  --max-workers "$MAX_WORKERS" \
  --clip-mode "$CLIP_MODE" \
  "${CUDA_ARG[@]}" \
  "$@"
