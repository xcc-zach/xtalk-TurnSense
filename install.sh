#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
MODEL_DIR="${TURNSENSE_MODEL_DIR:-models}"
HF_REPO_ID="${TURNSENSE_HF_REPO_ID:-brgroup/TurnSense}"

run_no_proxy() {
  env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY "$@"
}

if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

run_no_proxy python -m pip install --upgrade pip setuptools wheel
run_no_proxy python -m pip install huggingface_hub

REQ_FILE="requirements.txt"
TMP_REQ="$(mktemp)"
if command -v nvidia-smi >/dev/null 2>&1; then
  cp "$REQ_FILE" "$TMP_REQ"
else
  python - "$REQ_FILE" "$TMP_REQ" <<'PYREQ'
from pathlib import Path
import sys
src = Path(sys.argv[1])
dst = Path(sys.argv[2])
lines = []
for line in src.read_text(encoding="utf-8").splitlines():
    stripped = line.strip()
    if stripped.startswith("onnxruntime-gpu"):
        lines.append("onnxruntime==1.23.2")
    elif stripped.startswith("nvidia-cudnn-cu12"):
        continue
    else:
        lines.append(line)
dst.write_text("\n".join(lines) + "\n", encoding="utf-8")
PYREQ
fi

run_no_proxy python -m pip install -r "$TMP_REQ"
rm -f "$TMP_REQ"

mkdir -p "$MODEL_DIR"
if [ "${TURNSENSE_SKIP_MODEL_DOWNLOAD:-0}" != "1" ]; then
  run_no_proxy python - "$HF_REPO_ID" "$MODEL_DIR" <<'PYMODEL'
import shutil
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

repo_id = sys.argv[1]
model_dir = Path(sys.argv[2])
model_dir.mkdir(parents=True, exist_ok=True)

snapshot_dir = Path(snapshot_download(
    repo_id=repo_id,
    local_dir=model_dir,
    allow_patterns=["*.onnx", "*.mvn", "*.json"],
))

def ensure_at_root(name: str) -> None:
    target = model_dir / name
    if target.exists():
        return
    matches = list(snapshot_dir.rglob(name))
    if matches:
        shutil.copy2(matches[0], target)

ensure_at_root("model_fp32.onnx")
ensure_at_root("model_int8.onnx")
ensure_at_root("am.mvn")

missing = [name for name in ("model_fp32.onnx", "am.mvn") if not (model_dir / name).exists()]
if missing:
    raise SystemExit(f"model download finished but required files are missing: {missing}")
PYMODEL
fi

echo "Installation finished."
echo "Activate with: source $VENV_DIR/bin/activate"
echo "Start with: bash start.sh --port 8000"
