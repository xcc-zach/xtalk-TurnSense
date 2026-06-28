This README provides one-click TurnSense service deployment and client usage instructions. [Original README](./README.original.en.md)

## Requirements

- Linux or macOS
- Python 3.10+
- GPU deployment requires an NVIDIA driver. On machines without a GPU, the installer switches to CPU `onnxruntime` automatically.

## Installation

The installer creates `.venv`, installs dependencies, and downloads model files to `models/` by default. Every `pip` command runs with `http_proxy`, `https_proxy`, `HTTP_PROXY`, `HTTPS_PROXY`, `all_proxy`, and `ALL_PROXY` unset.

```bash
chmod +x install.sh start.sh
bash install.sh
```

Skip model download if needed:

```bash
TURNSENSE_SKIP_MODEL_DOWNLOAD=1 bash install.sh
```

Default model paths:

- ONNX: `models/model_fp32.onnx`
- CMVN: `models/am.mvn`

## Start the Service

```bash
bash start.sh --port 8000
```

Common environment variables:

```bash
TURNSENSE_ONNX_PATH=models/model_fp32.onnx
TURNSENSE_CMVN_FILE=models/am.mvn
TURNSENSE_MAX_CONCURRENCY=8
TURNSENSE_MAX_WORKERS=4
TURNSENSE_USE_CUDA=1
```

## Client Requests

The existing client usage is preserved. The service exposes `/infer/file`, `/infer/json`, and `/infer/bytes`.

### Upload Audio File

```bash
curl -X POST http://127.0.0.1:8000/infer/file \
  -F "file=@examples/complete.wav"
```

### JSON Base64

```bash
base64_audio="$(python - <<'PY'
import base64
print(base64.b64encode(open("examples/complete.wav", "rb").read()).decode())
PY
)"

curl -X POST http://127.0.0.1:8000/infer/json \
  -H "Content-Type: application/json" \
  -d '{"audio_base64":"'"${base64_audio}"'","source":"complete.wav"}'
```

### Binary Audio

```bash
curl -X POST http://127.0.0.1:8000/infer/bytes \
  -H "Content-Type: application/octet-stream" \
  -H "X-Audio-Source: complete.wav" \
  --data-binary @examples/complete.wav
```

### Raw PCM

`/infer/bytes` also accepts 16 kHz, mono, little-endian `int16` raw PCM without a WAV header.

```bash
python - <<'PY'
import wave
with wave.open("examples/complete.wav", "rb") as wav:
    open("/tmp/complete.pcm", "wb").write(wav.readframes(wav.getnframes()))
PY

curl -X POST http://127.0.0.1:8000/infer/bytes \
  -H "Content-Type: application/octet-stream" \
  -H "X-Audio-Source: complete.pcm" \
  --data-binary @/tmp/complete.pcm
```

Example response:

```json
{
  "audio_path": "complete.wav",
  "prediction_id": 0,
  "prediction": "complete",
  "probabilities": {
    "complete": 0.98,
    "incomplete": 0.01,
    "invalid": 0.01
  }
}
```

## Performance Testing

Start the service first, then run:

```bash
source .venv/bin/activate
python serving/performance_testing.py --url http://127.0.0.1:8000 --concurrency 4 --requests 32
```

Use audio from the Hugging Face dataset `xcczach/sample-data`:

```bash
python serving/performance_testing.py --download-sample-data --concurrency 4 --requests 32
```

Test raw PCM:

```bash
python serving/performance_testing.py --mode raw-pcm --concurrency 4 --requests 32
```

### Results

Validation environment: Ubuntu Linux 5.15, Python 3.13.9, dual Intel Xeon Gold 6530 CPUs with 128 threads, and 2 NVIDIA GeForce RTX 4090 GPUs available (Driver 570.124.06, 49140 MiB each). The service was started with default parameters and without `TURNSENSE_USE_CUDA=1`, so this run used the CPU provider. Service settings were `max_concurrency=8` and `max_workers=4`; test audio was `examples/complete.wav`; endpoint was `/infer/bytes`.

| Concurrency | Requests | Success | Avg latency/s | p50/s | p90/s | Throughput samples/s |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 16 | 16 | 0.314 | 0.303 | 0.379 | 3.180 |
| 4 | 16 | 16 | 0.466 | 0.483 | 0.580 | 8.472 |
| 16 | 16 | 16 | 1.190 | 1.392 | 1.805 | 8.443 |
| 64 | 64 | 64 | 4.006 | 4.183 | 6.913 | 8.416 |
| 256 | 256 | 256 | 14.716 | 14.820 | 25.808 | 8.684 |
