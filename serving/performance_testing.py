#!/usr/bin/env python3
import argparse
import base64
import concurrent.futures
import json
import statistics
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path


def load_payload(audio_path: Path, mode: str):
    if mode == "raw-pcm":
        with wave.open(str(audio_path), "rb") as wav:
            payload = wav.readframes(wav.getnframes())
        return "/infer/bytes", payload, {
            "Content-Type": "application/octet-stream",
            "X-Audio-Source": f"{audio_path.name}.pcm",
        }

    data = audio_path.read_bytes()
    if mode == "json":
        body = json.dumps({
            "audio_base64": base64.b64encode(data).decode("ascii"),
            "source": audio_path.name,
        }).encode("utf-8")
        return "/infer/json", body, {"Content-Type": "application/json"}

    return "/infer/bytes", data, {
        "Content-Type": "application/octet-stream",
        "X-Audio-Source": audio_path.name,
    }


def post_once(base_url: str, endpoint: str, payload: bytes, headers: dict, timeout: float):
    req = urllib.request.Request(
        base_url.rstrip("/") + endpoint,
        data=payload,
        headers=headers,
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            status = resp.status
    except urllib.error.HTTPError as exc:
        body = exc.read()
        status = exc.code
    latency = time.perf_counter() - start
    ok = 200 <= status < 300
    return ok, latency, status, body.decode("utf-8", errors="replace")


def percentile(values, ratio: float):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * ratio)))
    return ordered[index]


def maybe_download_sample_data(target_dir: Path):
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(
        repo_id="xcczach/sample-data",
        repo_type="dataset",
        local_dir=target_dir,
        allow_patterns=["*.wav", "*.flac", "*.mp3"],
    ))
    for suffix in ("*.wav", "*.flac", "*.mp3"):
        matches = sorted(snapshot.rglob(suffix))
        if matches:
            return matches[0]
    raise FileNotFoundError("No audio file found in xcczach/sample-data")


def main():
    parser = argparse.ArgumentParser(description="TurnSense HTTP service benchmark")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--audio", default="examples/complete.wav")
    parser.add_argument("--mode", choices=["bytes", "json", "raw-pcm"], default="bytes")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--download-sample-data", action="store_true")
    parser.add_argument("--sample-data-dir", default="serving/sample-data")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if args.download_sample_data:
        audio_path = maybe_download_sample_data(Path(args.sample_data_dir))
    if not audio_path.exists():
        raise FileNotFoundError(f"audio file not found: {audio_path}")

    endpoint, payload, headers = load_payload(audio_path, args.mode)

    results = []
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(post_once, args.url, endpoint, payload, headers, args.timeout)
            for _ in range(args.requests)
        ]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    elapsed = time.perf_counter() - started

    successes = [item for item in results if item[0]]
    failures = [item for item in results if not item[0]]
    latencies = [item[1] for item in successes]

    report = {
        "url": args.url,
        "endpoint": endpoint,
        "audio": str(audio_path),
        "mode": args.mode,
        "concurrency": args.concurrency,
        "requests": args.requests,
        "success": len(successes),
        "failed": len(failures),
        "elapsed_sec": round(elapsed, 3),
        "throughput_samples_per_sec": round(len(successes) / elapsed, 3) if elapsed > 0 else 0.0,
        "latency_avg_sec": round(statistics.mean(latencies), 3) if latencies else 0.0,
        "latency_p50_sec": round(percentile(latencies, 0.50), 3),
        "latency_p90_sec": round(percentile(latencies, 0.90), 3),
        "latency_p99_sec": round(percentile(latencies, 0.99), 3),
    }
    if failures:
        report["first_failure"] = {
            "status": failures[0][2],
            "body": failures[0][3][:500],
        }

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
