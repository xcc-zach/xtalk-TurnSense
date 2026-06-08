import argparse
import asyncio
import base64
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional, Tuple
from urllib.request import urlopen

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from infer import (
    DEFAULT_AUDIO_SECONDS,
    DEFAULT_CLIP_MODE,
    DEFAULT_FRONTEND_CONF,
    LABELS,
    MAX_AUDIO_SECONDS,
    SAMPLING_RATE,
    AudioClassifierInfer,
)

DEFAULT_SERVICE_ONNX_PATH = "models/model_int8.onnx"
DEFAULT_SERVICE_CMVN_PATH = "models/am.mvn"
MODEL_BASE_URL = "https://huggingface.co/brgroup/TurnSense/resolve/main/TurnSense1.1"
DEFAULT_MODEL_URLS = {
    DEFAULT_SERVICE_ONNX_PATH: f"{MODEL_BASE_URL}/model_int8.onnx",
    DEFAULT_SERVICE_CMVN_PATH: f"{MODEL_BASE_URL}/am.mvn",
}


class JsonInferenceRequest(BaseModel):
    audio_base64: str = Field(..., description="Base64 encoded audio bytes")
    source: str = Field(default="base64")


class InferenceService:
    def __init__(
        self,
        onnx_path: str,
        cmvn_file: str,
        use_cuda: bool,
        max_concurrency: int,
        max_workers: int,
        audio_seconds: int,
        clip_mode: str,
    ):
        frontend_conf = dict(DEFAULT_FRONTEND_CONF)
        frontend_conf["cmvn_file"] = cmvn_file

        self.classifier = AudioClassifierInfer(
            onnx_path=onnx_path,
            labels=LABELS,
            use_cuda=use_cuda,
            audio_seconds=audio_seconds,
            max_audio_seconds=MAX_AUDIO_SECONDS,
            sampling_rate=SAMPLING_RATE,
            frontend_conf=frontend_conf,
            clip_mode=clip_mode,
        )
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    async def predict_bytes(self, audio_bytes: bytes, source: str) -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        async with self.semaphore:
            return await loop.run_in_executor(
                self.executor,
                lambda: self.classifier.predict_bytes(audio_bytes, source=source),
            )

    async def predict_pcm_bytes(
        self,
        audio_bytes: bytes,
        source: str,
        input_sr: int,
        channels: int,
        pcm_format: str,
    ) -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        async with self.semaphore:
            return await loop.run_in_executor(
                self.executor,
                lambda: self.classifier.predict_pcm_bytes(
                    audio_bytes,
                    source=source,
                    input_sr=input_sr,
                    channels=channels,
                    pcm_format=pcm_format,
                ),
            )

    async def predict_file(self, path: str) -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        async with self.semaphore:
            return await loop.run_in_executor(
                self.executor,
                lambda: self.classifier.predict_file(path),
            )

    def close(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)


def download_file(url: str, target_path: str) -> None:
    target_dir = os.path.dirname(target_path)
    if target_dir:
        os.makedirs(target_dir, exist_ok=True)
    tmp_path = f"{target_path}.tmp"
    try:
        with urlopen(url, timeout=120) as response, open(tmp_path, "wb") as output:
            shutil.copyfileobj(response, output)
        os.replace(tmp_path, target_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def ensure_default_model_assets(
    onnx_path: Optional[str],
    cmvn_file: Optional[str],
) -> Tuple[str, str]:
    if onnx_path:
        return onnx_path, cmvn_file or DEFAULT_SERVICE_CMVN_PATH

    resolved_onnx_path = DEFAULT_SERVICE_ONNX_PATH
    resolved_cmvn_file = cmvn_file or DEFAULT_SERVICE_CMVN_PATH
    required_files = [
        (resolved_onnx_path, DEFAULT_MODEL_URLS[DEFAULT_SERVICE_ONNX_PATH]),
        (resolved_cmvn_file, DEFAULT_MODEL_URLS[DEFAULT_SERVICE_CMVN_PATH]),
    ]

    for path, url in required_files:
        if os.path.exists(path):
            continue
        print(f"缺少默认模型文件，开始下载: {path}")
        download_file(url, path)

    return resolved_onnx_path, resolved_cmvn_file


def build_app(
    onnx_path: str,
    cmvn_file: str,
    use_cuda: bool = False,
    max_concurrency: int = 8,
    max_workers: int = 4,
    audio_seconds: int = DEFAULT_AUDIO_SECONDS,
    clip_mode: str = DEFAULT_CLIP_MODE,
) -> FastAPI:
    service_holder: Dict[str, Optional[InferenceService]] = {"service": None}

    def parse_content_type(value: str) -> tuple[str, Dict[str, str]]:
        parts = [part.strip() for part in value.split(";") if part.strip()]
        if not parts:
            return "", {}

        media_type = parts[0].lower()
        params: Dict[str, str] = {}
        for part in parts[1:]:
            if "=" not in part:
                continue
            key, raw_value = part.split("=", 1)
            params[key.strip().lower()] = raw_value.strip().strip("\"")
        return media_type, params

    def parse_positive_int(value: str, header_name: str) -> int:
        try:
            parsed = int(value)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"{header_name} 必须是正整数") from exc
        if parsed <= 0:
            raise HTTPException(status_code=400, detail=f"{header_name} 必须是正整数")
        return parsed

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"模型不存在: {onnx_path}")
        if not os.path.exists(cmvn_file):
            raise FileNotFoundError(f"CMVN 文件不存在: {cmvn_file}")

        service_holder["service"] = InferenceService(
            onnx_path=onnx_path,
            cmvn_file=cmvn_file,
            use_cuda=use_cuda,
            max_concurrency=max_concurrency,
            max_workers=max_workers,
            audio_seconds=audio_seconds,
            clip_mode=clip_mode,
        )
        yield
        service_holder["service"].close()

    app = FastAPI(title="TurnSense Service", version="0.1.0", lifespan=lifespan)

    def get_service() -> InferenceService:
        service = service_holder["service"]
        if service is None:
            raise RuntimeError("服务尚未初始化")
        return service

    @app.get("/healthz")
    async def healthz() -> Dict[str, Any]:
        return {
            "status": "ok",
            "model_loaded": service_holder["service"] is not None,
            "labels": LABELS,
        }

    @app.post("/infer/file")
    async def infer_file(file: UploadFile = File(...)) -> Dict[str, Any]:
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="上传文件为空")
        try:
            return await get_service().predict_bytes(audio_bytes, source=file.filename or "upload")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/infer/json")
    async def infer_json(payload: JsonInferenceRequest) -> Dict[str, Any]:
        try:
            audio_bytes = base64.b64decode(payload.audio_base64, validate=True)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"base64 解析失败: {exc}") from exc

        if not audio_bytes:
            raise HTTPException(status_code=400, detail="音频字节流为空")

        try:
            return await get_service().predict_bytes(audio_bytes, source=payload.source)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/infer/bytes")
    async def infer_bytes(request: Request) -> Dict[str, Any]:
        audio_bytes = await request.body()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="请求体为空")

        source = request.headers.get("x-audio-source", "octet-stream")
        media_type, media_params = parse_content_type(request.headers.get("content-type", ""))
        pcm_format = request.headers.get("x-audio-format", "").strip().lower()
        if not pcm_format:
            if media_type in {"audio/pcm", "audio/raw"}:
                pcm_format = "pcm_s16le"
            elif media_type == "audio/l16":
                pcm_format = "pcm_s16be"

        if pcm_format:
            sample_rate_value = request.headers.get("x-audio-sample-rate") or media_params.get("rate")
            channels_value = request.headers.get("x-audio-channels") or media_params.get("channels") or "1"

            input_sr = SAMPLING_RATE if sample_rate_value is None else parse_positive_int(
                sample_rate_value,
                "X-Audio-Sample-Rate",
            )
            channels = parse_positive_int(channels_value, "X-Audio-Channels")

            try:
                return await get_service().predict_pcm_bytes(
                    audio_bytes,
                    source=source,
                    input_sr=input_sr,
                    channels=channels,
                    pcm_format=pcm_format,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc

        try:
            return await get_service().predict_bytes(audio_bytes, source=source)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TurnSense async HTTP service")
    parser.add_argument("--onnx-path", default=os.getenv("TURNSENSE_ONNX_PATH"))
    parser.add_argument("--cmvn-file", default=os.getenv("TURNSENSE_CMVN_FILE"))
    parser.add_argument("--host", default=os.getenv("TURNSENSE_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("TURNSENSE_PORT", "8000")))
    parser.add_argument("--max-concurrency", type=int, default=int(os.getenv("TURNSENSE_MAX_CONCURRENCY", "8")))
    parser.add_argument("--max-workers", type=int, default=int(os.getenv("TURNSENSE_MAX_WORKERS", "4")))
    parser.add_argument("--audio-seconds", type=int, default=int(os.getenv("TURNSENSE_AUDIO_SECONDS", str(DEFAULT_AUDIO_SECONDS))))
    parser.add_argument("--clip-mode", choices=["head", "tail"], default=os.getenv("TURNSENSE_CLIP_MODE", DEFAULT_CLIP_MODE))
    parser.add_argument("--use-cuda", action="store_true", default=os.getenv("TURNSENSE_USE_CUDA", "").lower() in {"1", "true", "yes"})
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    onnx_path, cmvn_file = ensure_default_model_assets(args.onnx_path, args.cmvn_file)
    app = build_app(
        onnx_path=onnx_path,
        cmvn_file=cmvn_file,
        use_cuda=args.use_cuda,
        max_concurrency=args.max_concurrency,
        max_workers=args.max_workers,
        audio_seconds=args.audio_seconds,
        clip_mode=args.clip_mode,
    )

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
