import argparse
import asyncio
import base64
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from infer import (
    DEFAULT_AUDIO_SECONDS,
    DEFAULT_CLIP_MODE,
    DEFAULT_FRONTEND_CONF,
    DEFAULT_ONNX_PATH,
    LABELS,
    MAX_AUDIO_SECONDS,
    SAMPLING_RATE,
    AudioClassifierInfer,
)


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

    async def predict_file(self, path: str) -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        async with self.semaphore:
            return await loop.run_in_executor(
                self.executor,
                lambda: self.classifier.predict_file(path),
            )

    def close(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)


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

        try:
            return await get_service().predict_bytes(audio_bytes, source=source)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TurnSense async HTTP service")
    parser.add_argument("--onnx-path", default=os.getenv("TURNSENSE_ONNX_PATH", DEFAULT_ONNX_PATH))
    parser.add_argument("--cmvn-file", default=os.getenv("TURNSENSE_CMVN_FILE", "models/am.mvn"))
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
    app = build_app(
        onnx_path=args.onnx_path,
        cmvn_file=args.cmvn_file,
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
