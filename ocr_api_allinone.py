#!/usr/bin/env python3
# FastAPI server for streaming DeepSeek-OCR raw text from a single /infer endpoint.

from __future__ import annotations

import io
import os
import time
import shutil
import uuid
import asyncio
from pathlib import Path
from typing import Dict, Tuple, Optional, Iterable
from contextlib import asynccontextmanager, redirect_stdout, redirect_stderr

import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.utils import logging as hf_logging
from pathlib import Path
from dotenv import load_dotenv

hf_logging.set_verbosity_error()

# -------------------------
# Config via environment
# -------------------------


# Load .env from /app/.env by default (or wherever you mount it)
load_dotenv(dotenv_path=os.environ.get("ENV_FILE", str(Path("/app/.env"))))

# now read envs as before, e.g.:
MODEL_ID   = os.environ.get("MODEL_ID", "Jalea96/DeepSeek-OCR-bnb-4bit-NF4")
ATTN_IMPL  = os.environ.get("ATTN_IMPL", "eager")
DTYPE_STR  = os.environ.get("TORCH_DTYPE", "bf16")
DEVICE_MAP = os.environ.get("DEVICE_MAP", "auto")
PRESET_NAME = os.environ.get("PRESET", "gundam")
PROMPT     = os.environ.get("PROMPT", "<image>\n<|grounding|>Convert the document to markdown.")
HF_HOME    = os.environ.get("HF_HOME", str(Path.cwd() / ".hf_cache"))
PORT       = int(os.environ.get("PORT", "8000"))

os.environ.setdefault("HF_HOME", HF_HOME)
os.environ.setdefault("BITSANDBYTES_NOWELCOME", "1")

UPLOADS_DIR = Path.cwd() / "uploads"
RESULTS_DIR = Path.cwd() / "results"   # the model may write text files here; we can harvest them
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Resolution presets (base_size, image_size, crop_mode)
PRESETS: Dict[str, Tuple[int, int, bool]] = {
    "tiny":   (512,  512,  False),
    "small":  (640,  640,  False),
    "base":   (1024, 1024, False),
    "large":  (1280, 1280, False),
    "gundam": (1024, 640,  True),  # recommended
}

_TEXT_FILES = (".mmd", ".md", ".markdown", ".txt")

def _pick_dtype(s: str):
    s = s.lower()
    if s == "bf16": return torch.bfloat16
    if s == "fp16": return torch.float16
    return torch.float32

def _preset_tuple(name: str) -> Tuple[int, int, bool]:
    return PRESETS.get(name, PRESETS["gundam"])

def _extract_text_from_res(res) -> Optional[str]:
    if isinstance(res, dict):
        for k in ("markdown", "text", "result", "output", "md"):
            v = res.get(k)
            if isinstance(v, (str, bytes)):
                return v.decode() if isinstance(v, bytes) else v
    if isinstance(res, str):
        return res
    return None

def _read_first_text_file(paths: Iterable[Path]) -> Optional[str]:
    for p in sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True):
        if p.suffix.lower() in _TEXT_FILES:
            try:
                return p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                pass
    return None

# ---------------------------------
# Writer that streams to a queue
# ---------------------------------
class QueueWriter(io.TextIOBase):
    """Buffers lines and pushes them into an asyncio.Queue (thread-safe)."""
    def __init__(self, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop, prefix: str = ""):
        self.queue = queue
        self.loop = loop
        self.buf = ""
        self.prefix = prefix
        self.full = []  # capture everything for dedup

    def write(self, s):
        if not isinstance(s, str):
            s = str(s)
        self.full.append(s)
        self.buf += s
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            self.loop.call_soon_threadsafe(self.queue.put_nowait, self.prefix + line + "\n")
        return len(s)

    def flush(self):
        if self.buf:
            self.loop.call_soon_threadsafe(self.queue.put_nowait, self.prefix + self.buf)
            self.buf = ""

    def get_full_text(self) -> str:
        return "".join(self.full)

# -------------------------
# Model wrapper
# -------------------------
class DeepSeekOCR:
    def __init__(self):
        dtype = _pick_dtype(DTYPE_STR)

        # Configure attention on config if the field exists; don't pass as kwarg to from_pretrained
        cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
        for key in ("attn_implementation", "_attn_implementation"):
            if hasattr(cfg, key):
                setattr(cfg, key, ATTN_IMPL)

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

        attempts = [
            dict(config=cfg, device_map=DEVICE_MAP, torch_dtype=dtype),
            dict(config=AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True),
                 device_map=DEVICE_MAP, torch_dtype=dtype),
            dict(config=AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True),
                 device_map="cpu", torch_dtype=torch.float32),
        ]
        last_error = None
        for kw in attempts:
            try:
                self.model = AutoModel.from_pretrained(
                    MODEL_ID,
                    trust_remote_code=True,
                    use_safetensors=True,
                    low_cpu_mem_usage=True,
                    **kw,
                ).eval()
                last_error = None
                break
            except Exception as e:
                last_error = e
        if last_error is not None:
            raise RuntimeError(f"Failed to load model: {last_error}") from last_error

    def run_and_stream(
        self,
        image_path: Path,
        preset: Optional[str],
        prompt: Optional[str],
        out_dir: Path,
        q: asyncio.Queue,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """Run inference, streaming stdout/stderr. Append model/harvested text at the end if not printed."""
        base_size, image_size, crop_mode = _preset_tuple(preset or PRESET_NAME)
        prompt = prompt if prompt is not None else PROMPT

        out_dir.mkdir(parents=True, exist_ok=True)
        before = {p.resolve() for p in out_dir.glob("*")}
        t0 = time.time()

        out_writer = QueueWriter(q, loop)
        err_writer = QueueWriter(q, loop)

        try:
            with torch.inference_mode(), redirect_stdout(out_writer), redirect_stderr(err_writer):
                res = self.model.infer(
                    self.tokenizer,
                    prompt=prompt,
                    image_file=str(image_path),
                    output_path=str(out_dir),
                    base_size=base_size,
                    image_size=image_size,
                    crop_mode=crop_mode,
                    save_results=True,
                    test_compress=False,
                )

            out_writer.flush()
            err_writer.flush()

            # Harvest any text files written by the model
            after = {p.resolve() for p in out_dir.glob("*")}
            created = [p for p in (after - before) if p.suffix.lower() in _TEXT_FILES]
            touched = [
                p for p in after
                if p.suffix.lower() in _TEXT_FILES and p.stat().st_mtime >= t0 - 1e-3
            ]
            harvested = _read_first_text_file(created) or _read_first_text_file(touched)
            model_text = _extract_text_from_res(res)
            captured = out_writer.get_full_text() + err_writer.get_full_text()

            # Separate, then append deduped final texts
            if captured and not captured.endswith("\n"):
                loop.call_soon_threadsafe(q.put_nowait, "\n")
            if isinstance(model_text, str) and model_text.strip() and model_text not in captured:
                loop.call_soon_threadsafe(q.put_nowait, model_text if model_text.endswith("\n") else model_text + "\n")
            if isinstance(harvested, str) and harvested.strip():
                if (harvested != model_text) and (harvested not in captured):
                    loop.call_soon_threadsafe(q.put_nowait, harvested if harvested.endswith("\n") else harvested + "\n")

        except Exception as e:
            loop.call_soon_threadsafe(q.put_nowait, f"[ERROR] Inference failed: {e}\n")
        finally:
            loop.call_soon_threadsafe(q.put_nowait, None)  # sentinel

# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(title="DeepSeek OCR (streaming raw text)", version="0.2.0")
_ocr: Optional[DeepSeekOCR] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _ocr
    _ocr = DeepSeekOCR()
    print("[INFO] Model loaded.")
    try:
        yield
    finally:
        _ocr = None

app.router.lifespan_context = lifespan

def _save_upload(file: UploadFile) -> Path:
    ext = Path(file.filename).suffix.lower() or ".png"
    if ext not in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}:
        raise HTTPException(status_code=400, detail="Unsupported image format")
    uid = uuid.uuid4().hex
    tmp_path = UPLOADS_DIR / f"{uid}{ext}"
    with tmp_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return tmp_path

@app.post("/infer", response_class=StreamingResponse)
async def infer(file: UploadFile = File(...)):
    if _ocr is None:
        raise HTTPException(status_code=503, detail="Model not ready")

    try:
        tmp_path = _save_upload(file)
    finally:
        await file.close()

    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()

    async def streamer():
        # Start inference in a thread (do NOT await here) so we can yield concurrently.
        fut = loop.run_in_executor(
            None,
            _ocr.run_and_stream,
            tmp_path,
            PRESET_NAME,
            PROMPT,
            RESULTS_DIR,
            q,
            loop,
        )

        try:
            while True:
                item = await q.get()
                if item is None:
                    break
                yield item
        finally:
            try:
                await fut
            except Exception:
                pass
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    headers = {
        "X-Accel-Buffering": "no",
        "Cache-Control": "no-cache",
    }
    return StreamingResponse(streamer(), media_type="text/plain; charset=utf-8", headers=headers)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT, reload=False)

