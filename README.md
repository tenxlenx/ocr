# DeepSeek‑OCR Streaming API (CUDA 13 / cu130)

A tiny, Dockerized **FastAPI** server around the Hugging Face model **`Jalea96/DeepSeek-OCR-bnb-4bit-NF4`** that:

* exposes **one endpoint**: `POST /infer`
* **streams raw text** (`text/plain`) as the model emits it (use `curl -N`)
* is configurable entirely via a **`.env`** file (port, model, dtype, preset, etc.)
* ships a lightweight **client CLI** (`ocr`) that can send an image *or* let you **draw a screen region across any monitor** and stream the OCR text

Hardware target: NVIDIA GPU with CUDA 13.0+ (Docker base: `nvidia/cuda:13.0.1-cudnn-runtime-ubuntu22.04`) and **PyTorch cu130** wheels.

---

## Quick Start

### 0) Requirements

* NVIDIA Driver + **NVIDIA Container Toolkit** (for GPU access)
* Docker & docker-compose
* (Optional client) Python 3.10+ on your host with `requests`, `Pillow`, `mss`, `tkinter`

  > On minimal distros you may need to install `python3-tk`.

### 1) Configure `.env`

Create a file **`.env`** next to your `docker-compose.yml`:

```env
# Server
PORT=8000

# Model/runtime
MODEL_ID=Jalea96/DeepSeek-OCR-bnb-4bit-NF4
ATTN_IMPL=eager            # or flash_attention_2 (if installed)
TORCH_DTYPE=bf16           # bf16|fp16|fp32
DEVICE_MAP=auto
PRESET=gundam
PROMPT=<image>\n<|grounding|>Convert the document to markdown.

# Hugging Face cache inside the container
HF_HOME=/data/hf_cache
HF_TOKEN=

# Optional quantization toggles (if wired in code)
FORCE_BNB_4BIT=1
BNB_DOUBLE_QUANT=1
```

> Tip: keep `PRESET=gundam` unless you know you want other sizes.

### 2) Build & Run

```bash
docker compose up --build -d
```

This builds the CUDA 13 image, installs **PyTorch (cu130)** and friends, and starts the API.

> If your `docker-compose.yml` maps `${PORT:-8000}:8000`, the container always listens on **8000 inside**, while the **host port** is taken from `.env`.

### 3) Test the API (streaming)

Use `curl -N` to avoid buffering:

```bash
curl -s -N -F "file=@test.png" http://localhost:8000/infer
```

You should see tokens/lines flowing until inference completes.

---

## API

### `POST /infer`

* **Request**: `multipart/form-data` with a single part named `file` (png/jpg/webp/tiff…)
* **Response**: `text/plain` **stream** (chunked); the server pipes the model’s raw output as it happens, then exits.

**Example:**

```bash
curl -s -N -F "file=@docs/sample.png" http://localhost:8000/infer
```

> **Note**: there is intentionally **no JSON** and no “final file” here; the contract is *stream raw text*. If you need markdown files or boxes, build those on the client from the stream.

---

## Client CLI (`ocr`)

A tiny Python script that:

* accepts an image path **or** lets you **select a region** across **any monitor** and uploads the crop
* **streams cleaned text** by default (removes `<|det|>…</|det|>`, `<|ref|>…</|ref|>`, ANSI, tqdm bars, tensor-size logs like `BASE:` / `PATCHES: torch.Size([...])`, etc.)
* `--raw` to see the unfiltered stream

### Install

```bash
# install deps
python3 -m pip install --upgrade requests Pillow mss

# put the script somewhere in PATH, e.g.
install -Dm755 scripts/ocr ~/.local/bin/ocr
# or just make it executable and add folder to PATH
chmod +x scripts/ocr
```

> On some Linux distros you may need `sudo apt install python3-tk` (for `tkinter`) if it isn’t preinstalled.

### Usage

```bash
# Select a screen region and stream cleaned OCR
ocr

# Send a file
ocr test.png

# Raw (no cleaning)
ocr --raw test.png

# Different server
ocr --url http://server:8000 test.png
```

---

## Project Layout

```
.
├─ Dockerfile                 # CUDA 13.0.1 base, PyTorch cu130 wheels, FastAPI server
├─ docker-compose.yml         # uses .env for port/model/dtype/etc.
├─ .env                       # runtime configuration
├─ ocr_api_allinone.py        # FastAPI server; loads env via python-dotenv
├─ scripts/
│   └─ ocr                    # Client CLI (upload file or screen selection; stream text)
├─ uploads/                   # (mounted) temporary uploads
└─ results/                   # (mounted) optional output dir for model side-effects
```

---

## Configuration Reference (.env)

| Key                | Meaning                                            | Default                             |
| ------------------ | -------------------------------------------------- | ----------------------------------- |
| `PORT`             | Host port mapping for the API                      | `8000`                              |
| `MODEL_ID`         | HF repo id of the model                            | `Jalea96/DeepSeek-OCR-bnb-4bit-NF4` |
| `ATTN_IMPL`        | Attention backend (`eager`, `flash_attention_2`)   | `eager`                             |
| `TORCH_DTYPE`      | Torch dtype (`bf16`, `fp16`, `fp32`)               | `bf16`                              |
| `DEVICE_MAP`       | `auto` or device mapping string                    | `auto`                              |
| `PRESET`           | Resolution preset (`tiny/small/base/large/gundam`) | `gundam`                            |
| `PROMPT`           | Initial prompt text                                | see `.env`                          |
| `HF_HOME`          | HF cache dir in container                          | `/data/hf_cache`                    |
| `HF_TOKEN`         | HF token (not required for this public model)      | *(empty)*                           |
| `FORCE_BNB_4BIT`   | (optional) force BitsAndBytes 4-bit quant          | `1`                                 |
| `BNB_DOUBLE_QUANT` | (optional) BnB double-quant toggle                 | `1`                                 |

---

## Tips, Performance & Notes

* **GPU access:** ensure the host has NVIDIA Container Toolkit installed and your compose service uses `gpus: all`.
* **flash-attn:** the Dockerfile attempts to install `flash-attn` (2.7.x). For CUDA 13 it may not have prebuilt wheels yet; failure is ignored. If you enable it (`ATTN_IMPL=flash_attention_2`) without a working install, the code falls back to `eager`.
* **Quantization messages:** warnings like `Unused kwargs: ['_load_in_4bit' ...]` are expected when the model config already contains quantization; they are harmless.
* **Streaming client cleaning:** by default the `ocr` client removes noisy lines such as `<|det|>...`, `<|ref|>...`, tqdm bars (`other: 0%|...`), and tensor size logs (`BASE:` / `PATCHES:`). Use `--raw` to see unfiltered output.

---

## Troubleshooting

**Q: Server says `python-multipart` required / 422 on upload**
A: The Dockerfile installs `python-multipart`. If you run locally, ensure:

```bash
python3 -m pip install python-multipart
```

**Q: `Error loading ASGI app. Could not import module`**
A: Make sure the filename in the container matches the `CMD` or uvicorn module path (this repo uses `ocr_api_allinone.py` with `CMD ["python3", "-u", "ocr_api_allinone.py"]`).

**Q: No GPU found in container**
A: Install NVIDIA Container Toolkit on the host; run compose with `gpus: all`.

**Q: Client won’t open region selector**
A: Install `tkinter` (often `python3-tk` package) and `Pillow`. On Wayland, some DEs restrict global screenshots; try X11 session or grant screen-capture permissions.

**Q: Stream shows only raw tokens**
A: That’s expected while the model is generating. The client stitches small token lines and removes noise unless `--raw` is passed.

---

## Security

* This is a development-focused server that accepts arbitrary image uploads and runs a large model in the same process. Place it behind a trusted network or reverse proxy if exposed.
* Use separate volumes for `uploads/` and `results/` if you care about persistence.

---

## Acknowledgements

* Model: **Jalea96/DeepSeek-OCR-bnb-4bit-NF4** on Hugging Face.
* Thanks to the Hugging Face ecosystem, FastAPI, and the PyTorch team.

---

## License

This repository’s scripts are provided under the MIT License (see `LICENSE`).
Model weights have their **own license**—please review the corresponding Hugging Face model card.
