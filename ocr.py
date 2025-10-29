#!/usr/bin/env python3
"""
ocr — stream OCR from your API across ALL monitors.
CLEAN by default (no det/ref tags, no progress/warnings) and de-duplicates lines so
end-of-run “full replay” from the server won’t be printed again.

Usage:
  ocr                   # select a region across any monitor, upload & stream cleaned text
  ocr test.png          # send an image file
  ocr --raw test.png    # raw stream (no cleaning, no dedup)
  ocr --allow-duplicates  # keep cleaning but allow duplicates
  ocr --url http://host:8000

Deps: Pillow, mss, requests, tkinter
"""
from __future__ import annotations
import argparse, io, os, re, sys
from typing import Optional, Tuple
from collections import deque

missing = []
try: import requests
except Exception: missing.append("requests")
try:
    from PIL import Image, ImageTk
except Exception: missing.append("Pillow")
try: import mss
except Exception: missing.append("mss")
try: import tkinter as tk
except Exception: missing.append("tkinter (usually preinstalled with Python)")
if missing:
    print(
        "Missing dependencies:\n  - " + "\n  - ".join(missing) +
        "\nInstall with:\n  python3 -m pip install " +
        " ".join([m for m in missing if not m.startswith('tkinter')]),
        file=sys.stderr
    ); sys.exit(2)

DEFAULT_URL = os.environ.get("OCR_URL", "http://localhost:8000")
TIMEOUT = 3600  # seconds

# ---- Cleaning primitives ----
RE_ANSI = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
RE_TAGS = re.compile(r"<\|(?:ref|det)\|>.*?<\|/(?:ref|det)\|>", re.DOTALL)

# tqdm variants (bars and "it" lines)
RE_TQDM_IT  = re.compile(r"^\s*(?:[\w\-]+:\s*)?\d+\s*it\s*\[.*?\]\s*$", re.I)
RE_TQDM_BAR = re.compile(r"^\s*(?:[\w\-]+:\s*)?\d{1,3}%\|.*\|\s*\d+/\d+\s*\[.*\]\s*$")

# Specific noise lines:
# Specific noise lines:
RE_BASE_TENSOR    = re.compile(r"^\s*BASE:\s*torch\.Size\(\[[0-9,\s]+\]\)\s*$", re.I)
RE_PATCHES_TENSOR = re.compile(r"^\s*PATCHES:\s*torch\.Size\(\[[0-9,\s]+\]\)\s*$", re.I)
RE_NOPATCH        = re.compile(r"^\s*NO\s+PATCHES\s*$", re.I)
RE_ANY_TENSOR_SZ = re.compile(r"^\s*[\w\-]+:\s*torch\.Size\(\[[0-9,\s]+\]\)\s*$", re.I)


NOISE_PREFIXES = (
    "The attention mask", "Setting `pad_token_id`",
    "The attention layers in this model",
    "====================", "===============save results:===============",
    "INFO:", "WARNING:", "[INFO]", "[WARN]", "[ERROR]",
    "CUDA Version", "Container image Copyright",
)

PUNCT_START = tuple(".,;:!?)]}’”\"")

def is_noise_line(s: str) -> bool:
    if not s:
        return True
    if "<|det|>" in s or "<|ref|>" in s:
        return True
    if RE_TQDM_IT.match(s) or RE_TQDM_BAR.match(s):
        return True
    if RE_BASE_TENSOR.match(s) or RE_PATCHES_TENSOR.match(s) or RE_NOPATCH.match(s):
        return True
    for p in NOISE_PREFIXES:
        if s.startswith(p):
            return True
    if s == "None":
        return True
    if RE_ANY_TENSOR_SZ.match(s) or RE_NOPATCH.match(s):
        return True
    return False

def stitch_and_print(line: str, token_buf: list[str],
                     seen_set: set[str] | None, seen_q: deque[str] | None,
                     allow_duplicates: bool) -> None:
    s = line.strip()
    if not s:
        if token_buf:
            out = "".join(token_buf)
            token_buf.clear()
            if out: _print_once(out, seen_set, seen_q, allow_duplicates)
        _print_once("", seen_set, seen_q, allow_duplicates)  # newline
        return
    # Tiny token (<=2 chars) -> glue
    if len(s) <= 2 and s not in ("-", "*", "#"):
        if token_buf and s not in PUNCT_START:
            token_buf.append("")  # glue with no extra space
        token_buf.append(s)
        return
    # Flush buffered tokens into this line
    if token_buf:
        glued = "".join(token_buf)
        token_buf.clear()
        if glued:
            if s and s[0] not in PUNCT_START and not glued.endswith(tuple(" \n")):
                glued += " "
            s = glued + s
    _print_once(s, seen_set, seen_q, allow_duplicates)

def _print_once(s: str, seen_set: set[str] | None, seen_q: deque[str] | None,
                allow_duplicates: bool) -> None:
    key = re.sub(r"[ \t]+", " ", s.strip())
    if not allow_duplicates:
        if seen_set is not None and key:
            if key in seen_set:
                return
            seen_set.add(key)
            if seen_q is not None:
                seen_q.append(key)
                if len(seen_q) > 4096:
                    old = seen_q.popleft()
                    seen_set.discard(old)
    if s == "":
        sys.stdout.write("\n")
    else:
        sys.stdout.write(s + "\n")
    sys.stdout.flush()

def clean_stream_chunks(chunks, raw: bool, allow_duplicates: bool):
    """
    Consume streamed chunks, apply cross-chunk cleaning, stitch tokens,
    and suppress duplicate lines unless --allow-duplicates is set.
    """
    if raw:
        for chunk in chunks:
            if chunk:
                sys.stdout.write(chunk); sys.stdout.flush()
        return

    acc = ""
    token_buf: list[str] = []
    seen_set: set[str] = set()
    seen_q: deque[str] = deque()

    for chunk in chunks:
        if not chunk: continue
        acc += chunk.replace("\r", "\n")  # normalize CR

        # Strip ANSI and any <|det|>/<|ref|> blocks across accumulator
        acc = RE_ANSI.sub("", acc)
        acc = RE_TAGS.sub("", acc)

        # Emit complete lines
        while "\n" in acc:
            line, acc = acc.split("\n", 1)
            s = line.strip()
            if is_noise_line(s): continue
            # Condense whitespace unless Markdown structural line
            if not (s.startswith("#") or s.startswith("- ") or s.startswith("* ") or s.startswith(">")):
                s = re.sub(r"[ \t]+", " ", s)
            s = RE_TAGS.sub("", s)
            if "<|det|>" in s or "<|ref|>" in s:  # belt & suspenders
                continue
            stitch_and_print(s, token_buf, seen_set, seen_q, allow_duplicates)

    # Flush remainder/tokens
    tail = acc.strip()
    if tail and not is_noise_line(tail):
        tail = re.sub(r"[ \t]+", " ", RE_TAGS.sub("", tail))
        stitch_and_print(tail, token_buf, seen_set, seen_q, allow_duplicates)
    if token_buf:
        _print_once("".join(token_buf), seen_set, seen_q, allow_duplicates)
        token_buf.clear()

# ---- API call with streaming ----
def stream_infer(url: str, img_name: str, img_bytes: bytes, raw: bool, allow_duplicates: bool) -> int:
    endpoint = url.rstrip("/") + "/infer"
    files = {"file": (img_name, io.BytesIO(img_bytes), "image/png")}
    headers = {"Expect": ""}

    try:
        with requests.post(endpoint, files=files, headers=headers, stream=True, timeout=TIMEOUT) as r:
            if r.status_code != 200:
                sys.stderr.write(f"/infer failed: HTTP {r.status_code}\n{r.text}\n")
                return 1
            def chunk_iter():
                for c in r.iter_content(chunk_size=4096, decode_unicode=True):
                    if c:
                        yield c
            clean_stream_chunks(chunk_iter(), raw=raw, allow_duplicates=allow_duplicates)
    except KeyboardInterrupt:
        return 130
    except Exception as e:
        sys.stderr.write(f"Request error: {e}\n")
        return 1
    return 0

# ---- Screen capture across ALL monitors ----
def grab_full_desktop() -> tuple[Image.Image, dict]:
    """
    Returns a composite screenshot spanning ALL monitors (mss.monitors[0])
    AND the bounding box dict with keys: left, top, width, height.
    """
    with mss.mss() as sct:
        mon_all = sct.monitors[0]  # union of all monitors
        shot = sct.grab(mon_all)
        img = Image.frombytes("RGB", shot.size, shot.rgb)
        return img, mon_all  # mon_all has left/top which may be negative

def select_region(img_full: Image.Image, bbox: dict) -> Optional[Tuple[int, int, int, int]]:
    """
    Full-desktop selection overlay across ALL monitors.
    Places a borderless, topmost window at (+left,+top) with size width×height.
    """
    root = tk.Tk()
    root.overrideredirect(True)       # borderless
    root.attributes("-topmost", True)

    # Some WMs need this to avoid scaling surprises
    try: root.tk.call('tk', 'scaling', 1.0)
    except Exception: pass

    left = int(bbox.get("left", 0))
    top  = int(bbox.get("top", 0))
    w    = int(bbox.get("width", img_full.size[0]))
    h    = int(bbox.get("height", img_full.size[1]))

    # Try to position the window over the full virtual desktop (negatives allowed)
    try:
        root.geometry(f"{w}x{h}{'+' if left>=0 else ''}{left}{'+' if top>=0 else ''}{top}")
    except Exception:
        # Fallback: center-ish
        root.geometry(f"{w}x{h}+0+0")

    canvas = tk.Canvas(root, width=w, height=h, highlightthickness=0, bd=0)
    canvas.pack(fill="both", expand=True)
    photo = ImageTk.PhotoImage(img_full)
    canvas.create_image(0, 0, image=photo, anchor="nw")

    # Dim the background a bit
    canvas.create_rectangle(0, 0, w, h, fill="#000000", stipple="gray50", outline="")
    sel = canvas.create_rectangle(0, 0, 0, 0, outline="#00ff00", width=2)

    start = {"x": 0, "y": 0}; end = {"x": 0, "y": 0}; done = {"ok": False}
    canvas.create_text(20, 20, anchor="nw",
                       text="Drag to select area across any monitor. Release to confirm. ESC to cancel.",
                       fill="#ffffff", font=("TkDefaultFont", 16, "bold"))

    def on_press(e):
        start["x"], start["y"] = e.x, e.y
        end["x"], end["y"] = e.x, e.y
        canvas.coords(sel, start["x"], start["y"], end["x"], end["y"])

    def on_drag(e):
        # clamp to canvas
        ex = max(0, min(e.x, w-1)); ey = max(0, min(e.y, h-1))
        end["x"], end["y"] = ex, ey
        canvas.coords(sel, start["x"], start["y"], end["x"], end["y"])

    def on_release(e):
        ex = max(0, min(e.x, w-1)); ey = max(0, min(e.y, h-1))
        end["x"], end["y"] = ex, ey
        done["ok"] = True
        root.quit()

    def on_key(e):
        if e.keysym == "Escape":
            done["ok"] = False
            root.quit()

    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<B1-Motion>", on_drag)
    canvas.bind("<ButtonRelease-1>", on_release)
    root.bind("<Escape>", on_key)

    root.mainloop()
    try: root.destroy()
    except Exception: pass

    if not done["ok"]:
        return None

    x1, y1 = start["x"], start["y"]
    x2, y2 = end["x"], end["y"]
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return (x1, y1, x2, y2)

# ---- CLI ----
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Select a region across ALL monitors (or pass an image) and stream OCR text.")
    p.add_argument("image", nargs="?", help="Optional path to image; if omitted, select a screen region.")
    p.add_argument("--url", "-u", default=DEFAULT_URL, help=f"Base URL of the API (default: {DEFAULT_URL})")
    p.add_argument("--raw", action="store_true", help="Do not clean the streamed output; print raw text.")
    p.add_argument("--allow-duplicates", action="store_true",
                   help="Keep cleaning but allow duplicate lines (disables end-of-run dedup).")
    args = p.parse_args(argv)

    if args.image:
        if not os.path.isfile(args.image):
            print(f"File not found: {args.image}", file=sys.stderr); return 2
        with open(args.image, "rb") as f: data = f.read()
        return stream_infer(args.url, os.path.basename(args.image), data, raw=args.raw, allow_duplicates=args.allow_duplicates)

    # interactive grab across ALL monitors
    try:
        full, bbox = grab_full_desktop()
    except Exception as e:
        print(f"Failed to capture screen: {e}", file=sys.stderr); return 1

    box = select_region(full, bbox)
    if box is None:
        print("Cancelled.", file=sys.stderr); return 1

    x1, y1, x2, y2 = box
    if x2 - x1 < 2 or y2 - y1 < 2:
        print("Selection too small.", file=sys.stderr); return 1

    crop = full.crop((x1, y1, x2, y2))
    buf = io.BytesIO(); crop.save(buf, format="PNG")
    return stream_infer(args.url, "selection.png", buf.getvalue(), raw=args.raw, allow_duplicates=args.allow_duplicates)

if __name__ == "__main__":
    sys.exit(main())
