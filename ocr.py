#!/usr/bin/env python3
"""
Live OCR → clean → ncurses → live code autoformat + autocolor → FINAL CLEAN TO CLIPBOARD.

What this version does extra vs. the previous:
- still strips <|det|>...</|det|>, <|ref|>...</|ref|>
- still does word-level streaming into ncurses
- still detects code and pretty-prints it live
- BUT: when the stream ends (hard-stop / q / error), it:
    1) collects the FULL cleaned text
    2) runs a FINAL FILTER to drop model/noise/progress crap:
       - "===================="
       - "BASE: ..."
       - "NO PATCHES"
       - "image: ... it/s"
       - "other: ... it/s"
       - lines with "torch.Size("
       - everything after "===============save results:==============="
    3) puts the result to the system clipboard (xclip/xsel/wl-copy/pbcopy/powershell)
- if clipboard fails, it just warns.

Comments are in English, as requested.
"""

from __future__ import annotations
import argparse
import io
import os
import sys
import re
import json
import shutil
import subprocess
import platform
from typing import Iterable, Optional, Tuple, List

# ---------------------------------------------------------------------
# dependency checks
# ---------------------------------------------------------------------
missing = []
try:
    import requests
except Exception:
    missing.append("requests")
try:
    from PIL import Image, ImageTk
except Exception:
    missing.append("Pillow")
try:
    import mss
except Exception:
    missing.append("mss")
try:
    import tkinter as tk
except Exception:
    missing.append("tkinter (usually preinstalled with Python)")
try:
    import curses
except Exception:
    curses = None  # fallback

# optional libs
try:
    from langdetect import detect as ld_detect
    HAS_LANGDETECT = True
except Exception:
    HAS_LANGDETECT = False

try:
    from pygments import lex
    from pygments.lexers import guess_lexer
    from pygments.token import Token
    from pygments.util import ClassNotFound
    HAS_PYGMENTS = True
except Exception:
    HAS_PYGMENTS = False

try:
    import black
    HAS_BLACK = True
except Exception:
    HAS_BLACK = False

try:
    import autopep8
    HAS_AUTOPEP8 = True
except Exception:
    HAS_AUTOPEP8 = False

if missing:
    print(
        "Missing deps:\n  - " + "\n  - ".join(missing)
        + "\nInstall with:\n  python3 -m pip install "
        + " ".join([m for m in missing if not m.startswith("tkinter")]),
        file=sys.stderr,
    )
    sys.exit(2)

# ---------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------
DEFAULT_URL = os.environ.get("OCR_URL", "http://localhost:8000")
TIMEOUT = 3600
TAB_WIDTH = 4
ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

# tags we strip from the OCR stream
OPEN_TAGS = ("<|det|>", "<|ref|>")
CLOSE_TAGS = {
    "<|det|>": "<|/det|>",
    "<|ref|>": "<|/ref|>",
}
MAX_OPEN_LEN = max(len(t) for t in OPEN_TAGS)

# noise patterns
NOISE_LINE_PREFIXES = (
    "image:",
    "other:",
    "video:",
    "audio:",
    "base:",
    "patches:",
    "====================",
    "===============save",
    "===============",
)
NOISE_SUBSTRINGS = (
    "torch.size(",
    " it/s]",
    "it/s]",
    "progress:",
)


# ---------------------------------------------------------------------
# final cleanup (the one you wanted)
# ---------------------------------------------------------------------
def final_cleanup(text: str) -> str:
    """
    Remove model/status/progress garbage and cut at the 'save results' marker.
    This is the final text that goes to clipboard.
    """
    if not text:
        return ""

    # cut everything from the hard stop marker
    lower = text.lower()
    marker = "===============save results:==============="
    cut_pos = lower.find(marker)
    if cut_pos != -1:
        text = text[:cut_pos]

    lines = text.splitlines()
    out: List[str] = []

    drop_prefixes = (
        "====================",
        "===============save",
        "base:",
        "no patches",
        "image:",
        "other:",
        "video:",
        "audio:",
        "patches:",
    )
    # tqdm-ish progress line
    progress_re = re.compile(r".*\b\d{1,3}%\|#+\|.*")

    for ln in lines:
        s = ln.strip()
        if not s:
            # skip pure empty, we will compact later
            continue

        sl = s.lower()

        # drop lines starting with known garbage
        if any(sl.startswith(p) for p in drop_prefixes):
            continue

        # drop torch.Size(...) spam
        if "torch.size(" in sl:
            continue

        # drop tqdm-like progress
        if progress_re.match(s):
            continue

        out.append(ln)

    # compact consecutive duplicates
    deduped: List[str] = []
    last = None
    for ln in out:
        if ln == last:
            continue
        deduped.append(ln)
        last = ln

    return "\n".join(deduped).strip()


# ---------------------------------------------------------------------
# clipboard helper
# ---------------------------------------------------------------------
class ClipBuffer:
    """Collects cleaned text parts before final write to clipboard."""

    def __init__(self):
        self.parts: List[str] = []

    def write(self, chunk: str):
        if chunk:
            self.parts.append(chunk)

    def get_text(self) -> str:
        return "".join(self.parts)

    def wrap(self, iterable: Iterable[str]) -> Iterable[str]:
        """Wrap an iterator so we can intercept all text flowing through it."""
        for part in iterable:
            self.write(part)
            yield part


def copy_to_clipboard(text: str) -> bool:
    """Try multiple OS-specific clipboard commands."""
    if not text:
        return False

    system = platform.system().lower()

    if system == "linux":
        # xclip
        if shutil.which("xclip"):
            p = subprocess.run(
                ["xclip", "-selection", "clipboard"],
                input=text.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if p.returncode == 0:
                return True
        # xsel
        if shutil.which("xsel"):
            p = subprocess.run(
                ["xsel", "--clipboard", "--input"],
                input=text.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if p.returncode == 0:
                return True
        # wl-copy
        if shutil.which("wl-copy"):
            p = subprocess.run(
                ["wl-copy"],
                input=text.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if p.returncode == 0:
                return True

    if system == "darwin" and shutil.which("pbcopy"):
        p = subprocess.run(
            ["pbcopy"],
            input=text.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if p.returncode == 0:
            return True

    if system == "windows":
        ps = shutil.which("powershell") or shutil.which("pwsh")
        if ps:
            p = subprocess.run(
                [ps, "-NoProfile", "-Command", "Set-Clipboard"],
                input=text.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if p.returncode == 0:
                return True

    return False


# ---------------------------------------------------------------------
# tag stripper: remove <|det|>...</|det|> streaming-wise
# ---------------------------------------------------------------------
def strip_tags_stream(chunks: Iterable[str]) -> Iterable[str]:
    pending = ""
    in_tag = False
    current_close = ""

    for chunk in chunks:
        if not chunk:
            continue
        chunk = ANSI_RE.sub("", chunk)

        for ch in chunk:
            pending += ch

            if not in_tag:
                opened = False
                for ot in OPEN_TAGS:
                    if pending.endswith(ot):
                        before = pending[:-len(ot)]
                        if before:
                            yield before
                        in_tag = True
                        current_close = CLOSE_TAGS[ot]
                        pending = ""
                        opened = True
                        break
                if opened:
                    continue
                if len(pending) > MAX_OPEN_LEN:
                    # flush oldest char
                    yield pending[0]
                    pending = pending[1:]
            else:
                if pending.endswith(current_close):
                    pending = ""
                    in_tag = False
                    current_close = ""
                else:
                    if len(pending) > len(current_close) * 2:
                        pending = pending[-len(current_close):]

    if not in_tag and pending:
        yield pending
    # if in_tag → drop


# ---------------------------------------------------------------------
# natural language detector
# ---------------------------------------------------------------------
class NaturalLang:
    HU_DIAC = set("áéíóöőúüű")

    def __init__(self):
        self.fallback = {"hu": 0, "en": 0, "de": 0}
        self.total = 0
        self.current = "unk"
        self.buf: List[str] = []

    def feed(self, tok: str):
        self.total += 1
        self.buf.append(tok)
        if len(self.buf) > 200:
            self.buf = self.buf[-200:]

        if not HAS_LANGDETECT:
            t = tok.lower()
            if any(c in t for c in self.HU_DIAC):
                self.fallback["hu"] += 2
            if t in ("the", "and", "is", "are", "to", "of"):
                self.fallback["en"] += 2
            if t in ("und", "die", "der", "das", "ist"):
                self.fallback["de"] += 2

            if self.total % 20 == 0:
                lang, sc = max(self.fallback.items(), key=lambda x: x[1])
                self.current = lang if sc > 0 else "unk"

    def get(self) -> str:
        if HAS_LANGDETECT and self.buf and self.total % 25 == 0:
            try:
                text = " ".join(self.buf)
                lang = ld_detect(text)
                self.current = lang
            except Exception:
                pass
        return self.current or "unk"


# ---------------------------------------------------------------------
# code formatting helpers
# ---------------------------------------------------------------------
def has_clang_format() -> bool:
    return shutil.which("clang-format") is not None


def run_clang_format(code: str) -> Optional[str]:
    if not has_clang_format():
        return None
    try:
        p = subprocess.run(
            ["clang-format"],
            input=code.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if p.returncode == 0:
            return p.stdout.decode("utf-8", "replace")
    except Exception:
        return None
    return None


def format_python(code: str) -> str:
    if HAS_BLACK:
        try:
            return black.format_str(code, mode=black.FileMode())
        except Exception:
            pass
    if HAS_AUTOPEP8:
        try:
            return autopep8.fix_code(code)
        except Exception:
            pass
    return code.expandtabs(TAB_WIDTH)


def format_json_text(code: str) -> str:
    try:
        obj = json.loads(code)
        return json.dumps(obj, indent=4, ensure_ascii=False)
    except Exception:
        return code.expandtabs(TAB_WIDTH)


def format_cpp(code: str) -> str:
    cf = run_clang_format(code)
    if cf is not None:
        return cf
    return code.expandtabs(TAB_WIDTH)


def guess_code_lang(code: str) -> str:
    if HAS_PYGMENTS:
        try:
            lexr = guess_lexer(code)
            name = lexr.name.lower()
            if "python" in name:
                return "code:python"
            if "json" in name:
                return "code:json"
            if "c++" in name or "cpp" in name or "c " in name or name == "c":
                return "code:cpp"
            if "bash" in name or "shell" in name:
                return "code:sh"
            if "java" in name:
                return "code:java"
            return "code:" + name.replace(" ", "_")
        except ClassNotFound:
            return "code:generic"
        except Exception:
            return "code:generic"
    return "code:generic"


def format_code(code: str, lang_hint: str) -> str:
    if lang_hint == "code:generic":
        lang_hint = guess_code_lang(code)

    if lang_hint == "code:python":
        return format_python(code)
    if lang_hint == "code:json":
        return format_json_text(code)
    if lang_hint in ("code:cpp", "code:java"):
        return format_cpp(code)

    return code.expandtabs(TAB_WIDTH)


# ---------------------------------------------------------------------
# live code collector for curses
# ---------------------------------------------------------------------
class LiveCodeBlock:
    """
    Collect incoming code tokens and emit only new formatted lines to the sink.
    """

    def __init__(self, sink: "CursesSink"):
        self.sink = sink
        self.active = False
        self.text_parts: List[str] = []
        self.last_lines: List[str] = []
        self.lang_hint = "code:generic"

    def start(self):
        self.active = True
        self.text_parts.clear()
        self.last_lines = []
        self.lang_hint = "code:generic"
        self.sink.toggle_code()

    def feed(self, tok: str):
        if not self.active:
            return
        self.text_parts.append(tok)

        raw = "".join(self.text_parts)
        raw = raw.replace("\r", "\n")

        if self.lang_hint == "code:generic" and len(raw) > 12:
            self.lang_hint = guess_code_lang(raw)

        self.sink.set_code_label(self.lang_hint)

        formatted = format_code(raw, self.lang_hint)
        lines = formatted.splitlines()

        old_n = len(self.last_lines)
        new_n = len(lines)
        if new_n > old_n:
            for ln in lines[old_n:]:
                self.sink.add_newline()
                self.sink.add_code_line(ln)

        self.last_lines = lines

    def finish(self):
        if not self.active:
            return
        self.active = False
        self.text_parts.clear()
        self.last_lines = []
        self.sink.toggle_code()
        self.sink.add_newline()


# ---------------------------------------------------------------------
# curses sink
# ---------------------------------------------------------------------
class CursesSink:
    def __init__(self, stdscr, natdet: NaturalLang):
        self.stdscr = stdscr
        self.max_y, self.max_x = self.stdscr.getmaxyx()
        self.pad = curses.newpad(20000, self.max_x)
        self.line = 0
        self.col = 0
        self.scroll_top = 0
        self.in_code_block = False
        self.colors_ok = False
        self.natdet = natdet
        self.code_label = "unk"
        self._init_colors()

    def _init_colors(self):
        if not curses.has_colors():
            return
        try:
            curses.start_color()
            curses.use_default_colors()
        except curses.error:
            pass

        def sp(i, fg, bg=-1):
            try:
                curses.init_pair(i, fg, bg)
                return True
            except curses.error:
                try:
                    curses.init_pair(i, fg, 0)
                    return True
                except curses.error:
                    return False

        ok1 = sp(1, curses.COLOR_BLUE)
        ok2 = sp(2, curses.COLOR_CYAN)
        ok3 = sp(3, curses.COLOR_BLACK)
        ok4 = sp(4, curses.COLOR_MAGENTA)
        ok5 = sp(5, curses.COLOR_YELLOW)
        ok6 = sp(6, curses.COLOR_GREEN)
        ok7 = sp(7, curses.COLOR_RED)
        ok8 = sp(8, curses.COLOR_GREEN)
        ok9 = sp(9, curses.COLOR_YELLOW)
        ok10 = sp(10, curses.COLOR_BLUE)

        self.colors_ok = any([ok1, ok2, ok3, ok4, ok5, ok6, ok7, ok8, ok9, ok10])

    def set_code_label(self, lbl: str):
        self.code_label = lbl

    def draw_title(self):
        nat = self.natdet.get()
        label = self.code_label if self.in_code_block else nat
        title = " OCR stream — q=quit "
        tail = f" lang: {label or 'unk'} "
        try:
            if self.colors_ok:
                self.stdscr.attron(curses.color_pair(4) | curses.A_BOLD)
            self.stdscr.addstr(0, 0, title[: self.max_x - 1])
            if self.colors_ok:
                self.stdscr.attroff(curses.color_pair(4) | curses.A_BOLD)

            if self.colors_ok:
                if self.in_code_block or (label and label.startswith("code:")):
                    self.stdscr.attron(curses.color_pair(6) | curses.A_BOLD)
                else:
                    self.stdscr.attron(curses.color_pair(5) | curses.A_BOLD)
            self.stdscr.addstr(0, len(title), tail[: self.max_x - len(title) - 1])
            if self.colors_ok:
                if self.in_code_block or (label and label.startswith("code:")):
                    self.stdscr.attroff(curses.color_pair(6) | curses.A_BOLD)
                else:
                    self.stdscr.attroff(curses.color_pair(5) | curses.A_BOLD)
        except curses.error:
            pass
        self.stdscr.refresh()

    def _newline(self):
        self.line += 1
        self.col = 0
        if self.line >= self.max_y - 1:
            self.scroll_top = self.line - (self.max_y - 2)

    def _refresh(self):
        try:
            self.pad.refresh(self.scroll_top, 0, 1, 0, self.max_y - 1, self.max_x - 1)
        except curses.error:
            pass
        self.draw_title()

    def toggle_code(self):
        self.in_code_block = not self.in_code_block

    def add_space(self):
        if self.col + 1 >= self.max_x - 1:
            self._newline()
        else:
            try:
                self.pad.addstr(self.line, self.col, " ")
            except curses.error:
                pass
            self.col += 1
        self._refresh()

    def add_newline(self):
        self._newline()
        self._refresh()

    def add_word(self, word: str, style: str | None = None):
        word = word.expandtabs(TAB_WIDTH)
        if len(word) > self.max_x - 1:
            chunks = [word[i:i + self.max_x - 1] for i in range(0, len(word), self.max_x - 1)]
        else:
            chunks = [word]

        for idx, chunk in enumerate(chunks):
            if self.col + len(chunk) >= self.max_x - 1:
                self._newline()

            attr = curses.A_NORMAL
            if self.colors_ok:
                if style == "heading":
                    attr = curses.color_pair(1) | curses.A_BOLD
                elif style == "list":
                    attr = curses.color_pair(2)
                elif style == "code":
                    attr = curses.color_pair(3)
            elif style == "heading":
                attr = curses.A_BOLD

            try:
                self.pad.addstr(self.line, self.col, chunk, attr)
            except curses.error:
                pass
            self.col += len(chunk)
            if idx < len(chunks) - 1:
                self._newline()
        self._refresh()

    def add_code_line(self, line: str):
        line = line.expandtabs(TAB_WIDTH)
        if not HAS_PYGMENTS or not self.colors_ok:
            self.add_word(line, style="code")
            return

        try:
            tokens = list(lex(line, guess_lexer(line)))
        except Exception:
            self.add_word(line, style="code")
            return

        if self.col != 0:
            self._newline()

        for tok_type, tok_val in tokens:
            if tok_type in Token.Keyword:
                attr = curses.color_pair(7) | curses.A_BOLD
            elif tok_type in Token.String:
                attr = curses.color_pair(8)
            elif tok_type in Token.Number:
                attr = curses.color_pair(9)
            elif tok_type in Token.Comment:
                attr = curses.color_pair(10)
            else:
                attr = curses.color_pair(3)

            for sub in tok_val.splitlines(True):
                if sub == "\n":
                    self._newline()
                    continue
                if self.col + len(sub) >= self.max_x - 1:
                    self._newline()
                try:
                    self.pad.addstr(self.line, self.col, sub, attr)
                except curses.error:
                    pass
                self.col += len(sub)
        self._refresh()


# ---------------------------------------------------------------------
# ncurses loop
# ---------------------------------------------------------------------
def emit_words_ncurses(stdscr, chunks: Iterable[str]) -> None:
    nat = NaturalLang()
    sink = CursesSink(stdscr, nat)
    code_live = LiveCodeBlock(sink)

    buf = ""
    at_line_start = True
    skip_line = False
    stop_all = False
    last_was_newline = False
    prev_word_lower = ""
    current_line_style = None

    stdscr.nodelay(True)

    for chunk in chunks:
        if not chunk or stop_all:
            continue
        chunk = chunk.expandtabs(TAB_WIDTH)
        buf += chunk
        i = 0
        L = len(buf)
        while i < L and not stop_all:
            try:
                key = stdscr.getch()
                if key in (ord("q"), ord("Q")):
                    stop_all = True
                    break
            except Exception:
                pass

            ch = buf[i]
            if ch.isspace():
                j = i
                has_nl = False
                while j < L and buf[j].isspace():
                    if buf[j] == "\n":
                        has_nl = True
                    j += 1
                if has_nl:
                    if code_live.active:
                        code_live.feed("\n")
                    else:
                        if not skip_line and not last_was_newline:
                            sink.add_newline()
                    at_line_start = True
                    skip_line = False
                    last_was_newline = True
                    prev_word_lower = ""
                    current_line_style = None
                else:
                    if code_live.active:
                        code_live.feed(" ")
                    else:
                        if not skip_line and not at_line_start:
                            sink.add_space()
                    at_line_start = False
                    last_was_newline = False
                i = j
                continue

            j = i
            while j < L and not buf[j].isspace():
                j += 1
            word = buf[i:j]
            lower = word.lower().lstrip()

            # hard stop
            if "save results" in lower:
                stop_all = True
                break
            if prev_word_lower == "save" and lower.startswith("result"):
                stop_all = True
                break

            # code fence
            if lower == "```":
                if code_live.active:
                    code_live.finish()
                else:
                    code_live.start()
                nat.feed(lower)
                at_line_start = False
                last_was_newline = False
                prev_word_lower = lower
                i = j
                continue

            # line start noise
            if at_line_start and not code_live.active:
                tl = lower.lstrip("#").lstrip()
                for pfx in NOISE_LINE_PREFIXES:
                    if tl.startswith(pfx):
                        skip_line = True
                        break

            # inline noise
            if not skip_line and not code_live.active:
                if any(s in lower for s in NOISE_SUBSTRINGS):
                    skip_line = True

            nat.feed(lower)

            if code_live.active:
                code_live.feed(word)
            else:
                if at_line_start and not skip_line and not sink.in_code_block:
                    stripped = word.lstrip()
                    if stripped.startswith("#"):
                        current_line_style = "heading"
                    elif stripped in ("-", "*", "+") or re.match(r"\d+\.$", stripped):
                        current_line_style = "list"
                    else:
                        current_line_style = None

                if not skip_line:
                    if current_line_style == "heading":
                        sink.add_word(word, style="heading")
                    elif current_line_style == "list":
                        sink.add_word(word, style="list")
                    else:
                        sink.add_word(word)

            at_line_start = False
            last_was_newline = False
            prev_word_lower = lower
            i = j

        buf = buf[i:]
        if stop_all:
            break

    if code_live.active:
        code_live.finish()


# ---------------------------------------------------------------------
# HTTP streaming
# ---------------------------------------------------------------------
def stream_infer(url: str, img_name: str, img_bytes: bytes, use_curses: bool = True) -> int:
    endpoint = url.rstrip("/") + "/infer"
    files = {"file": (img_name, io.BytesIO(img_bytes), "image/png")}
    headers = {"Expect": ""}

    collector = ClipBuffer()

    try:
        with requests.post(endpoint, files=files, headers=headers, stream=True, timeout=TIMEOUT) as r:
            if r.status_code != 200:
                sys.stderr.write(f"/infer failed: HTTP {r.status_code}\n{r.text}\n")
                return 1

            def chunk_iter():
                for c in r.iter_content(chunk_size=4096, decode_unicode=True):
                    if c:
                        # normalize CRLF
                        yield c.replace("\r", "\n")

            tagless = strip_tags_stream(chunk_iter())
            mirrored = collector.wrap(tagless)

            if use_curses and curses is not None and sys.stdout.isatty():
                curses.wrapper(emit_words_ncurses, mirrored)
            else:
                for part in mirrored:
                    sys.stdout.write(part)
                    sys.stdout.flush()

    except KeyboardInterrupt:
        text = collector.get_text()
        clean = final_cleanup(text)
        if clean:
            copy_to_clipboard(clean)
        return 130
    except Exception as e:
        sys.stderr.write(f"Request error: {e}\n")
        text = collector.get_text()
        clean = final_cleanup(text)
        if clean:
            copy_to_clipboard(clean)
        return 1

    # normal exit → final cleanup → clipboard
    text = collector.get_text()
    clean = final_cleanup(text)
    if clean:
        if not copy_to_clipboard(clean):
            print("[warn] could not copy text to clipboard — install xclip/xsel/wl-clipboard", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------
# screen capture
# ---------------------------------------------------------------------
def grab_full_desktop() -> tuple["Image.Image", dict]:
    with mss.mss() as sct:
        mon_all = sct.monitors[0]
        shot = sct.grab(mon_all)
        img = Image.frombytes("RGB", shot.size, shot.rgb)
        return img, mon_all


def select_region(img_full: "Image.Image", bbox: dict) -> Optional[Tuple[int, int, int, int]]:
    root = tk.Tk()
    root.overrideredirect(True)
    root.attributes("-topmost", True)

    left = int(bbox.get("left", 0))
    top = int(bbox.get("top", 0))
    w = int(bbox.get("width", img_full.size[0]))
    h = int(bbox.get("height", img_full.size[1]))

    try:
        root.geometry(f"{w}x{h}{'+' if left >= 0 else ''}{left}{'+' if top >= 0 else ''}{top}")
    except Exception:
        root.geometry(f"{w}x{h}+0+0")

    canvas = tk.Canvas(root, width=w, height=h, highlightthickness=0, bd=0)
    canvas.pack(fill="both", expand=True)

    photo = ImageTk.PhotoImage(img_full)
    canvas.create_image(0, 0, image=photo, anchor="nw")
    canvas.create_rectangle(0, 0, w, h, fill="#000000", stipple="gray50", outline="")

    sel = canvas.create_rectangle(0, 0, 0, 0, outline="#00ff00", width=2)

    start = {"x": 0, "y": 0}
    result = {"ok": False, "x1": 0, "y1": 0, "x2": 0, "y2": 0}

    canvas.create_text(
        20,
        20,
        anchor="nw",
        text="Drag to select area. Release to confirm. ESC to cancel.",
        fill="#ffffff",
        font=("TkDefaultFont", 14, "bold"),
    )

    def on_press(e):
        start["x"], start["y"] = e.x, e.y
        canvas.coords(sel, e.x, e.y, e.x, e.y)

    def on_drag(e):
        ex = max(0, min(e.x, w - 1))
        ey = max(0, min(e.y, h - 1))
        canvas.coords(sel, start["x"], start["y"], ex, ey)

    def on_release(e):
        ex = max(0, min(e.x, w - 1))
        ey = max(0, min(e.y, h - 1))
        canvas.coords(sel, start["x"], start["y"], ex, ey)
        x1, y1, x2, y2 = canvas.coords(sel)
        result["ok"] = True
        result["x1"] = int(x1)
        result["y1"] = int(y1)
        result["x2"] = int(x2)
        result["y2"] = int(y2)
        root.quit()

    def on_key(e):
        if e.keysym == "Escape":
            result["ok"] = False
            root.quit()

    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<B1-Motion>", on_drag)
    canvas.bind("<ButtonRelease-1>", on_release)
    root.bind("<Escape>", on_key)

    root.mainloop()
    try:
        root.destroy()
    except Exception:
        pass

    if not result["ok"]:
        return None

    x1 = result["x1"]
    y1 = result["y1"]
    x2 = result["x2"]
    y2 = result["y2"]
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return (x1, y1, x2, y2)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser("ocr live streamer (with cleaning + clipboard)")
    p.add_argument("image", nargs="?", help="Image path; if missing, capture from screen.")
    p.add_argument("--url", "-u", default=DEFAULT_URL, help=f"OCR API base url (default: {DEFAULT_URL})")
    p.add_argument("--no-curses", action="store_true", help="Disable ncurses.")
    args = p.parse_args(argv)

    use_curses = not args.no_curses

    if args.image:
        if not os.path.isfile(args.image):
            print(f"File not found: {args.image}", file=sys.stderr)
            return 2
        with open(args.image, "rb") as f:
            data = f.read()
        return stream_infer(args.url, os.path.basename(args.image), data, use_curses=use_curses)

    try:
        img, bbox = grab_full_desktop()
    except Exception as e:
        print(f"Failed to capture screen: {e}", file=sys.stderr)
        return 1

    box = select_region(img, bbox)
    if box is None:
        print("Cancelled.", file=sys.stderr)
        return 1

    x1, y1, x2, y2 = box
    if x2 - x1 < 2 or y2 - y1 < 2:
        print("Selection too small.", file=sys.stderr)
        return 1

    crop = img.crop((x1, y1, x2, y2))
    buf = io.BytesIO()
    crop.save(buf, format="PNG")

    return stream_infer(args.url, "selection.png", buf.getvalue(), use_curses=use_curses)


if __name__ == "__main__":
    sys.exit(main())

