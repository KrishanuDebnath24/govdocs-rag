"""
GovDocs1 pilot v2: Tika first for every file, Docling only where shape pays off.

For each file in one extracted GovDocs1 ZIP folder:
  1. Tika detects the real type (images are skipped).
  2. Tika extracts text + metadata for EVERY other file (fast, consistent baseline).
  3. Gates decide whether the file is worth Docling's GPU time:
       - duplicates of a file already seen       -> skipped (in this ZIP or any earlier ZIP processed
                                                    into the same --out folder: <out>/seen_texts.sqlite)
       - little or no text (scanned/empty)       -> Tika text (OCR could be added later)
       - more pages than --max-pages              -> Tika text
       - HTML with more text than --max-html-chars -> Tika text (huge pages: statistics, dumps, books)
       - spreadsheets / CSV                       -> a small "profile" (first rows of each sheet)
  4. Docling rebuilds headings and tables (old Office files are converted by LibreOffice first),
     inside a worker process with a hard timeout. If it times out, fails or loses much of the
     text Tika found, the file falls back to Tika text.
  5. Chunking:
       - Docling documents: real headings (breadcrumb) + natural breaks (HybridChunker).
       - Tables in documents: a table that fits in 512 tokens is embedded whole (cleaned of
         markup that wastes tokens); a bigger one gets a "table card" with its column names, the
         label of every row and as many full rows as fit (if the labels don't all fit: a card
         with the first rows, plus up to 3 cards listing the other rows' labels).
         Spreadsheets/CSV: header + first rows of each sheet. Either way the whole table, as
         compact Markdown, is stored in every chunk's "table_md" field for the RAG app.
       - Tika text: Unstructured finds Title / NarrativeText / ListItem, titles become
         sections and breadcrumbs; inside a section text splits at paragraph/sentence breaks,
         or semantically with LlamaIndex (--chunker structural | hybrid | semantic).
       - All text is Unicode-normalized and broken "fi" glyphs from old PDFs are repaired.
       - Tika text of multi-page documents loses running headers/footers and page numbers
         (so they can't become fake section titles); junk metadata titles such as
         "Microsoft Word - report.doc" are cleaned; tiny chunks are merged with a neighbour;
         number-heavy chunks (flattened tables) are limited to --max-data-chunks per document.
       - Caps: --max-text-chunks for plain-text files, --max-doc-chunks for real documents.

Output (mirrors the S3 layout):
    <out>/parsed/<zip>/<doc_id>.tika.json      Tika text + metadata, every file
    <out>/parsed/<zip>/<doc_id>.docling.json   Docling structure, when used
    <out>/chunks/<zip>.jsonl                   one chunk record per line
    <out>/log_<zip>.csv                        one row per file

Usage (PowerShell):
    python pilot.py "D:\\rag corpus\\001" --zip-id 001 --sample 100
    python pilot.py "D:\\rag corpus\\001" --zip-id 001                 # whole folder

Needs: Tika server (Docker) on http://localhost:9998, LibreOffice, and
       pip install docling requests tqdm xlrd unstructured llama-index-core llama-index-embeddings-huggingface
       (+ CUDA torch for the GPU). Unstructured and LlamaIndex are optional: without them the
       script still runs and says what it is skipping.
"""

import argparse
import csv
import hashlib
import json
import logging
import multiprocessing as mp
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unicodedata
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import requests
from tqdm import tqdm

# ---------------------------------------------------------------- settings
TIKA = os.environ.get("TIKA_URL", "http://localhost:9998")
BUCKET = os.environ.get("RAG_BUCKET", "my-rag-corpus")   # only used to build source links
EMBED_MODEL = "BAAI/bge-base-en-v1.5"                    # tokenizer must match your embedder
EMBED_DIMS = 768
MAX_TOKENS = 512
TIKA_TIMEOUT = 120          # seconds per Tika call
LO_TIMEOUT = 90             # seconds per LibreOffice conversion
CORPUS_FILES = 986_000      # GovDocs1 size, for the projections in the summary
SPLIT_MODEL = "BAAI/bge-small-en-v1.5"   # small, fast model used only to find semantic breakpoints
TITLE_SCAN_MAX_CHARS = 1_000_000         # longer texts skip Unstructured title detection (speed)

DEFAULTS = {
    "docling_timeout": 90,   # hard limit per file; the worker is killed and restarted after it
    "max_pages": 60,         # longer documents use Tika text instead of Docling
    "min_text": 200,         # fewer Tika characters than this = scanned/empty, skip Docling
    "max_text_chunks": 100,  # cap for plain-text files (logs, data dumps)
    "max_doc_chunks": 1000,  # cap for real documents that fell back to Tika text (long PDFs etc.)
    "max_data_chunks": 3,    # number-heavy text chunks (flattened tables) kept per document
    "min_chunk_tokens": 64,  # smaller text chunks are merged with a neighbour when they fit
    "sheet_rows": 30,        # rows kept per sheet / CSV
    "max_sheets": 10,        # sheets kept per workbook
    "max_table_chars": 30000,  # whole tables up to this size are stored with their chunk
    "max_html_chars": 200000,  # HTML pages with more Tika text than this skip Docling (Tika text instead)
}
SHEET_FULL_MAX_ROWS = 2000   # sheets/CSVs longer than this are stored cut short (flagged)

warnings.filterwarnings("ignore")
os.environ.setdefault("UNSTRUCTURED_DISABLE_NOTICE", "1")

# ---------------------------------------------------------------- routing
DOCLING_MIMES = {
    "application/pdf", "text/html", "application/xhtml+xml",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}
LEGACY = {  # old Office formats -> LibreOffice -> Docling
    "application/msword": "docx",
    "application/rtf": "docx",
    "text/rtf": "docx",
    "application/vnd.ms-powerpoint": "pptx",
}
HTML_MIMES = {"text/html", "application/xhtml+xml"}
SHEET_MIMES = {
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}


def base_mime(mime):
    return mime.split(";")[0].strip().lower()


def kind_for(m, ext):
    if m.startswith("image/"):
        return "skip"
    if m == "text/csv" or (m == "text/plain" and ext == ".csv"):
        return "csv"
    if m in SHEET_MIMES:
        return "sheet"
    if m in DOCLING_MIMES:
        return "docling"
    if m in LEGACY:
        return "legacy"
    return "text"


# ---------------------------------------------------------------- tokenizer
from transformers import AutoTokenizer  # noqa: E402  (installed with docling)

HF_TOK = AutoTokenizer.from_pretrained(EMBED_MODEL)
HF_TOK.model_max_length = 10**9  # we only count tokens; silences length warnings


def ntok(text):
    return len(HF_TOK.encode(text, add_special_tokens=False))


# ---------------------------------------------------------------- Tika
DETECT_PATH = "/detect"  # Tika 4.x; set to /detect/stream at startup on 3.x


def tika_put(endpoint, data, name, accept, ok=(200,)):
    """PUT bytes to Tika, retrying briefly when the 4.x fork pool is busy (429/503)."""
    headers = {"Accept": accept, "Content-Disposition": f'attachment; filename="{name}"'}
    for attempt in range(3):
        r = requests.put(f"{TIKA}{endpoint}", data=data, headers=headers, timeout=TIKA_TIMEOUT)
        if r.status_code in (429, 503) and attempt < 2:
            time.sleep(float(r.headers.get("Retry-After", 2)))
            continue
        break
    if r.status_code not in ok:
        r.raise_for_status()
    return r


def find_detect_path():
    """Tika 4.x renamed /detect/stream to /detect. Try both so either version works."""
    status = None
    for path in ("/detect", "/detect/stream"):
        r = requests.put(f"{TIKA}{path}", data=b"%PDF-1.4\n%probe\n",
                         headers={"Content-Disposition": 'attachment; filename="probe.pdf"'},
                         timeout=30)
        status = r.status_code
        if status == 200:
            return path, r.text.strip()
    raise RuntimeError(f"neither /detect nor /detect/stream worked (last status {status})")


def detect(data, name):
    return tika_put(DETECT_PATH, data, name, "text/plain").text.strip()


def first_field(meta, *names, suffix):
    """Read a metadata field by name, or by suffix, since 4.x renamed some X-TIKA:* keys."""
    for n in names:
        if meta.get(n):
            return meta[n]
    for k, v in meta.items():
        if k.lower().endswith(suffix) and v:
            return v
    return ""


def tika_parse(data, name):
    """Text + metadata of a file and anything embedded in it (e.g. inside .gz)."""
    # 4.x answers 422 when a parse partly fails but still returns what it extracted.
    items = tika_put("/rmeta/text", data, name, "application/json", ok=(200, 422)).json() or []
    if isinstance(items, dict):  # an error body, not a list of documents
        raise RuntimeError(f"Tika: {items.get('status')} {items.get('message', '')}".strip())
    parts = [str(first_field(it, "X-TIKA:content", "tk:content", suffix=":content")).strip()
             for it in items]
    meta = items[0] if items else {}
    return "\n\n".join(p for p in parts if p).strip(), meta


TITLE_PREFIX = re.compile(r"^(microsoft\s+(office\s+)?(word|excel|powerpoint)|wordperfect|"
                          r"adobe\s+\w+)\s*[-–:]\s*", re.I)
TITLE_EXT = re.compile(r"\.(docx?|xlsx?|pptx?|pps|pdf|wpd|rtf|txt|html?)$", re.I)
JUNK_TITLE = re.compile(r"^(untitled|no title|title|document\s*\d*|slide\s*\d*|powerpoint presentation|"
                        r"presentation\s*\d*|book\s*\d*|sheet\s*\d*|microsoft word)$", re.I)


def clean_title(title):
    """'Microsoft Word - NCSTAR 1-4D Smoke.doc' -> 'NCSTAR 1-4D Smoke'; junk titles -> ''."""
    t = " ".join(str(title).split())
    t = TITLE_EXT.sub("", TITLE_PREFIX.sub("", t)).strip(" -_")
    if len(t) < 3 or not re.search(r"[A-Za-z]{2}", t) or JUNK_TITLE.match(t):
        return ""
    return t[:150]


def tika_title(meta):
    title = first_field(meta, "dc:title", "title", suffix=":title")
    if isinstance(title, list):
        title = title[0] if title else ""
    return clean_title(title)


def page_count(meta):
    for k, v in meta.items():
        if k.lower().endswith(("npages", "page-count", "pagecount", "slide-count", "slidecount")):
            try:
                return int(str(v[0] if isinstance(v, list) else v))
            except ValueError:
                pass
    return 0


def text_hash(text):
    return hashlib.sha1(" ".join(text.lower().split()).encode("utf-8")).hexdigest()


class SeenStore:
    """Text hashes of documents already processed, shared by every run that writes to the same output
    folder: later ZIPs, and parallel runs too (SQLite does the locking). The first copy of a text wins;
    later copies, in any ZIP, are logged as duplicates and skip Docling, chunking and embedding."""

    def __init__(self, path):
        import sqlite3
        self.path = path
        self.db = sqlite3.connect(str(path), timeout=60, isolation_level=None)  # autocommit
        self.db.execute("PRAGMA journal_mode=WAL")  # parallel runs read and write without blocking
        self.db.execute("CREATE TABLE IF NOT EXISTS seen (hash TEXT PRIMARY KEY, zip TEXT, file TEXT)")
        self.db.execute("CREATE TABLE IF NOT EXISTS registered (zip TEXT PRIMARY KEY)")

    def claim(self, h, zip_id, name):
        """None if this file is the first with this text (it is recorded now, atomically, so two parallel
        runs can't both win), else 'zip/file' of the first copy. Re-running a ZIP finds its own files."""
        if self.db.execute("INSERT OR IGNORE INTO seen VALUES (?, ?, ?)", (h, zip_id, name)).rowcount == 1:
            return None
        z, f = self.db.execute("SELECT zip, file FROM seen WHERE hash = ?", (h,)).fetchone()
        return None if (z, f) == (zip_id, name) else f"{z}/{f}"

    def register_existing(self, out, current_zip, min_text):
        """Record the documents of ZIPs processed into this folder before (once per ZIP, from their
        logs and Tika JSON), including ZIPs run before this check existed. Returns rows added."""
        done = {r[0] for r in self.db.execute("SELECT zip FROM registered")}
        added = 0
        for log in sorted(Path(out).glob("log_*.csv")):
            zip_id = log.stem[len("log_"):]
            if zip_id == current_zip or zip_id in done:
                continue
            with open(log, newline="", encoding="utf-8") as f:
                rows = [r for r in csv.DictReader(f) if r.get("status") == "ok"]
            for r in rows:
                tj = Path(out) / "parsed" / zip_id / f"{Path(r['file']).stem}.tika.json"
                try:
                    text = json.loads(tj.read_text(encoding="utf-8")).get("text") or ""
                except (OSError, ValueError):
                    continue
                if len(text.strip()) >= min_text:
                    added += self.db.execute("INSERT OR IGNORE INTO seen VALUES (?, ?, ?)",
                                             (text_hash(text), zip_id, r["file"])).rowcount
            self.db.execute("INSERT OR IGNORE INTO registered VALUES (?)", (zip_id,))
        return added

    def count(self):
        return self.db.execute("SELECT COUNT(*) FROM seen").fetchone()[0]


def first_copy(ctx, h, name):
    """'file' (this ZIP) or 'zip/file' (an earlier ZIP) of the first document with the same text,
    or None when this file is the first."""
    if h in ctx.seen:
        return ctx.seen[h]
    first = ctx.seen_db.claim(h, ctx.zip_id, name) if ctx.seen_db else None
    ctx.seen[h] = first or name
    return first


# ---------------------------------------------------------------- text cleanup
# Some older PDFs map the "fi" ligature to a wrong glyph, e.g. "signi›cant", "›re", "conÞrm".
BROKEN_FI = re.compile(r"(?<=[a-z])[›Þ](?=[a-z])|(?<![A-Za-z])[›Þ](?=[a-z]{2,})")


def clean_texts(texts):
    """NFKC-normalize (turns real ligatures like 'ﬁ' into 'fi', odd spaces into spaces).
    If a document shows the broken-'fi' pattern 3+ times, repair it. Returns (texts, fixes)."""
    texts = [unicodedata.normalize("NFKC", t) for t in texts]
    hits = sum(len(BROKEN_FI.findall(t)) for t in texts)
    if hits < 3:  # one or two could be genuine characters; only fix a systematic pattern
        return texts, 0
    return [BROKEN_FI.sub("fi", t) for t in texts], hits


def compact_md_table(md):
    """Docling pads table cells with spaces to line columns up; that wastes tokens."""
    out = []
    for line in md.splitlines():
        s = line.strip()
        if not s.startswith("|"):
            if s:
                out.append(s)
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if any(cells) and all(re.fullmatch(r":?-+:?", c) for c in cells if c):
            out.append("|" + "---|" * len(cells))  # separator row
        else:
            out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


# ---------------------------------------------------------------- chunking helpers
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def hard_split(text, budget):
    """Last resort for runs with no sentence breaks: cut on token boundaries."""
    offs = HF_TOK(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    out = []
    for i in range(0, len(offs), budget):
        j = min(i + budget, len(offs)) - 1
        out.append(text[offs[i][0]:offs[j][1]])
    return out


def pack(pieces, budget, sep):
    chunks, cur, cur_t = [], [], 0
    for p in pieces:
        t = ntok(p)
        if cur and cur_t + t > budget:
            chunks.append(sep.join(cur))
            cur, cur_t = [], 0
        cur.append(p)
        cur_t += t
    if cur:
        chunks.append(sep.join(cur))
    return chunks


def text_chunks(text, budget):
    """Boundary detector for plain text: paragraphs, then sentences, then tokens."""
    pieces = []
    for para in re.split(r"\n\s*\n", text):
        para = re.sub(r"[ \t]+", " ", para).strip()
        if not para:
            continue
        if ntok(para) <= budget:
            pieces.append(para)
            continue
        for sent in SENT_SPLIT.split(para):
            pieces.extend([sent] if ntok(sent) <= budget else hard_split(sent, budget))
    return pack(pieces, budget, "\n\n")


def table_preview(md, budget):
    """The part of a table that gets embedded: header + as many rows as fit + a 'more rows' note."""
    md = md.strip()
    if ntok(md) <= budget:
        return md
    lines = [ln for ln in md.splitlines() if ln.strip()]
    head, rows = lines[:2], lines[2:]
    used = ntok("\n".join(head)) + 16  # room for the note
    if used >= budget or not rows:     # header alone is too big: cut on tokens
        return hard_split(md, budget)[0]
    kept = []
    for row in rows:
        t = ntok(row)
        if used + t > budget:
            break
        kept.append(row)
        used += t
    if not kept:                        # first row alone is too big: cut it
        kept = [hard_split(rows[0], budget - used)[0]]
    more = len(rows) - len(kept)
    return "\n".join(head + kept + ([f"… ({more} more rows in the full table)"] if more else []))


# ---------------------------------------------------------------- tables: what gets embedded
# A table chunk's "text" is embedded, searched by BM25 and read by the re-ranker; its "table_md" (the
# whole table) is what the RAG app gives the LLM. "text" must fit in MAX_TOKENS, so:
#   - a table that fits is embedded whole, without markup that wastes tokens (bold, links, the
#     |---| line, empty columns, a title Docling repeats in every column name);
#   - a bigger table gets a TABLE CARD: column names + the label (first cell) of EVERY row + as many
#     full rows as still fit, so a question naming any row ("Van Nuys, CA") can find the table.
#     If the labels don't all fit, the first card keeps the first rows in full (like the old preview)
#     and up to MAX_TABLE_CARDS more cards list the other rows' labels. All cards carry the same
#     table_md, so the RAG app's near-duplicate step keeps only the best card of a table.
# Spreadsheets and CSVs keep their profile (header + first rows of each sheet).
MAX_TABLE_CARDS = 3
MAX_LABEL_CHARS = 80
MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
CELL_SPLIT = re.compile(r"(?<!\\)\|")
LEADER = re.compile(r"(?:\s*\.){4,}|_{4,}|-{4,}")  # ". . . . ." / "_____" / "-----" fillers in cells
BULLETS = {"•", "·", "▪", "◦", "o", "-", "*", "–", "—"}


def clean_cell(cell):
    """'**Lab 1**' -> 'Lab 1';  '[NW\\_001838064.2](http://...)' -> 'NW_001838064.2';
    'Overview . . . . . . 1' -> 'Overview 1'."""
    cell = MD_LINK.sub(r"\1", cell).replace("**", "").replace("\\_", "_").replace("\\|", "/")
    return " ".join(LEADER.sub(" ", cell).split())


def md_cells(line):
    s = line.strip()
    s = s[1:] if s.startswith("|") else s
    s = s[:-1] if s.endswith("|") and not s.endswith("\\|") else s
    return [clean_cell(c) for c in CELL_SPLIT.split(s)]


def is_separator(line):
    return "---" in line and not re.sub(r"[\s|:\-]", "", line)


def all_bold(line):
    cells = [c.strip() for c in CELL_SPLIT.split(line.strip().strip("|")) if c.strip()]
    return len(cells) >= 2 and all(c.startswith("**") and c.endswith("**") for c in cells)


def shared_title(names):
    """Docling copies a title spanning the whole table into every column name
    ("ATTACHMENT B: ... CASH PRICES 1 - Drug (B)"). Returns it when most columns start with it."""
    starts = [n.split(" - ", 1)[0] for n in names if " - " in n]
    if not starts:
        return ""
    title, count = Counter(starts).most_common(1)[0]
    return title if len(title) >= 15 and count >= 3 and count >= len(names) / 2 else ""


def parse_md_table(md):
    """Compact Markdown table -> (title, column names, rows), or None if it has no data rows.
    Header rows that Docling put below the separator (all bold, or repeating the first header cell)
    are merged into the column names: 'MM6/IL-6' over 'Lab 1' -> 'MM6/IL-6 Lab 1'. Columns empty in
    every row are dropped. If the 'header' is really data (bullets, long sentences), the table gets
    no column names and that line becomes its first row."""
    lines = [ln for ln in (md or "").splitlines() if ln.strip()]
    sep = next((i for i, ln in enumerate(lines) if is_separator(ln)), None)
    if not sep:  # no separator, or nothing above it
        return None
    head, data = [md_cells(ln) for ln in lines[:sep]], lines[sep + 1:]
    while len(data) > 1 and len(head) < 3:
        cells = md_cells(data[0])
        repeats = len(cells[0]) >= 3 and re.search(r"[A-Za-z]", cells[0]) and cells[0] == head[0][0]
        if not (all_bold(data[0]) or repeats):
            break
        head.append(cells)
        data.pop(0)
    rows = [r for r in (md_cells(ln) for ln in data) if any(r)]
    if not rows:
        return None
    width = max(len(r) for r in head + rows)
    names = []
    for j in range(width):
        parts = []
        for r in head:
            v = r[j] if j < len(r) else ""
            if v and v not in parts:
                parts.append(v)
        names.append(" ".join(parts))
    title = shared_title(names)
    if title:
        names = [n[len(title) + 3:] if n.startswith(title + " - ") else n for n in names]
    if len(head) == 1 and (any(n in BULLETS for n in names) or sum(len(n) > 60 for n in names) > width / 2):
        title, rows, names = "", [head[0]] + rows, [""] * width
    keep = [j for j in range(width) if names[j] or any(j < len(r) and r[j] for r in rows)]
    names = [names[j] for j in keep]
    rows = [[r[j] if j < len(r) else "" for j in keep] for r in rows]
    return title, names, rows


def row_line(cells):
    return "| " + " | ".join(cells) + " |"


def row_label(cells):
    label = next((c for c in cells if c and c not in BULLETS), "")
    return label if len(label) <= MAX_LABEL_CHARS else label[:MAX_LABEL_CHARS].rsplit(" ", 1)[0] + "..."


def table_texts(md, head):
    """Text(s) to embed for a table, each at most MAX_TOKENS with `head` (breadcrumb + caption) on
    top: the whole cleaned table if it fits, otherwise table card(s).
    Returns (kind, texts) with kind "whole", "card" or "cards", or None if the table can't be read."""
    # Normalize first (as clean_texts will later), so the token counts below are final:
    # NFKC turns e.g. a "…" dot leader into "...", three tokens instead of one.
    parsed = parse_md_table(unicodedata.normalize("NFKC", md))
    if not parsed:
        return None
    head = unicodedata.normalize("NFKC", head)
    title, names, rows = parsed
    budget = MAX_TOKENS - 8
    top = "\n".join(x for x in [head, title, row_line(names) if any(names) else ""] if x)
    if ntok(top) > budget // 2:  # very wide header or long breadcrumb: keep its start
        top = hard_split(top, budget // 2)[0]
    whole = top + "\n" + "\n".join(row_line(r) for r in rows)
    if ntok(whole) <= budget:
        return "whole", [whole]

    def labels_of(rs, skip=()):
        out = []
        for r in rs:
            lab = row_label(r)
            if lab and lab not in out and lab not in skip:
                out.append(lab)
        return out

    def card(k):  # first k rows in full, the labels of all the others
        shown = rows[:k]
        rest = labels_of(rows[k:], skip=set(labels_of(shown)))
        label = f"Other rows ({len(rows) - k} of {len(rows)})" if k else f"Rows ({len(rows)})"
        body = [row_line(r) for r in shown] + ([f"{label}: " + "; ".join(rest)] if rest else [])
        return top + "\n" + "\n".join(body)

    if ntok(card(0)) <= budget:  # every label fits: add full rows while they still fit
        lo, hi = 0, len(rows)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            lo, hi = (mid, hi) if ntok(card(mid)) <= budget else (lo, mid - 1)
        return "card", [card(lo)]

    # The labels don't all fit in one card. First card: the first rows in full (like the old preview).
    # Then up to MAX_TABLE_CARDS cards listing the labels of the other rows.
    lo, hi = 0, len(rows) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        lo, hi = (mid, hi) if ntok(top + "\n" + "\n".join(row_line(r) for r in rows[:mid])) <= budget else (lo, mid - 1)
    if lo:
        first = top + "\n" + "\n".join(row_line(r) for r in rows[:lo])
    else:  # even the first row is too long: keep its start
        first, lo = top + "\n" + hard_split(row_line(rows[0]), budget - ntok(top) - 1)[0], 1
    labels = labels_of(rows[lo:], skip=set(labels_of(rows[:lo])))
    cards, cur, used = [], [], 0
    room = budget - ntok(top) - 24
    for lab in labels:
        t = ntok(lab) + 1
        if cur and used + t > room:
            cards.append(cur)
            cur, used = [], 0
            if len(cards) == MAX_TABLE_CARDS:
                break
        cur.append(lab)
        used += t
    if cur and len(cards) < MAX_TABLE_CARDS:
        cards.append(cur)
    left = len(labels) - sum(len(c) for c in cards)
    texts = [first]
    for i, part in enumerate(cards, 1):
        more = f"; and {left} more" if left and i == len(cards) else ""
        texts.append(f"{top}\nOther rows ({len(rows) - lo} of {len(rows)}), part {i} of {len(cards)}: "
                     + "; ".join(part) + more)
    return "cards", texts


def make_table_chunks(md, crumb, caption, max_chars, preview_md=None):
    """Table preserver. Returns the chunk(s) for one table: one, or up to MAX_TABLE_CARDS cards for a
    big table. Every chunk carries the whole table as table_md, so the RAG app can return all of it."""
    head = f"{crumb}\n{caption}" if caption else crumb
    complete = len(md) <= max_chars
    full = md if complete else md[:max_chars].rsplit("\n", 1)[0]  # cut at a row boundary
    built = None if preview_md else table_texts(full, head)
    # spreadsheet/CSV profile, or a table with no data rows: header + first rows, as before
    texts = built[1] if built else [with_crumb(head, table_preview(preview_md or md, budget_for(head)))]
    return [{"type": "table", "breadcrumb": crumb, "text": t, "table_md": full, "table_complete": complete}
            for t in texts]


def with_crumb(crumb, text):
    return f"{crumb}\n{text}" if crumb else text


def budget_for(crumb):
    return MAX_TOKENS - ntok(crumb) - 8


# ---------------------------------------------------------------- Tika-path chunking
def find_sections(text, ctx):
    """Heading detector for plain text, using Unstructured's element labels
    (Title / NarrativeText / ListItem). Returns [(title, body), ...] or None."""
    if not ctx.partition_text or len(text) > TITLE_SCAN_MAX_CHARS:
        return None
    try:
        elements = ctx.partition_text(text=text)
    except Exception:
        return None
    is_title = [e.category == "Title" and 3 <= len(e.text.strip()) <= 120
                and re.search(r"[A-Za-z]{3}", e.text) is not None for e in elements]
    n_titles = sum(is_title)
    if n_titles == 0 or n_titles > 0.4 * len(elements):
        return None  # nothing found, or so many "titles" that it's probably a list, form or log
    sections, title, body = [], "", []
    for e, t in zip(elements, is_title):
        txt = " ".join(e.text.split()) if t else e.text.strip()
        if t:
            if body:
                sections.append((title, "\n\n".join(body)))
                title, body = txt, []
            else:  # title right after a title: keep both levels, e.g. "PART II › Methods"
                title = " › ".join(([title.split(" › ")[-1]] if title else []) + [txt])
        elif txt:
            body.append(f"- {txt}" if e.category == "ListItem" else txt)
    if body:
        sections.append((title, "\n\n".join(body)))
    return sections or None


def semantic_split(text, budget, ctx):
    """Embedding-based boundary detector: split where meaning shifts between sentences."""
    from llama_index.core import Document
    out = []
    for node in ctx.semantic.get_nodes_from_documents([Document(text=text)]):
        piece = node.get_content().strip()
        if piece:  # semantic pieces have no size limit, so re-split any that are too big
            out.extend([piece] if ntok(piece) <= budget else text_chunks(piece, budget))
    return out


NUM_TOKEN = re.compile(r"^[-+($]?\d[\d.,:/%]*[)%]?$")


def numeric_share(text):
    """Share of words that are numbers; high for flattened tables and data dumps."""
    words = text.split()
    return sum(bool(NUM_TOKEN.match(w)) for w in words) / max(len(words), 1)


def line_key(line):
    """Normalize a line so a running header/footer looks the same on every page."""
    k = " ".join(line.strip().lower().split())
    k = re.sub(r"\d+", "#", k)
    return re.sub(r"^[ivxlcdm]{1,7}\b|\b[ivxlcdm]{1,7}$", "#", k)  # roman page numbers


PAGE_NUMBER = re.compile(r"^(page\s*)?#(\s*(of|/)\s*#)?$|^-\s*#\s*-$")


def strip_repeated_lines(text, pages):
    """Remove running headers/footers and page numbers from a multi-page document's text.
    A wordy line repeated on at least ~30% of pages is treated as a header or footer."""
    lines = text.splitlines()
    keys = [line_key(ln) for ln in lines]
    counts = Counter(k for k, ln in zip(keys, lines) if 3 <= len(ln.strip()) <= 120)
    threshold = max(4, int(0.3 * pages))
    kept, removed = [], 0
    for k, ln in zip(keys, lines):
        s = ln.strip()
        boiler = bool(s) and (
            bool(PAGE_NUMBER.match(k))
            or (counts.get(k, 0) >= threshold and len(re.findall(r"[a-z]", k)) >= 3
                and numeric_share(s) < 0.5))  # repeated data rows are content, not headers
        if boiler:
            removed += 1
        else:
            kept.append(ln)
    return "\n".join(kept), removed


def merge_small(chunks, min_tokens):
    """Join a tiny text chunk with its neighbour (same document) when both fit in one chunk."""
    out = []
    for c in chunks:
        prev = out[-1] if out else None
        if (prev and prev["type"] == c["type"] == "text"
                and min(ntok(prev["text"]), ntok(c["text"])) < min_tokens
                and ntok(prev["text"]) + ntok(c["text"]) <= MAX_TOKENS):
            out[-1] = {**prev, "text": prev["text"] + "\n\n" + c["text"]}
        else:
            out.append(c)
    return out


def plain_text_chunks(text, label, cap, ctx, pages=0):
    """Chunk Tika text: drop running headers/footers, let Unstructured find titles (sections +
    breadcrumbs), split each section at natural breaks or semantically (see --chunker), merge
    tiny chunks, keep only a few number-heavy chunks. Returns (chunks, info)."""
    cfg = ctx.cfg
    info = {"truncated": False, "method": "", "boilerplate": 0, "data_dropped": 0}
    if pages >= 4:
        text, info["boilerplate"] = strip_repeated_lines(text, pages)
    # Only the start of very long texts is even looked at: about twice what `cap` full chunks need
    # (~4 characters per token). Title detection and semantic splitting cost time per character, so a
    # larger window (it was 8 per token) only made long files slower; they still keep `cap` chunks.
    limit = cap * MAX_TOKENS * 4
    info["truncated"] = len(text) > limit
    text = text[:limit]
    sections = find_sections(text, ctx)
    has_titles = bool(sections)
    sections = sections or [("", text)]
    chunks, used_semantic, data_kept = [], False, 0
    for title, body in sections:
        crumb = f"{label} › {title}" if title else label
        budget = budget_for(crumb)
        want_semantic = ctx.semantic is not None and (
            ctx.chunker == "semantic"
            or (ctx.chunker == "hybrid" and not has_titles and ntok(body) > 3 * budget))
        pieces = semantic_split(body, budget, ctx) if want_semantic else text_chunks(body, budget)
        used_semantic = used_semantic or want_semantic
        for p in pieces:
            if numeric_share(p) > 0.6:  # flattened table / data dump: keep a few as a sample
                if data_kept >= cfg["max_data_chunks"]:
                    info["data_dropped"] += 1
                    continue
                data_kept += 1
            chunks.append({"type": "text", "breadcrumb": crumb, "text": with_crumb(crumb, p)})
        if len(chunks) > cap:
            break
    chunks = merge_small(chunks, cfg["min_chunk_tokens"])
    info["truncated"] = info["truncated"] or len(chunks) > cap
    info["method"] = ("titles" if has_titles else "plain") + ("+semantic" if used_semantic else "")
    return chunks[:cap], info


# ---------------------------------------------------------------- spreadsheets and CSV
def rows_to_markdown(rows):
    rows = [[str(c).replace("|", "\\|").replace("\n", " ").strip() for c in r] for r in rows]
    rows = [r for r in rows if any(r)]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    keep = [j for j in range(width) if any(j < len(r) and r[j] for r in rows)]  # drop empty columns
    if not keep:
        return ""
    rows = [[(r[j] if j < len(r) else "") for j in keep] for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(keep)]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


def profile_chunks(sheets, name, cfg):
    """One chunk per sheet: its first rows are embedded, the whole sheet is stored with it."""
    chunks = []
    for sheet_name, rows in sheets:
        rows = [r for r in rows if any(str(c).strip() for c in r)]
        preview_md = rows_to_markdown(rows[: cfg["sheet_rows"] + 1])
        if not preview_md:
            continue
        full_md = rows_to_markdown(rows[: SHEET_FULL_MAX_ROWS + 1])
        crumb = f"{name} › {sheet_name}" if sheet_name else name
        for chunk in make_table_chunks(full_md, crumb, "", cfg["max_table_chars"], preview_md):
            if len(rows) > SHEET_FULL_MAX_ROWS + 1:
                chunk["table_complete"] = False
            chunks.append(chunk)
    return chunks


def sheet_profile(path, name, tika_text, cfg):
    limit = SHEET_FULL_MAX_ROWS + 2  # one row more than we keep, so "cut short" can be detected
    try:
        import pandas as pd  # installed with docling; .xls also needs: pip install xlrd
        book = pd.read_excel(path, sheet_name=None, header=None, nrows=limit, dtype=str)
        sheets = [(str(s), df.fillna("").values.tolist()) for s, df in list(book.items())[: cfg["max_sheets"]]]
        how = "pandas"
    except Exception:  # unreadable by pandas: Tika's tab-separated text
        lines = [ln for ln in tika_text.splitlines() if ln.strip()][:limit]
        sheets = [("", [ln.split("\t") for ln in lines])]
        how = "tika text"
    return profile_chunks(sheets, name, cfg), how


def csv_profile(path, name, cfg):
    raw = path.read_bytes()[: 2_000_000].decode("utf-8", errors="replace")
    try:
        dialect = csv.Sniffer().sniff(raw[:5000])
    except csv.Error:
        dialect = csv.excel
    rows = [r for r in csv.reader(raw.splitlines(), dialect) if any(c.strip() for c in r)]
    return profile_chunks([("", rows[: SHEET_FULL_MAX_ROWS + 2])], name, cfg)


# ---------------------------------------------------------------- LibreOffice
def find_soffice():
    for cand in (os.environ.get("SOFFICE"), shutil.which("soffice"),
                 r"C:\Program Files\LibreOffice\program\soffice.exe",
                 r"C:\Program Files (x86)\LibreOffice\program\soffice.exe"):
        if cand and Path(cand).exists():
            return cand
    return None


def libreoffice_convert(path, target, soffice, tmp):
    if not soffice:
        raise RuntimeError("LibreOffice not found")
    outdir = Path(tmp) / "lo_out"
    outdir.mkdir(exist_ok=True)
    profile = (Path(tmp) / "lo_profile").resolve().as_uri()  # private profile avoids lock clashes
    try:
        subprocess.run(
            [soffice, f"-env:UserInstallation={profile}", "--headless",
             "--convert-to", target, "--outdir", str(outdir), str(path)],
            check=True, timeout=LO_TIMEOUT, capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"LibreOffice could not convert the file (exit code {e.returncode})") from None
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"LibreOffice took over {LO_TIMEOUT}s") from None
    out = outdir / f"{path.stem}.{target}"
    if not out.exists():
        raise RuntimeError("LibreOffice produced no output")
    return out


# ---------------------------------------------------------------- Docling (in a worker process)
def make_converter(cfg):
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    opts = PdfPipelineOptions()
    opts.do_ocr = False              # scanned PDFs are caught by the min-text gate instead
    opts.do_table_structure = True   # TableFormer rebuilds table cells
    if hasattr(opts, "document_timeout"):
        opts.document_timeout = max(10, cfg["docling_timeout"] - 10)
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})


def make_chunker():
    from docling.chunking import HybridChunker
    try:
        from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
        # leave room for the document title we add to each chunk's breadcrumb
        tok = HuggingFaceTokenizer(tokenizer=HF_TOK, max_tokens=MAX_TOKENS - 32)
        return HybridChunker(tokenizer=tok, merge_peers=True)
    except ImportError:  # older docling-core
        return HybridChunker(tokenizer=EMBED_MODEL, max_tokens=MAX_TOKENS - 32, merge_peers=True)


def label_of(item):
    lab = getattr(item, "label", "")
    return str(getattr(lab, "value", lab)).lower()


def docling_title(doc, meta_title):
    """The document's title: cleaned metadata title, else the title Docling found on page 1."""
    if meta_title:
        return meta_title
    for item in getattr(doc, "texts", []) or []:
        if label_of(item) == "title":
            t = clean_title(getattr(item, "text", ""))
            if t:
                return t
    return ""


def docling_chunks(doc, chunker, meta_title, file_name, max_table_chars):
    """Heading detector + boundary detector (HybridChunker) + table preserver.
    Breadcrumbs start with the document title, so 'Summary' becomes 'Report title › Summary'."""
    title = docling_title(doc, meta_title)
    root = title or file_name
    out, table_crumbs = [], {}
    for ch in chunker.chunk(dl_doc=doc):
        items = ch.meta.doc_items or []
        heads = [h for h in (ch.meta.headings or []) if h]
        add_title = bool(title) and not (heads and heads[0].strip().lower() == title.lower())
        crumb = " › ".join(([title] if add_title else []) + heads) or root
        labels = [label_of(it) for it in items]
        for it, lab in zip(items, labels):
            if lab == "table":
                table_crumbs[it.self_ref] = crumb
        if items and all(lab == "table" for lab in labels):
            continue  # tables are added whole below, so don't index them twice
        text = chunker.contextualize(chunk=ch)  # = heading lines + text
        if not heads:
            text = with_crumb(crumb, text)
        elif add_title:
            text = with_crumb(title, text)
        out.append({"type": "text", "breadcrumb": crumb, "text": text})

    for tbl in doc.tables:
        try:
            md = tbl.export_to_markdown(doc=doc)
        except TypeError:
            md = tbl.export_to_markdown()
        md = compact_md_table(md)
        if not md.strip():
            continue
        crumb = table_crumbs.get(tbl.self_ref, root)
        try:
            caption = (tbl.caption_text(doc) or "").strip()
        except Exception:
            caption = ""
        out.extend(make_table_chunks(md, crumb, caption, max_table_chars))
    return out


def _docling_worker(conn, cfg):
    """Runs in a child process so a stuck file can be killed without losing the whole run."""
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.ERROR)
    conv, chunker = make_converter(cfg), make_chunker()
    conn.send(("ready", None))
    while True:
        job = conn.recv()
        if job is None:
            break
        src, meta_title, file_name, json_path = job
        try:
            doc = conv.convert(src).document
            md_len = len(doc.export_to_markdown())
            chunks = []
            if md_len >= 50:
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(doc.export_to_dict(), f, ensure_ascii=False, default=str)
                chunks = docling_chunks(doc, chunker, meta_title, file_name, cfg["max_table_chars"])
            conn.send(("ok", (md_len, chunks)))
        except Exception as e:
            conn.send(("error", f"{type(e).__name__}: {e}"))


class DoclingRunner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.proc = self.conn = None
        self.restarts = 0
        self._start()

    def _start(self):
        parent, child = mp.Pipe()
        self.proc = mp.Process(target=_docling_worker, args=(child, self.cfg), daemon=True)
        self.proc.start()
        self.conn = parent
        if not self.conn.poll(1800):  # first run downloads models
            raise RuntimeError("Docling worker did not start")
        self.conn.recv()

    def _restart(self):
        self.proc.kill()
        self.proc.join()
        self.restarts += 1
        self._start()

    def run(self, src, meta_title, file_name, json_path):
        timeout = self.cfg["docling_timeout"]
        self.conn.send((str(src), meta_title, file_name, str(json_path)))
        if not self.conn.poll(timeout):
            self._restart()
            raise TimeoutError(f"Docling took over {timeout}s")
        try:
            status, payload = self.conn.recv()
        except EOFError:  # the worker crashed (e.g. out of GPU memory)
            self._restart()
            raise RuntimeError("Docling worker crashed")
        if status != "ok":
            raise RuntimeError(payload)
        return payload

    def close(self):
        try:
            self.conn.send(None)
            self.proc.join(timeout=10)
        finally:
            if self.proc.is_alive():
                self.proc.kill()


# ---------------------------------------------------------------- one file
FIELDS = ["file", "ext", "bytes", "mime", "route", "chunker", "status", "seconds", "pages",
          "tika_chars", "docling_chars", "chunks", "note", "error"]


def add_note(rec, note):
    rec["note"] = f"{rec['note']}; {note}" if rec["note"] else note


def process(path, ctx):
    cfg = ctx.cfg
    name, ext, doc_id = path.name, path.suffix.lower(), path.stem
    rec = dict.fromkeys(FIELDS, "")
    rec.update(file=name, ext=ext, bytes=path.stat().st_size, status="ok", pages=0,
               tika_chars=0, docling_chars=0, chunks=0)
    chunks, info = [], None
    t0 = time.perf_counter()
    try:
        data = path.read_bytes()
        rec["mime"] = detect(data, name)
        m = base_mime(rec["mime"])
        kind = rec["route"] = kind_for(m, ext)
        if kind == "skip":
            rec["status"] = "skipped"
            return rec, chunks, {}

        # 1. Tika baseline for every file (cleaned: Unicode normalized, broken "fi" repaired)
        text, meta = tika_parse(data, name)
        (text,), fixes = clean_texts([text])
        if fixes:
            add_note(rec, f"Tika text: repaired {fixes} broken 'fi' glyphs")
        title, pages = tika_title(meta), page_count(meta)
        label = title or name
        rec.update(tika_chars=len(text), pages=pages)
        with open(ctx.parsed_dir / f"{doc_id}.tika.json", "w", encoding="utf-8") as f:
            json.dump({"mime": m, "title": title, "pages": pages, "metadata": meta, "text": text},
                      f, ensure_ascii=False, default=str)

        # 2. Duplicates: same text as a file already processed, in this ZIP or an earlier one
        if len(text.strip()) >= cfg["min_text"]:
            first = first_copy(ctx, text_hash(text), name)
            if first:
                rec.update(status="duplicate", note=f"same text as {first}")
                return rec, chunks, {}

        # 3. Shape where it pays off, else Tika text
        cap = cfg["max_text_chunks"]
        if kind == "csv":
            chunks = csv_profile(path, name, cfg)
            rec["chunker"] = "profile"
        elif kind == "sheet":
            chunks, how = sheet_profile(path, name, text, cfg)
            rec["chunker"] = f"profile ({how})"
        elif kind in ("docling", "legacy"):
            reason = None
            if len(text.strip()) < cfg["min_text"]:
                reason = "little text in Tika (scanned or empty?)"
            elif pages > cfg["max_pages"]:
                reason = f"{pages} pages, over the {cfg['max_pages']}-page limit"
            elif m in HTML_MIMES and len(text) > cfg["max_html_chars"]:
                # Huge pages (access statistics, data dumps, whole books) are slow in Docling and often
                # time out; Tika text keeps up to --max-doc-chunks and drops number-heavy chunks
                reason = f"HTML page of {len(text):,} characters, over the {cfg['max_html_chars']:,} limit"
            else:
                converted = None
                try:
                    src = path
                    if kind == "legacy":
                        converted = src = libreoffice_convert(path, LEGACY[m], ctx.soffice, ctx.tmp)
                    md_len, chunks = ctx.docling.run(src, title, name,
                                                     ctx.parsed_dir / f"{doc_id}.docling.json")
                    rec["docling_chars"] = md_len
                    if md_len < 50:
                        reason = "Docling found no text"
                    elif len(text) > 2000 and md_len < 0.5 * len(text):
                        reason = f"Docling kept only {md_len / len(text):.0%} of Tika's text"
                    else:
                        cleaned, dfix = clean_texts([c["text"] for c in chunks])
                        for c, t in zip(chunks, cleaned):
                            c["text"] = t
                        if dfix:
                            add_note(rec, f"Docling text: repaired {dfix} broken 'fi' glyphs")
                        chunks = merge_small(chunks, cfg["min_chunk_tokens"])
                        rec["chunker"] = "docling"
                except Exception as e:
                    reason = f"{type(e).__name__}: {e}"
                finally:
                    if converted and converted.exists():
                        converted.unlink()
            if reason:
                rec["route"] = f"{kind}->tika"
                add_note(rec, reason[:200])
                cap = cfg["max_doc_chunks"]  # a real document: keep (nearly) all of it
                chunks, info = plain_text_chunks(text, label, cap, ctx, pages)
        else:
            chunks, info = plain_text_chunks(text, label, cap, ctx, pages)

        if info:
            rec["chunker"] = info["method"]
            if info["boilerplate"]:
                add_note(rec, f"removed {info['boilerplate']} header/footer lines")
            if info["data_dropped"]:
                add_note(rec, f"dropped {info['data_dropped']} number-heavy chunks")
            if info["truncated"]:
                add_note(rec, f"truncated to {cap} chunks")
        if not chunks:
            rec["status"] = "empty"
        return rec, chunks, {"title": title, "pages": pages}
    except Exception as e:
        rec["status"] = "failed"
        rec["error"] = f"{type(e).__name__}: {e}"[:300]
        return rec, [], {}
    finally:
        rec["seconds"] = round(time.perf_counter() - t0, 2)
        rec["chunks"] = len(chunks) if rec["status"] not in ("failed",) else 0


# ---------------------------------------------------------------- main
def main():
    global DETECT_PATH
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="folder with the extracted files of one GovDocs1 ZIP")
    ap.add_argument("--zip-id", default=None, help="ZIP number, e.g. 001 (default: folder name)")
    ap.add_argument("--out", default="pilot_out", help="output folder (default: pilot_out)")
    ap.add_argument("--sample", type=int, default=0, help="process a random sample of N files")
    ap.add_argument("--chunker", choices=["structural", "hybrid", "semantic"], default="hybrid",
                    help="how Tika-path text is split inside each section: structural = paragraph/"
                         "sentence breaks only; hybrid = semantic only for long text with no titles "
                         "(default); semantic = semantic everywhere on the Tika path. "
                         "Docling documents always use their real structure.")
    ap.add_argument("--per-zip-dedupe", action="store_true",
                    help="check duplicates only within this ZIP (default: also against every earlier ZIP "
                         "processed into the same --out folder, via <out>/seen_texts.sqlite)")
    for key, val in DEFAULTS.items():
        ap.add_argument(f"--{key.replace('_', '-')}", type=int, default=val, help=f"default {val}")
    args = ap.parse_args()
    cfg = {k: getattr(args, k) for k in DEFAULTS}

    logging.basicConfig(level=logging.ERROR)

    folder = Path(args.folder)
    zip_id = args.zip_id or folder.name
    out = Path(args.out)
    parsed_dir = out / "parsed" / zip_id
    parsed_dir.mkdir(parents=True, exist_ok=True)
    (out / "chunks").mkdir(parents=True, exist_ok=True)

    # Preflight checks
    try:
        requests.get(f"{TIKA}/tika", timeout=10).raise_for_status()
        DETECT_PATH, probe = find_detect_path()
    except Exception as e:
        sys.exit(f"Tika problem at {TIKA}: {type(e).__name__}: {e}\nStart it with: docker start tika")
    try:
        version = requests.get(f"{TIKA}/version", timeout=10).text.strip()
    except Exception:
        version = "unknown version"
    soffice = find_soffice()
    print(f"Tika: OK ({version}; detect via {DETECT_PATH})")
    print(f"LibreOffice: {soffice or 'NOT FOUND - old .doc/.ppt files will use Tika text'}")
    try:
        import xlrd  # noqa: F401
        print("xlrd: OK (.xls sheets read directly)")
    except ImportError:
        print("xlrd: missing - .xls profiles will use Tika text (pip install xlrd)")
    gpu = None
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        print(f"GPU: {gpu or 'not available - Docling will run on CPU (slower)'}")
    except ImportError:
        print("GPU: torch not installed")

    # Heading detector for plain text (Unstructured)
    partition_text = None
    try:
        from unstructured.partition.text import partition_text
        partition_text(text="Introduction\n\nThis sentence warms up the title detector.")  # fetches NLTK data once
        print("Unstructured: OK (finds titles in plain-text files)")
    except Exception as e:
        partition_text = None
        print(f"Unstructured: not available ({type(e).__name__}: {str(e)[:80]}) - "
              "plain-text files get no section titles (pip install unstructured)")

    # Semantic boundary detector (LlamaIndex + a small bge model)
    semantic = None
    chunker = args.chunker
    if chunker != "structural":
        try:
            from llama_index.core.node_parser import SemanticSplitterNodeParser
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            embed = HuggingFaceEmbedding(model_name=SPLIT_MODEL, device="cuda" if gpu else "cpu")
            semantic = SemanticSplitterNodeParser(buffer_size=1, breakpoint_percentile_threshold=95,
                                                  embed_model=embed)
            print(f"LlamaIndex semantic splitter: OK ({SPLIT_MODEL}, mode={chunker})")
        except Exception as e:
            print(f"LlamaIndex semantic splitter: not available ({type(e).__name__}: {str(e)[:80]}) - "
                  "using structural splitting (pip install llama-index-core llama-index-embeddings-huggingface)")
            chunker = "structural"
    print("Limits: " + ", ".join(f"{k}={v}" for k, v in cfg.items()) + f", chunker={chunker}")

    files = sorted(p for p in folder.rglob("*") if p.is_file())
    if args.sample and args.sample < len(files):
        files = sorted(random.Random(42).sample(files, args.sample))
    print(f"Files: {len(files)} from {folder}")

    seen_db = None
    if not args.per_zip_dedupe:
        seen_db = SeenStore(out / "seen_texts.sqlite")
        added = seen_db.register_existing(out, zip_id, cfg["min_text"])
        print(f"Duplicates across ZIPs: checked against {seen_db.count():,} documents in {seen_db.path}"
              + (f" ({added:,} registered now from earlier ZIPs)" if added else ""))
    else:
        print("Duplicates: checked within this ZIP only (--per-zip-dedupe)")

    print("Starting the Docling worker (the first run downloads its models)...")
    ctx = SimpleNamespace(cfg=cfg, soffice=soffice, parsed_dir=parsed_dir, seen={}, seen_db=seen_db,
                          zip_id=zip_id, docling=DoclingRunner(cfg), tmp=None, partition_text=partition_text,
                          semantic=semantic, chunker=chunker)

    log_path = out / f"log_{zip_id}.csv"
    chunk_path = out / "chunks" / f"{zip_id}.jsonl"
    stats = defaultdict(lambda: defaultdict(float))
    methods = defaultdict(lambda: [0, 0])  # chunking method -> [files, chunks]
    notes = Counter()
    t_start = time.perf_counter()

    try:
        with tempfile.TemporaryDirectory() as tmp, \
                open(log_path, "w", newline="", encoding="utf-8") as logf, \
                open(chunk_path, "w", encoding="utf-8") as chf:
            ctx.tmp = tmp
            log = csv.DictWriter(logf, fieldnames=FIELDS)
            log.writeheader()
            for path in tqdm(files, unit="file"):
                rec, chunks, extra = process(path, ctx)
                log.writerow(rec)
                logf.flush()
                source = f"s3://{BUCKET}/raw/govdocs1/zipfiles/{zip_id}.zip#{path.name}"
                for i, c in enumerate(chunks):
                    chf.write(json.dumps({
                        "chunk_id": f"{path.stem}-{i:03d}", "doc_id": path.stem, "zip": zip_id,
                        "source": source, "mime": base_mime(rec["mime"]), "route": rec["route"],
                        "title": extra.get("title", ""), "pages": extra.get("pages", 0),
                        "type": c["type"], "breadcrumb": c["breadcrumb"],
                        "tokens": ntok(c["text"]), "text": c["text"],
                        # tables: "text" is what gets embedded; "table_md" is what the RAG app returns
                        **({"table_md": c["table_md"], "table_complete": c["table_complete"]}
                           if "table_md" in c else {}),
                    }, ensure_ascii=False) + "\n")

                s = stats[rec["route"] or "?"]
                s["files"] += 1
                s[rec["status"]] += 1
                s["seconds"] += rec["seconds"]
                s["chunks"] += rec["chunks"]
                if rec["chunker"]:
                    methods[rec["chunker"]][0] += 1
                    methods[rec["chunker"]][1] += rec["chunks"]
                for msg in filter(None, [rec["error"]] + rec["note"].split("; ")):
                    notes[re.sub(r"\d+", "N", msg)[:110]] += 1

                done = sum(int(v["files"]) for v in stats.values())
                failed = sum(int(v["failed"]) for v in stats.values())
                if done == 10 and failed == 10:
                    tqdm.write("\nThe first 10 files all failed, so the run stopped. Error:")
                    tqdm.write("  " + rec["error"])
                    break
    finally:
        ctx.docling.close()

    # Summary
    total = time.perf_counter() - t_start
    head = f"{'route':<16}{'files':>6}{'ok':>5}{'empty':>7}{'dupl':>6}{'failed':>8}{'skip':>6}{'s/file':>8}{'chunks':>8}"
    print("\n" + head)
    for route, s in sorted(stats.items(), key=lambda kv: -kv[1]["files"]):
        print(f"{route:<16}{int(s['files']):>6}{int(s['ok']):>5}{int(s['empty']):>7}{int(s['duplicate']):>6}"
              f"{int(s['failed']):>8}{int(s['skipped']):>6}{s['seconds'] / s['files']:>8.2f}{int(s['chunks']):>8}")

    if methods:
        print("\nHow files were chunked (files / chunks):")
        for method, (nf, nc) in sorted(methods.items(), key=lambda kv: -kv[1][0]):
            print(f"  {method:<22}{nf:>6} files {nc:>8} chunks")

    if notes:
        print("\nFallbacks, caps, repairs and errors (count: reason):")
        for msg, count in notes.most_common(10):
            print(f"  {count:>4}: {msg}")

    n = sum(int(v["files"]) for v in stats.values())
    n_chunks = sum(int(v["chunks"]) for v in stats.values())
    failed = sum(int(v["failed"]) for v in stats.values())
    print(f"\nTotal: {n} files in {total / 60:.1f} min ({total / max(n, 1):.2f} s/file), "
          f"{n_chunks} chunks ({n_chunks / max(n, 1):.1f} per file)")
    if ctx.docling.restarts:
        print(f"Docling worker restarted {ctx.docling.restarts} time(s) after timeouts or crashes.")
    if n and failed <= n * 0.5:  # projections from mostly failed files would be meaningless
        proj_chunks = n_chunks / n * CORPUS_FILES
        print(f"Projection for all of GovDocs1 (rough, from this sample):")
        print(f"  ~{total / n * CORPUS_FILES / 3600:,.0f} hours on one machine like this one")
        print(f"  ~{proj_chunks / 1e6:,.0f}M chunks, ~{proj_chunks * EMBED_DIMS * 4 / 1e9:,.0f} GB of "
              f"float32 vectors at {EMBED_DIMS} dimensions")
    print(f"Log:    {log_path}\nChunks: {chunk_path}\nParsed: {parsed_dir}")


if __name__ == "__main__":
    main()