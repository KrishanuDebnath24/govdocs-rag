"""
Rebuild the table chunks of one ZIP with pilot.py's table logic, without re-parsing anything:
every table chunk already holds its whole table in "table_md".

What changes (only the "text" of table chunks from documents; spreadsheets/CSV stay as they are):
  - a table that fits in 512 tokens is embedded whole, cleaned of markup that wastes tokens
    (bold markers, link URLs, the |---| line, empty columns, a title repeated in every column name);
  - a bigger table gets a table card: column names + the label (first cell) of every row + as many
    full rows as fit. If the labels don't all fit, the first card keeps the first rows in full (like
    the old preview) and up to 3 more cards list the other rows' labels.
table_md (what the LLM reads) is unchanged, and the first card keeps the old chunk_id
(extra cards get -2, -3), so eval sets keep matching.

Usage (from D:\\rag corpus, same folder as pilot.py):
  python rebuild_tables.py pilot_out_full --zip-id 001 --dry-run     # report only, writes nothing
  python rebuild_tables.py pilot_out_full --zip-id 001
  python rag_opensearch.py index pilot_out_full --zip-id 001         # then re-index

The first run saves the original as chunks/<zip>.before_tables.jsonl and always rebuilds from it,
so running this again is safe. To undo: copy that file back over chunks/<zip>.jsonl and re-index.
"""

import argparse
import json
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    import pilot as P  # loads the bge tokenizer, so token counts match the embedder
except ImportError as e:
    sys.exit(f"Put rebuild_tables.py in the same folder as pilot.py ({e})")

SPREADSHEET_ROUTES = {"sheet", "csv"}


def norm(text):
    return " ".join(re.sub(r"[^0-9a-z]+", " ", (text or "").lower()).split())


def head_of(c):
    """Breadcrumb + caption: the lines above the table in the old text."""
    lines = []
    for ln in c["text"].splitlines():
        if ln.lstrip().startswith("|"):
            break
        lines.append(ln)
    crumb = c.get("breadcrumb", "")
    caption = "\n".join(lines[1:] if crumb and lines else lines).strip()
    return f"{crumb}\n{caption}" if caption else crumb


def rows_named(md, texts):
    """How many of the table's rows have their label in the embedded text(s)."""
    parsed = P.parse_md_table(md)
    if not parsed:
        return 0, 0
    joined = " " + norm("\n".join(texts)) + " "
    labels = [norm(P.row_label(r)) for r in parsed[2]]
    return sum(1 for lab in labels if lab and f" {lab} " in joined), len(labels)


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", help="pilot output folder, e.g. pilot_out_full")
    ap.add_argument("--zip-id", default="001")
    ap.add_argument("--dry-run", action="store_true", help="report what would change, write nothing")
    args = ap.parse_args()

    chunk_path = Path(args.out) / "chunks" / f"{args.zip_id}.jsonl"
    backup = chunk_path.with_name(f"{args.zip_id}.before_tables.jsonl")
    if not chunk_path.exists() and not backup.exists():
        sys.exit(f"Not found: {chunk_path}")
    source = backup if backup.exists() else chunk_path
    chunks = load(source)
    print(f"Read {len(chunks):,} chunks from {source}" + (" (the original, saved by an earlier run)" if source == backup else ""))

    t0 = time.perf_counter()
    out, stats = [], Counter()
    named_before = named_after = rows_total = 0
    for c in chunks:
        if c.get("type") != "table" or not c.get("table_md"):
            out.append(c)
            continue
        if c.get("route") in SPREADSHEET_ROUTES:
            stats["spreadsheet/CSV (unchanged)"] += 1
            out.append(c)
            continue
        built = P.table_texts(c["table_md"], head_of(c))
        if not built:
            stats["no data rows (unchanged)"] += 1
            out.append(c)
            continue
        kind, texts = built
        texts, _ = P.clean_texts(texts)  # same Unicode/"fi" cleanup the pilot applies to chunk text
        b, n = rows_named(c["table_md"], [c["text"]])
        a, _ = rows_named(c["table_md"], texts)
        named_before, named_after, rows_total = named_before + b, named_after + a, rows_total + n
        stats[{"whole": "embedded whole", "card": "one table card"}.get(kind, f"split over {len(texts)} cards")] += 1
        for j, text in enumerate(texts):
            out.append({**c, "chunk_id": c["chunk_id"] if j == 0 else f"{c['chunk_id']}-{j + 1}",
                        "text": text, "tokens": P.ntok(text)})

    tables = sum(1 for c in chunks if c.get("type") == "table")
    print(f"\n{tables:,} table chunks, rebuilt in {time.perf_counter() - t0:.0f}s:")
    for k, v in sorted(stats.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<32}{v:>6}")
    if rows_total:
        print(f"Rows whose label is in the embedded text: {named_before:,} -> {named_after:,} of {rows_total:,} "
              f"({named_before / rows_total:.0%} -> {named_after / rows_total:.0%})")
    longest = max((c["tokens"] for c in out if c.get("type") == "table"), default=0)
    print(f"Chunks: {len(chunks):,} -> {len(out):,} (+{len(out) - len(chunks):,});  longest table text: {longest} tokens")

    if args.dry_run:
        print("\nDry run: nothing written.")
        return
    if not backup.exists():
        shutil.copyfile(chunk_path, backup)
        print(f"\nOriginal saved as {backup}")
    tmp = chunk_path.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for c in out:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    tmp.replace(chunk_path)
    print(f"Wrote {chunk_path}")
    print(f"Next: python rag_opensearch.py index {args.out} --zip-id {args.zip_id}")


if __name__ == "__main__":
    main()