"""
Document catalog for the SQL filter step: one row per indexed document in a SQLite database,
<out>/documents.sqlite, so a question can be limited to the documents an SQL condition picks before
hybrid search runs.

  question -> 0. SQL filter: SELECT doc_id FROM documents WHERE <condition>     (this file)
           -> 1. hybrid search (BM25 + HNSW) over those documents' chunks only   } rag_opensearch.py
           -> 2. re-rank -> 3. answer with citations                             }

Build or refresh it (rag_opensearch.py index also refreshes the ZIP it loads):
  python doc_catalog.py build pilot_out_full                 # every ZIP with a log and a chunks file
  python doc_catalog.py build pilot_out_full --zip-id 008    # one ZIP
Look around:
  python doc_catalog.py fields
  python doc_catalog.py query "SELECT kind, COUNT(*) AS docs FROM documents GROUP BY kind ORDER BY docs DESC"
Use it:
  python rag_opensearch.py ask "..." --sql "kind = 'pdf' AND year BETWEEN 2004 AND 2006"
  python rag_opensearch.py search "..." --sql "organization LIKE '%USDA%'"
  interactive mode: type  sql: year >= 2005  to set it, and  sql:  to clear it

The condition is ordinary SQLite SQL. It can also be a whole SELECT (or WITH) whose first column is
doc_id, e.g. to keep only the newest file of each title:
  --sql "SELECT doc_id FROM documents d WHERE modified = (SELECT MAX(modified) FROM documents WHERE title = d.title)"
Filtering opens the database read-only, so a condition can't change it.
"""

import argparse
import csv
import json
import re
import sqlite3
import sys
import time
from pathlib import Path

try:
    import rag_opensearch as R
except ImportError:
    sys.exit("Put doc_catalog.py in the same folder as rag_opensearch.py")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CATALOG = "documents.sqlite"
DEFAULT_PATH = str(Path("pilot_out_full") / CATALOG)
MAX_TERMS = 65536   # OpenSearch's index.max_terms_count: the most doc_ids one terms filter may list

COLUMNS = [  # name, SQLite type, what it holds
    ("doc_id", "TEXT PRIMARY KEY", "file number, e.g. 001644 (the same doc_id as in OpenSearch)"),
    ("zip", "TEXT", "GovDocs1 ZIP, e.g. 001"),
    ("file", "TEXT", "file name, e.g. 001644.pdf"),
    ("ext", "TEXT", "file extension, e.g. pdf, doc, xls, html"),
    ("kind", "TEXT", "pdf, word, excel, powerpoint, csv, html, text or other"),
    ("mime", "TEXT", "file type Tika detected"),
    ("bytes", "INTEGER", "file size"),
    ("title", "TEXT", "document title"),
    ("author", "TEXT", "author from the file's metadata"),
    ("organization", "TEXT", "company or publisher from the file's metadata, e.g. USDA (often empty)"),
    ("software", "TEXT", "program that made the file, e.g. Microsoft Word 10.0"),
    ("year", "INTEGER", "year from the file's metadata (the same as OpenSearch's year)"),
    ("created", "TEXT", "creation date, YYYY-MM-DD"),
    ("modified", "TEXT", "last-modified date, YYYY-MM-DD"),
    ("pages", "INTEGER", "pages (PDF and Office files)"),
    ("chars", "INTEGER", "characters of text Tika extracted"),
    ("route", "TEXT", "how it was parsed: docling, legacy, text, sheet, csv, docling->tika, ..."),
    ("chunks", "INTEGER", "chunks in the index"),
    ("tables", "INTEGER", "table chunks in the index"),
]
NAMES = [c[0] for c in COLUMNS]
INDEXED = ["zip", "kind", "year", "created", "organization", "author"]

DATE = re.compile(r"\b(19[5-9]\d|20[0-4]\d)-([01]\d)-([0-3]\d)")
CREATED_KEYS = ("dcterms:created", "pdf:docinfo:created", "meta:creation-date", "creation-date",
                "xmp:createdate", "created", "date")
MODIFIED_KEYS = ("dcterms:modified", "pdf:docinfo:modified", "last-modified", "meta:save-date",
                 "xmp:modifydate", "modified", "last-save-date")


# ---------------------------------------------------------------- metadata
def _lower(meta):
    return {k.lower(): v for k, v in meta.items()}


def meta_date(meta, keys):
    low = _lower(meta)
    for key in keys:
        if key in low:
            m = DATE.search(R.first_value(low[key]))
            if m:
                return m.group(0)
    return None


def meta_ending(meta, endings, limit=200):
    """First non-empty value whose key ends with one of `endings`, tried in that order."""
    low = _lower(meta)
    for end in endings:
        for k, v in low.items():
            if k.endswith(end):
                val = " ".join(R.first_value(v).split())
                if val:
                    return val[:limit]
    return None


def as_int(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- build
def zip_ids(out):
    """ZIPs the pilot has a log for (cmd_build skips those without a chunks file)."""
    return sorted(p.stem[len("log_"):] for p in Path(out).glob("log_*.csv"))


def zip_rows(out, zip_id):
    """One row per document that has chunks in this ZIP (the documents the index holds)."""
    out = Path(out)
    log = {}
    with open(out / f"log_{zip_id}.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            log[Path(r.get("file", "")).stem] = r
    docs = {}
    with open(out / "chunks" / f"{zip_id}.jsonl", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            c = json.loads(line)
            d = docs.setdefault(c["doc_id"], {"file": R.file_of(c), "title": c.get("title") or "",
                                               "mime": c.get("mime", ""), "route": c.get("route", ""),
                                               "pages": c.get("pages") or None, "chunks": 0, "tables": 0})
            d["chunks"] += 1
            d["tables"] += c.get("type") == "table"

    rows = []
    for doc_id, d in docs.items():
        rec = log.get(doc_id, {})
        meta, tika_title = {}, ""
        tj = out / "parsed" / zip_id / f"{doc_id}.tika.json"
        if tj.exists():
            try:
                j = json.loads(tj.read_text(encoding="utf-8"))
                meta, tika_title = j.get("metadata") or {}, j.get("title") or ""
            except (ValueError, OSError):
                pass
        ext = Path(d["file"]).suffix.lower().lstrip(".")
        rows.append({
            "doc_id": doc_id, "zip": zip_id, "file": d["file"], "ext": ext,
            "kind": R.kind_of(d["mime"], ext), "mime": d["mime"], "bytes": as_int(rec.get("bytes")),
            "title": (d["title"] or tika_title or "").strip()[:300] or None,
            "author": R.doc_author(meta),
            "organization": meta_ending(meta, ("company", "publisher")),
            "software": meta_ending(meta, ("creatortool", "creator_tool", ":application", "producer")),
            "year": R.doc_year(meta), "created": meta_date(meta, CREATED_KEYS),
            "modified": meta_date(meta, MODIFIED_KEYS),
            "pages": as_int(d["pages"]) or as_int(rec.get("pages")), "chars": as_int(rec.get("tika_chars")),
            "route": d["route"] or rec.get("route") or None, "chunks": d["chunks"], "tables": d["tables"],
        })
    return rows


def open_rw(path):
    con = sqlite3.connect(str(path))
    con.execute(f"CREATE TABLE IF NOT EXISTS documents ({', '.join(f'{n} {t}' for n, t, _ in COLUMNS)})")
    have = {r[1] for r in con.execute("PRAGMA table_info(documents)")}
    for n, t, _ in COLUMNS:       # a catalog from an older version of this script gets the new columns
        if n not in have:
            con.execute(f"ALTER TABLE documents ADD COLUMN {n} {t.replace(' PRIMARY KEY', '')}")
    for n in INDEXED:
        con.execute(f"CREATE INDEX IF NOT EXISTS documents_{n} ON documents ({n})")
    return con


def update_zip(out, zip_id, path=None):
    """Replace one ZIP's rows. Returns the number of documents written."""
    rows = zip_rows(out, zip_id)
    con = open_rw(path or Path(out) / CATALOG)
    try:
        with con:
            con.execute("DELETE FROM documents WHERE zip = ?", (zip_id,))
            con.executemany(f"INSERT OR REPLACE INTO documents ({', '.join(NAMES)}) "
                            f"VALUES ({', '.join('?' * len(NAMES))})", [tuple(r[n] for n in NAMES) for r in rows])
    finally:
        con.close()
    return len(rows)


# ---------------------------------------------------------------- filter
def open_ro(path):
    p = Path(path)
    if not p.exists():
        raise ValueError(f"no document catalog at {p}. Build it with: python doc_catalog.py build pilot_out_full")
    return sqlite3.connect(p.resolve().as_uri() + "?mode=ro", uri=True)


def to_select(condition):
    """A WHERE condition becomes SELECT doc_id FROM documents WHERE ...; a whole SELECT/WITH is kept."""
    sql = condition.strip().rstrip(";").strip()
    if not re.match(r"(?is)^(select|with)\b", sql):
        sql = f"SELECT doc_id FROM documents WHERE {sql}"
    return sql


def select_doc_ids(condition, path):
    """Run the condition read-only. Returns (doc_ids in order, documents in the catalog, SQL run)."""
    sql = to_select(condition)
    con = open_ro(path)
    try:
        ids = list(dict.fromkeys(str(r[0]) for r in con.execute(sql) if r and r[0] is not None))
        total = con.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        if ids:   # a whole SELECT must return doc_ids, not some other column
            known = {r[0] for r in con.execute(
                f"SELECT doc_id FROM documents WHERE doc_id IN ({','.join('?' * min(len(ids), 500))})",
                ids[:500])}
            if not known:
                raise ValueError("the SQL's first column doesn't look like doc_id (none of its values is a "
                                 "doc_id in the catalog)")
    except sqlite3.Error as e:
        raise ValueError(f"SQL error: {e}\n  SQL run: {sql}") from None
    finally:
        con.close()
    return ids, total, sql


def all_doc_ids(path):
    con = open_ro(path)
    try:
        return [r[0] for r in con.execute("SELECT doc_id FROM documents")]
    finally:
        con.close()


def opensearch_filter(condition, path=DEFAULT_PATH):
    """SQL condition -> (OpenSearch filter on doc_id or None, one-line summary). The filter goes inside
    both the BM25 and the k-NN part of the hybrid query, so both search only the chosen documents' chunks."""
    t0 = time.perf_counter()
    ids, total, _ = select_doc_ids(condition, path)
    ms = (time.perf_counter() - t0) * 1000
    note = f"SQL filter: {len(ids):,} of {total:,} documents ({ms:.0f} ms)"
    if not ids:
        return {"bool": {"must_not": [{"match_all": {}}]}}, note + "; nothing matches, so searches find nothing"
    if len(ids) >= total:
        return None, note + "; that is every document, so no filter is needed"
    if len(ids) <= MAX_TERMS:
        return {"terms": {"doc_id": ids}}, note
    rest = sorted(set(all_doc_ids(path)) - set(ids))
    if len(rest) <= MAX_TERMS:       # shorter to list the documents left out
        return {"bool": {"must_not": [{"terms": {"doc_id": rest}}]}}, note
    parts = [ids[i:i + MAX_TERMS] for i in range(0, len(ids), MAX_TERMS)]
    return {"bool": {"should": [{"terms": {"doc_id": p}} for p in parts], "minimum_should_match": 1}}, note


# ---------------------------------------------------------------- commands
def print_table(cur, rows, width=40):
    names = [d[0] for d in cur.description]
    cells = [[("" if v is None else str(v)).replace("\n", " ")[:width] for v in r] for r in rows]
    widths = [max([len(n)] + [len(r[i]) for r in cells]) for i, n in enumerate(names)]
    print("  " + "  ".join(n.ljust(w) for n, w in zip(names, widths)))
    print("  " + "  ".join("-" * w for w in widths))
    for r in cells:
        print("  " + "  ".join(v.ljust(w) for v, w in zip(r, widths)))


def cmd_build(args):
    out = Path(args.out)
    path = Path(args.catalog) if args.catalog else out / CATALOG
    ids = [args.zip_id] if args.zip_id else zip_ids(out)
    if not ids:
        sys.exit(f"No log_<zip>.csv files in {out}")
    t0 = time.perf_counter()
    for z in ids:
        if not (out / "chunks" / f"{z}.jsonl").exists():
            print(f"  ZIP {z}: no chunks file (the pilot didn't finish it), skipped")
            continue
        print(f"  ZIP {z}: {update_zip(out, z, path):,} documents")
    con = sqlite3.connect(str(path))
    n, zips = con.execute("SELECT COUNT(*), COUNT(DISTINCT zip) FROM documents").fetchone()
    con.close()
    print(f"Catalog {path}: {n:,} documents from {zips} ZIP(s) ({time.perf_counter() - t0:.0f} s)")
    print("Build it after indexing, so it lists the same ZIPs as the index; "
          "rag_opensearch.py index keeps it up to date from then on.")


def cmd_fields(args):
    con = open_ro(args.catalog)
    try:
        total = con.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        print(f"documents: {total:,} rows in {args.catalog}\n")
        for n, t, desc in COLUMNS:
            filled = con.execute(f"SELECT COUNT({n}) FROM documents").fetchone()[0]
            examples = [str(r[0])[:30] for r in con.execute(
                f"SELECT {n}, COUNT(*) c FROM documents WHERE {n} IS NOT NULL AND {n} != '' "
                f"GROUP BY {n} ORDER BY c DESC LIMIT 3")]
            print(f"  {n:<13}{t.split()[0]:<8}{filled / max(total, 1):>5.0%} filled  {desc}")
            if examples and n not in ("doc_id", "file", "chars", "bytes"):
                print(f"  {'':<21}e.g. {', '.join(examples)}")
    finally:
        con.close()


def cmd_query(args):
    con = open_ro(args.catalog)
    try:
        cur = con.execute(args.sql)
        rows = cur.fetchmany(args.limit)
        more = cur.fetchone() is not None
        print_table(cur, rows, args.width)
        print(f"  ({len(rows)} rows{'; more not shown, raise --limit' if more else ''})")
    except sqlite3.Error as e:
        sys.exit(f"SQL error: {e}")
    finally:
        con.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("build", help="create or refresh the catalog from the pilot output")
    sp.add_argument("out", help="pilot output folder, e.g. pilot_out_full")
    sp.add_argument("--zip-id", default=None, help="only this ZIP (default: all)")
    sp.add_argument("--catalog", default=None, help=f"database file (default <out>/{CATALOG})")
    sp.set_defaults(func=cmd_build)
    sp = sub.add_parser("fields", help="list the columns you can filter on, with examples")
    sp.add_argument("--catalog", default=DEFAULT_PATH)
    sp.set_defaults(func=cmd_fields)
    sp = sub.add_parser("query", help="run a read-only SQL query and print the result")
    sp.add_argument("sql")
    sp.add_argument("--catalog", default=DEFAULT_PATH)
    sp.add_argument("--limit", type=int, default=50)
    sp.add_argument("--width", type=int, default=40, help="characters shown per cell")
    sp.set_defaults(func=cmd_query)
    args = ap.parse_args()
    try:
        args.func(args)
    except ValueError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()