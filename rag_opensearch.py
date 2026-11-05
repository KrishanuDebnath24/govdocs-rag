"""
RAG over the GovDocs1 pilot with OpenSearch: HNSW vectors + BM25, hybrid search fused by RRF
inside OpenSearch, metadata filters, cross-encoder re-ranking, and an LLM answer with citations.
No LLM-generated document metadata (summaries/keywords/questions) is used.

  question -> 0. SQL filter (--sql): documents.sqlite picks the documents  (doc_catalog.py)
           -> 1. filter (kind, year, pages, ...) + those documents   } one OpenSearch hybrid query,
           -> 2. hybrid search: k-NN (HNSW) + BM25, RRF              } filters applied inside both parts
           -> 3. re-rank top 50 with a cross-encoder, keep 5
           -> 4. LLM answers from those 5 chunks (each with its neighbouring chunks, --expand), citing [1]..[5]

Start OpenSearch in Docker (PowerShell, one line; security off = local development only):
  docker run -d --name opensearch -p 9200:9200 -e "discovery.type=single-node" -e "DISABLE_INSTALL_DEMO_CONFIG=true" -e "DISABLE_SECURITY_PLUGIN=true" -e "OPENSEARCH_JAVA_OPTS=-Xms2g -Xmx2g" -v opensearch-data:/usr/share/opensearch/data opensearchproject/opensearch:latest

Commands (from D:\\rag corpus):
  python rag_opensearch.py setup                                   # create the index + RRF pipeline
  python rag_opensearch.py index  pilot_out_full --zip-id 001      # embed (or reuse vectors) and load
  python rag_opensearch.py search "stair pressurization" --filter "kind=pdf year>=2005"
  python rag_opensearch.py ask    "Would pressurized stairwells have helped in the WTC towers?"
  python rag_opensearch.py ask                                     # interactive
  python rag_opensearch.py eval   pilot_out_full --zip-id 001      # uses rag_local.py's test questions

Filters: space-separated conditions, e.g.  kind=pdf,word  year>=2005  pages<50  has_tables=true
         title~"forest health"  author~nist  zip=001  ext!=txt
Fields:  kind, ext, type, doc_id, zip, year, pages, tokens, has_tables, title, author
SQL filter: any SQLite condition on the document catalog (doc_catalog.py; index keeps it up to date), e.g.
  python rag_opensearch.py ask "..." --sql "kind = 'pdf' AND year BETWEEN 2004 AND 2006"
  python rag_opensearch.py search "..." --sql "organization LIKE '%USDA%' AND pages < 50"
  Columns: python doc_catalog.py fields. Interactive mode: 'sql: <condition>' sets it, 'sql:' clears it.

Needs: pip install opensearch-py sentence-transformers requests tqdm
       answers: Claude API (pip install anthropic; $env:ANTHROPIC_API_KEY = "sk-ant-..."), the default;
                or Ollama (--llm ollama; ollama pull qwen2.5:7b) or Bedrock (--llm bedrock; boto3 + a model ID)
Vectors are stored once (left out of _source). An index made before that change: setup --recreate, then
index each ZIP again; saved vectors in <out>/embeddings are reused, nothing is re-embedded.
"""

import argparse
import csv
import json
import os
import re
import shlex
import sys
import textwrap
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

OPENSEARCH_URL = "http://localhost:9200"
INDEX = "govdocs"
PIPELINE = "govdocs-rrf"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"      # same model and tokenizer as pilot.py / rag_local.py
DIM = 768
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # bge query instruction
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"   # faster / smaller: BAAI/bge-reranker-base
CANDIDATES = 50     # results from each retriever, and the hybrid list passed to the re-ranker
RRF_RANK_CONSTANT = 60

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------- small helpers
def load_jsonl(path):
    if not Path(path).exists():
        sys.exit(f"Not found: {path}")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def file_of(c):
    return c.get("file") or c["source"].split("#")[-1]


def body_of(c):
    """What a person reads: the whole table for tables, the text minus breadcrumb lines otherwise;
    for a passage (a chunk with its neighbours, see expand_hits) all of its chunks in order."""
    if c["type"] == "passage":
        return stitch(c["_chunks"], crumb=c["breadcrumb"])
    if c["type"] == "table":
        return (c.get("table_md") or c["text"]).strip()
    parts = {p.strip() for p in c["breadcrumb"].split("›")} | {c["breadcrumb"].strip()}
    lines = c["text"].split("\n")
    while lines and lines[0].strip() in parts:
        lines.pop(0)
    return "\n".join(lines).strip()


def kind_of(mime, ext):
    m = (mime or "").lower()
    if "pdf" in m:
        return "pdf"
    if "wordprocessing" in m or "msword" in m or "rtf" in m:
        return "word"
    if "presentation" in m or "powerpoint" in m:
        return "powerpoint"
    if "spreadsheet" in m or "excel" in m:
        return "excel"
    if "csv" in m or ext == "csv":
        return "csv"
    if "html" in m:
        return "html"
    if m.startswith("text/"):
        return "text"
    return "other"


def first_value(v):
    return str(v[0] if isinstance(v, list) and v else v)


def doc_year(meta):
    for needle in ("created", "creation", "date", "modified", "save"):
        for k, v in meta.items():
            if needle in k.lower():
                m = re.search(r"\b(19[89]\d|20[0-2]\d)\b", first_value(v))
                if m:
                    return int(m.group(1))
    return None


def doc_author(meta):
    for k, v in meta.items():
        if k.lower().endswith(("creator", "author")) and first_value(v).strip():
            return first_value(v).strip()[:120]
    return None


def route_group(route):
    if route in ("docling", "legacy"):
        return "docling"
    if route in ("sheet", "csv"):
        return "spreadsheet/csv"
    return "tika text"


# ---------------------------------------------------------------- models
def load_model():
    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError:
        sys.exit("Missing packages. Run: pip install sentence-transformers")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(EMBED_MODEL, device=device)
    model.max_seq_length = 512
    if device == "cuda":
        model.half()
    return model, device


def encode(model, texts, batch_size=64, progress=False):
    return model.encode(texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True,
                        show_progress_bar=progress).astype(np.float32)


def load_reranker(name, device):
    try:
        import torch
        from sentence_transformers import CrossEncoder
        kw = {"model_kwargs": {"torch_dtype": torch.float16}} if device == "cuda" else {}
        try:
            return CrossEncoder(name, device=device, max_length=1024, **kw)
        except TypeError:  # older sentence-transformers
            return CrossEncoder(name, device=device, max_length=1024)
    except Exception as e:
        print(f"Re-ranker '{name}' could not load ({type(e).__name__}: {str(e)[:100]}); continuing without it.")
        return None


def shingles(text, n=5):
    words = re.findall(r"\w+", text.lower())
    return {" ".join(words[i:i + n]) for i in range(max(1, len(words) - n + 1))}


NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def dedupe_signature(h):
    """Words AND numbers of the whole chunk, breadcrumb included, so pages built from the same
    template (another species, another month's bulletin) don't look like copies of each other."""
    text = h.get("text", "")
    if h.get("type") == "table":
        text = f"{h.get('breadcrumb', '')}\n{h.get('table_md') or text}"
    text = text[:3000]
    return shingles(text), set(NUMBER.findall(text))


def is_near_duplicate(a, b, threshold=0.8):
    """a, b: signatures. A copy shares >= 80% of the smaller chunk's 5-word sequences (so shifted chunk
    boundaries still count) and contains every number of the chunk with fewer numbers."""
    (sa, na), (sb, nb) = a, b
    if len(sa & sb) / max(1, min(len(sa), len(sb))) < threshold:
        return False
    small, big = (na, nb) if len(na) <= len(nb) else (nb, na)
    return small <= big


def drop_near_duplicates(hits, scores, threshold=0.8):
    """Skip a chunk that is a copy of a higher-ranked kept chunk, e.g. the draft and final versions of
    one report. Same text with different names or numbers is NOT a copy. Returns (hits, scores, dropped)."""
    kept, kept_scores, sigs, dropped = [], [], [], 0
    for h, s in zip(hits, scores):
        sig = dedupe_signature(h)
        if any(is_near_duplicate(sig, o, threshold) for o in sigs):
            dropped += 1
            continue
        kept.append(h)
        kept_scores.append(s)
        sigs.append(sig)
    return kept, kept_scores, dropped


def fuse_ranks(*orders, k=RRF_RANK_CONSTANT):
    """Merge several rankings of the same chunks by reciprocal rank (1 / (k + rank)), e.g. the hybrid
    order and the re-ranker order, so a chunk both stages rank well can't be sunk by one of them."""
    scores, by_id = defaultdict(float), {}
    for order in orders:
        for r, h in enumerate(order):
            by_id[h["chunk_id"]] = h
            scores[h["chunk_id"]] += 1.0 / (k + r + 1)
    return [by_id[i] for i in sorted(scores, key=lambda i: -scores[i])]


def rerank(reranker, query, hits):
    """hits: list of _source dicts. Returns (reordered hits, scores aligned with them)."""
    if not reranker or not hits:
        return hits, [None] * len(hits)
    scores = np.asarray(reranker.predict([(query, h["text"]) for h in hits], batch_size=16,
                                         show_progress_bar=False), dtype=np.float32).reshape(-1)
    order = np.argsort(-scores)
    return [hits[i] for i in order], [float(scores[i]) for i in order]


# ---------------------------------------------------------------- OpenSearch: connection, index, pipeline
def connect(url):
    try:
        from opensearchpy import OpenSearch
    except ImportError:
        sys.exit("Missing package. Run: pip install opensearch-py")
    client = OpenSearch(hosts=[url], http_compress=True, use_ssl=url.startswith("https"),
                        verify_certs=False, ssl_show_warn=False, timeout=120)
    info, waited = None, 0
    while info is None:  # OpenSearch needs 30-60 s after "docker start" before it answers
        try:
            info = client.info()
        except Exception as e:
            if waited >= 90:
                sys.exit(f"OpenSearch isn't reachable at {url} ({type(e).__name__}).\n"
                         "Check it: docker ps -a --filter name=opensearch   (STATUS should be 'Up')\n"
                         "If it says Exited, see why: docker logs opensearch --tail 30\n"
                         "If it isn't there at all, create it with the docker run line at the top of this file.")
            if waited == 0:
                print(f"Waiting for OpenSearch at {url} to start", end="", flush=True)
            print(".", end="", flush=True)
            time.sleep(5)
            waited += 5
    if waited:
        print(" ready")
    version = info["version"]["number"]
    major, minor = (int(x) for x in version.split(".")[:2])
    if (major, minor) < (2, 19):
        sys.exit(f"OpenSearch {version} is too old: RRF hybrid search needs 2.19 or later.")
    return client, version


def index_body():
    text_en = {"type": "text", "analyzer": "english"}  # stemming + English stop words, like rag_local's BM25
    return {
        "settings": {"index": {"number_of_shards": 1, "number_of_replicas": 0,
                               "knn": True, "knn.algo_param.ef_search": 100}},
        "mappings": {
          # Vectors are kept once, in the k-NN files that search uses. Without this line each one is also
          # stored as text in _source (about half the index); embeddings/<zip>.npy keeps them anyway.
          "_source": {"excludes": ["embedding"]},
          "properties": {
            "chunk_id": {"type": "keyword"}, "doc_id": {"type": "keyword"}, "zip": {"type": "keyword"},
            "file": {"type": "keyword"}, "ext": {"type": "keyword"}, "kind": {"type": "keyword"},
            "mime": {"type": "keyword"}, "route": {"type": "keyword"}, "type": {"type": "keyword"},
            "title": {**text_en, "fields": {"raw": {"type": "keyword", "ignore_above": 256}}},
            "author": {**text_en, "fields": {"raw": {"type": "keyword", "ignore_above": 256}}},
            "year": {"type": "integer"}, "pages": {"type": "integer"}, "tokens": {"type": "integer"},
            "has_tables": {"type": "boolean"}, "table_complete": {"type": "boolean"},
            "breadcrumb": text_en,
            "text": text_en,                                   # BM25 searches this field
            "table_md": {"type": "text", "index": False},      # whole table: returned, not searched
            "source": {"type": "keyword", "index": False},
            "embedding": {                                     # k-NN searches this field
                "type": "knn_vector", "dimension": DIM,
                "method": {"name": "hnsw", "engine": "faiss",
                           "space_type": "innerproduct",       # = cosine, since bge vectors are normalized
                           "parameters": {"m": 16, "ef_construction": 128}},
            },
        }},
    }


def pipeline_body():
    return {"description": "Hybrid search: reciprocal rank fusion of BM25 and k-NN results",
            "phase_results_processors": [{"score-ranker-processor": {
                "combination": {"technique": "rrf", "rank_constant": RRF_RANK_CONSTANT}}}]}


def ensure_setup(client, recreate=False):
    if recreate and client.indices.exists(index=INDEX):
        client.indices.delete(index=INDEX)
        print(f"Deleted index '{INDEX}'")
    client.transport.perform_request("PUT", f"/_search/pipeline/{PIPELINE}", body=pipeline_body())
    if not client.indices.exists(index=INDEX):
        client.indices.create(index=INDEX, body=index_body())
        print(f"Created index '{INDEX}' (HNSW, faiss, {DIM} dims) and search pipeline '{PIPELINE}' (RRF)")
        return True
    warn_if_vectors_in_source(client)
    return False


def warn_if_vectors_in_source(client):
    """An index created before the _source change still stores every vector twice."""
    try:
        mapping = client.indices.get_mapping(index=INDEX)[INDEX]["mappings"]
    except Exception:
        return
    if "embedding" not in (mapping.get("_source") or {}).get("excludes", []):
        print(f"Note: index '{INDEX}' was created before vectors were left out of _source, so it stores each "
              "vector twice (about half its disk). To shrink it: python rag_opensearch.py setup --recreate, "
              "then index each ZIP again (saved vectors are reused, nothing is re-embedded).")


def cmd_setup(args):
    client, version = connect(args.url)
    print(f"OpenSearch {version} at {args.url}")
    try:
        plugins = {p.get("component") for p in client.cat.plugins(format="json")}
        missing = {"opensearch-knn", "opensearch-neural-search"} - plugins
        if missing:
            sys.exit(f"Missing plugins: {', '.join(sorted(missing))}. Use the official opensearchproject/opensearch image.")
    except Exception:
        pass  # plugin listing isn't essential
    created = ensure_setup(client, recreate=args.recreate)
    if not created:
        n = client.count(index=INDEX)["count"]
        print(f"Index '{INDEX}' already exists with {n:,} chunks (use --recreate to start empty). "
              f"Search pipeline '{PIPELINE}' updated.")


# ---------------------------------------------------------------- index (load chunks)
def doc_fields(chunks, parsed_dir):
    """Document-level fields copied onto every chunk: kind, year, author, pages, has_tables."""
    docs = {}
    for c in chunks:
        f = file_of(c)
        d = docs.setdefault(c["doc_id"], {"ext": Path(f).suffix.lower().lstrip("."), "mime": c.get("mime", ""),
                                          "pages": c.get("pages") or None, "has_tables": False})
        d["has_tables"] = d["has_tables"] or c["type"] == "table"
    for doc_id, d in docs.items():
        d["kind"] = kind_of(d["mime"], d["ext"])
        meta = {}
        tj = Path(parsed_dir) / f"{doc_id}.tika.json"
        if tj.exists():
            try:
                meta = json.loads(tj.read_text(encoding="utf-8")).get("metadata") or {}
            except (ValueError, OSError):
                meta = {}
        d["year"], d["author"] = doc_year(meta), doc_author(meta)
    return docs


def chunk_vectors(out, zip_id, chunks, batch_size):
    """Reuse rag_local.py's vectors when they match the chunks; otherwise embed and save them."""
    emb_path = Path(out) / "embeddings" / f"{zip_id}.npy"
    ids_path = Path(out) / "embeddings" / f"{zip_id}.ids.json"
    ids = [c["chunk_id"] for c in chunks]
    if emb_path.exists() and ids_path.exists() and json.loads(ids_path.read_text(encoding="utf-8")) == ids:
        emb = np.load(emb_path)
        if emb.shape == (len(chunks), DIM):
            print(f"Reusing {len(chunks):,} chunk vectors from {emb_path}")
            return emb
    model, device = load_model()
    print(f"Embedding {len(chunks):,} chunks with {EMBED_MODEL} on {device}...")
    t0 = time.perf_counter()
    emb = encode(model, [c["text"] for c in chunks], batch_size, progress=True)
    print(f"Embedded in {time.perf_counter() - t0:.0f}s")
    emb_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(emb_path, emb)
    ids_path.write_text(json.dumps(ids), encoding="utf-8")
    return emb


def cmd_index(args):
    from opensearchpy import helpers
    from tqdm import tqdm
    out = Path(args.out)
    chunks = load_jsonl(out / "chunks" / f"{args.zip_id}.jsonl")
    if not chunks:
        sys.exit("No chunks to index.")
    client, version = connect(args.url)
    print(f"OpenSearch {version}; {len(chunks):,} chunks from ZIP {args.zip_id}")
    ensure_setup(client)
    emb = chunk_vectors(out, args.zip_id, chunks, args.batch_size)
    docs = doc_fields(chunks, out / "parsed" / args.zip_id)

    removed = client.delete_by_query(index=INDEX, body={"query": {"term": {"zip": args.zip_id}}},
                                     refresh=True, conflicts="proceed").get("deleted", 0)
    if removed:
        print(f"Removed {removed:,} chunks from an earlier load of ZIP {args.zip_id}")

    def actions():
        for c, v in zip(chunks, emb):
            d = docs[c["doc_id"]]
            src = {
                "chunk_id": c["chunk_id"], "doc_id": c["doc_id"], "zip": c.get("zip", args.zip_id),
                "file": file_of(c), "ext": d["ext"], "kind": d["kind"], "mime": c.get("mime", ""),
                "route": c.get("route", ""), "type": c["type"], "title": c.get("title") or None,
                "author": d["author"], "year": d["year"], "pages": d["pages"], "has_tables": d["has_tables"],
                "breadcrumb": c["breadcrumb"], "text": c["text"], "tokens": c.get("tokens"),
                "source": c.get("source", ""), "embedding": v.tolist(),
            }
            if "table_md" in c:
                src["table_md"], src["table_complete"] = c["table_md"], c.get("table_complete", True)
            yield {"_index": INDEX, "_id": f"{src['zip']}-{c['chunk_id']}", "_source": src}

    ok = failed = 0
    t0 = time.perf_counter()
    for success, info in tqdm(helpers.streaming_bulk(client, actions(), chunk_size=200, request_timeout=300,
                                                     max_retries=3, raise_on_error=False),
                              total=len(chunks), unit="chunk", desc="Indexing"):
        if success:
            ok += 1
        else:
            failed += 1
            if failed <= 3:
                print(f"  failed: {str(info)[:200]}")
    client.indices.refresh(index=INDEX)
    total = client.count(index=INDEX)["count"]
    print(f"\nIndexed {ok:,} chunks ({failed} failed) in {time.perf_counter() - t0:.0f}s; "
          f"the index now holds {total:,} chunks.")
    try:   # the SQL filter's document catalog lists the same documents as the index
        import doc_catalog
        n = doc_catalog.update_zip(out, args.zip_id)
        print(f"Document catalog: {n:,} documents from ZIP {args.zip_id} in {out / doc_catalog.CATALOG}")
    except ImportError:
        print("(doc_catalog.py isn't in this folder, so the SQL filter's catalog wasn't updated)")
    except Exception as e:
        print(f"Document catalog not updated: {type(e).__name__}: {str(e)[:150]}")
    print('Try it: python rag_opensearch.py search "your question"')


# ---------------------------------------------------------------- filters
FILTER_FIELDS = {"kind": "keyword", "ext": "keyword", "type": "keyword", "doc_id": "keyword", "zip": "keyword",
                 "year": "int", "pages": "int", "tokens": "int", "has_tables": "bool",
                 "title": "text", "author": "text"}
FILTER_RE = re.compile(r"^(\w+)\s*(>=|<=|!=|=|>|<|~)\s*(.+)$")
FILTER_HELP = ("Use conditions like: kind=pdf,word  year>=2005  pages<50  has_tables=true  "
               "title~\"forest health\"  author~nist  ext!=txt  (fields: " + ", ".join(FILTER_FIELDS) + ")")


def parse_filter(spec):
    """'kind=pdf year>=2005' -> an OpenSearch bool filter (or None)."""
    if not spec or not spec.strip():
        return None
    must, must_not = [], []
    for part in shlex.split(spec):
        m = FILTER_RE.match(part.strip())
        if not m:
            raise ValueError(f"can't read '{part}'. {FILTER_HELP}")
        field, op, value = m.group(1).lower(), m.group(2), m.group(3).strip()
        kind = FILTER_FIELDS.get(field)
        if not kind:
            raise ValueError(f"unknown field '{field}'. {FILTER_HELP}")
        if kind == "keyword" and op in ("=", "!="):
            clause = {"terms": {field: [v.strip() for v in value.split(",") if v.strip()]}}
        elif kind == "int" and op in ("=", "!=", ">", ">=", "<", "<="):
            try:
                n = int(value)
            except ValueError:
                raise ValueError(f"'{field}' needs a whole number, got '{value}'") from None
            clause = {"term": {field: n}} if op in ("=", "!=") else \
                {"range": {field: {{">": "gt", ">=": "gte", "<": "lt", "<=": "lte"}[op]: n}}}
        elif kind == "bool" and op in ("=", "!="):
            truth = value.lower() in ("1", "true", "yes", "y")
            clause = {"term": {field: truth}}
        elif kind == "text" and op == "~":
            clause = {"match": {field: {"query": value, "operator": "and"}}}
        else:
            raise ValueError(f"'{op}' doesn't work with '{field}'. {FILTER_HELP}")
        (must_not if op == "!=" else must).append(clause)
    out = {"bool": {}}
    if must:
        out["bool"]["filter"] = must
    if must_not:
        out["bool"]["must_not"] = must_not
    return out


# ---------------------------------------------------------------- queries
SOURCE_FIELDS = {"excludes": ["embedding"]}


def keyword_clause(text, flt):
    q = {"bool": {"must": [{"match": {"text": {"query": text}}}]}}
    if flt:
        q["bool"]["filter"] = [flt]
    return q


def knn_clause(vector, k, flt):
    inner = {"vector": [float(x) for x in vector], "k": k}
    if flt:
        inner["filter"] = flt  # efficient filtering: HNSW search only visits allowed chunks
    return {"knn": {"embedding": inner}}


def keyword_query(text, flt, k=CANDIDATES):
    return {"size": k, "_source": SOURCE_FIELDS, "query": keyword_clause(text, flt)}


def vector_query(vector, flt, k=CANDIDATES):
    return {"size": k, "_source": SOURCE_FIELDS, "query": knn_clause(vector, k, flt)}


def hybrid_query(text, vector, flt, k=CANDIDATES):
    # The filter goes inside both sub-queries (works on OpenSearch 2.19+; 3.0+ also accepts a top-level filter).
    return {"size": k, "_source": SOURCE_FIELDS,
            "query": {"hybrid": {"queries": [keyword_clause(text, flt), knn_clause(vector, k, flt)]}}}


class Searcher:
    def __init__(self, url, reranker_name=None):
        self.client, self.version = connect(url)
        if not self.client.indices.exists(index=INDEX):
            sys.exit("The index doesn't exist yet. Run: python rag_opensearch.py index pilot_out_full --zip-id 001")
        self.count = self.client.count(index=INDEX)["count"]
        self.model, self.device = load_model()
        self.reranker = load_reranker(reranker_name, self.device) if reranker_name else None
        self.reranker_name = reranker_name if self.reranker else None

    def _run(self, body, pipeline=False):
        params = {"search_pipeline": PIPELINE} if pipeline else {}
        hits = self.client.search(index=INDEX, body=body, params=params)["hits"]["hits"]
        return [h["_source"] for h in hits]

    def embed_queries(self, queries):
        return encode(self.model, [QUERY_PREFIX + q for q in queries])

    def vector(self, vec, flt, k=CANDIDATES):
        return self._run(vector_query(vec, flt, k))

    def keyword(self, text, flt, k=CANDIDATES):
        return self._run(keyword_query(text, flt, k))

    def hybrid(self, text, vec, flt, k=CANDIDATES):
        return self._run(hybrid_query(text, vec, flt, k), pipeline=True)

    def retrieve(self, question, flt, top=5, dedupe=True, fuse=False):
        """The funnel: filtered hybrid search (top 50) -> re-rank (optionally fused with the hybrid
        order) -> drop near-duplicates -> top N.
        Returns (top hits, their re-rank scores, number of near-duplicates dropped)."""
        vec = self.embed_queries([question])[0]
        hits = self.hybrid(question, vec, flt)
        ranked, scores = rerank(self.reranker, question, hits)
        if fuse and self.reranker:
            by_id = {h["chunk_id"]: sc for h, sc in zip(ranked, scores)}
            ranked = fuse_ranks(hits, ranked)
            scores = [by_id[h["chunk_id"]] for h in ranked]
        dropped = 0
        if dedupe:
            ranked, scores, dropped = drop_near_duplicates(ranked, scores)
        return ranked[:top], scores[:top], dropped


# ---------------------------------------------------------------- neighbouring chunks
def chunk_position(c):
    """A chunk's place in its document: chunk_id is <doc_id>-NNN (a table's extra cards add -2, -3)."""
    cid, doc = c.get("chunk_id", ""), c.get("doc_id", "")
    if not doc or not cid.startswith(doc + "-"):
        return None
    head = cid[len(doc) + 1:].split("-")[0]
    return int(head) if head.isdigit() else None


def stitch(parts, max_table_chars=None, crumb=None):
    """Chunks of one document, in order, as one text. A line in brackets marks where the section differs
    from `crumb` (the section named in the source's heading) or from the chunk before."""
    out, seen = [], set()
    crumb = parts[0]["breadcrumb"] if crumb is None and parts else crumb
    for c in parts:
        body = body_of(c)
        if not body or body in seen:       # a table's extra cards repeat the same table
            continue
        seen.add(body)
        if c["type"] == "table" and max_table_chars and len(body) > max_table_chars:
            body = body[:max_table_chars].rsplit("\n", 1)[0] + "\n… (table continues)"
        if c["breadcrumb"] != crumb:
            out.append(f"[{c['breadcrumb']}]")
            crumb = c["breadcrumb"]
        out.append(body)
    return "\n\n".join(out)


def expand_hits(client, hits, scores, window=1):
    """Each hit plus up to `window` chunks before and after it in the same document, so text cut at a
    chunk boundary (a list, a sentence, the explanation under a table) reaches the LLM whole. Windows of one
    document that overlap or touch become one passage, so nothing is sent twice. This runs after
    re-ranking and fetches the neighbours by id in one request (a few ms); the ranking doesn't change.
    Returns (passages, scores): a passage is the best hit's fields with type "passage" and _chunks."""
    if window <= 0 or not hits:
        return hits, scores
    groups = []
    for h, sc in zip(hits, scores):
        pos = chunk_position(h)
        if pos is None:
            groups.append({"hit": h, "score": sc})
        else:
            groups.append({"hit": h, "score": sc, "doc": h["doc_id"], "zip": h.get("zip"),
                           "lo": max(0, pos - window), "hi": pos + window, "matched": {pos: h}})
    merged = True
    while merged:       # groups are in rank order; a lower-ranked window joins the higher-ranked one
        merged = False
        for i, a in enumerate(groups):
            for b in groups[i + 1:]:
                if ("doc" in a and "doc" in b and (a["doc"], a["zip"]) == (b["doc"], b["zip"])
                        and b["lo"] <= a["hi"] + 1 and a["lo"] <= b["hi"] + 1):
                    a["lo"], a["hi"] = min(a["lo"], b["lo"]), max(a["hi"], b["hi"])
                    for p, h in b["matched"].items():
                        a["matched"].setdefault(p, h)
                    groups.remove(b)
                    merged = True
                    break
            if merged:
                break

    def doc_id_of(g, p):
        return f"{g['zip']}-{g['doc']}-{p:03d}"    # the _id rag_opensearch.py index gives each chunk

    ids = [doc_id_of(g, p) for g in groups if "doc" in g for p in range(g["lo"], g["hi"] + 1)
           if p not in g["matched"]]
    found = {}
    if ids:
        try:
            for d in client.mget(index=INDEX, body={"ids": ids}).get("docs", []):
                if d.get("found"):
                    found[d["_id"]] = d["_source"]
        except Exception as e:
            print(f"(couldn't fetch neighbouring chunks: {type(e).__name__}; sending the chunks alone)")
    out, out_scores = [], []
    for g in groups:
        parts = [] if "doc" not in g else [g["matched"].get(p) or found.get(doc_id_of(g, p))
                                           for p in range(g["lo"], g["hi"] + 1)]
        parts = [c for c in parts if c]
        if len(parts) <= 1:
            out.append(g["hit"])
        else:
            first, last = chunk_position(parts[0]), chunk_position(parts[-1])
            out.append({**g["hit"], "type": "passage", "_chunks": parts,
                        "_span": f"chunks {first:03d}-{last:03d}",
                        "_matched": [g["matched"][p]["chunk_id"] for p in sorted(g["matched"])]})
        out_scores.append(g["score"])
    return out, out_scores


# ---------------------------------------------------------------- search
def show_result(rank, c, info, width):
    print(f"\n[{rank}] {c['breadcrumb']}")
    print(f"    {c['type']} · {file_of(c)} (ZIP {c.get('zip', '?')}) · {info}")
    body = body_of(c)
    cut = bool(width) and len(body) > width
    shown = body[:width].rsplit(" ", 1)[0] + " …" if cut else body
    print(textwrap.indent(shown, "    "))
    if c["type"] == "table":
        note = f"table is {len(body):,} characters; first {width:,} shown" if cut else "whole table shown"
        if c.get("table_complete") is False:
            note += "; the stored table was cut at the size limit"
        print(f"    ({note})")
    elif cut:
        print(f"    (display cut at {width:,} of {len(body):,} characters; --width 0 shows it all. "
              "ask sends the whole text)")


def interactive(prompt_text, on_question, on_filter, on_sql=None):
    print(f"{prompt_text} 'filter: <conditions>' sets a filter, 'filter:' clears it"
          + (", 'sql: <condition>' sets the SQL filter, 'sql:' clears it" if on_sql else "")
          + "; an empty line quits.")
    while True:
        try:
            q = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        if q.lower().startswith("filter:"):
            on_filter(q[7:].strip())
            continue
        if on_sql and q.lower().startswith("sql:"):
            on_sql(q[4:].strip())
            continue
        on_question(q)


def combine_filters(*filters):
    """All of the given OpenSearch filters (None = no filter)."""
    parts = [f for f in filters if f]
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else {"bool": {"filter": parts}}


def make_filter_state(spec, sql=None, catalog=None):
    """Holds the field filter (--filter) and the SQL filter (--sql); state["flt"] is both together.
    The SQL runs once when it is set, not for every question: the documents it picks don't change."""
    state = {"flt": None, "spec": "", "meta": None, "sql_flt": None, "sql": ""}

    def update():
        state["flt"] = combine_filters(state["meta"], state["sql_flt"])

    def set_filter(new_spec):
        try:
            state["meta"], state["spec"] = parse_filter(new_spec), new_spec
            update()
            print(f"Filter: {new_spec or 'none'}")
            return True
        except ValueError as e:
            print(f"Filter not applied: {e}")
            return False

    def set_sql(condition):
        if not condition:
            state["sql_flt"], state["sql"] = None, ""
            update()
            print("SQL filter: none")
            return True
        try:
            import doc_catalog
            flt, note = doc_catalog.opensearch_filter(condition, catalog or doc_catalog.DEFAULT_PATH)
        except ImportError:
            print("SQL filter not applied: doc_catalog.py isn't in this folder")
            return False
        except ValueError as e:
            print(f"SQL filter not applied: {e}")
            return False
        state["sql_flt"], state["sql"] = flt, condition
        update()
        print(note)
        return True

    if spec and not set_filter(spec):
        sys.exit(1)
    if sql and not set_sql(sql):
        sys.exit(1)
    return state, set_filter, set_sql


def cmd_search(args):
    s = Searcher(args.url, None if args.no_rerank else args.reranker)
    print(f"OpenSearch {s.version} · {s.count:,} chunks · re-rank: {s.reranker_name or 'off'} · model on {s.device}")
    state, set_filter, set_sql = make_filter_state(args.filter, args.sql, args.catalog)

    def run(question):
        flt = state["flt"]
        vec = s.embed_queries([question])[0]
        if args.mode == "vector":
            hits = s.vector(vec, flt)
        elif args.mode == "keyword":
            hits = s.keyword(question, flt)
        else:
            hits = s.hybrid(question, vec, flt)
        # Separate vector and keyword queries, only to show where each result came from
        vpos = {h["chunk_id"]: r + 1 for r, h in enumerate(s.vector(vec, flt))}
        kpos = {h["chunk_id"]: r + 1 for r, h in enumerate(s.keyword(question, flt))}
        hpos = {h["chunk_id"]: r + 1 for r, h in enumerate(hits)}
        ranked, scores = rerank(s.reranker, question, hits)
        if args.fuse_hybrid and s.reranker:
            by_id = {h["chunk_id"]: sc for h, sc in zip(ranked, scores)}
            ranked = fuse_ranks(hits, ranked)
            scores = [by_id[h["chunk_id"]] for h in ranked]
        if not args.keep_duplicates:
            ranked, scores, dropped = drop_near_duplicates(ranked, scores)
            if dropped:
                print(f"(hid {dropped} near-duplicate chunks, such as other versions of the same document; "
                      "--keep-duplicates shows them)")
        if not ranked:
            print("No results" + (" (the filter or SQL filter may be too strict)." if flt else "."))
        for r, (h, score) in enumerate(zip(ranked[:args.k], scores[:args.k]), 1):
            found = [f"vector #{vpos[h['chunk_id']]}" if h["chunk_id"] in vpos else "",
                     f"keyword #{kpos[h['chunk_id']]}" if h["chunk_id"] in kpos else ""]
            info = "found by " + (", ".join(x for x in found if x) or "hybrid only")
            if score is not None:
                info = f"re-rank {score:.2f} (was {args.mode} #{hpos[h['chunk_id']]}) · " + info
            if args.expand:     # this result with its neighbouring chunks, as ask would send it
                h = expand_hits(s.client, [h], [score], args.expand)[0][0]
                if h["type"] == "passage":
                    info += f" · {h['_span']} (found: {h['chunk_id']})"
            show_result(r, h, info, args.width)

    if args.query:
        run(" ".join(args.query))
    else:
        interactive("Type a question and press Enter.", run, set_filter, set_sql)


# ---------------------------------------------------------------- ask (the answer step)
ANSWER_PROMPT = """You answer questions using the numbered sources below, which are excerpts \
from US government documents.

Rules:
- Base your answer only on the sources. Do not add facts from your own knowledge.
- The sources may use different words than the question: formal names, abbreviations or \
technical terms instead of everyday ones (for example "occupants" for "people"). Treat those as \
matching.
- If the sources answer the question fully or partly, give the answer they support, and say \
briefly what they don't cover. A source that states a conclusion (for example that something \
"would have been ineffective") answers a "would it have helped" question.
- Only if none of the sources is relevant to the question, reply exactly: "I couldn't find this \
in the documents."
- Cite the sources you use with their numbers in square brackets, like [1] or [2][4], right after \
the sentence they support.
- Be concise: a short paragraph or a few bullet points. Copy numbers and names exactly as written.

Question: {question}

Sources:
{sources}

Answer:"""


def build_context(hits, max_chars=20000, max_table_chars=4000):
    blocks, used = [], 0
    for n, h in enumerate(hits, 1):
        body = stitch(h["_chunks"], max_table_chars, h["breadcrumb"]) if h["type"] == "passage" else body_of(h)
        if h["type"] == "table" and len(body) > max_table_chars:
            body = body[:max_table_chars].rsplit("\n", 1)[0] + "\n… (table continues)"
        block = f"[{n}] {h['breadcrumb']} ({file_of(h)})\n{body}"
        if blocks and used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks), len(blocks)


class OllamaLLM:
    def __init__(self, model, url):
        import requests
        self.requests, self.model, self.url = requests, model, url.rstrip("/")
        try:
            tags = requests.get(f"{self.url}/api/tags", timeout=10).json()
        except Exception:
            sys.exit(f"Ollama isn't reachable at {self.url}. Install it from ollama.com and make sure it's running.")
        names = {m.get("name", "") for m in tags.get("models", [])}
        if model not in names and f"{model}:latest" not in names:
            sys.exit(f"The model '{model}' isn't in Ollama yet. Run: ollama pull {model}")
        self.name = f"ollama:{model}"

    def stream(self, prompt):
        body = {"model": self.model, "prompt": prompt, "stream": True,
                "options": {"temperature": 0.1, "num_predict": 600, "num_ctx": 8192}}
        with self.requests.post(f"{self.url}/api/generate", json=body, stream=True, timeout=600) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                d = json.loads(line)
                if d.get("response"):
                    yield d["response"]
                if d.get("done"):
                    break


CLAUDE_MODEL = "claude-sonnet-5-5"   # or claude-haiku-5-5: much cheaper, a little less careful


class ClaudeLLM:
    """Claude through the Claude API (pip install anthropic; key in ANTHROPIC_API_KEY)."""

    def __init__(self, model):
        try:
            import anthropic
        except ImportError:
            sys.exit("The Claude API needs the anthropic package: pip install anthropic")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit('Set your Claude API key first (PowerShell): $env:ANTHROPIC_API_KEY = "sk-ant-..."')
        self.client = anthropic.Anthropic()
        self.model, self.name = model, f"claude:{model}"

    def stream(self, prompt):
        with self.client.messages.stream(model=self.model, max_tokens=1024,
                                         messages=[{"role": "user", "content": prompt}]) as s:
            yield from s.text_stream


class BedrockLLM:
    def __init__(self, model, region):
        if not model:
            sys.exit("Bedrock needs a model ID: copy one from the Bedrock console and pass --model <id>")
        try:
            import boto3
        except ImportError:
            sys.exit('Bedrock needs boto3: pip install -U "boto3[crt]"')
        self.client = boto3.client("bedrock-runtime", region_name=region)
        self.model, self.name = model, f"bedrock:{model}"

    def stream(self, prompt):
        r = self.client.converse(modelId=self.model, messages=[{"role": "user", "content": [{"text": prompt}]}],
                                 inferenceConfig={"maxTokens": 600, "temperature": 0.1})
        yield "".join(part.get("text", "") for part in r["output"]["message"]["content"])


# How much text each answer model is given. A 7B model in an 8K context needs tables cut short;
# Claude reads up to 1M tokens, so it gets whole tables (pilot.py stores up to 30,000 characters each).
CONTEXT_LIMITS = {"ollama": (20000, 4000), "bedrock": (20000, 4000), "claude": (150000, 30000)}


def cmd_ask(args):
    if args.llm == "claude":
        llm = ClaudeLLM(args.model or CLAUDE_MODEL)
    elif args.llm == "ollama":
        llm = OllamaLLM(args.model or "qwen2.5:7b", args.ollama_url)
    else:
        llm = BedrockLLM(args.model, args.region)
    max_chars, max_table_chars = CONTEXT_LIMITS[args.llm]
    # Neighbouring chunks: on by default for Claude, which has room for them; small local models don't.
    expand = args.expand if args.expand is not None else (1 if args.llm == "claude" else 0)
    s = Searcher(args.url, None if args.no_rerank else args.reranker)
    print(f"OpenSearch {s.version} · {s.count:,} chunks · re-rank: {s.reranker_name or 'off'} · answers: {llm.name}")
    if args.llm == "ollama" and s.device == "cuda" and s.reranker:
        print("Tip: the re-ranker and Ollama share your GPU. If answers are slow, try "
              "--reranker BAAI/bge-reranker-base or --model qwen2.5:3b")
    state, set_filter, set_sql = make_filter_state(args.filter, args.sql, args.catalog)

    def run(question):
        t0 = time.perf_counter()
        top, scores, dropped = s.retrieve(question, state["flt"], top=args.k, dedupe=not args.keep_duplicates,
                                          fuse=args.fuse_hybrid)
        top, scores = expand_hits(s.client, top, scores, expand)
        if not top:
            print("No matching chunks" + (" (the filter or SQL filter may be too strict)." if state["flt"] else "."))
            return
        context, used = build_context(top, max_chars=max_chars, max_table_chars=max_table_chars)
        t1 = time.perf_counter()
        if args.show_context:
            print("\n--- context sent to the LLM ---\n" + context + "\n--- end of context ---")
        print(f"\nAnswer (retrieval {t1 - t0:.1f}s):\n")
        try:
            for piece in llm.stream(ANSWER_PROMPT.format(question=question, sources=context)):
                print(piece, end="", flush=True)
        except Exception as e:
            print(f"\n[LLM error: {type(e).__name__}: {str(e)[:150]}]")
        note = f"; {dropped} near-duplicate chunks skipped" if dropped else ""
        print(f"\n\nSources (answer {time.perf_counter() - t1:.1f}s{note}):")
        for n, (h, score) in enumerate(zip(top[:used], scores[:used]), 1):
            sc = f" · re-rank {score:.2f}" if score is not None else ""
            span = f" · {h['_span']}" if h["type"] == "passage" else ""
            print(f"  [{n}] {h['breadcrumb']} · {file_of(h)} (ZIP {h.get('zip', '?')}){span}{sc}")

    if args.question:
        run(" ".join(args.question))
    else:
        interactive("Ask a question about the documents.", run, set_filter, set_sql)


# ---------------------------------------------------------------- eval
def first_rank(ranked, pred, top=10):
    for r, h in enumerate(ranked[:top], 1):
        if pred(h):
            return r
    return None


def summarize(rows):
    def hit(ranks, k):
        return sum(1 for r in ranks if r and r <= k) / max(len(ranks), 1)

    chunk = [r for r, _ in rows]
    docs = [d for _, d in rows]
    return {"n": len(rows), "hit1": hit(chunk, 1), "hit5": hit(chunk, 5), "hit10": hit(chunk, 10),
            "mrr": sum(1 / r for r in chunk if r) / max(len(chunk), 1),
            "doc1": hit(docs, 1), "doc5": hit(docs, 5)}


def cmd_eval(args):
    from tqdm import tqdm
    out = Path(args.out)
    qpath = Path(args.questions) if args.questions else out / "eval" / f"{args.zip_id}_questions.jsonl"
    if not qpath.exists():
        sys.exit(f"No test questions at {qpath}. Make them with: python rag_local.py make-eval {args.out} "
                 f"--zip-id {args.zip_id}")
    qs = load_jsonl(qpath)
    s = Searcher(args.url, None if args.no_rerank else args.reranker)
    print(f"OpenSearch {s.version} · {s.count:,} chunks · {len(qs)} questions from {qpath.name}")
    vecs = s.embed_queries([q["question"] for q in qs])

    runs = defaultdict(list)
    for q, vec in tqdm(list(zip(qs, vecs)), unit="q", desc="Searching"):
        text = q["question"]
        runs["vector (HNSW)"].append(s.vector(vec, None))
        runs["keyword (BM25)"].append(s.keyword(text, None))
        hyb = s.hybrid(text, vec, None)
        runs["hybrid (RRF)"].append(hyb)
        if s.reranker:
            runs["hybrid + rerank"].append(rerank(s.reranker, text, hyb)[0])

    print(f"\n{'mode':<20}{'Hit@1':>7}{'Hit@5':>7}{'Hit@10':>8}{'MRR@10':>8}   {'doc Hit@1':>9}{'doc Hit@5':>10}")
    ranks = {}
    for mode, lists in runs.items():
        ranks[mode] = [(first_rank(r, lambda h, q=q: h["chunk_id"] == q["chunk_id"]),
                        first_rank(r, lambda h, q=q: h["doc_id"] == q["doc_id"])) for q, r in zip(qs, lists)]
        m = summarize(ranks[mode])
        print(f"{mode:<20}{m['hit1']:>7.0%}{m['hit5']:>7.0%}{m['hit10']:>8.0%}{m['mrr']:>8.2f}"
              f"   {m['doc1']:>9.0%}{m['doc5']:>10.0%}")
    print("Hit@k: the exact source chunk is in the top k. doc Hit@k: any chunk of the source document is.")

    best = list(runs)[-1]
    print(f"\n{best}, by kind of chunk:")
    groups = defaultdict(list)
    for q, r in zip(qs, ranks[best]):
        groups[f"{q['type']} · {route_group(q.get('route', ''))}"].append(r)
    for g, rows in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        m = summarize(rows)
        print(f"  {g:<28}{m['n']:>4} questions   Hit@5 {m['hit5']:>5.0%}   doc Hit@5 {m['doc5']:>5.0%}")

    res_path = out / "eval" / f"{args.zip_id}_results_opensearch.csv"
    with open(res_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["qid", "question", "type", "route"] + [f"rank: {m}" for m in runs] + ["source_breadcrumb"])
        for j, q in enumerate(qs):
            w.writerow([q.get("qid", j), q["question"], q["type"], q.get("route", "")]
                       + [ranks[m][j][0] or "" for m in runs] + [q["breadcrumb"]])
    print(f"\nPer-question ranks: {res_path} (blank = not in the top 10)")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=OPENSEARCH_URL, help=f"OpenSearch address (default {OPENSEARCH_URL})")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("setup", help="create the index (HNSW + BM25 fields) and the RRF search pipeline")
    sp.add_argument("--recreate", action="store_true", help="delete the index first (removes all chunks)")
    sp.set_defaults(func=cmd_setup)

    sp = sub.add_parser("index", help="embed (or reuse vectors) and load one ZIP's chunks")
    sp.add_argument("out", help="pilot output folder, e.g. pilot_out_full")
    sp.add_argument("--zip-id", default="001")
    sp.add_argument("--batch-size", type=int, default=64, help="embedding batch size (lower if GPU runs out)")
    sp.set_defaults(func=cmd_index)

    def retrieval_args(sp):
        sp.add_argument("--filter", default=None, help='e.g. "kind=pdf year>=2005"')
        sp.add_argument("--sql", default=None,
                        help="SQL condition on the document catalog, e.g. \"kind = 'pdf' AND year >= 2005\"")
        sp.add_argument("--catalog", default=str(Path("pilot_out_full") / "documents.sqlite"),
                        help="document catalog for --sql (default pilot_out_full\\documents.sqlite)")
        sp.add_argument("--no-rerank", action="store_true")
        sp.add_argument("--reranker", default=RERANK_MODEL, help=f"cross-encoder (default {RERANK_MODEL})")
        sp.add_argument("--keep-duplicates", action="store_true",
                        help="don't hide near-duplicate chunks (e.g. draft and final versions of a report)")
        sp.add_argument("--fuse-hybrid", action="store_true",
                        help="merge the re-ranker order with the hybrid order (protects e.g. numeric tables)")

    sp = sub.add_parser("search", help="see what retrieval returns (no LLM)")
    sp.add_argument("query", nargs="*", help="question (leave out for interactive mode)")
    retrieval_args(sp)
    sp.add_argument("--mode", choices=["hybrid", "vector", "keyword"], default="hybrid")
    sp.add_argument("-k", type=int, default=5, help="results to show")
    sp.add_argument("--width", type=int, default=700, help="characters shown per result (0 = all)")
    sp.add_argument("--expand", type=int, default=0,
                    help="also show N chunks before and after each result, as ask sends them")
    sp.set_defaults(func=cmd_search)

    sp = sub.add_parser("ask", help="RAG: retrieve, re-rank, and let an LLM answer with citations")
    sp.add_argument("question", nargs="*", help="question (leave out for interactive mode)")
    retrieval_args(sp)
    sp.add_argument("-k", type=int, default=5, help="chunks given to the LLM (default 5)")
    sp.add_argument("--expand", type=int, default=None,
                    help="send each chunk with N chunks before and after it from the same document, so text "
                         "cut at a chunk boundary arrives whole (default 1 with Claude, 0 otherwise)")
    sp.add_argument("--llm", choices=["claude", "ollama", "bedrock"], default="claude",
                    help="claude = Claude API (needs ANTHROPIC_API_KEY), ollama = local, bedrock = AWS")
    sp.add_argument("--model", default=None,
                    help=f"Claude model (default {CLAUDE_MODEL}), Ollama model (default qwen2.5:7b) or Bedrock model ID")
    sp.add_argument("--ollama-url", default="http://localhost:11434")
    sp.add_argument("--region", default="us-east-1")
    sp.add_argument("--show-context", action="store_true", help="print the sources sent to the LLM")
    sp.set_defaults(func=cmd_ask)

    sp = sub.add_parser("eval", help="score vector, keyword, hybrid and hybrid+rerank on test questions")
    sp.add_argument("out", help="pilot output folder, e.g. pilot_out_full")
    sp.add_argument("--zip-id", default="001")
    sp.add_argument("--questions", default=None, help="questions file (default: <out>/eval/<zip>_questions.jsonl)")
    sp.add_argument("--no-rerank", action="store_true")
    sp.add_argument("--reranker", default=RERANK_MODEL)
    sp.set_defaults(func=cmd_eval)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()