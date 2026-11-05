"""
Recall check for the OpenSearch RAG pipeline in rag_opensearch.py (the current setup: no document
metadata, hybrid BM25 + HNSW search fused by RRF, cross-encoder re-ranking, near-duplicate removal).

  Recall@50  is an answering chunk among the 50 hybrid results that go to the re-ranker?
  Recall@5   is it among the 5 chunks the LLM receives (after re-ranking and near-duplicate removal)?

A retrieved chunk counts as answering when it IS the question's source chunk, OR it contains the
question's evidence quote. So another version of the same report (draft vs final) also counts, and
the eval set keeps working if you re-chunk later (chunk ids change, the quoted text doesn't).

Usage (from D:\\rag corpus; OpenSearch running; rag_opensearch.py in the same folder):
  python eval_recall.py eval_set_001.jsonl
  python eval_recall.py eval_set_001.jsonl --no-rerank                       # retrieval only
  python eval_recall.py eval_set_001.jsonl --reranker BAAI/bge-reranker-base # compare re-rankers
  python eval_recall.py eval_set_001.jsonl --sql "zip = '001'"               # with an SQL filter first

Eval set: JSON lines with at least  question, chunk_id, doc_id, evidence
          (optional: qid, style, answer, breadcrumb, type, route, file)
"""

import argparse
import csv
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

try:
    import rag_opensearch as R
except ImportError:
    sys.exit("Put eval_recall.py in the same folder as rag_opensearch.py")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def norm(text):
    """Lowercase, keep letters and digits only: makes quotes match across spacing, punctuation and table pipes."""
    return " ".join(re.sub(r"[^0-9a-z]+", " ", (text or "").lower()).split())


def answers(h, q, evidence):
    if h.get("chunk_id") == q["chunk_id"]:
        return True
    return bool(evidence) and evidence in norm(h.get("text", "") + " " + (h.get("table_md") or ""))


def first_rank(hits, pred, k):
    for r, h in enumerate(hits[:k], 1):
        if pred(h):
            return r
    return None


def share(values):
    return sum(1 for v in values if v) / max(len(values), 1)


def machine_note(device):
    """The re-ranker's speed depends on the GPU's state, so say what it was. A laptop on battery, or
    another program holding GPU memory, makes re-ranking slower and the timings incomparable."""
    notes, warn = [f"models on {device}"], []
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            notes.append(f"{torch.cuda.get_device_name(0)}, {free / 2**30:.1f} of {total / 2**30:.1f} GB free")
            if free < 1.0 * 2**30:
                warn.append("little GPU memory left: another program may be using the GPU (check nvidia-smi)")
    except Exception:
        pass
    try:
        import psutil
        battery = psutil.sensors_battery()
        if battery is not None:
            notes.append("plugged in" if battery.power_plugged else "ON BATTERY")
            if not battery.power_plugged:
                warn.append("on battery the GPU runs slower, so times aren't comparable with plugged-in runs")
    except Exception:
        pass
    return "  " + " · ".join(notes) + "".join(f"\n  WARNING: {w}" for w in warn)


def load_eval_set(path):
    if not Path(path).exists():
        sys.exit(f"Not found: {path}")
    qs = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            q = json.loads(line)
            missing = [k for k in ("question", "chunk_id", "doc_id", "evidence") if not q.get(k)]
            if missing:
                sys.exit(f"Line {n} of {path} is missing: {', '.join(missing)}")
            q.setdefault("qid", f"q{n:03d}")
            q["_ev"] = norm(q["evidence"])
            qs.append(q)
    if not qs:
        sys.exit(f"No questions in {path}")
    return qs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("eval_set", help="JSON lines file of questions, e.g. eval_set_001.jsonl")
    ap.add_argument("--url", default=R.OPENSEARCH_URL)
    ap.add_argument("--no-rerank", action="store_true", help="measure retrieval only")
    ap.add_argument("--reranker", default=R.RERANK_MODEL)
    ap.add_argument("--out", default=None, help="CSV of per-question results (default: next to the eval set)")
    ap.add_argument("--sql", default=None, help="SQL filter on the document catalog, applied to every question")
    ap.add_argument("--catalog", default=None, help="document catalog for --sql (default pilot_out_full\\documents.sqlite)")
    args = ap.parse_args()

    from tqdm import tqdm
    qs = load_eval_set(args.eval_set)
    s = R.Searcher(args.url, None if args.no_rerank else args.reranker)
    present = s.client.count(index=R.INDEX, body={"query": {"terms": {"chunk_id": [q["chunk_id"] for q in qs]}}})["count"]
    print(f"OpenSearch {s.version} · {s.count:,} chunks · re-rank: {s.reranker_name or 'off'} · "
          f"{len(qs)} questions ({present} source chunks found in the index)")
    if present < len(qs):
        print("  Questions whose source chunk is missing are matched by their evidence quote only.")
    print(machine_note(s.device))
    flt = None
    if args.sql:
        import doc_catalog
        try:
            flt, note = doc_catalog.opensearch_filter(args.sql, args.catalog or doc_catalog.DEFAULT_PATH)
        except ValueError as e:
            sys.exit(f"SQL filter: {e}")
        print("  " + note)

    vecs = s.embed_queries([q["question"] for q in qs])
    rows = []
    for q, vec in tqdm(list(zip(qs, vecs)), unit="q", desc="Evaluating"):
        hit = lambda h, q=q: answers(h, q, q["_ev"])           # noqa: E731
        same_doc = lambda h, q=q: h.get("doc_id") == q["doc_id"] or hit(h)  # noqa: E731
        t0 = time.perf_counter()
        hyb = s.hybrid(q["question"], vec, flt, k=R.CANDIDATES)
        t1 = time.perf_counter()
        if s.reranker:
            ranked, scores = R.rerank(s.reranker, q["question"], hyb)
            final, _, dropped = R.drop_near_duplicates(ranked, scores)
            fused = R.drop_near_duplicates(R.fuse_ranks(hyb, ranked), [None] * len(hyb))[0]
        else:
            ranked, final, dropped = hyb, R.drop_near_duplicates(hyb, [None] * len(hyb))[0], 0
            fused = final
        t2 = time.perf_counter()
        vec50 = s.vector(vec, flt)                  # diagnostics: each retriever on its own
        kw50 = s.keyword(q["question"], flt)
        rows.append({
            "q": q,
            "r50": first_rank(hyb, hit, 50),         # in the re-ranker's input?
            "r5_hybrid": first_rank(hyb, hit, 5),    # top 5 without re-ranking
            "r_reranked": first_rank(ranked, hit, 50),
            "r5": first_rank(final, hit, 5),         # what the LLM sees
            "r5_fused": first_rank(fused, hit, 5),   # same, with --fuse-hybrid
            "doc5": first_rank(final, same_doc, 5),
            "vec50": first_rank(vec50, hit, 50),
            "kw50": first_rank(kw50, hit, 50),
            "top1": final[0]["breadcrumb"] if final else "",
            "top1_file": final[0].get("file", "") if final else "",
            "dropped": dropped,
            "t_search": t1 - t0,                     # OpenSearch hybrid query (BM25 + HNSW + RRF)
            "t_rank": t2 - t1,                       # re-ranking + duplicate removal
        })

    # ---- summary
    def line(label, key, sub=rows):
        print(f"  {label:<58}{share([r[key] for r in sub]):>7.0%}")

    print(f"\nRecall ({len(rows)} questions)")
    line("Recall@50  hybrid -> input to the re-ranker", "r50")
    line("Recall@5   hybrid alone (no re-ranking)", "r5_hybrid")
    if s.reranker:
        line("Recall@5   after re-ranking = what the LLM sees", "r5")
        line("Recall@5   re-ranking fused with hybrid order (--fuse-hybrid)", "r5_fused")
    line("doc Recall@5  (any chunk of the source document)", "doc5")
    print("  diagnostics:")
    line("  Recall@50  vector (HNSW) alone", "vec50")
    line("  Recall@50  keyword (BM25) alone", "kw50")
    found = [r for r in rows if r["r50"]]
    if s.reranker and found:
        kept = share([r["r5"] for r in found])
        print(f"  {'  re-ranker keeps it in the top 5, when it is in the 50':<58}{kept:>7.0%}")
        mrr = sum(1 / r["r5"] for r in rows if r["r5"]) / len(rows)
        print(f"  {'  MRR@5 (1 = answer always ranked first)':<58}{mrr:>7.2f}")
    n = len(rows)
    t_search = sum(r["t_search"] for r in rows) / n
    t_rank = sum(r["t_rank"] for r in rows) / n
    totals = sorted(r["t_search"] + r["t_rank"] for r in rows)
    print(f"  average retrieval time: {t_search + t_rank:.2f} s per question")
    second = "re-ranking + duplicate removal" if s.reranker else "duplicate removal"
    print(f"    OpenSearch search {t_search:.2f} s · {second} {t_rank:.2f} s · "
          f"median {totals[n // 2]:.2f} s · slowest {totals[-1]:.2f} s · "
          f"first question {rows[0]['t_search'] + rows[0]['t_rank']:.2f} s")

    final_key = "r5" if s.reranker else "r5_hybrid"
    for field, title in (("type", "chunk type"), ("style", "question style"), ("route", "parser route")):
        groups = defaultdict(list)
        for r in rows:
            val = r["q"].get(field) or "?"
            groups[R.route_group(val) if field == "route" else val].append(r)
        if len(groups) <= 1:
            continue
        print(f"\nBy {title}:")
        for g, sub in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            fused = f"   fused {share([r['r5_fused'] for r in sub]):>5.0%}" if s.reranker else ""
            print(f"  {g:<22}{len(sub):>4} q   Recall@50 {share([r['r50'] for r in sub]):>5.0%}"
                  f"   Recall@5 {share([r[final_key] for r in sub]):>5.0%}{fused}")

    # ---- failures
    missed = [r for r in rows if not r["r50"]]
    lost = [r for r in rows if r["r50"] and not r[final_key]]
    if missed:
        print(f"\nNot in the top 50 at all ({len(missed)}): a retrieval problem (wording, chunking, parsing)")
        for r in missed[:6]:
            print(f"  {r['q']['qid']}: {r['q']['question']}\n        source: {r['q'].get('breadcrumb', r['q']['chunk_id'])}"
                  f"\n        got:    {r['top1']}")
    if lost:
        print(f"\nIn the top 50 but not the top 5 ({len(lost)}): a ranking problem (re-ranker or duplicates)")
        for r in lost[:6]:
            pos = f"re-ranked #{r['r_reranked']}" if r["r_reranked"] else "re-ranked out"
            print(f"  {r['q']['qid']}: {r['q']['question']}\n        hybrid #{r['r50']} -> {pos}"
                  f"   source: {r['q'].get('breadcrumb', r['q']['chunk_id'])}")

    out = Path(args.out) if args.out else Path(args.eval_set).with_name(Path(args.eval_set).stem + "_recall.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["qid", "style", "type", "route", "question", "hybrid_rank@50", "hybrid_rank@5",
                    "reranked_rank", "final_rank@5", "fused_rank@5", "doc_rank@5", "vector_rank@50", "keyword_rank@50",
                    "source_breadcrumb", "top1_breadcrumb", "top1_file", "search_s", "rerank_s"])
        for r in rows:
            q = r["q"]
            w.writerow([q["qid"], q.get("style", ""), q.get("type", ""), q.get("route", ""), q["question"],
                        r["r50"] or "", r["r5_hybrid"] or "", r["r_reranked"] or "", r[final_key] or "",
                        r["r5_fused"] or "",
                        r["doc5"] or "", r["vec50"] or "", r["kw50"] or "", q.get("breadcrumb", ""),
                        r["top1"], r["top1_file"], f"{r['t_search']:.3f}", f"{r['t_rank']:.3f}"])
    print(f"\nPer-question results: {out} (blank = not found within that cut-off)")


if __name__ == "__main__":
    main()