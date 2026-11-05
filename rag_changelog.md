# GovDocs1 RAG: change log

What went wrong, why, and what we did about it. Newest items are at the bottom of each section.

**Current setup:** Tika (Docker) for type detection and text → Docling for structure (LibreOffice converts old Office files) → structure-aware chunks → bge-base-en-v1.5 vectors → optional SQL filter (`--sql`, SQLite document catalog) → OpenSearch 2.19 (HNSW + BM25, hybrid search fused with RRF) → bge-reranker-v2-m3 → near-duplicate removal → top 5 → Claude API (claude-sonnet-5-5) answers with citations, reading whole tables (Ollama and Bedrock still available). No LLM-generated document metadata.

**Latest retrieval score** (ZIP 001, 33,848 chunks, 111 eval questions, bge-reranker-v2-m3, 30 Oct 2026):

| Stage | Recall |
|---|---|
| Recall@50, hybrid (what the re-ranker receives) | 99% |
| Recall@5, hybrid alone | 83% |
| Recall@5 after re-ranking (what the LLM sees) | **95%** (text 97%, tables 89%) |
| doc Recall@5 (any chunk of the right document) | 100% |
| MRR@5 | 0.83 |
| Retrieval time | 1.39 s per question |

Misses: q088 (not in the top 50), q021 and q030 (tables), q067 and q077 (vague questions).

**Scaling check, ZIPs 001–007** (5 Nov 2026, same 111 questions, all from ZIP 001, so this measures whether 6 more ZIPs crowd out ZIP 001's answers). The 1-ZIP column is the last run before it (30 Oct, after table cards and the `_source` rebuild):

| Stage | 1 ZIP | 7 ZIPs |
|---|---|---|
| Recall@50, hybrid | 99% | 99% |
| Recall@5, hybrid alone | 84% | 81% |
| Recall@5 after re-ranking | 95% (106 of 111) | 95% (105 of 111) |
| text / tables | 97% / 89% | 96% / 89% |
| doc Recall@5 | 100% | 100% |
| MRR@5 | 0.83 | 0.82 |
| Retrieval time | 1.39 s | 3.71 s in many segments; **1.35 s** after a force merge (OpenSearch 0.07 s, re-ranking 1.28 s; fixed item 19) |

Recall holds: the answer still reaches the re-ranker for 110 of 111 questions, and the re-ranker recovers all but one of the questions that new chunks pushed out of the hybrid top 5. New miss: q049 (hybrid #1 → re-ranked #6), a question naming "RD Instruction 1940-T", outranked by chunks from other documents. The other misses are the 1-ZIP ones (q088, q030, q067, q077, q085).

---

## Fixed

| # | What went wrong | Why | Fix | Where |
|---|---|---|---|---|
| 1 | Every file failed in the first pilot run (404) | Tika 4.x renamed `/detect/stream` to `/detect` | The script finds the endpoint itself; it also handles 4.x partial parses (422), busy retries (429/503) and renamed metadata keys | pilot.py |
| 2 | Pilot too slow and too many chunks: 12 s/file, 144 chunks/file; one .xls gave 5,450 chunks, another took 7 min | Spreadsheets fully indexed; long PDFs sent to Docling; tables indexed twice; no per-file time limit | Tika first for every file; spreadsheets get a 30-row profile per sheet; PDFs over 60 pages use Tika text; Docling runs in a worker killed after 90 s; tables stored once; plain text capped at 100 chunks. **Result: 4.6 s/file, 33 chunks/file** | pilot.py |
| 3 | Long documents cut off at 100 chunks after falling back to Tika | The plain-text cap also applied to real documents | Separate cap for documents (`--max-doc-chunks`, 1,000) | pilot.py |
| 4 | Table chunks mostly spaces and dashes | Docling pads cells to line up columns | Padding stripped (`compact_md_table`) | pilot.py |
| 5 | Words like "signi›cantly" and "›re" | Old PDFs map the "fi" ligature to the wrong character | Unicode normalization, plus repair when the pattern appears 3+ times in a document | pilot.py |
| 6 | Plain-text files had no section headings | Tika returns flat text | Unstructured finds titles, which become sections and breadcrumbs (ignored when over 40% of blocks look like titles, as in logs) | pilot.py |
| 7 | Big tables split into many row-group chunks | Embedding model reads only 512 tokens | One chunk per table: a preview is embedded and the whole table is stored in `table_md` and returned to the LLM | pilot.py |
| 8 | Breadcrumbs like "Microsoft Word - X.doc", running headers ("Draft for Public Comment"), "Page 195" | Junk metadata titles; repeated page headers taken as section titles | Titles cleaned; repeated header/footer lines and page numbers removed before title detection | pilot.py |
| 9 | Tiny chunks (27 tokens) and dozens of number-dump chunks | Short sections; flattened tables in Tika text | Chunks under 64 tokens merged with a neighbour; max 3 number-heavy chunks per document | pilot.py |
| 10 | Generic breadcrumbs like "INTRODUCTION" or "Summary" | Docling's heading path excludes the document title | Document title added to the start of every breadcrumb | pilot.py |
| 11 | `setup` couldn't reach OpenSearch; container showed `Exited (143)` | It was stopped from outside (normal shutdown), and the script didn't wait for startup | `docker start opensearch`; `docker update --restart unless-stopped opensearch`; script now waits up to 90 s | rag_opensearch.py |
| 12 | Top 5 filled with two copies of the same passage | Draft and final versions of the same NIST report (001644/001647) | Near-duplicate chunks hidden after re-ranking (`--keep-duplicates` shows them) | rag_opensearch.py |
| 13 | LLM said "couldn't find this" when source [5] had the answer | Prompt too strict: "WTC 1 and 2" vs "Twin Towers", "occupants" vs "people" | Prompt accepts different wording and partial answers; still no outside facts | rag_opensearch.py |
| 14 | Duplicate removal dropped correct chunks (q042, q102) | 2005 and 2006 grape bulletins share wording but not numbers; tiny ITIS table looked like a subset of other chunks | A copy must share 80% of text **including the breadcrumb** and contain the **same numbers**. **Result: Recall@5 94% → 95%** | rag_opensearch.py |
| 15 | Copies of a document in later ZIPs would be parsed, embedded and indexed again | The pilot checked for duplicates only within one ZIP; GovDocs1 has many copies across ZIPs | Shared store of text hashes, `<out>/seen_texts.sqlite`: the first copy wins and later ones are logged as duplicates. ZIPs processed earlier (ZIP 001) are registered from their logs on the next run. Safe for parallel runs. `--per-zip-dedupe` restores the old behaviour | pilot.py |
| 16 | Every vector stored twice, about half of the index | Each chunk's JSON in `_source` held its 768 numbers as text, though the script never reads them back | Mapping leaves `embedding` out of `_source`; `setup` warns while an older index is in use. Rebuilt with `setup --recreate` and indexing again (saved vectors reused, 89 s). **Result: ZIP 001's index 474 MB → 242 MB** after a force merge (499 MB right after loading, in 13 segments) | rag_opensearch.py |
| 17 | Huge HTML pages and long text files made parsing slow: in a 100-file sample, 5 such files took 4.3 of 7 minutes | Docling ran, or timed out after 90 s, on pages with hundreds of thousands of characters. Long text files had 410,000 characters chunked to keep 100 chunks. 11 HTML pages held 16% of ZIP 001's chunks, mostly statistics and data dumps | HTML with over 200,000 characters of text (`--max-html-chars`) skips Docling and goes to Tika text: up to 1,000 chunks, at most 3 number-heavy ones. Statistics pages shrink to a few chunks; reports and books keep their text. The text window is halved to about 205,000 characters, which still holds the 100 kept chunks of all 29 capped text files in ZIP 001 | pilot.py |
| 18 | ZIP 010 failed: `Expand-Archive` stopped partway and the pilot crashed listing files (`WinError 225`) | Windows Defender flagged `010473.xls` as malware and blocked it; one unreadable file stopped the whole ZIP | Files the system won't let the pilot read are skipped and logged as "blocked by the system". Extract with `tar`, which carries on past a blocked file. Defender stays on | pilot.py |
| 19 | Retrieval time 1.39 s → 3.71 s per question at 7 ZIPs | Six overnight bulk loads left the index in many segments, and a search runs the HNSW and BM25 parts once per segment. The GPU was ruled out (nothing else on it). A run straight after the merge took 2.18 s, paying one-off loading of the merged graph and files | Force merge to one segment once a batch of ZIPs is loaded (`_forcemerge?max_num_segments=1`); time a run only after a warm-up. `eval_recall.py` now splits the time (OpenSearch vs re-ranking, median, slowest, first question) and notes free GPU memory and battery. **Result: 1.35 s** (OpenSearch 0.07 s, re-ranking 1.28 s), no slower than 1 ZIP | OpenSearch; eval_recall.py |

---

## Open / in progress

| # | Problem | Evidence | What we're trying |
|---|---|---|---|
| A | Re-ranker weak on numeric tables | Tables Recall@5 89% vs text 97%; q021 went from hybrid #4 to re-ranked #14, q030 from #9 to #7. **Cause found:** the re-ranker reads a table's preview (first rows that fit in 512 tokens), and 22% of tables (581 of 2,682) lose rows that way. Both table misses (q021, q030) had their answer row cut: by `**bold**` header markup (q021) and by Docling copying the table title into all 13 column names (q030) | **Table cards, at ingestion only** (`rebuild_tables.py`; same logic in `pilot.py` for new ZIPs). Tables that fit in 512 tokens are embedded whole without wasted tokens (bold, link URLs, the `\|---\|` line, dot leaders, empty columns, a title repeated in every column; two-row headers merged). Bigger tables get a card: column names + every row's label + the full rows that fit; the biggest get a first-rows card + up to 3 label cards. Spreadsheets unchanged. Offline on ZIP 001: rows named in searchable text **60% → 97%**; +46 chunks; 18 of 19 eval tables have the answer text in the chunk (was 16), q030 has its row label; no query-time change. **Next: re-index and measure** against tables 89%. Tried before: `--fuse-hybrid` (worse, 95% → 90%) and other re-rankers (slower, see Decisions) |
| B | Vague questions in crowded topics miss | q088 not in the top 50 with any re-ranker; q067 (hybrid #50 → re-ranked #49) and q077 (hybrid #23 → #18) | Query rewriting or HyDE (search with both versions), or fine-tune the re-ranker on question–chunk pairs |
| C | Huge HTML pages produce thousands of chunks | `001346.html` alone made 3,070 chunks (9% of ZIP 001); HTML has no page count, so it skips the 60-page limit | **Done, fixed item 17.** Check the effect in ZIP 002's log |
| D | Retrieval takes about 1.35 s per question | Measured at 7 ZIPs: re-ranking 50 candidates 1.28 s, OpenSearch 0.07 s | `bench_rerank.py` re-ranks the same 50 candidates per question under each setting and measures time and Recall@5: bigger batches, 512-token cut, top 40 / top 30 only, ONNX Runtime with fused kernels (attention, layer norm, GELU), TensorRT. Hand-written CUDA kernels ruled out: re-ranking is mostly matrix multiplication that cuBLAS already runs on the tensor cores, and TensorRT / ONNX Runtime already ship the fused kernels for the rest. **ONNX Runtime with fused kernels: 1.22 s vs 1.28 s (5% faster), same Recall@5 (95%), not adopted** (see Decisions). GPU power limit 90 W, so not throttled. **Next: the PyTorch settings, especially re-ranking only the top 30–40** |
| E | Wrong header row on some multi-page tables | Docling misreads tables that span pages | Not fixed yet |
| F | Answers not measured yet | Eval checks retrieval only | Script to run `ask` on the 111 questions and score answers and citations against the reference answers |
| G | Web pages laid out as one big table are mostly unsearchable | 102 "tables" in ZIP 001 (e.g. `001115`, `001260`) hold a whole HTML/Word page in one or two cells, with no rows; only their first 504 tokens are embedded | Treat them as text: split like normal text in pilot.py (not done yet) |

---

## Decisions

- **No LLM document metadata for now.** Per-document summaries, keywords and questions take hours locally, and per-chunk metadata would take days. Revisit if eval misses are wording gaps.
- **Embedding model bge-base-en-v1.5** is the baseline: free, fast, 768 dimensions, 512-token input. Later, compare a cheaper model (bge-small) and a stronger one on the eval set.
- **OpenSearch** as the vector store, because it has HNSW, BM25 and RRF hybrid search in one query, and it's the same engine as Amazon OpenSearch on AWS.
- **Eval matching uses a verbatim evidence quote**, so draft and final versions both count as hits and the eval set survives re-chunking.
- **Eval questions were written from the chunks**, so the scores are likely optimistic; use them to compare changes, and add real user questions later.
- **Re-ranker stays bge-reranker-v2-m3** (30 Oct 2026). Compared on the 111 questions, same index and settings:

  | Re-ranker | Recall@5 | Text / tables | Paraphrase questions | MRR@5 | Time per question |
  |---|---|---|---|---|---|
  | **bge-reranker-v2-m3** (568M) | **95%** | 97% / 89% | **96%** | 0.83 | **1.39 s** |
  | mxbai-rerank-base-v2 (0.5B) | 94% | 93% / 95% | 91% | 0.84 | 2.45 s |
  | Qwen3-Reranker-0.6B | 92% | 91% / 95% | 85% | 0.82 | 4.12 s |

  mxbai and Qwen3 find q021 (table) but lose text questions that bge gets (mxbai: q004, q075, q104; Qwen3: at least q033, q048, q057, q073), and take 2–3× as long. Not worth switching. `--fuse-hybrid` was worse with all three.
- **Answers by the Claude API** (30 Oct 2026), default model `claude-sonnet-5-5` (`claude-haiku-5-5` is much cheaper). It frees the GPU for the re-ranker, as Ollama and the re-ranker didn't fit together in 6 GB. It also reads whole tables, up to 30,000 characters each, where the local model got the first 4,000. About $0.02 per question with Sonnet 5.5 at ~8,000 tokens in.
- **Table views for the re-ranker** (it reads the rows matching the question instead of the preview) were built, then dropped; the pipeline stays as it was. Open item A has the cause they addressed.
- **Re-ranker stays on PyTorch, not ONNX Runtime** (5 Nov 2026). On the same 50 candidates per question at 7 ZIPs, ONNX Runtime with fused attention / layer-norm / GELU kernels took 1.22 s against 1.28 s, with the same Recall@5 (95%) and MRR@5 (0.82), and the same top 5 for 109 of 111 questions. That confirms the time is the model's matrix multiplications, which cuBLAS already runs at speed, so hand-written kernels would gain little too. Not worth 60 ms: onnxruntime-gpu 1.31 needs its own CUDA 13 libraries (1.2 GB), which clash with PyTorch's CUDA 12 in one process, so the re-ranker would have to run in a separate process. TensorRT not tried; expected to be similar.
- **SQL filter before retrieval** (5 Nov 2026), the funnel's first stage. `doc_catalog.py` keeps one row per indexed document in SQLite (`pilot_out_full/documents.sqlite`): zip, file, kind, mime, size, title, author, organization (Office "Company" / publisher), software, year, created and modified dates, pages, characters, parser route, chunks, tables. `rag_opensearch.py index` refreshes it for the ZIP it loads. `--sql "<condition>"` on `search`, `ask` and `eval_recall.py` (or `sql:` in interactive mode) runs the condition read-only, and the matching doc_ids become a filter inside both the BM25 and the k-NN part of the hybrid query (k-NN uses efficient filtering, so the HNSW search only visits those documents). Up to 65,536 documents go in one terms filter; larger sets use the complement or several filters. SQLite because it needs no server; on AWS the same table lives in Aurora PostgreSQL. GovDocs has no permissions or departments, so those columns don't exist; a real deployment would add them (or an access table to join) and put the user's permissions into every condition.