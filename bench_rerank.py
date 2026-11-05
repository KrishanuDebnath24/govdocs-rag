"""
Re-ranker speed test: which GPU settings make re-ranking faster without losing recall?

Every setting re-ranks exactly the same candidates (each question's 50 hybrid results, fetched from
OpenSearch once at the start), so only the re-ranking step is timed and every setting sees the same
input. Recall@5 and MRR@5 are measured as in eval_recall.py, after near-duplicate removal.

Settings:
  current     what rag_opensearch.py does now (bge-reranker-v2-m3, batches of 16, up to 1024 tokens)
  batch32     batches of 32
  batch64     all 50 candidates in one batch
  max512      each (question, chunk) pair cut at 512 tokens instead of 1024
  top40       re-rank only the first 40 hybrid results; the rest stay behind them in hybrid order
  top30       the same with 30
  onnx        ONNX Runtime on the GPU, fp16, with fused kernels (attention, layer norm, GELU)  [--onnx]
  tensorrt    an NVIDIA TensorRT engine built for this GPU, fp16, through ONNX Runtime         [--tensorrt]
(plus fp16 / sdpa when the current setup turns out not to use them)

Usage (from D:\\rag corpus; OpenSearch running; laptop plugged in; nothing else using the GPU):
  python bench_rerank.py eval_set_001.jsonl                      # PyTorch settings, about 15 minutes
  python bench_rerank.py eval_set_001.jsonl --only current,onnx  # + ONNX Runtime, see below
  python bench_rerank.py eval_set_001.jsonl --only current,tensorrt

ONNX Runtime and TensorRT run in a second process that doesn't load PyTorch, because onnxruntime-gpu
1.31 uses CUDA 13 and its own copy of the CUDA libraries, while PyTorch may use CUDA 12. They need:
  pip uninstall -y onnxruntime
  pip install "onnxruntime-gpu[cuda,cudnn]" onnx
TensorRT also needs TensorRT 10 for CUDA 13 (pip install "tensorrt-cu13<11").
The ONNX files go to reranker_onnx\\ (about 2.5 GB; built once, reused on later runs).
"""

import argparse
import csv
import gc
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

try:
    import eval_recall as E
    import rag_opensearch as R
except ImportError:
    sys.exit("Put bench_rerank.py in the same folder as rag_opensearch.py and eval_recall.py")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MAX_LENGTH = 1024    # rag_opensearch.load_reranker uses 1024 tokens per (question, chunk) pair
BATCH = 16           # and batches of 16 (rag_opensearch.rerank)
ALL_KEYS = ["current", "fp16", "sdpa", "batch32", "batch64", "max512", "top40", "top30", "onnx", "tensorrt"]
LABELS = {"onnx": "ONNX Runtime, CUDA, fp16, fused kernels", "tensorrt": "TensorRT, fp16"}
TRT_HELP = ("  TensorRT needs TensorRT 10 for CUDA 13, which onnxruntime-gpu 1.31 is built against:\n"
            "    pip install \"tensorrt-cu13<11\"\n"
            "  If pip can't find it for Windows, skip TensorRT: NVIDIA's other route is a manual download.")
ONNX_HELP = ("pip uninstall -y onnxruntime\n"
             "    pip install \"onnxruntime-gpu[cuda,cudnn]\" onnx\n"
             "  (the [cuda,cudnn] part brings the CUDA 13 libraries onnxruntime-gpu needs; it also runs on the\n"
             "  CPU, so anything else that used onnxruntime keeps working)")


# ---------------------------------------------------------------- helpers
def free_gpu():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def sync():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def gpu_power():
    """Power limit (and the most it can be set to) from nvidia-smi: laptops lower it in quiet/battery modes."""
    vals = {}
    for field in ("enforced.power.limit", "power.limit", "power.max_limit"):
        try:
            out = subprocess.run(["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=15).stdout.strip().splitlines()[0]
            vals[field] = float(out)
        except Exception:
            pass
    limit = vals.get("enforced.power.limit") or vals.get("power.limit")
    if not limit:
        return ""
    top = vals.get("power.max_limit")
    return f"power limit {limit:.0f} W" + (f" (max {top:.0f} W)" if top else "")


def hf_model(ce):
    """The Hugging Face model inside a CrossEncoder (its location differs between versions)."""
    import torch
    for root in (getattr(ce, "model", None), ce):
        if isinstance(root, torch.nn.Module):
            for m in root.modules():
                if hasattr(getattr(m, "config", None), "model_type"):
                    return m
    return None


def describe(ce):
    m = hf_model(ce)
    if m is None:
        return "?", "?"
    dtype = str(next(m.parameters()).dtype).replace("torch.", "")
    return dtype, getattr(m.config, "_attn_implementation", None) or "?"


# ---------------------------------------------------------------- PyTorch re-rankers
def load_ce(name, device, max_length=MAX_LENGTH, attn=None):
    """Like rag_opensearch.load_reranker, but always fp16 on the GPU and with the settings as arguments."""
    import torch
    from sentence_transformers import CrossEncoder
    kw = {"torch_dtype": torch.float16} if device == "cuda" else {}
    if attn:
        kw["attn_implementation"] = attn
    try:
        ce = CrossEncoder(name, device=device, max_length=max_length, model_kwargs=kw)
    except TypeError:  # older sentence-transformers: no model_kwargs
        if attn:
            raise
        ce = CrossEncoder(name, device=device, max_length=max_length)
    m = hf_model(ce)
    if device == "cuda" and m is not None and next(m.parameters()).dtype != torch.float16:
        m.half()
    return ce


def torch_scorer(ce, batch_size):
    def score(pairs):
        return np.asarray(ce.predict(pairs, batch_size=batch_size, show_progress_bar=False),
                          dtype=np.float32).reshape(-1)
    return score


# ---------------------------------------------------------------- ONNX Runtime / TensorRT
def ort_has_gpu():
    """Is a GPU build of ONNX Runtime installed? (Only checks; the worker process loads its CUDA.)"""
    try:
        import onnxruntime as ort
    except ImportError:
        print(f"  ONNX Runtime isn't installed. To add it:\n    {ONNX_HELP}")
        return False
    print(f"  onnxruntime {ort.__version__}")
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        print(f"  This ONNX Runtime has no GPU support (providers: {', '.join(ort.get_available_providers())}).\n"
              f"  To fix:\n    {ONNX_HELP}")
        return False
    return True


def add_tensorrt_dlls():
    """pip's TensorRT keeps its DLLs inside its packages; ONNX Runtime has to be able to find them."""
    import site
    roots = [Path(p) for p in site.getsitepackages() + [site.getusersitepackages()] if Path(p).is_dir()]
    dirs = {f.parent for r in roots for pat in ("tensorrt*/nvinfer_*.dll", "tensorrt*/*/nvinfer_*.dll",
                                                "tensorrt*/libnvinfer.so.*") for f in r.glob(pat)}
    for d in dirs:
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(str(d))
            except OSError:
                pass
        os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
    return sorted(str(d) for d in dirs)


def build_onnx(name, out_dir, device):
    """Export the re-ranker to ONNX once, in fp16 (about 1.1 GB): plain_fp16.onnx, which TensorRT turns
    into its own engine, and fused\\model.onnx, where ONNX Runtime's transformer optimizer has replaced the
    attention, layer-norm and GELU steps with its fused GPU kernels.
    Returns (fused path, plain path, tokenizer, input names)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    out_dir = out_dir.resolve()
    plain, info = out_dir / "plain_fp16.onnx", out_dir / "model_info.txt"
    fdir = out_dir / "fused"          # model.onnx + weights.data + fused.data
    fused = fdir / "model.onnx"
    tok = AutoTokenizer.from_pretrained(name)

    if not info.exists():
        print("  Exporting the re-ranker to ONNX in fp16 (once; a few minutes)...")
        kw = {"torch_dtype": torch.float16}
        try:   # plain attention exports as MatMul/Softmax, the pattern the fusion step recognizes
            model = AutoModelForSequenceClassification.from_pretrained(name, attn_implementation="eager", **kw)
        except (TypeError, ValueError):
            model = AutoModelForSequenceClassification.from_pretrained(name, **kw)
        model = model.half().to(device).eval()
        sample = tok(["what is the question?"] * 2, ["a passage of text " * 8] * 2, padding=True,
                     truncation="longest_first", max_length=MAX_LENGTH, return_tensors="pt")
        names = [n for n in ("input_ids", "attention_mask", "token_type_ids") if n in sample]

        class Logits(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, *xs):
                return self.m(**dict(zip(names, xs))).logits

        dyn = {n: {0: "batch", 1: "seq"} for n in names}
        dyn["logits"] = {0: "batch"}
        kw = dict(input_names=names, output_names=["logits"], dynamic_axes=dyn, opset_version=17,
                  do_constant_folding=True)
        args = tuple(sample[n].to(device) for n in names)
        with torch.no_grad():
            try:
                torch.onnx.export(Logits(model), args, str(plain), dynamo=False, **kw)
            except TypeError:  # torch older than 2.5 has no `dynamo` argument
                torch.onnx.export(Logits(model), args, str(plain), **kw)
        config = model.config
        model = None
        free_gpu()
        info.write_text(f"{config.num_attention_heads} {config.hidden_size} {' '.join(names)}")

    heads, hidden, *names = info.read_text().split()
    if not (fdir / "done").exists():
        print("  Fusing attention / layer norm / GELU into ONNX Runtime's GPU kernels (once)...")
        # The fusion step works on the graph with the weights left on disk (weights.data), so it needs
        # little memory; it reads only the weights it merges (each layer's query/key/value) from there.
        import onnx
        from onnxruntime.transformers import optimizer
        shutil.rmtree(fdir, ignore_errors=True)
        fdir.mkdir()
        m = onnx.load(str(plain))
        onnx.save_model(m, str(fdir / "graph.onnx"), save_as_external_data=True, all_tensors_to_one_file=True,
                        location="weights.data", size_threshold=1024)
        m = None
        gc.collect()
        cwd = os.getcwd()
        os.chdir(fdir)
        try:
            opt = optimizer.optimize_model(onnx.load("graph.onnx", load_external_data=False), model_type="bert",
                                           num_heads=int(heads), hidden_size=int(hidden), opt_level=0, use_gpu=True)
            stats = {k: v for k, v in opt.get_fused_operator_statistics().items() if v}
            opt.topological_sort()
            onnx.save_model(opt.model, "model.onnx", save_as_external_data=True, all_tensors_to_one_file=True,
                            location="fused.data", size_threshold=1024)
        finally:
            os.chdir(cwd)
        opt = None
        gc.collect()
        (fdir / "done").write_text(", ".join(f"{k} {v}" for k, v in stats.items()) or "nothing")
    tok.backend_tokenizer.save(str(out_dir / "tokenizer.json"))   # the worker tokenizes without transformers
    done = (fdir / "done").read_text()
    print(f"  fused kernels: {done}")
    if "Attention" not in done:
        print("  (attention wasn't fused, so expect little gain from this setting)")
    return fused, plain, tok, names


def ort_session(ort, path, providers, want):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(path), so, providers=providers)
    used = sess.get_providers()
    if used[0] != want:
        raise RuntimeError(f"ONNX Runtime couldn't use {want} and fell back to {used[0]}")
    return sess


def trt_providers(cache_dir, input_names, batch_size, max_length):
    """TensorRT needs the range of input shapes up front, or it rebuilds its engine for every new shape."""
    def shapes(b, n):
        return ",".join(f"{name}:{b}x{n}" for name in input_names)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    opts = {"device_id": 0, "trt_fp16_enable": True,
            "trt_engine_cache_enable": True, "trt_engine_cache_path": str(cache_dir),
            "trt_timing_cache_enable": True, "trt_timing_cache_path": str(cache_dir),
            "trt_max_workspace_size": 2 * 2**30,
            "trt_profile_min_shapes": shapes(1, 1),
            "trt_profile_opt_shapes": shapes(batch_size, 512),
            "trt_profile_max_shapes": shapes(batch_size, max_length)}
    return [("TensorrtExecutionProvider", opts), ("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]


def ort_scorer(sess, tk, batch_size=BATCH):
    """Same input as CrossEncoder.predict: (question, chunk) pairs, longest first, padded per batch.
    tk: a `tokenizers` Tokenizer with truncation (longest first) and padding switched on."""
    names = [i.name for i in sess.get_inputs()]

    def score(pairs):
        order = sorted(range(len(pairs)), key=lambda i: -(len(pairs[i][0]) + len(pairs[i][1])))
        out = np.empty(len(pairs), dtype=np.float32)
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            enc = tk.encode_batch([(pairs[i][0], pairs[i][1]) for i in idx])
            arrays = {"input_ids": [e.ids for e in enc], "attention_mask": [e.attention_mask for e in enc],
                      "token_type_ids": [e.type_ids for e in enc]}
            feed = {n: np.asarray(arrays[n], dtype=np.int64) for n in names}
            out[idx] = np.asarray(sess.run(None, feed)[0], dtype=np.float32).reshape(len(idx), -1)[:, 0]
        return out
    return score


def worker(job_file):
    """ONNX Runtime / TensorRT timing, in its own process without PyTorch, so ONNX Runtime loads its own
    CUDA libraries (onnxruntime-gpu 1.31 uses CUDA 13; PyTorch may have loaded CUDA 12 ones)."""
    job = json.loads(Path(job_file).read_text(encoding="utf-8"))
    if "tensorrt" in job["settings"]:
        found = add_tensorrt_dlls()
        print(f"  TensorRT libraries: {', '.join(found) if found else 'not found'}", flush=True)
    import onnxruntime as ort
    if hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls()
        except Exception as e:
            print(f"  preload_dlls: {type(e).__name__}: {e}")
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(job["tokenizer"])
    tk.no_padding()
    tk.no_truncation()
    tk.enable_truncation(max_length=job["max_length"], strategy="longest_first")
    tk.enable_padding(pad_id=job["pad_id"], pad_token=job["pad_token"])
    if "torch" in sys.modules:
        print("  (note: PyTorch got loaded in the worker too)")

    results = {}
    for key in job["settings"]:
        print(f"\n[{key}] {LABELS[key]}", flush=True)
        sess = None
        try:
            if key == "onnx":
                sess = ort_session(ort, job["fused"], [("CUDAExecutionProvider", {"device_id": 0}),
                                                       "CPUExecutionProvider"], "CUDAExecutionProvider")
            else:
                print(f"  TensorRT builds its engine on the first questions: first run only, can take 10+ minutes;"
                      f" cached in {job['trt_cache']}", flush=True)
                sess = ort_session(ort, job["plain"], trt_providers(job["trt_cache"], job["names"], job["batch"],
                                                                    job["max_length"]), "TensorrtExecutionProvider")
            score = ort_scorer(sess, tk, job["batch"])
            for q, texts in job["questions"][:3]:          # untimed warm-up (and the TensorRT engine build)
                score([(q, t) for t in texts])
            times, scores = [], []
            for q, texts in job["questions"]:
                t0 = time.perf_counter()
                sc = score([(q, t) for t in texts])
                times.append(time.perf_counter() - t0)
                scores.append([float(x) for x in sc])
            print(f"  {np.mean(times):.2f} s per question (median {np.median(times):.2f} s)", flush=True)
            results[key] = {"times": times, "scores": scores}
        except Exception as e:
            msg = f"{type(e).__name__}: {str(e)[:400]}"
            print(f"  failed: {msg}")
            print(TRT_HELP if key == "tensorrt" else f"  If CUDA didn't load:\n    {ONNX_HELP}")
            results[key] = {"error": msg}
        sess = None
        gc.collect()
    Path(job["out"]).write_text(json.dumps(results), encoding="utf-8")


# ---------------------------------------------------------------- the benchmark
def rank_and_dedupe(hits, sc, topk=R.CANDIDATES):
    """Order the first `topk` hits by score (the rest follow in hybrid order), then drop near-duplicates."""
    head, tail = hits[:topk], hits[topk:]
    order = np.argsort(-np.asarray(sc, dtype=np.float32))
    ranked = [head[i] for i in order] + tail
    scores = [float(sc[i]) for i in order] + [None] * len(tail)
    return R.drop_near_duplicates(ranked, scores)[0]


def run_setting(score, qs, cands, topk=R.CANDIDATES, warmup=3):
    """Re-rank every question's candidates; time only the scoring. Returns (times, final lists, dedupe s)."""
    for q, hits in list(zip(qs, cands))[:warmup]:     # untimed: CUDA start-up
        score([(q["question"], h["text"]) for h in hits[:topk]])
    times, finals, t_dedupe = [], [], 0.0
    for q, hits in zip(qs, cands):
        pairs = [(q["question"], h["text"]) for h in hits[:topk]]
        sync()
        t0 = time.perf_counter()
        sc = score(pairs)
        sync()
        times.append(time.perf_counter() - t0)
        t1 = time.perf_counter()
        finals.append(rank_and_dedupe(hits, sc, topk))
        t_dedupe += time.perf_counter() - t1
    return times, finals, t_dedupe / max(len(qs), 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("eval_set", help="JSON lines file of questions, e.g. eval_set_001.jsonl")
    ap.add_argument("--url", default=R.OPENSEARCH_URL)
    ap.add_argument("--reranker", default=R.RERANK_MODEL)
    ap.add_argument("--onnx", action="store_true", help="also try ONNX Runtime with fused kernels")
    ap.add_argument("--tensorrt", action="store_true", help="also try NVIDIA TensorRT (through ONNX Runtime)")
    ap.add_argument("--only", default="", help=f"comma-separated settings to run, from: {', '.join(ALL_KEYS)}")
    ap.add_argument("--onnx-dir", default="reranker_onnx")
    ap.add_argument("--out", default="bench_rerank.csv")
    args = ap.parse_args()

    only = {k.strip() for k in args.only.split(",") if k.strip()}
    if args.only:
        bad = only - set(ALL_KEYS)
        if bad:
            sys.exit(f"Unknown setting(s): {', '.join(sorted(bad))}. Choose from: {', '.join(ALL_KEYS)}")
        args.onnx = args.onnx or "onnx" in only
        args.tensorrt = args.tensorrt or "tensorrt" in only
    wanted = (lambda k: k in only) if only else (lambda k: True)

    import torch
    import sentence_transformers
    import transformers
    qs = E.load_eval_set(args.eval_set)
    print("Fetching each question's candidates from OpenSearch (once)...")
    s = R.Searcher(args.url, None)
    vecs = s.embed_queries([q["question"] for q in qs])
    cands = [s.hybrid(q["question"], v, None, k=R.CANDIDATES) for q, v in zip(qs, vecs)]
    device = s.device
    print(f"OpenSearch {s.version} · {s.count:,} chunks · {len(qs)} questions · {R.CANDIDATES} candidates each")
    del s
    free_gpu()
    print(f"torch {torch.__version__} (CUDA {torch.version.cuda}) · transformers {transformers.__version__} · "
          f"sentence-transformers {sentence_transformers.__version__}")
    print(E.machine_note(device) + (f" · {gpu_power()}" if device == "cuda" else ""))
    if device != "cuda":
        print("  WARNING: no GPU found; these times say nothing about the GPU.")

    results = []   # (key, label, times, finals)

    def record(key, label, score, topk=R.CANDIDATES):
        print(f"\n[{key}] {label}")
        times, finals, dedupe = run_setting(score, qs, cands, topk=topk)
        print(f"  {np.mean(times):.2f} s per question (median {np.median(times):.2f} s)")
        results.append((key, label, times, finals))
        return dedupe

    # ---- PyTorch: the current loader first, then variations of it
    cur = R.load_reranker(args.reranker, device)
    if cur is None:
        sys.exit("The re-ranker didn't load; see the message above.")
    dtype, attn = describe(cur)
    print(f"Current re-ranker: {args.reranker}, {dtype}, attention: {attn}, batches of {BATCH}, "
          f"up to {MAX_LENGTH} tokens")
    dedupe_s = None
    if wanted("current"):
        dedupe_s = record("current", f"current ({dtype}, batch {BATCH}, {MAX_LENGTH} tokens, top 50)",
                          torch_scorer(cur, BATCH))
    base = cur
    if device == "cuda" and dtype != "float16":
        cur = base = None
        free_gpu()
        base = load_ce(args.reranker, device)
        if wanted("fp16"):
            record("fp16", "fp16 (the current setup isn't; all settings below are)", torch_scorer(base, BATCH))
    if attn == "eager" and wanted("sdpa"):
        try:
            ce = load_ce(args.reranker, device, attn="sdpa")
            record("sdpa", "PyTorch fused attention (SDPA)", torch_scorer(ce, BATCH))
            del ce
            free_gpu()
        except Exception as e:
            print(f"\n[sdpa] skipped: {type(e).__name__}: {str(e)[:150]}")
    for key, label, bs, topk in (("batch32", "batches of 32", 32, R.CANDIDATES),
                                 ("batch64", "one batch of all 50", 64, R.CANDIDATES),
                                 ("top40", "re-rank only the top 40 hybrid results", BATCH, 40),
                                 ("top30", "re-rank only the top 30 hybrid results", BATCH, 30)):
        if wanted(key):
            record(key, label, torch_scorer(base, bs), topk=topk)
    cur = base = None     # free the GPU for the next model
    free_gpu()
    if wanted("max512"):
        ce = load_ce(args.reranker, device, max_length=512)
        record("max512", "pairs cut at 512 tokens", torch_scorer(ce, BATCH))
        ce = None
        free_gpu()

    # ---- ONNX Runtime with fused kernels, and TensorRT: built here, timed in a process without PyTorch
    if args.onnx or args.tensorrt:
        print("\nONNX Runtime / TensorRT")
        built = None
        if ort_has_gpu():
            try:
                built = build_onnx(args.reranker, args.onnx_dir, device)
            except Exception as e:
                print(f"  ONNX export failed: {type(e).__name__}: {str(e)[:300]}")
        if built:
            fused, plain, tok, names = built
            free_gpu()
            keys = [k for k, on in (("onnx", args.onnx), ("tensorrt", args.tensorrt)) if on]
            work = Path(args.onnx_dir).resolve()
            job = {"settings": keys, "fused": str(fused), "plain": str(plain),
                   "tokenizer": str(work / "tokenizer.json"), "pad_id": tok.pad_token_id, "pad_token": tok.pad_token,
                   "max_length": MAX_LENGTH, "batch": BATCH, "names": names, "trt_cache": str(work / "trt_cache"),
                   "out": str(work / "worker_results.json"),
                   "questions": [[q["question"], [h["text"] for h in hits]] for q, hits in zip(qs, cands)]}
            job_file = work / "worker_job.json"
            job_file.write_text(json.dumps(job), encoding="utf-8")
            Path(job["out"]).unlink(missing_ok=True)
            print("  Timing in a separate process without PyTorch...", flush=True)
            subprocess.run([sys.executable, os.path.abspath(__file__), "--worker", str(job_file)])
            if Path(job["out"]).exists():
                res = json.loads(Path(job["out"]).read_text(encoding="utf-8"))
                for key in keys:
                    if "scores" in res.get(key, {}):
                        finals = [rank_and_dedupe(hits, sc) for hits, sc in zip(cands, res[key]["scores"])]
                        results.append((key, LABELS[key], res[key]["times"], finals))
            else:
                print("  The ONNX Runtime process stopped without results (see its messages above).")

    if not results:
        sys.exit("No setting ran.")

    # ---- summary
    def ranks(finals):
        return [E.first_rank(f, lambda h, q=q: E.answers(h, q, q["_ev"]), 5) for f, q in zip(finals, qs)]

    ref_key, _, ref_times, ref_finals = results[0]
    ref_ranks, ref_t = ranks(ref_finals), float(np.mean(ref_times))
    ref_top5 = [[h["chunk_id"] for h in f[:5]] for f in ref_finals]
    print(f"\nRe-ranking only (scoring the candidates), {len(qs)} questions; compared with [{ref_key}]:")
    print(f"  {'setting':<10}{'mean':>8}{'median':>8}{'speed-up':>10}{'Recall@5':>10}{'MRR@5':>7}"
          f"{'same top 5':>12}")
    table = []
    for key, label, times, finals in results:
        rk = ranks(finals)
        same = sum([h["chunk_id"] for h in f[:5]] == r for f, r in zip(finals, ref_top5))
        mrr = sum(1 / r for r in rk if r) / len(rk)
        print(f"  {key:<10}{np.mean(times):>7.2f}s{np.median(times):>7.2f}s{ref_t / np.mean(times):>9.2f}x"
              f"{E.share(rk):>10.0%}{mrr:>7.2f}{same:>8} of {len(qs)}")
        gained = [q["qid"] for q, a, b in zip(qs, ref_ranks, rk) if b and not a]
        lost = [q["qid"] for q, a, b in zip(qs, ref_ranks, rk) if a and not b]
        if gained or lost:
            print(f"  {'':<10}  vs [{ref_key}]: found {', '.join(gained) or 'none'}; lost {', '.join(lost) or 'none'}")
        table.append((key, rk, times))
    if dedupe_s is not None:
        print(f"  (duplicate removal, on the CPU, adds {dedupe_s * 1000:.0f} ms per question to every setting)")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["qid", "question"] + [c for key, _, _ in table for c in (f"{key}_rank@5", f"{key}_s")])
        for i, q in enumerate(qs):
            w.writerow([q["qid"], q["question"]] + [c for _, rk, times in table
                                                    for c in (rk[i] or "", f"{times[i]:.3f}")])
    print(f"\nPer-question ranks and times: {args.out}")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        worker(sys.argv[2])
    else:
        main()