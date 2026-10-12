"""Answer quality, LazyGraphRAG-style: uv run python bench/lazygraphrag.py [data_dir] [out_dir] [queries_per_class]

Follows the method of Microsoft's LazyGraphRAG post (microsoft.com/en-us/research/blog/lazygraphrag-...), scaled down:
- an LLM writes synthetic queries from the corpus, half local (answered by one or two documents), half global
  (sensemaking over the whole corpus); the post used 50 + 50 over 5,590 AP articles, the default here is 10 + 10
- each condition retrieves context for a query; the same LLM answers from that context only
- an LLM judge compares two answers head to head on comprehensiveness, diversity, empowerment and relevance, in both
  orders to cancel position bias; win rate = (wins + ties / 2) / comparisons, so 50% is a draw
Conditions (pensieve-docs is compared with each baseline at the same context budget, as the post's vector RAG 8K/64K):
- pensieve 8K:  Index.related(query), the sections its reasons cite, then the rest of those documents, up to 8K tokens
- bm25 8K:      Okapi BM25 over chunks, best first, up to 8K tokens
- vector 8K/64K: all-MiniLM-L6-v2 cosine over chunks (needs `uv sync --extra embeddings`)
The LLM is the local `claude` CLI (no API key needed); BENCH_MODEL picks the model (default sonnet). Every LLM reply is
cached in out_dir/llm-cache.json, so a rerun resumes where it stopped.
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from compare import bm25_rank  # noqa: E402

from pensieve_docs import config  # noqa: E402
from pensieve_docs.build import build  # noqa: E402
from pensieve_docs.parse import location, parse_file  # noqa: E402
from pensieve_docs.related import Index  # noqa: E402

MODEL = os.environ.get("BENCH_MODEL", "sonnet")
METRICS = {  # definitions from the GraphRAG paper (Edge et al. 2024), which the LazyGraphRAG post reuses
    "comprehensiveness": "How much detail does the answer provide to cover all aspects and details of the question?",
    "diversity": "How varied and rich is the answer in providing different perspectives and insights on the question?",
    "empowerment": "How well does the answer help the reader understand and make informed judgements about the topic?",
    "relevance": "How clearly and specifically does the answer address the question?",
}
_lock = threading.Lock()


def tokens(text: str) -> int:
    return max(1, len(text) // 4)


def llm(prompt: str, cache: dict, path: Path) -> str:
    key = hashlib.sha256(f"{MODEL}\n{prompt}".encode()).hexdigest()
    if key not in cache:
        r = subprocess.run(["claude", "-p", "--model", MODEL, "--tools", "", "--no-session-persistence"],
                           input=prompt, capture_output=True, text=True, timeout=600)
        if r.returncode:
            raise RuntimeError(r.stderr or r.stdout)
        with _lock:
            cache[key] = r.stdout.strip()
            path.write_text(json.dumps(cache))
    return cache[key]


def parse_json(text: str):
    return json.loads(re.search(r"[\[{].*[\]}]", text, re.S).group())


def fill(passages, budget: int) -> str:
    out, used = [], 0
    for p in passages:
        if p not in out and used + tokens(p) <= budget:
            out.append(p)
            used += tokens(p)
    return "\n\n---\n\n".join(out)


def main(data: Path, out: Path, n: int):
    build(data, config.load(folder=data), out)
    idx = Index(out / "graph.json")
    parsed = {f: parse_file(data / f) for f in idx.docs}
    chunks = {f"{f}#{i}": f"[{f}]\n{c['text']}" for f, p in parsed.items() for i, c in enumerate(p["chunks"])}
    cache_path = out / "llm-cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    ask = lambda prompt: llm(prompt, cache, cache_path)

    outline = "\n".join(f"- {f}: {' '.join(c['text'] for c in p['chunks'])[:300]}" for f, p in parsed.items())
    queries = parse_json(ask(
        f"Below is an outline of a document collection. Write {n} LOCAL questions, each answered by facts in one or "
        f"two documents, and {n} GLOBAL questions, which need sensemaking across many documents (themes, contrasts, "
        f"trends) and have no single document that answers them. Reply with JSON only: "
        f'{{"local": [...], "global": [...]}}\n\n{outline}'))

    def pensieve(q, budget):
        hits = idx.related(q, k=10, min_score=0)
        by_loc = {(f, location(c, p["doc_type"])): f"[{f}]\n{c['text']}" for f, p in parsed.items() for c in p["chunks"]}
        cited = [by_loc[(h["source_file"], r["where"].split(" ", 1)[1])] for h in hits for r in h["reasons"]
                 if (h["source_file"], r["where"].split(" ", 1)[1]) in by_loc]
        rest = [f"[{h['source_file']}]\n{c['text']}" for h in hits for c in parsed[h["source_file"]]["chunks"]]
        return fill(cited + rest, budget)

    allq = [(q, "local") for q in queries["local"]] + [(q, "global") for q in queries["global"]]
    bm25 = bm25_rank({q: q for q, _ in allq}, chunks)
    conditions = {"pensieve 8K": {q: pensieve(q, 8000) for q, _ in allq},
                  "bm25 8K": {q: fill((chunks[c] for c in bm25[q]), 8000) for q, _ in allq}}
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        ids = list(chunks)
        mat = model.encode([chunks[c] for c in ids], normalize_embeddings=True)  # ponytail: model reads 256 tokens
        for budget in (8000, 64000):
            conditions[f"vector {budget // 1000}K"] = {
                q: fill((chunks[ids[i]] for i in np.argsort(-(mat @ model.encode(q, normalize_embeddings=True)))), budget)
                for q, _ in allq}
    except ImportError:
        print("vector: skipped (uv sync --extra embeddings)")

    def answer(q, ctx):
        return ask(f"Answer the question using only the context below. Be thorough.\n\nQuestion: {q}\n\nContext:\n{ctx}")

    def judge(q, a, b):
        crit = "\n".join(f"- {m}: {d}" for m, d in METRICS.items())
        return parse_json(ask(
            f"Compare two answers to the question on each criterion. Reply with JSON only, mapping each criterion to "
            f'"1", "2" or "tie".\n\nCriteria:\n{crit}\n\nQuestion: {q}\n\nAnswer 1:\n{a}\n\nAnswer 2:\n{b}'))

    jobs = [(name, q) for name in conditions for q, _ in allq]
    with ThreadPoolExecutor(8) as ex:
        answers = dict(zip(jobs, ex.map(lambda j: answer(j[1], conditions[j[0]][j[1]]), jobs)))
        pairs = [(base, q, order) for base in conditions if base != "pensieve 8K" for q, _ in allq for order in (0, 1)]

        def run(p):
            base, q, order = p
            a, b = answers[("pensieve 8K", q)], answers[(base, q)]
            return judge(q, *((a, b) if order == 0 else (b, a)))
        verdicts = dict(zip(pairs, ex.map(run, pairs)))

    kind = dict(allq)
    print(f"{len(parsed)} documents, {len(queries['local'])} local + {len(queries['global'])} global queries, "
          f"model {MODEL}\n")
    print("Win rate of pensieve 8K against each condition (50% = draw):\n")
    print("| vs | queries | " + " | ".join(METRICS) + " | avg context tokens |")
    print("| --- | --- |" + " ---: |" * (len(METRICS) + 1))
    for base in (c for c in conditions if c != "pensieve 8K"):
        for cls in ("local", "global"):
            row = []
            for m in METRICS:
                s = [1.0 if v.get(m) == ("1" if o == 0 else "2") else 0.5 if v.get(m) == "tie" else 0.0
                     for (b, q, o), v in verdicts.items() if b == base and kind[q] == cls]
                row.append(f"{sum(s) / len(s):.0%}")
            ctx = [tokens(conditions[base][q]) for q, k in allq if k == cls]
            print(f"| {base} | {cls} | " + " | ".join(row) + f" | {sum(ctx) / len(ctx):,.0f} |")
    ctx = [tokens(conditions["pensieve 8K"][q]) for q, _ in allq]
    print(f"\npensieve 8K used {sum(ctx) / len(ctx):,.0f} context tokens per query on average. "
          f"Indexing used no LLM calls.")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "bench/data/wiki"),
         Path(sys.argv[2] if len(sys.argv) > 2 else "bench/data/out"),
         int(sys.argv[3]) if len(sys.argv) > 3 else 10)
