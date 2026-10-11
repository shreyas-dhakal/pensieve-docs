"""Related-document ranking: pensieve-docs vs standard baselines, on a labelled corpus.

    uv run python bench/compare.py [data_dir] [out_dir]     # MiniLM needs `uv sync --extra embeddings`

Every method ranks the other documents for each document (its full text as the query):
- concepts:       summed min(TF-IDF) over shared concepts, from graph.json mentions (what bench/eval.py scores)
- Index.related:  the same score through the query path (spaCy on the query text), as the product calls it
- tfidf:          cosine of word TF-IDF vectors
- bm25:           Okapi BM25 (k1 1.2, b 0.75), query = the document's words
- minilm:         cosine of all-MiniLM-L6-v2 embeddings, averaged over ~800-character passages
Relevance is either "same topic" or "linked on Wikipedia" (either direction). Same topic is easy with six unrelated
topics; links are the harder test, since they pick a few documents out of each topic.
"""
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

from pensieve_docs import config
from pensieve_docs.build import build
from pensieve_docs.edges import overlap
from pensieve_docs.parse import parse_file
from pensieve_docs.related import Index

K = 3
_TOKEN = re.compile(r"[a-z][a-z0-9]+")


def words(text: str) -> list[str]:
    from spacy.lang.en.stop_words import STOP_WORDS

    return [w for w in _TOKEN.findall(text.lower()) if w not in STOP_WORDS]


def tfidf_rank(queries: dict, docs: dict) -> dict:
    """{query id: doc ids best first}; IDF from the docs. A query never ranks a doc with its own id."""
    df, n = Counter(w for t in docs.values() for w in set(words(t))), len(docs)

    def vec(text):
        v = {w: (1 + math.log(k)) * math.log(n / df[w]) for w, k in Counter(words(text)).items() if w in df}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {w: x / norm for w, x in v.items()}

    dv = {d: vec(t) for d, t in docs.items()}
    return {q: sorted((d for d in docs if d != q), key=lambda d: (-sum(x * dv[d].get(w, 0) for w, x in qv.items()), d))
            for q, qv in ((q, vec(t)) for q, t in queries.items())}


def bm25_rank(queries: dict, docs: dict, k1=1.2, b=0.75) -> dict:
    tf = {d: Counter(words(t)) for d, t in docs.items()}
    df = Counter(w for c in tf.values() for w in c)
    n, avg = len(docs), sum(sum(c.values()) for c in tf.values()) / len(docs)
    idf = {w: math.log(1 + (n - k + 0.5) / (k + 0.5)) for w, k in df.items()}

    def score(qtf, d):
        c, dl = tf[d], sum(tf[d].values())
        return sum(qn * idf[w] * c[w] * (k1 + 1) / (c[w] + k1 * (1 - b + b * dl / avg)) for w, qn in qtf.items() if w in c)

    return {q: sorted((d for d in docs if d != q), key=lambda d: (-score(qtf, d), d))
            for q, qtf in ((q, Counter(words(t))) for q, t in queries.items())}


def minilm_rank(queries: dict, docs: dict) -> dict:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # HF_HUB_OFFLINE=1: cached copy only

    def embed(t):
        paras = [p for p in re.split(r"\n\s*\n", t) if p.strip()] or [t]
        passages, cur = [], ""
        for p in paras:  # ~800-character passages: the model reads 256 tokens at most
            if cur and len(cur) + len(p) > 800:
                passages.append(cur)
                cur = ""
            cur += p + "\n"
        passages.append(cur)
        v = model.encode(passages, normalize_embeddings=True).mean(axis=0)
        return v / np.linalg.norm(v)

    de = {d: embed(t) for d, t in docs.items()}
    ids = sorted(de)
    mat = np.stack([de[d] for d in ids])
    out = {}
    for q, t in queries.items():
        sims = mat @ (de[q] if q in de else embed(t))
        out[q] = [ids[i] for i in sorted(range(len(ids)), key=lambda i: (-sims[i], ids[i])) if ids[i] != q]
    return out


def metrics(rank: dict, relevant) -> dict:
    """P@3, MAP and nDCG@10 for relevant(query, doc) -> bool, averaged over queries that have any relevant doc."""
    p3, ap, ndcg = [], [], []
    for q, ranked in rank.items():
        rel = [relevant(q, d) for d in ranked]
        if not any(rel):
            continue
        p3.append(sum(rel[:K]) / K)
        hits, prec = 0, []
        for i, r in enumerate(rel, 1):
            if r:
                hits += 1
                prec.append(hits / i)
        ap.append(sum(prec) / sum(rel))
        dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rel[:10]))
        ideal = sum(1 / math.log2(i + 2) for i in range(min(10, sum(rel))))
        ndcg.append(dcg / ideal)
    mean = lambda xs: sum(xs) / len(xs)
    return {"P@3": mean(p3), "MAP": mean(ap), "nDCG@10": mean(ndcg)}


def main(data: Path, out: Path):
    truth = json.loads((data / "truth.json").read_text())
    topic, linked = truth["topics"], {frozenset(p) for p in truth["links"]}
    build(data, config.load(folder=data), out)
    g = json.loads((out / "graph.json").read_text())
    docs = {n["id"]: n["source_file"] for n in g["nodes"] if n["kind"] == "document"}
    texts = {f: "\n\n".join(c["text"] for c in parse_file(data / f)["chunks"]) for f in sorted(docs.values())}

    w = {}
    for e in g["links"]:
        if e["relation"] == "mentions" and e["source"] in docs:
            w.setdefault(docs[e["source"]], {})[e["target"]] = e["weight"]
    ranks, secs = {}, {}
    t = time.perf_counter()
    ranks["concepts (graph)"] = {f: sorted((o for o in texts if o != f),
                                           key=lambda o: (-sum(x for x, _ in overlap(w.get(f, {}), w.get(o, {}))), o))
                                 for f in texts}
    secs["concepts (graph)"] = time.perf_counter() - t
    idx, t = Index(out / "graph.json"), time.perf_counter()
    ranks["Index.related"] = {}
    for f, text in texts.items():
        hits = [h["source_file"] for h in idx.related(text, k=len(texts), min_score=0, exclude=f)]
        ranks["Index.related"][f] = hits + sorted(o for o in texts if o != f and o not in hits)  # unscored: last
    secs["Index.related"] = time.perf_counter() - t
    for name, fn in (("tfidf", tfidf_rank), ("bm25", bm25_rank), ("minilm", minilm_rank)):
        t = time.perf_counter()
        try:
            ranks[name] = fn(texts, texts)
        except ImportError:
            print(f"{name}: skipped (uv sync --extra embeddings)")
            continue
        secs[name] = time.perf_counter() - t

    same = lambda q, d: topic[q] == topic[d]
    link = lambda q, d: frozenset((q, d)) in linked
    print(f"{len(texts)} documents, {len(set(topic.values()))} topics, {len(linked)} linked pairs\n")
    print(f"| method | same topic P@{K} | MAP | nDCG@10 | wiki-linked P@{K} | MAP | nDCG@10 | seconds |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    out_rows = {}
    for name, rank in ranks.items():
        a, b = metrics(rank, same), metrics(rank, link)
        out_rows[name] = (a, b)
        print(f"| {name} | {a['P@3']:.1%} | {a['MAP']:.3f} | {a['nDCG@10']:.3f} "
              f"| {b['P@3']:.1%} | {b['MAP']:.3f} | {b['nDCG@10']:.3f} | {secs[name]:.1f} |")
    return out_rows


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "bench/data/wiki"),
         Path(sys.argv[2] if len(sys.argv) > 2 else "bench/data/out"))
