"""Related papers on BEIR SCIDOCS: pensieve-docs vs TF-IDF, BM25 and MiniLM.

    curl -L -o bench/data/scidocs.zip https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scidocs.zip
    unzip bench/data/scidocs.zip -d bench/data
    HF_HUB_OFFLINE=1 uv run python bench/scidocs.py [n_queries]

The query is a paper's title; relevant papers are ones it cites or that are co-read with it, 5 of ~30 judged per query.
pensieve-docs compares all document pairs, so this ranks a sub-corpus: every paper judged for the first n queries
(2933 papers for 100). Every method ranks the same pool, so they compare with each other, but the absolute numbers are
higher than full-corpus BEIR results.
"""
import csv
import json
import sys
import time
from pathlib import Path

from compare import bm25_rank, metrics, minilm_rank, tfidf_rank

from pensieve_docs import config
from pensieve_docs.build import build
from pensieve_docs.related import Index

DATA = Path("bench/data/scidocs")


def recall_at(rank: dict, relevant, k=100) -> float:
    rs = [sum(relevant(q, d) for d in r[:k]) / n for q, r in rank.items() if (n := sum(relevant(q, d) for d in r))]
    return sum(rs) / len(rs)


def main(n: int):
    qrels = {}
    with open(DATA / "qrels" / "test.tsv") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            qrels.setdefault(r["query-id"], {})[r["corpus-id"]] = int(r["score"])
    qids = sorted(qrels)[:n]
    pool = {d for q in qids for d in qrels[q]}
    titles = {}
    for line in open(DATA / "queries.jsonl"):
        q = json.loads(line)
        if q["_id"] in qids:
            titles[q["_id"]] = q["text"]
    docs = {}
    for line in open(DATA / "corpus.jsonl"):
        d = json.loads(line)
        if d["_id"] in pool:
            docs[d["_id"]] = f"{d['title']}\n\n{d['text']}"

    folder, out = Path(f"bench/data/scidocs-{n}"), Path(f"bench/data/scidocs-{n}-out")
    folder.mkdir(parents=True, exist_ok=True)
    for d, text in docs.items():
        p = folder / f"{d}.txt"
        if not p.exists():
            p.write_text(text, encoding="utf-8")
    t = time.perf_counter()
    build(folder, config.load(), out)
    build_secs = time.perf_counter() - t

    ranks, secs = {}, {}
    idx, t = Index(out / "graph.json"), time.perf_counter()
    ranks["pensieve Index.related"] = {}
    for q, title in titles.items():
        hits = [h["source_file"][:-4] for h in idx.related(title, k=len(docs), min_score=0)]
        seen = set(hits)
        ranks["pensieve Index.related"][q] = hits + sorted(d for d in docs if d not in seen)  # unscored: last
    secs["pensieve Index.related"] = time.perf_counter() - t
    for name, fn in (("tfidf", tfidf_rank), ("bm25", bm25_rank), ("minilm", minilm_rank)):
        t = time.perf_counter()
        ranks[name] = fn(titles, docs)
        secs[name] = time.perf_counter() - t

    rel = lambda q, d: qrels[q].get(d, 0) > 0
    print(f"SCIDOCS: {len(titles)} title queries over {len(docs)} papers; pensieve build {build_secs:.0f}s\n")
    print("| method | nDCG@10 | MAP | P@3 | Recall@100 | seconds |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for name, r in ranks.items():
        m = metrics(r, rel)
        print(f"| {name} | {m['nDCG@10']:.3f} | {m['MAP']:.3f} | {m['P@3']:.1%} | {recall_at(r, rel):.3f} | {secs[name]:.1f} |")
    found = sum(bool(idx.concepts(t)) for t in titles.values())
    print(f"\npensieve found at least one known concept in {found} of {len(titles)} titles")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 100)
