"""Score a built graph against a labelled corpus: python bench/eval.py [data_dir] [out_dir]

Builds the graph, then reports
- communities vs. topics: purity (is each community one topic?) and inverse purity (is each topic one community?)
- related docs: for each doc, the top 3 other docs by shared concepts (min TF-IDF weight, summed);
  how many share its topic, and how many it links to on Wikipedia (either direction)
- the same for the similar_to edges
- concept count, and the most widespread concepts to eyeball for junk
"""
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from pensieve_docs import config
from pensieve_docs.build import build
from pensieve_docs.parse import parse_file
from pensieve_docs.related import Index

K = 3


def idx_threshold():
    from pensieve_docs.related import MIN_SCORE

    return MIN_SCORE


def main(data: Path, out: Path):
    truth = json.loads((data / "truth.json").read_text())
    topic, linked = truth["topics"], {frozenset(p) for p in truth["links"]}
    t0 = time.perf_counter()
    stats = build(data, config.load(folder=data), out)
    secs = time.perf_counter() - t0
    g = json.loads((out / "graph.json").read_text())
    nodes = {n["id"]: n for n in g["nodes"]}
    docs = {n["id"]: n["source_file"] for n in g["nodes"] if n["kind"] == "document"}

    # communities vs topics
    comm = defaultdict(list)
    for d, f in docs.items():
        comm[nodes[d]["community"]].append(topic[f])
    n = len(docs)
    purity = sum(Counter(ts).most_common(1)[0][1] for ts in comm.values()) / n
    by_topic = defaultdict(Counter)
    for c, ts in comm.items():
        for t in ts:
            by_topic[t][c] += 1
    inverse = sum(cs.most_common(1)[0][1] for cs in by_topic.values()) / n

    # related docs by shared concepts (what a recall badge would show)
    w = defaultdict(dict)
    sim = defaultdict(dict)
    for e in g["links"]:
        if e["relation"] == "mentions":
            w[e["source"]][e["target"]] = e["weight"]
        elif e["relation"] == "similar_to":
            sim[e["source"]][e["target"]] = sim[e["target"]][e["source"]] = e["weight"]

    scored = []  # (score, same topic) for every concept suggestion, to pick a badge threshold

    def precision(rank):
        same = link = total = 0
        misses = []
        for d, f in docs.items():
            for o in rank(d)[:K]:
                if rank is by_concepts:
                    scored.append((score(d, o), topic[docs[o]] == topic[f]))
                total += 1
                same += topic[docs[o]] == topic[f]
                link += frozenset((f, docs[o])) in linked
                if topic[docs[o]] != topic[f]:
                    misses.append(f"{f} -> {docs[o]}")
        return same / max(total, 1), link / max(total, 1), total, misses

    def score(d, o):
        return sum(min(w[d][c], w[o][c]) for c in w[d].keys() & w[o].keys())

    def by_concepts(d):
        sc = {o: score(d, o) for o in docs if o != d}
        return [o for o, s in sorted(sc.items(), key=lambda x: -x[1]) if s > 0]

    def by_similarity(d):
        return sorted(sim[d], key=lambda o: -sim[d][o])

    cp = precision(by_concepts)
    sp = precision(by_similarity)
    # the real query path: each article's full text through Index.related, as the product would call it
    idx, hits, same, link, t1 = Index(out / "graph.json"), 0, 0, 0, time.perf_counter()
    for f in docs.values():
        text = "\n".join(c["text"] for c in parse_file(data / f)["chunks"])
        for h in idx.related(text, k=K, exclude=f):
            hits += 1
            same += topic[h["source_file"]] == topic[f]
            link += frozenset((f, h["source_file"])) in linked
    qms = (time.perf_counter() - t1) / len(docs) * 1000
    concepts = sorted((x for x in g["nodes"] if x["kind"] == "concept"), key=lambda x: -x["doc_count"])

    print(f"build: {stats['detected']} files -> {len(g['nodes'])} nodes, {len(g['links'])} edges in {secs:.1f}s")
    print(f"communities: {len(comm)} for {len(by_topic)} topics | purity {purity:.2f} | inverse purity {inverse:.2f}")
    print(f"related@{K} by concepts:  same topic {cp[0]:.0%}, wiki-linked {cp[1]:.0%} ({cp[2]} suggestions)")
    print(f"related@{K} by similar_to: same topic {sp[0]:.0%}, wiki-linked {sp[1]:.0%} ({sp[2]} suggestions)")
    print(f"Index.related (threshold {idx_threshold()}): {hits / (K * len(docs)):.0%} of slots filled, "
          f"same topic {same / max(hits, 1):.0%}, wiki-linked {link / max(hits, 1):.0%}, {qms:.0f} ms per query")
    for th in (0.1, 0.2, 0.3, 0.5):
        kept = [ok for s, ok in scored if s >= th]
        print(f"  badge threshold {th}: shows {len(kept) / len(scored):.0%} of suggestions, "
              f"{sum(kept) / max(len(kept), 1):.0%} same topic")
    print(f"concepts: {len(concepts)} ({sum(' ' not in c['norm'] for c in concepts)} single-word)")
    print("most widespread:", ", ".join(f"{c['label']}({c['doc_count']})" for c in concepts[:40]))
    print("cross-topic concept suggestions:", *cp[3][:12], sep="\n  ")
    return {"purity": purity, "inverse_purity": inverse, "concept_p": cp[:2], "similar_p": sp[:2]}


if __name__ == "__main__":
    data = Path(sys.argv[1] if len(sys.argv) > 1 else "bench/data/wiki")
    main(data, Path(sys.argv[2] if len(sys.argv) > 2 else "bench/data/out"))
