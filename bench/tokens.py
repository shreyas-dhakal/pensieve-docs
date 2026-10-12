"""How many tokens an agent saves by asking pensieve-docs first: uv run python bench/tokens.py [data_dir] [out_dir]

For each question about the wiki bench, compares reading the whole corpus with three ways an agent can use the graph:
- graph slice: the 3 nodes whose labels best match the question plus everything within 3 hops, written as one
  line per node and edge. This is the measure graphify's `graphify benchmark` reports, so the numbers compare.
- related + sections: Index.related(question, k=3), then only the sections its reasons point to
- related + documents: Index.related(question, k=3), then the 3 whole documents
Tokens are estimated as characters / 4 throughout. Each question names the article that answers it, so the table
also says whether that article reached the agent: a slice that saves tokens but drops the answer is no saving.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

from spacy.lang.en.stop_words import STOP_WORDS

from pensieve_docs import config
from pensieve_docs.build import build
from pensieve_docs.parse import location, parse_file
from pensieve_docs.related import Index

QUESTIONS = {  # question -> the article that answers it
    "how is malaria transmitted by mosquitoes": "malaria/malaria.md",
    "what does caffeine do to adenosine receptors": "coffee/caffeine.md",
    "how does a heat pump work": "home-energy/heat-pump.md",
    "how does backpropagation compute the gradient": "machine-learning/backpropagation.md",
    "who was augustus": "roman-history/augustus.md",
    "what is bebop": "jazz/bebop.md",
    "how is espresso made": "coffee/espresso.md",
    "what is net metering": "home-energy/net-metering.md",
    "how do mosquito nets prevent malaria": "malaria/mosquito-net.md",
    "what is overfitting": "machine-learning/overfitting.md",
}


def tokens(text: str) -> int:
    return max(1, len(text) // 4)


def graph_slice(g: dict, question: str, depth: int = 3) -> tuple[str, set]:
    """The text an agent would be handed, and the source files it covers."""
    nodes = {n["id"]: n for n in g["nodes"]}
    adj = defaultdict(list)
    for e in g["links"]:
        adj[e["source"]].append((e["target"], e["relation"]))
        adj[e["target"]].append((e["source"], e["relation"]))
    terms = [w for w in question.lower().split() if len(w) > 2 and w not in STOP_WORDS]
    scored = sorted(((sum(t in nodes[n]["label"].lower() for t in terms), n) for n in nodes), reverse=True)
    seen = frontier = {n for s, n in scored[:3] if s > 0}
    seen, lines = set(seen), []
    for _ in range(depth):
        nxt = set()
        for n in sorted(frontier):
            for m, rel in adj[n]:
                if m not in seen:
                    nxt.add(m)
                    lines.append(f"EDGE {nodes[n]['label']} --{rel}--> {nodes[m]['label']}")
        seen |= nxt
        frontier = nxt
    lines += [f"NODE {nodes[n]['label']} src={nodes[n]['source_file']} loc={nodes[n].get('source_location')}"
              for n in sorted(seen)]
    return "\n".join(lines), {nodes[n]["source_file"] for n in seen}


def main(data: Path, out: Path):
    build(data, config.load(folder=data), out)
    g = json.loads((out / "graph.json").read_text())
    idx = Index(out / "graph.json")
    parsed = {f: parse_file(data / f) for f in idx.docs}
    full = {f: "\n\n".join(c["text"] for c in p["chunks"]) for f, p in parsed.items()}
    corpus = sum(tokens(t) for t in full.values())

    rows = []
    for q, answer in QUESTIONS.items():
        text, files = graph_slice(g, q)
        hits = idx.related(q, k=3, min_score=0)
        listing = json.dumps(hits, ensure_ascii=False)
        cited = {(h["source_file"], r["where"].split(" ", 1)[1]) for h in hits for r in h["reasons"]}
        sections = "\n\n".join(c["text"] for f, p in parsed.items() for c in p["chunks"]
                               if (f, location(c, p["doc_type"])) in cited)
        top = {h["source_file"] for h in hits}
        rows.append((q, (tokens(text), answer in files), (tokens(listing) + tokens(sections), answer in top),
                     (tokens(listing) + sum(tokens(full[f]) for f in top), answer in top)))

    print(f"corpus: {len(full)} documents, ~{corpus:,} tokens if an agent reads them all\n")
    print("| question | graph slice | related + sections | related + documents |")
    print("| --- | ---: | ---: | ---: |")
    cell = lambda t, ok: f"{t:,} ({corpus / t:.0f}x){'' if ok else ' MISSED'}"
    for q, *ms in rows:
        print(f"| {q} | " + " | ".join(cell(*m) for m in ms) + " |")
    avg = [sum(r[i][0] for r in rows) / len(rows) for i in (1, 2, 3)]
    found = [sum(r[i][1] for r in rows) for i in (1, 2, 3)]
    print("| **average** | " + " | ".join(f"**{a:,.0f} ({corpus / a:.0f}x)**, answer in {k}/{len(rows)}"
                                         for a, k in zip(avg, found)) + " |")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "bench/data/wiki"),
         Path(sys.argv[2] if len(sys.argv) > 2 else "bench/data/out"))
