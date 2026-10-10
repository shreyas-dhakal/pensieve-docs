"""Which documents relate to a piece of text, and why. Load once, query many times:

    idx = Index("pensieve-out/graph.json")
    idx.related("thinking about a home battery for backup power")
    # [{"id", "label", "source_file", "score", "reasons": [{"concept", "where"}, ...]}, ...]

The score is the one bench/eval.py measures: summed min(query weight, doc weight) over shared concepts,
with both sides TF-IDF weighted and L2-normalised, so the same thresholds apply.
"""
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from . import config
from .concepts import nlp
from .edges import _words

MIN_SCORE = 0.2  # bench: keeps ~2/3 of top-3 suggestions, all on topic


class Index:
    def __init__(self, graph_json: str | Path, cfg: dict | None = None):
        g = json.loads(Path(graph_json).read_text(encoding="utf-8"))
        self.cfg = cfg or config.load()
        self.root = Path(g["graph"].get("root", "."))
        self.nodes = {n["id"]: n for n in g["nodes"]}
        self.docs = {n["source_file"]: n["id"] for n in g["nodes"] if n["kind"] == "document"}
        self.mentions = defaultdict(dict)  # doc id -> concept id -> (weight, source_location)
        for e in g["links"]:
            if e["relation"] == "mentions":
                self.mentions[e["source"]][e["target"]] = (e["weight"], e["source_location"])
        n = len(self.docs)
        self.forms, self.idf = {}, {}
        for c in (x for x in g["nodes"] if x["kind"] == "concept"):
            self.idf[c["id"]] = math.log((1 + n) / (1 + c["doc_count"])) + 1
            for s in (c["norm"], c["label"]):
                if k := tuple(w for w, _ in _words(s)):
                    self.forms.setdefault(k, c["id"])
        self.maxn = max(map(len, self.forms), default=0)

    def concepts(self, text: str) -> Counter:
        """Known concepts in the text, longest match first, on both surface words and lemmas."""
        model = nlp(self.cfg["concepts"]["spacy_model"])
        with model.select_pipes(disable=[p for p in ("parser", "ner") if p in model.pipe_names]):  # lemmas only
            doc = model(text)
        lemmas = " ".join("|" if t.is_punct and t.text != "-" else t.lemma_ for t in doc)
        found = Counter()
        for ws in (_words(text), _words(lemmas)):
            hits, i = Counter(), 0
            while i < len(ws):
                for n in range(min(self.maxn, len(ws) - i), 0, -1):
                    k = tuple(w for w, _ in ws[i:i + n])
                    if k in self.forms and not any(b for _, b in ws[i + 1:i + n]):
                        hits[self.forms[k]] += 1
                        i += n
                        break
                else:
                    i += 1
            found |= hits  # max per concept: the same word seen as surface and lemma counts once
        return found

    def related(self, text: str, k: int = 5, min_score: float = MIN_SCORE, exclude: str | None = None) -> list[dict]:
        """Top-k documents sharing concepts with `text`, best first. `exclude` is a source_file to skip."""
        found = self.concepts(text)
        q = {c: (1 + math.log(n)) * self.idf[c] for c, n in found.items()}
        norm = math.sqrt(sum(x * x for x in q.values())) or 1.0
        q = {c: x / norm for c, x in q.items()}
        out = []
        for f, d in self.docs.items():
            if f == exclude:
                continue
            shared = sorted(((min(q[c], m[0]), c, m[1]) for c, m in self.mentions[d].items() if c in q), reverse=True)
            score = sum(s for s, _, _ in shared)
            if score >= min_score:
                out.append({"id": d, "label": self.nodes[d]["label"], "source_file": f, "score": round(score, 3),
                            "reasons": [{"concept": self.nodes[c]["label"], "where": f"{f} {loc}"} for _, c, loc in shared[:3]]})
        return sorted(out, key=lambda r: -r["score"])[:k]
