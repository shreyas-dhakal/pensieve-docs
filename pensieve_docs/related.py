"""Which documents relate to a piece of text, and why. Load once, query many times:

    idx = Index("pensieve-out/graph.json")
    idx.related("thinking about a home battery for backup power")
    # [{"id", "label", "source_file", "score", "reasons": [{"concept", "where"}, ...]}, ...]

The score is the one bench/eval.py measures: summed min(query weight, doc weight) over shared concepts,
with both sides TF-IDF weighted and L2-normalised, so the same thresholds apply. Documents need that score to be
listed at all; they are ordered by reciprocal-rank fusion of it with BM25 over lemmas (bow.json, written by build),
the concept side weighted by how many concepts the query has (see FULL_CONCEPTS).
"""
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from . import config
from .concepts import nlp
from .edges import _words, overlap

MIN_SCORE = 0.2  # bench: keeps ~2/3 of top-3 suggestions, all on topic
RRF_K = 60  # reciprocal-rank fusion constant (Cormack et al. 2009)
# the concept ranking gets full say only from queries with this many known concepts. A title has ~2 and ranks
# better on BM25 alone (SCIDOCS nDCG@10 0.326 vs 0.258 at full weight); a whole article has ~80 and gains from it
# (wiki MAP 0.970 vs 0.945). Tuned on those two benches.
FULL_CONCEPTS = 40


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
        bow_path = Path(graph_json).with_name("bow.json")
        self.bow = json.loads(bow_path.read_text(encoding="utf-8")) if bow_path.exists() else {}  # older builds: none
        self.dl = {f: sum(c.values()) for f, c in self.bow.items()}
        self.avgdl = sum(self.dl.values()) / len(self.dl) if self.dl else 0.0
        df = Counter(w for c in self.bow.values() for w in c)
        self.bm25_idf = {w: math.log(1 + (len(self.bow) - k + 0.5) / (k + 0.5)) for w, k in df.items()}

    def _parse(self, text: str):
        model = nlp(self.cfg["concepts"]["spacy_model"])
        with model.select_pipes(disable=[p for p in ("parser", "ner") if p in model.pipe_names]):  # lemmas only
            return model(text)

    def bm25(self, doc, k1: float = 1.2, b: float = 0.75) -> dict:
        """source_file -> Okapi BM25 of the query's lemmas (the extractor's bag-of-words filter) against each doc."""
        q = Counter(t.lemma_.lower() for t in doc if t.is_alpha and not t.is_stop and len(t) > 2)
        out = {}
        for f, c in self.bow.items():
            norm = k1 * (1 - b + b * self.dl[f] / (self.avgdl or 1))
            out[f] = sum(n * self.bm25_idf[w] * c[w] * (k1 + 1) / (c[w] + norm) for w, n in q.items() if w in c)
        return out

    def concepts(self, text: str, doc=None) -> Counter:
        """Known concepts in the text, longest match first, on both surface words and lemmas."""
        doc = doc if doc is not None else self._parse(text)
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
        doc = self._parse(text)
        found = self.concepts(text, doc)
        bm25 = self.bm25(doc)
        q = {c: (1 + math.log(n)) * self.idf[c] for c, n in found.items()}
        norm = math.sqrt(sum(x * x for x in q.values())) or 1.0
        q = {c: x / norm for c, x in q.items()}
        out = []
        for f, d in self.docs.items():
            if f == exclude:
                continue
            m = self.mentions[d]
            shared = overlap(q, {c: w for c, (w, _) in m.items()})
            score = sum(s for s, _ in shared)
            if score >= min_score:
                out.append({"id": d, "label": self.nodes[d]["label"], "source_file": f, "score": round(score, 3),
                            "bm25": round(bm25.get(f, 0.0), 3), "_s": score,
                            "reasons": [{"concept": self.nodes[c]["label"], "where": f"{f} {m[c][1]}"} for _, c in shared[:3]]})
        weight = min(1.0, len(found) / FULL_CONCEPTS)
        return [{k: v for k, v in r.items() if k != "_s"} for r in self._fuse(out, weight)[:k]]

    @staticmethod
    def _fuse(out: list, concept_weight: float) -> list:
        """Order by w/(RRF_K + concept rank) + 1/(RRF_K + BM25 rank); a zero score earns no rank on that list."""
        fused = Counter()
        for key, w in (("_s", concept_weight), ("bm25", 1.0)):
            ranked = sorted((r for r in out if r[key] > 0), key=lambda r: (-r[key], r["source_file"]))
            for i, r in enumerate(ranked):
                fused[r["source_file"]] += w / (RRF_K + i + 1)
        return sorted(out, key=lambda r: (-fused[r["source_file"]], -r["_s"], r["source_file"]))
