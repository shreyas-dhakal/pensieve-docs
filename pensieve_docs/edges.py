"""Corpus-level graph construction: nodes, typed/tiered edges, hyperedges.

Input records (one per parsed file): {"rel", "hash", "mtime", "size", "parsed", "concepts"}.
One edge per node pair; on a clash the stronger tier wins (EXTRACTED > INFERRED > AMBIGUOUS).
"""
import difflib
import math
import os
import posixpath
import re
import string
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import PurePosixPath
from urllib.parse import unquote

from wordfreq import zipf_frequency

from .concepts import clean
from .parse import location

RANK = {"EXTRACTED": 0, "INFERRED": 1, "AMBIGUOUS": 2}
GENERIC_TAIL = 4.8  # wordfreq Zipf; "work", "rule", "support" - not "capacity" or "efficiency"
RUBRIC = (0.55, 0.65, 0.75, 0.85, 0.95)  # graphify's discrete INFERRED confidence scale


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_") or "x"


def bucket(x: float) -> float:
    """Map a 0..1 strength onto the discrete INFERRED rubric."""
    return RUBRIC[min(4, max(0, int(x * 5)))]


def ambiguous_score(x: float) -> float:
    """Map a 0..1 position inside the ambiguous band onto 0.1..0.3."""
    return round(0.1 + 0.2 * min(1.0, max(0.0, x)), 2)


class Graph:
    def __init__(self):
        self.nodes: dict[str, dict] = {}
        self.edges: dict[frozenset, dict] = {}
        self.hyperedges: list[dict] = []

    def node(self, nid: str, **attrs):
        self.nodes.setdefault(nid, {"id": nid}).update(attrs)

    def edge(self, src, tgt, relation, tier, score, weight, source_file, chunk=None, source_location=None, **extra):
        if src == tgt:
            return
        key = frozenset((src, tgt))
        old = self.edges.get(key)
        if old and RANK[old["confidence"]] <= RANK[tier]:
            return
        self.edges[key] = {
            "source": src, "target": tgt, "relation": relation,
            "confidence": tier, "confidence_score": round(score, 2), "weight": round(weight, 4),
            "source_file": source_file, "source_location": source_location, "chunk": chunk, **extra,
        }


def build(records: list[dict], root_name: str, cfg: dict) -> Graph:
    g = Graph()
    doc_ids = _doc_ids([r["rel"] for r in records])
    _structure(g, records, doc_ids, root_name)
    _links(g, records, doc_ids)
    _concepts(g, records, doc_ids, cfg)
    _similarity(g, records, doc_ids, cfg)
    return g


def _doc_ids(rels: list[str]) -> dict[str, str]:
    ids = {r: slug(str(PurePosixPath(r).with_suffix(""))) for r in rels}
    dupes = {v for v, k in Counter(ids.values()).items() if k > 1}
    ids = {r: slug(r) if v in dupes else v for r, v in ids.items()}  # a.md + a.txt -> keep extension
    # concept-notes.md must not take concept "notes"'s id
    return {r: "doc_" + v if v.startswith(("concept_", "folder_")) else v for r, v in ids.items()}


# ---------- EXTRACTED: folders, documents, sections ----------

def _folder_id(rel_dir: str) -> str:
    return "folder_root" if rel_dir in ("", ".") else "folder_" + slug(rel_dir)


def _structure(g: Graph, records, doc_ids, root_name):
    g.node("folder_root", label=root_name, file_type="folder", kind="folder", source_file=".", source_location=None)
    for r in records:
        rel, p = r["rel"], r["parsed"]
        did = doc_ids[rel]
        parts = PurePosixPath(rel).parts[:-1]
        parent = "folder_root"
        for i in range(len(parts)):
            d = "/".join(parts[: i + 1])
            fid = _folder_id(d)
            g.node(fid, label=parts[i], file_type="folder", kind="folder", source_file=d, source_location=None)
            g.edge(parent, fid, "contains", "EXTRACTED", 1.0, 1.0, d)
            parent = fid
        g.node(did, label=p["title"], file_type="paper" if p["doc_type"] == "pdf" else "document", kind="document",
               source_file=rel, source_location=None, doc_type=p["doc_type"], hash=r["hash"], mtime=r["mtime"],
               size=r["size"], n_chunks=len(p["chunks"]), doi=p["doi"], redactions=p.get("redactions") or {})
        g.edge(parent, did, "contains", "EXTRACTED", 1.0, 1.0, rel)
        seen = Counter()
        for i, c in enumerate(p["chunks"]):
            if not c["heading"] or c.get("cont"):
                continue
            sid = f"{did}#{slug(c['heading'])}"  # "#": slugs never contain it, so no clash with a doc id
            seen[sid] += 1
            if seen[sid] > 1:
                sid += f"_{seen[sid]}"
            loc = location(c, p["doc_type"])
            g.node(sid, label=c["heading"], file_type="document", kind="section", level=c["level"],
                   source_file=rel, source_location=loc, chunk=i)
            g.edge(did, sid, "has_section", "EXTRACTED", 1.0, 1.0, rel, i, loc)


# ---------- EXTRACTED: links and citations between documents ----------

def _links(g: Graph, records, doc_ids):
    by_path = {r["rel"].lower(): doc_ids[r["rel"]] for r in records}
    by_stem = {}
    for r in records:
        by_stem.setdefault(PurePosixPath(r["rel"]).stem.lower(), doc_ids[r["rel"]])
        by_stem.setdefault(r["parsed"]["title"].lower(), doc_ids[r["rel"]])
    by_doi = {r["parsed"]["doi"]: doc_ids[r["rel"]] for r in records if r["parsed"]["doi"]}
    titles = [(r["parsed"]["title"].lower(), doc_ids[r["rel"]]) for r in records if len(r["parsed"]["title"]) >= 12]

    for r in records:
        rel, p, did = r["rel"], r["parsed"], doc_ids[r["rel"]]
        external = 0
        for ref in p["refs"]:
            kind, t = ref["kind"], ref["target"]
            loc = location(p["chunks"][ref["chunk"]], p["doc_type"])
            target, relation = None, "links_to"
            if kind == "link":
                path = unquote(t.split("#")[0])
                if path:
                    full = posixpath.normpath(posixpath.join(posixpath.dirname(rel), path)).lower()
                    target = by_path.get(full) or by_path.get(full + ".md")
            elif kind == "wikilink":
                target = by_stem.get(t.lower()) or by_path.get(t.lower()) or by_path.get(t.lower() + ".md")
            elif kind == "doi":
                target, relation = by_doi.get(t), "cites"
            elif kind == "reference":
                low = t.lower()
                hits = [(len(title), d) for title, d in titles if title in low]
                target = max(hits)[1] if hits else None  # longest title wins ("Battery Storage Economics" over "Battery Storage")
                relation = "cites"
            elif kind == "url":
                external += 1
            if target and target != did:
                g.edge(did, target, relation, "EXTRACTED", 1.0, 1.0, rel, ref["chunk"], loc, evidence=t[:200])
        g.nodes[did]["external_links"] = external


# ---------- INFERRED / AMBIGUOUS: concepts ----------

def _term_counts(records) -> tuple[Counter, Counter]:
    """Raw (pre-merge) document frequency and total count per concept norm."""
    df, tot = Counter(), Counter()
    for r in records:
        terms = Counter()
        for chunk in r["concepts"]["terms"]:
            terms.update(chunk)
        df.update(terms.keys())
        tot.update(terms)
    return df, tot


def _merge_map(df: Counter, tot: Counter, ratio: float) -> dict[str, str]:
    """Fold near-duplicate norms (e.g. 'lithium-ion battery' / 'lithium ion battery') onto the most frequent."""
    # ponytail: bucketed by first 2 alnum chars, so O(n * bucket); fine to ~50k terms
    canon, buckets = {}, defaultdict(list)
    for t in sorted(df, key=lambda t: (-df[t], -tot[t], t)):
        key = re.sub(r"[^a-z0-9]", "", t)[:2]
        hit = difflib.get_close_matches(t, buckets[key], 1, ratio)
        canon[t] = hit[0] if hit else t
        if not hit:
            buckets[key].append(t)

    known = set(canon.values())

    def fold(t):
        """'shortlisted heat pump' -> 'heat pump' (known multi-word head, more widespread);
        'net metering rule' -> 'net metering' (known term + a very common trailing word)."""
        toks = t.split()
        for k in range(1, len(toks) - 1):
            head = " ".join(toks[k:])
            if head in known and df[head] > df[t]:
                return head
        stem = " ".join(toks[:-1])
        if stem in known and zipf_frequency(toks[-1], "en") >= GENERIC_TAIL:
            return stem
        return t

    for t in canon:
        while (f := fold(canon[t])) != canon[t]:
            canon[t] = f
    return canon


def _generic(t: str, zipf: float) -> bool:
    """Every word is everyday English ('day', 'research note'): too vague to be a concept."""
    return zipf > 0 and min(zipf_frequency(w, "en") for w in t.split()) >= zipf


_WORD = re.compile(r"[^\s-]+")  # words split on whitespace and hyphens: "lithium-ion" == "lithium ion"
_PUNCT = string.punctuation + "\u201c\u201d\u2018\u2019\u2026"


def _words(text: str) -> list[tuple[str, bool]]:
    """Lowercased words, each with whether a phrase must break before it (punctuation in between)."""
    out, brk = [], True
    for m in _WORD.finditer(text.lower()):
        w = m.group()
        core = w.strip(_PUNCT)
        if not core:
            brk = True
            continue
        out.append((core, brk or not w.startswith(core)))
        brk = not w.endswith(core)
    return out


def _backfill(records) -> list[dict]:
    """Count every known term wherever its text appears, not only where spaCy happened to chunk it.
    Longest match wins, so "net metering export rate" does not also count as "net metering"."""
    forms = {}
    for r in records:
        for t, surf in r["concepts"]["labels"].items():
            for s in (t, *surf):
                if k := tuple(w for w, _ in _words(s)):
                    forms.setdefault(k, t)
    if not forms:
        return records
    maxn = max(map(len, forms))
    out = []
    for r in records:
        terms = []
        for ch, old in zip(r["parsed"]["chunks"], r["concepts"]["terms"]):
            ws, found, i = _words(clean(ch)), Counter(), 0
            while i < len(ws):
                for n in range(min(maxn, len(ws) - i), 0, -1):
                    k = tuple(w for w, _ in ws[i:i + n])
                    if k in forms and not any(b for _, b in ws[i + 1:i + n]):
                        found[forms[k]] += 1
                        i += n
                        break
                else:
                    i += 1
            terms.append({t: max(old.get(t, 0), found[t]) for t in old.keys() | found.keys()})
        out.append({**r, "concepts": {**r["concepts"], "terms": terms}})
    return out


def _concepts(g: Graph, records, doc_ids, cfg):
    c = cfg["concepts"]
    records = _backfill(records)
    df0, tot0 = _term_counts(records)
    canon = _merge_map(df0, tot0, c["merge_ratio"])

    labels = defaultdict(Counter)
    doc_terms, chunk_terms = {}, []  # chunk_terms: (rel, chunk index, set of terms)
    for r in records:
        dt = Counter()
        for i, chunk in enumerate(r["concepts"]["terms"]):
            ct = Counter()
            for t, n in chunk.items():
                ct[canon[t]] += n
            dt.update(ct)
            chunk_terms.append((r["rel"], i, ct))
        for t, forms in r["concepts"]["labels"].items():
            labels[canon[t]].update(forms)
        doc_terms[r["rel"]] = dt

    n = len(records)
    df = Counter(t for dt in doc_terms.values() for t in dt)
    idf = {t: math.log((1 + n) / (1 + d)) + 1 for t, d in df.items()}
    tfidf = {}
    for rel, dt in doc_terms.items():
        v = {t: (1 + math.log(k)) * idf[t] for t, k in dt.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        tfidf[rel] = sorted(((t, x / norm) for t, x in v.items()), key=lambda kv: (-kv[1], kv[0]))

    keep = {t for t, d in df.items() if d >= c["min_docs"]}
    for ranked in tfidf.values():
        keep.update(t for t, _ in ranked[: c["keep_top_single"]])
    keep = {t for t in keep if not _generic(t, c["generic_zipf"])}
    # a term found only in the doc it titles is that doc, already a node ("Battery Storage Economics")
    only_in = {t: rel for rel, dt in doc_terms.items() for t in dt if df[t] == 1}
    keep = {t for t in keep if t not in only_in or t != g.nodes[doc_ids[only_in[t]]]["label"].lower()}

    docs_of = defaultdict(list)
    best_chunk = {}
    for rel, i, ct in chunk_terms:
        for t, k in ct.items():
            if k > best_chunk.get((rel, t), (None, 0))[1]:
                best_chunk[(rel, t)] = (i, k)
    recs = {r["rel"]: r for r in records}
    for rel, ranked in tfidf.items():
        mine = [(t, w) for t, w in ranked if t in keep][: c["max_per_doc"]]
        if not mine:
            continue
        top = mine[0][1]
        p = recs[rel]["parsed"]
        for t, w in mine:
            cid = "concept_" + slug(t)
            i = best_chunk[(rel, t)][0]
            loc = location(p["chunks"][i], p["doc_type"])
            g.edge(doc_ids[rel], cid, "mentions", "INFERRED", bucket(w / top), w, rel, i, loc)
            docs_of[t].append((rel, w, i, loc))

    for t, docs in docs_of.items():
        cid = "concept_" + slug(t)
        rel, w, i, loc = max(docs, key=lambda d: d[1])
        g.node(cid, label=labels[t].most_common(1)[0][0], file_type="concept", kind="concept", norm=t,
               doc_count=df[t], score=round(sum(d[1] for d in docs), 4), source_file=rel, source_location=loc)
        if df[t] >= cfg["hyperedge_min_docs"]:
            members = sorted({doc_ids[rel] for rel, dt in doc_terms.items() if t in dt})
            files = sorted(rel for rel, dt in doc_terms.items() if t in dt)
            g.hyperedges.append({
                "id": cid + "_docs", "label": f"{g.nodes[cid]['label']} ({len(members)} docs)",
                "nodes": [cid] + members, "relation": "participate_in", "confidence": "INFERRED",
                "confidence_score": bucket(sum(d[1] for d in docs) / len(docs) / max(d[1] for d in docs)),
                "source_file": files[0], "source_files": files,
            })

    _nested(g, set(docs_of))
    _cooccur(g, chunk_terms, set(docs_of), recs, cfg["cooccur"])


def _nested(g: Graph, concepts: set):
    """'net metering export rate' -> 'net metering': a concept whose words contain another one.
    Longest-match counting hides the shorter term inside the longer one; this edge keeps them connected."""
    for a in concepts:
        ta = f" {a} "
        for b in concepts:
            if b != a and f" {b} " in ta:
                n = g.nodes["concept_" + slug(a)]
                g.edge(n["id"], "concept_" + slug(b), "conceptually_related_to", "INFERRED", 0.85, 1.0,
                       n["source_file"], None, n["source_location"])


def _cooccur(g: Graph, chunk_terms, concepts: set, recs, cc):
    """CO_OCCURS between concepts that share chunks. weight = PMI, tier by NPMI."""
    total = len(chunk_terms)
    single, pair, first = Counter(), Counter(), {}
    for rel, i, ct in chunk_terms:
        present = sorted(t for t in ct if t in concepts)
        single.update(present)
        for a, b in combinations(present, 2):
            pair[(a, b)] += 1
            first.setdefault((a, b), (rel, i))
    stats = {}
    for (a, b), k in pair.items():
        if k < cc["min_count"]:
            continue  # one shared chunk is noise, not even AMBIGUOUS
        pab = k / total
        pmi = math.log(pab / ((single[a] / total) * (single[b] / total)))
        npmi = pmi / -math.log(pab) if pab < 1 else 1.0
        if pmi > 0:
            stats[(a, b)] = (pmi, npmi, k)
    partners = defaultdict(list)
    for (a, b), (pmi, npmi, k) in stats.items():
        partners[a].append((npmi, k, b))
        partners[b].append((npmi, k, a))
    chosen = set()
    for a, lst in partners.items():
        for npmi, k, b in sorted(lst, reverse=True)[: cc["top_k"]]:
            chosen.add(tuple(sorted((a, b))))
    thr = cc["npmi_threshold"]
    for a, b in sorted(chosen):
        pmi, npmi, k = stats[(a, b)]
        rel, i = first[(a, b)]
        p = recs[rel]["parsed"]
        tier = "INFERRED" if npmi >= thr else "AMBIGUOUS"
        score = bucket((npmi - thr) / (1 - thr)) if tier == "INFERRED" else ambiguous_score(npmi / thr)
        g.edge("concept_" + slug(a), "concept_" + slug(b), "co_occurs", tier, score, pmi, rel, i,
               location(p["chunks"][i], p["doc_type"]), npmi=round(npmi, 4), count=k)


# ---------- INFERRED / AMBIGUOUS: document similarity ----------

def _tfidf_vectors(records) -> list[dict]:
    n = len(records)
    df = Counter(t for r in records for t in r["concepts"]["bow"])
    out = []
    for r in records:
        v = {t: (1 + math.log(k)) * (math.log((1 + n) / (1 + df[t])) + 1) for t, k in r["concepts"]["bow"].items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        out.append({t: x / norm for t, x in v.items()})
    return out


def _minilm_vectors(records, model: str) -> list[dict]:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # never download at build time
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from sentence_transformers import SentenceTransformer

    texts = [" ".join(c["text"] for c in r["parsed"]["chunks"])[:5000] for r in records]
    embs = SentenceTransformer(model).encode(texts, normalize_embeddings=True)
    return [dict(enumerate(map(float, e))) for e in embs]


def _similarity(g: Graph, records, doc_ids, cfg):
    s = cfg["similarity"]
    vecs = _minilm_vectors(records, s["model"]) if s["method"] == "minilm" else _tfidf_vectors(records)
    thr, amin = s["threshold"], s["ambiguous_min"]
    # ponytail: all-pairs O(n^2) cosine; swap for an ANN index past ~10k docs
    sims = defaultdict(list)
    for i, j in combinations(range(len(records)), 2):
        a, b = vecs[i], vecs[j]
        if len(a) > len(b):
            a, b = b, a
        sim = sum(x * b.get(t, 0.0) for t, x in a.items())
        sims[i].append((sim, j))
        sims[j].append((sim, i))
    for i, lst in sims.items():
        for sim, j in sorted(lst, reverse=True)[: s["top_k"]]:
            if sim >= thr:
                tier, score = "INFERRED", bucket((sim - thr) / (1 - thr))
            elif sim >= amin:
                tier, score = "AMBIGUOUS", ambiguous_score((sim - amin) / (thr - amin))
            else:
                break
            a, b = sorted((i, j))
            g.edge(doc_ids[records[a]["rel"]], doc_ids[records[b]["rel"]], "similar_to", tier, score, sim,
                   records[a]["rel"], method=s["method"])
