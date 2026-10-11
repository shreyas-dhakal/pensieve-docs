"""Default concept extractor: spaCy entities + noun chunks, lemmatised.

Extractor interface (any dotted path in config `concepts.extractors`):
    extract(chunks: list[dict], cfg: dict) -> {
        "terms":  [ {norm: count}, ... ]   # one dict per chunk, same order as chunks
        "labels": {norm: {surface: count}}  # display forms
        "bow":    {lemma: count}            # bag of words for TF-IDF similarity
        "rels":   [[a, b, verb|None, negated, hedged, chunk], ...]  # optional: concept pairs close in a parse
    }
Results are cached per file, so extractors must be deterministic for a given file content.
"""
import re
from collections import Counter, defaultdict
from itertools import combinations

from .parse import REFLINE, is_ref_heading

VERSION = 7  # bump when extraction output changes, to invalidate the per-file cache
SKIP_ENTS = {"CARDINAL", "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL"}
_EDGE_POS = {"DET", "PRON", "ADP", "CCONJ", "PUNCT", "NUM", "PART", "SYM", "SPACE", "ADV"}
_CLEAN = [
    (re.compile(r"!?\[([^\]]*)\]\([^)]*\)"), r"\1"),  # [text](url) -> text
    (re.compile(r"\[\[[^\]]*\]\]"), " "),  # [[wikilinks]] are links, not prose
    (re.compile(r"https?://\S+|\b10\.\d{4,9}/\S+"), " "),
    (re.compile(r"[#*_`>|]+"), " "),
]
_nlp = {}


def nlp(name: str):
    if name not in _nlp:
        import spacy

        _nlp[name] = spacy.load(name)
    return _nlp[name]


def _norm(span, max_tokens: int, stop: set) -> tuple[str, str] | None:
    toks = list(span)
    breaks = [i for i, t in enumerate(toks) if "\n" in t.text]
    if breaks:  # spans glued across lines ("Abstract\nRooftop solar"): keep the part with the head noun
        toks = toks[breaks[-1] + 1:]
    while toks and (toks[0].is_stop or toks[0].pos_ in _EDGE_POS or not toks[0].text.strip("-'")):
        toks.pop(0)
    while toks and (toks[-1].is_stop or toks[-1].pos_ in _EDGE_POS or not toks[-1].text.strip("-'")):
        toks.pop()
    if not toks or len(toks) > max_tokens or not any(t.is_alpha for t in toks) or any(t.like_url or t.like_email for t in toks):
        return None
    ws = [t.whitespace_ and " " for t in toks[:-1]] + [""]  # keep "lithium-ion" hyphenation
    norm = "".join((t.lemma_ if t.pos_ != "PROPN" else t.text).lower() + w for t, w in zip(toks, ws))
    if len(norm) < 3 or norm in stop or any(ch in norm for ch in "\\{}%=^_|<>"):  # math / markup, not prose
        return None
    return norm, "".join(t.text + w for t, w in zip(toks, ws))


_NEG = {"no", "not", "never", "without", "n't"}


def _path(x, y) -> list | None:
    """Tokens on the dependency path from x up to their lowest common ancestor and down to y."""
    up = [x, *x.ancestors]
    seen = {t.i: k for k, t in enumerate(up)}
    down = [y]
    while down[-1].i not in seen:
        if down[-1].head.i == down[-1].i:
            return None  # different sentences' roots
        down.append(down[-1].head)
    return up[: seen[down[-1].i] + 1] + down[-2::-1]


def _rels(sent_spans, max_path: int, hedges: set, chunk: int) -> list:
    """[a, b, verb, negated, hedged, chunk] for concept spans of one sentence joined by a short dependency path."""
    out = []
    for (sa, a), (sb, b) in combinations(sent_spans, 2):
        if a == b:
            continue
        if b < a:
            (sa, a), (sb, b) = (sb, b), (sa, a)
        path = _path(sa.root, sb.root)
        if path is None or len(path) - 1 > max_path:
            continue
        verb = next((t.lemma_.lower() for t in path if t.pos_ == "VERB"), None)
        ctx = path + [c for t in path for c in t.children]  # "do *not* reduce", "*no* heat pump", "*may* reduce"
        neg = any(t.dep_ == "neg" or t.lower_ in _NEG or t.lemma_.lower() in _NEG for t in ctx)
        hedged = any(t.lemma_.lower() in hedges or t.lower_ in hedges for t in ctx)
        out.append([a, b, verb, neg, hedged, chunk])
    return out


def clean(ch: dict) -> str:
    """Chunk prose as the extractor sees it: heading + text, markup and links stripped.
    Reference lists are left out: they already become `cites` edges, and their titles are not this doc's ideas."""
    if is_ref_heading(ch["heading"]):
        return ""
    body = "\n".join(line for line in ch["text"].splitlines() if not REFLINE.match(line))
    t = (ch["heading"] + "\n" if ch["heading"] and not ch.get("cont") else "") + body
    for rx, rep in _CLEAN:
        t = rx.sub(rep, t)
    return t


def extract(chunks: list[dict], cfg: dict) -> dict:
    c = cfg["concepts"]
    texts = [clean(ch) for ch in chunks]
    # end of the heading line, when clean() put one at the top of the text
    heads = [t.find("\n") if t and ch["heading"] and not ch.get("cont") else 0 for ch, t in zip(chunks, texts)]
    from spacy.util import filter_spans

    stop = {w.lower() for w in c["stop_concepts"]}
    hedges = {w.lower() for w in c["hedge_words"]}
    terms, labels, bow, rels = [], {}, Counter(), []
    for ci, (doc, head) in enumerate(zip(nlp(c["spacy_model"]).pipe(texts, batch_size=32), heads)):
        counts, by_sent = Counter(), defaultdict(list)
        # longest non-overlapping spans, so an entity inside a noun chunk is not double counted
        nc = list(doc.noun_chunks)
        # "coefficient of performance": spaCy splits at "of"; join when the whole still fits max_tokens
        of = [doc[a.start:b.end] for a, b in zip(nc, nc[1:])
              if b.start == a.end + 1 and doc[a.end].lower_ == "of" and _norm(doc[a.start:b.end], c["max_tokens"], stop)]
        spans = filter_spans([e for e in doc.ents if e.label_ not in SKIP_ENTS] + nc + of)
        for span in spans:
            r = _norm(span, c["max_tokens"], stop)
            if r and " " not in r[0] and span.end_char <= head:
                continue  # a one-word heading ("Economics") names a section; it is a concept only if the body uses it
            if r:
                counts[r[0]] += 1
                labels.setdefault(r[0], Counter())[r[1]] += 1
                by_sent[span.sent.start].append((span, r[0]))
        terms.append(dict(counts))
        for k in sorted(by_sent):
            rels += _rels(by_sent[k], c["max_path"], hedges, ci)
        bow.update(t.lemma_.lower() for t in doc if t.is_alpha and not t.is_stop and len(t) > 2)
    return {"terms": terms, "labels": {k: dict(v) for k, v in labels.items()}, "bow": dict(bow), "rels": rels}
