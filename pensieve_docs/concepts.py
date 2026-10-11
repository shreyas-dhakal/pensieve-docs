"""Default concept extractor: spaCy entities + noun chunks, lemmatised.

Extractor interface (any dotted path in config `concepts.extractors`):
    extract(chunks: list[dict], cfg: dict) -> {
        "terms":  [ {norm: count}, ... ]   # one dict per chunk, same order as chunks
        "labels": {norm: {surface: count}}  # display forms
        "bow":    {lemma: count}            # bag of words for TF-IDF similarity
        "rels":   [[a, b, verb|None, negated, hedged, chunk], ...]  # optional: concept pairs close in a parse
        "abbrevs": [[short norm, long norm], ...]  # optional: "large language model (LLM)"
        "isa":    [[specific, general, chunk, sentence start, end], ...]  # optional: "X such as Y", "Y is a X"
    }
Results are cached per file, so extractors must be deterministic for a given file content.
"""
import re
from collections import Counter, defaultdict
from itertools import combinations

from .parse import REFLINE, is_ref_heading

VERSION = 9  # bump when extraction output changes, to invalidate the per-file cache
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


def _trim(span) -> list:
    """The span's tokens without leading/trailing stopwords and punctuation ("the other mosquitoes" -> mosquitoes)."""
    toks = list(span)
    breaks = [i for i, t in enumerate(toks) if "\n" in t.text]
    if breaks:  # spans glued across lines ("Abstract\nRooftop solar"): keep the part with the head noun
        toks = toks[breaks[-1] + 1:]
    while toks and (toks[0].is_stop or toks[0].pos_ in _EDGE_POS or not toks[0].text.strip("-'")):
        toks.pop(0)
    while toks and (toks[-1].is_stop or toks[-1].pos_ in _EDGE_POS or not toks[-1].text.strip("-'")):
        toks.pop()
    return toks


def _norm(span, max_tokens: int, stop: set) -> tuple[str, str] | None:
    toks = _trim(span)
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


# Hearst (1992) patterns, as the words between two concepts of one sentence
_SUCH = {("such", "as"), (",", "such", "as"), ("including",), (",", "including")}  # X such as Y
_OTHER = {("and", "other"), ("or", "other")}  # Y and other X
_ISA = {("is", "a"), ("is", "an")}  # Y is a X
_LIST = {(",",), ("and",), ("or",), (",", "and"), (",", "or")}  # Y, Z and W


def _hearst(doc, items: list, chunk: int) -> list:
    """[specific, general, chunk, sentence start, sentence end] from one sentence's concepts [(span, norm)]."""
    cores = [(t[0].i, t[-1].i + 1, n) for sp, n in items if (t := _trim(sp))]
    gap = lambda k: tuple(t.lower_ for t in doc[cores[k][1]:cores[k + 1][0]] if not t.is_space)
    pairs = []
    for k in range(len(cores) - 1):
        if gap(k) in _SUCH:  # X such as Y, Z and W
            j = k + 1
            pairs.append((j, k))
            while j + 1 < len(cores) and gap(j) in _LIST:
                j += 1
                pairs.append((j, k))
        elif gap(k) in _OTHER:  # Y, Z and other X
            j = k
            pairs.append((j, k + 1))
            while j > 0 and gap(j - 1) in _LIST:
                j -= 1
                pairs.append((j, k + 1))
        elif gap(k) in _ISA:  # Y is a X
            pairs.append((k, k + 1))
    sent = items[0][0].sent if items else None
    return [[cores[y][2], cores[x][2], chunk, sent.start_char, sent.end_char]
            for y, x in pairs if cores[y][2] != cores[x][2]]


_PAREN = re.compile(r"\(([^()]{2,10})\)")


def _long_start(short: str, cand: str) -> int | None:
    """Schwartz & Hearst (2003): where the shortest tail of `cand` starts that holds short's letters in order,
    with its first letter at the start of a word. None if there is no such tail."""
    s, i = len(short) - 1, len(cand) - 1
    while s >= 0:
        ch = short[s].lower()
        if not ch.isalnum():
            s -= 1
            continue
        while i >= 0 and (cand[i].lower() != ch or (s == 0 and i > 0 and cand[i - 1].isalnum())):
            i -= 1
        if i < 0:
            return None
        i, s = i - 1, s - 1
    return cand.rfind(" ", 0, i + 1) + 1


def _abbrevs(doc, max_tokens: int, stop: set) -> list:
    """[(short norm, short text, long norm, long text)] for each "long form (SHORT)" in the doc."""
    out = []
    for m in _PAREN.finditer(doc.text):
        short = m.group(1).strip()
        if not (short[0].isalnum() and any(ch.isupper() for ch in short) and len(short.split()) <= 2):
            continue
        before = doc.text[:m.start()].rstrip()
        words = list(re.finditer(r"\S+", before))[-min(len(short) + 5, 2 * len(short)):]
        if not words or (k := _long_start(short, cand := before[words[0].start():])) is None:
            continue
        a0 = m.start(1) + m.group(1).find(short)
        sn = _norm(doc.char_span(a0, a0 + len(short), alignment_mode="expand"), max_tokens, stop)
        ln = _norm(doc.char_span(words[0].start() + k, len(before), alignment_mode="expand"), max_tokens, stop)
        if sn and ln and len(ln[0]) > len(sn[0]) and sn[0] not in ln[0].split():
            out.append((*sn, *ln))
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
    terms, labels, bow, rels, abbrevs, isa = [], {}, Counter(), [], [], []
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
        for sn, st, ln, lt in _abbrevs(doc, c["max_tokens"], stop):
            abbrevs.append([sn, ln])
            for n, t in ((sn, st), (ln, lt)):  # known forms, so backfill finds both wherever they appear
                labels.setdefault(n, Counter())[t] += 0
        for k in sorted(by_sent):
            rels += _rels(by_sent[k], c["max_path"], hedges, ci)
            isa += _hearst(doc, sorted(by_sent[k], key=lambda x: x[0].start), ci)
        bow.update(t.lemma_.lower() for t in doc if t.is_alpha and not t.is_stop and len(t) > 2)
    return {"terms": terms, "labels": {k: dict(v) for k, v in labels.items()}, "bow": dict(bow), "rels": rels, "abbrevs": abbrevs, "isa": isa}
