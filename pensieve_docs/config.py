"""Defaults + optional TOML override. Every tunable lives here."""
import copy
import tomllib
from pathlib import Path

DEFAULTS = {
    "out_dir": "pensieve-out",
    "exclude": [".git", "node_modules", "__pycache__", ".venv", "pensieve-out", "graphify-out"],
    "chunk": {"max_chars": 4000},
    "concepts": {
        "spacy_model": "en_core_web_sm",
        "extractors": ["pensieve_docs.concepts:extract"],  # dotted paths; see README "Extractor interface"
        "min_docs": 2,          # concepts must appear in this many docs...
        "keep_top_single": 3,   # ...unless they are among the top N (by TF-IDF) of their doc
        "max_per_doc": 25,      # cap MENTIONS edges per doc
        "merge_ratio": 0.9,     # difflib ratio to fold near-duplicate concepts
        "max_tokens": 4,        # longest noun phrase kept
        "generic_zipf": 4.6,    # drop concepts whose every word is this common in English (wordfreq Zipf); 0 = off
        "stop_concepts": [      # boilerplate headings / filler that should never become concepts (lemmatised, lowercase)
            "abstract", "introduction", "summary", "overview", "background", "method", "result", "discussion",
            "conclusion", "appendix", "reference", "bibliography", "attendee", "decision", "action",
            "open question", "note", "example", "thing", "way", "lot", "part", "kind", "number",
        ],
    },
    "similarity": {
        "method": "tfidf",      # "tfidf" | "minilm" (needs the [embeddings] extra + a locally cached model)
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "top_k": 5,
        "threshold": 0.25,      # >= threshold -> INFERRED SIMILAR_TO
        "ambiguous_min": 0.08,  # [ambiguous_min, threshold) -> AMBIGUOUS
    },
    "cooccur": {
        "top_k": 5,
        "min_count": 2,         # pairs sharing fewer chunks are dropped as noise
        "npmi_threshold": 0.2,  # weaker positive associations -> AMBIGUOUS
    },
    "hyperedge_min_docs": 3,
    "max_evidence": 10,         # evidence entries kept per edge (and shared chunks listed per co_occurs)
    "report": {"top_n": 15},
    "enrich": [],               # future post-build hooks, dotted paths: fn(graph, cfg). Empty by default.
}


def _merge(base: dict, over: dict) -> dict:
    for k, v in over.items():
        base[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return base


def load(path: str | Path | None = None, folder: Path | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path is None and folder is not None and (folder / "pensieve.toml").exists():
        path = folder / "pensieve.toml"
    if path:
        with open(path, "rb") as f:
            _merge(cfg, tomllib.load(f))
    return cfg
