"""Pipeline: detect -> parse + extract (cached per file by content hash) -> graph -> analyze -> write."""
import hashlib
import importlib
import json
import logging
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from . import analyze, concepts, edges, output, parse
from .detect import detect

log = logging.getLogger("pensieve-docs")


def _load(path: str):
    mod, fn = path.split(":")
    return getattr(importlib.import_module(mod), fn)


def _extract(chunks, cfg) -> dict:
    """Run every configured extractor and sum their results."""
    out = {"terms": [Counter() for _ in chunks], "labels": {}, "bow": Counter(), "rels": [], "abbrevs": []}
    for path in cfg["concepts"]["extractors"]:
        r = _load(path)(chunks, cfg)
        for acc, terms in zip(out["terms"], r["terms"]):
            acc.update(terms)
        for k, forms in r["labels"].items():
            out["labels"].setdefault(k, Counter()).update(forms)
        out["bow"].update(r["bow"])
        out["rels"] += r.get("rels", [])
        out["abbrevs"] += r.get("abbrevs", [])
    return {"terms": [dict(t) for t in out["terms"]], "labels": {k: dict(v) for k, v in out["labels"].items()},
            "bow": dict(out["bow"]), "rels": out["rels"], "abbrevs": out["abbrevs"]}


def _cache_tag(cfg) -> str:
    c = cfg["concepts"]
    return json.dumps([parse.VERSION, concepts.VERSION, cfg["chunk"]["max_chars"], c["extractors"], c["spacy_model"], c["max_tokens"], c["stop_concepts"], c["max_path"], c["hedge_words"]])


def _git_head(root: Path):
    try:
        r = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        return r.stdout.strip() or None
    except Exception:
        return None


def build(folder: str | Path, cfg: dict, out: str | Path | None = None) -> dict:
    t0 = time.perf_counter()
    root = Path(folder).resolve()
    out = Path(out or cfg["out_dir"]).resolve()
    cache_dir = out / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    files, skipped = detect(root, cfg["exclude"])
    files = [f for f in files if out not in f.parents]
    skipped = [(str(f.relative_to(root)), why) for f, why in skipped if out not in f.parents]

    tag, used, records, parsed, cached = _cache_tag(cfg), set(), [], 0, 0
    for f in files:
        rel = f.relative_to(root).as_posix()
        data = f.read_bytes()
        key = hashlib.sha256(tag.encode() + data).hexdigest()
        entry_path = cache_dir / f"{key}.json"
        used.add(entry_path.name)
        if entry_path.exists():
            entry = json.loads(entry_path.read_text(encoding="utf-8"))
            cached += 1
        else:
            try:
                p = parse.parse_file(f, cfg["chunk"]["max_chars"])
                entry = {"parsed": p, "concepts": _extract(p["chunks"], cfg)}
            except parse.SkipFile as e:
                entry = {"skip": str(e)}
            except Exception as e:  # corrupt file: log, skip, keep building
                entry = {"skip": f"parse error: {type(e).__name__}: {e}"}
            output.write_atomic(entry_path, json.dumps(entry, ensure_ascii=False))
            parsed += 1
        if "skip" in entry:
            log.warning("skip %s: %s", rel, entry["skip"])
            skipped.append((rel, entry["skip"]))
            continue
        st = f.stat()
        records.append({"rel": rel, "hash": hashlib.sha256(data).hexdigest(), "size": st.st_size,
                        "mtime": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(timespec="seconds"),
                        **entry})
    for stale in cache_dir.glob("*.json"):  # drop entries for deleted/changed files
        if stale.name not in used:
            stale.unlink()

    if not records:
        raise SystemExit(f"No parseable documents in {root}; nothing written.")

    g = edges.build(records, root.name, cfg)
    for hook in cfg.get("enrich", []):  # future: e.g. LLM typed relations for top nodes
        _load(hook)(g, cfg)
    a = analyze.analyze(g, cfg)

    redacted = Counter()
    for r in records:
        redacted.update(r["parsed"].get("redactions", {}))
    stats = {"root": root.name, "detected": len(files), "parsed": parsed, "cached": cached, "redacted": dict(redacted),
             "redacted_files": sum(bool(r["parsed"].get("redactions")) for r in records),
             "skipped": sorted(skipped), "seconds": time.perf_counter() - t0}
    meta = {"root": str(root), "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "built_at_commit": _git_head(root), "generator": "pensieve-docs 0.1.0",
            "similarity_method": cfg["similarity"]["method"]}
    output.write_atomic(out / "graph.json", json.dumps(output.graph_json(g, a, meta), indent=2, ensure_ascii=False))
    stats["seconds"] = time.perf_counter() - t0
    output.write_atomic(out / "GRAPH_REPORT.md", output.report(g, a, stats, cfg))
    return {**stats, "nodes": len(g.nodes), "edges": len(g.edges), "hyperedges": len(g.hyperedges),
            "communities": len(a["members"]), "out": str(out)}
