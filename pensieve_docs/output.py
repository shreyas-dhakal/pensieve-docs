"""graph.json (graphify-compatible node-link) and GRAPH_REPORT.md."""
import json
import os
import unicodedata
from collections import Counter
from pathlib import Path


def _canonical(item: dict, lead: tuple) -> dict:
    keys = [k for k in lead if k in item] + sorted(k for k in item if k not in lead)
    return {k: item[k] for k in keys}


def _key(item: dict) -> str:
    return json.dumps(item, ensure_ascii=False, sort_keys=True)


def graph_json(g, a: dict, meta: dict) -> dict:
    """Same shape as graphify's graph.json: node-link, `links`, top-level + graph.hyperedges."""
    nodes = []
    for n in g.nodes.values():
        c = a["community"][n["id"]]
        norm = "".join(ch for ch in unicodedata.normalize("NFKD", n["label"]) if not unicodedata.combining(ch)).lower()
        nodes.append(_canonical({**n, "community": c, "community_name": a["names"][c], "norm_label": norm}, ("id", "label")))
    links = [_canonical(e, ("source", "target", "relation")) for e in g.edges.values()]
    hyper = sorted(g.hyperedges, key=lambda h: h["id"])
    return {
        "directed": False,
        "multigraph": False,
        "graph": {"hyperedges": hyper, **meta},
        "nodes": sorted(nodes, key=_key),
        "links": sorted(links, key=_key),
        "hyperedges": hyper,
        "built_at_commit": meta.get("built_at_commit"),
    }


def write_atomic(path: Path, text: str):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _row(*cells) -> str:
    return "| " + " | ".join(str(c).replace("|", "\\|").replace("\n", " ") for c in cells) + " |"


def report(g, a: dict, stats: dict, cfg: dict) -> str:
    top = cfg["report"]["top_n"]
    lab = lambda n: g.nodes[n]["label"]
    kinds = Counter(n["kind"] for n in g.nodes.values())
    rel_tier = Counter((e["relation"], e["confidence"]) for e in g.edges.values())
    tiers = Counter(e["confidence"] for e in g.edges.values())
    L = [f"# Graph report: {stats['root']}", "",
         "## Build stats", "",
         f"- Files: {stats['detected']} supported, {stats['parsed']} parsed, {stats['cached']} from cache, "
         f"{len(stats['skipped'])} skipped",
         f"- Nodes: {len(g.nodes)} ({', '.join(f'{k} {v}' for k, v in sorted(kinds.items()))})",
         f"- Edges: {len(g.edges)} ({', '.join(f'{t} {tiers[t]}' for t in ('EXTRACTED', 'INFERRED', 'AMBIGUOUS'))})",
         f"- Hyperedges: {len(g.hyperedges)}",
         f"- Communities: {len(a['members'])}",
         f"- Similarity: {cfg['similarity']['method']}",
         f"- Time: {stats['seconds']:.2f}s",
         "", _row("relation", "tier", "count"), _row("---", "---", "---:")]
    L += [_row(r, t, n) for (r, t), n in sorted(rel_tier.items())]

    L += ["", "## Communities", "", _row("#", "name", "size", "docs", "cohesion", "top documents"),
          _row("---:", "---", "---:", "---:", "---:", "---")]
    for c in sorted(a["members"]):
        ns = a["members"][c]
        docs = sorted((n for n in ns if g.nodes[n]["kind"] == "document"), key=lambda n: -a["pagerank"][n])
        L.append(_row(c, a["names"][c], len(ns), len(docs), a["cohesion"][c], ", ".join(lab(n) for n in docs[:4])))

    L += ["", "## God nodes", "", "Ranked by normalised degree + PageRank.", "",
          _row("label", "kind", "degree", "pagerank", "community"), _row("---", "---", "---:", "---:", "---:")]
    L += [_row(x["label"], x["kind"], x["degree"], x["pagerank"], x["community"]) for x in a["gods"]]

    L += ["", "## Surprising cross-cluster links", ""]
    if not a["surprises"]:
        L.append("None found.")
    else:
        L += [_row("from", "relation", "to", "communities", "confidence", "provenance"), _row("---", "---", "---", "---", "---:", "---")]
        for e in a["surprises"]:
            s, t = e["source"], e["target"]
            L.append(_row(lab(s), e["relation"], lab(t), f"{a['community'][s]} → {a['community'][t]}",
                          e["confidence_score"], f"{e['source_file']} {e['source_location'] or ''}".strip()))

    amb = sorted((e for e in g.edges.values() if e["confidence"] == "AMBIGUOUS"),
                 key=lambda e: (-e["confidence_score"], -e["weight"], e["source"], e["target"]))
    L += ["", "## AMBIGUOUS edges needing review", "",
          f"{len(amb)} weak edges are stored in graph.json with `confidence: AMBIGUOUS`. Strongest {min(top, len(amb))}:", ""]
    if amb:
        L += [_row("from", "relation", "to", "weight", "confidence", "provenance"), _row("---", "---", "---", "---:", "---:", "---")]
        L += [_row(lab(e["source"]), e["relation"], lab(e["target"]), e["weight"], e["confidence_score"],
                   f"{e['source_file']} {e['source_location'] or ''}".strip()) for e in amb[:top]]

    L += ["", "## Concepts dropped as too common", "",
          "Found in more than `max_doc_frac` of documents or `max_chunk_frac` of chunks, so they tell nothing apart.", ""]
    if g.dropped:
        L += [_row("concept", "docs", "share of chunks"), _row("---", "---:", "---:")]
        L += [_row(d["label"], d["docs"], d["chunk_frac"]) for d in g.dropped]
    else:
        L += ["None."]

    from .redact import RULES

    kinds = list(dict.fromkeys(k for k, *_ in RULES))
    L += ["", "## Redacted before indexing", "",
          f"These are blanked out of the text before it is cached or indexed (your files are unchanged): {', '.join(kinds)}.",
          ""]
    if stats.get("redacted"):
        L += [f"{sum(stats['redacted'].values())} found in {stats['redacted_files']} file(s):", "",
              _row("kind", "count"), _row("---", "---:")]
        L += [_row(k, n) for k, n in sorted(stats["redacted"].items())]
    else:
        L += ["None found."]
    if stats["skipped"]:
        L += ["", "## Skipped files", "", _row("file", "reason"), _row("---", "---")]
        L += [_row(f, r) for f, r in stats["skipped"]]
    return "\n".join(L) + "\n"
