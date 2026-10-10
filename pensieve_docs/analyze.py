"""Louvain communities, community names, god nodes, surprising cross-cluster links."""
from collections import Counter, defaultdict

import networkx as nx

# how interesting a cross-community edge of each type is
SURPRISE = {"cites": 1.0, "links_to": 1.0, "similar_to": 0.9, "co_occurs": 0.7, "conceptually_related_to": 0.3, "mentions": 0.5}


def to_nx(g, include_ambiguous=False) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from((nid, attrs) for nid, attrs in g.nodes.items())
    for e in g.edges.values():
        if include_ambiguous or e["confidence"] != "AMBIGUOUS":
            # confidence_score, not raw weight: PMI / TF-IDF / cosine live on different scales
            G.add_edge(e["source"], e["target"], weight=e["confidence_score"], relation=e["relation"])
    return G


def communities(G: nx.Graph, seed: int = 42) -> list[set]:
    return nx.community.louvain_communities(G, weight="weight", seed=seed)


def pagerank(G: nx.Graph, d: float = 0.85, tol: float = 1e-10) -> dict:
    """Weighted PageRank by power iteration (networkx's own needs scipy)."""
    n = len(G) or 1
    out = {u: sum(w for _, _, w in G.edges(u, data="weight")) for u in G}
    pr = dict.fromkeys(G, 1 / n)
    for _ in range(200):
        dangling = d * sum(pr[u] for u in G if not out[u]) / n
        new = {v: (1 - d) / n + dangling + d * sum(pr[u] * w / out[u] for _, u, w in G.edges(v, data="weight"))
               for v in G}
        if sum(abs(new[v] - pr[v]) for v in G) < tol:
            return new
        pr = new
    return pr


def analyze(g, cfg) -> dict:
    G = to_nx(g)
    # Folders are excluded from clustering (they would just reproduce the directory tree),
    # then join the community most of their children are in.
    core = G.subgraph(n for n, d in G.nodes(data=True) if d["kind"] != "folder")
    comms = sorted(communities(core), key=lambda c: (-len(c), min(c)))
    community = {n: i for i, c in enumerate(comms) for n in c}
    for fid in sorted((n for n, d in G.nodes(data=True) if d["kind"] == "folder"), key=lambda n: -len(n)):
        votes = Counter(community[c] for c in G[fid] if c in community)
        community[fid] = votes.most_common(1)[0][0] if votes else 0
    members = defaultdict(list)
    for n, c in community.items():
        members[c].append(n)

    pr = pagerank(G)
    names, cohesion = {}, {}
    for c, ns in members.items():
        kinds = lambda k: sorted((n for n in ns if G.nodes[n]["kind"] == k), key=lambda n: -pr[n])
        top = kinds("concept")[:3] or kinds("document")[:2] or ns[:1]
        names[c] = " / ".join(G.nodes[n]["label"] for n in top)
        s = set(ns)
        inside = sum(1 for u, v in G.edges(ns) if u in s and v in s)
        boundary = sum(1 for u, v in G.edges(ns) if (u in s) != (v in s))
        cohesion[c] = round(inside / (inside + boundary), 3) if inside + boundary else 1.0

    # like graphify: documents are file-level hubs that win on degree mechanically; rank the ideas
    cand = [n for n, d in G.nodes(data=True) if d["kind"] == "concept"]
    maxdeg = max((G.degree(n) for n in cand), default=1) or 1
    maxpr = max((pr[n] for n in cand), default=1) or 1
    gods = sorted(cand, key=lambda n: -(G.degree(n) / maxdeg + pr[n] / maxpr))[: cfg["report"]["top_n"]]
    gods = [{"id": n, "label": G.nodes[n]["label"], "kind": G.nodes[n]["kind"], "degree": G.degree(n),
             "pagerank": round(pr[n], 5), "community": community[n]} for n in gods]

    def top_folder(n):
        sf = G.nodes[n].get("source_file") or ""
        return sf.split("/")[0] if "/" in sf else ""

    surprises = []
    for e in g.edges.values():
        a, b = e["source"], e["target"]
        if e["confidence"] == "AMBIGUOUS" or e["relation"] not in SURPRISE or community[a] == community[b]:
            continue
        score = SURPRISE[e["relation"]] * e["confidence_score"] * (1.5 if top_folder(a) != top_folder(b) else 1.0)
        surprises.append({**e, "score": round(score, 3)})
    surprises.sort(key=lambda e: (-e["score"], e["source"], e["target"]))

    return {"community": community, "names": names, "cohesion": cohesion, "members": dict(members),
            "pagerank": pr, "gods": gods, "surprises": surprises[: cfg["report"]["top_n"]]}
