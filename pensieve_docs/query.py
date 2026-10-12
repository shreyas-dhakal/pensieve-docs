"""Ask the graph, the way graphify's query / path / explain do. Load once, ask many times:

    g = Graph("pensieve-out/graph.json")
    g.query("how does a heat pump save money")   # walk out from the matching nodes, within a token budget
    g.path("heat pump", "net metering")          # shortest chain of connections between two nodes
    g.explain("net metering")                    # one node: what it is, where it appears, what it connects to

Each returns plain text, one line per node or edge, with the file and location every connection came from, so an
agent can quote it or open just those lines. A name is matched exactly (label, file or id) first, then as known
concepts in the text, then as a substring of node labels.
"""
import json
from pathlib import Path

import networkx as nx

from .related import Index

SKIP = {"contains"}  # folder structure says nothing about content, and every document has one
TIER = {"EXTRACTED": 0, "INFERRED": 1, "AMBIGUOUS": 2}
KIND = {"concept": 0, "document": 1, "section": 2, "folder": 3}  # which node a shared name means
PER_NODE = 8  # ponytail: strongest edges followed per node, so one busy node can't spend the whole budget


def tokens(text: str) -> int:
    return max(1, len(text) // 4)


class Graph(Index):
    def __init__(self, graph_json: str | Path, cfg: dict | None = None):
        super().__init__(graph_json, cfg)
        self.g = nx.Graph()
        self.g.add_nodes_from(self.nodes)
        for e in json.loads(Path(graph_json).read_text(encoding="utf-8"))["links"]:
            if e["relation"] not in SKIP:
                self.g.add_edge(e["source"], e["target"], e=e,
                                cost=1 + TIER[e["confidence"]] + (1 - e["confidence_score"]))

    def resolve(self, text: str, n: int = 3) -> list[str]:
        """Node ids that `text` names, best first."""
        t = text.strip().lower()
        if t in {f.lower() for f in self.docs}:  # a path names the document, not every node found in that file
            return [d for f, d in self.docs.items() if f.lower() == t]
        exact = [i for i, x in self.nodes.items() if t in {i.lower(), x["label"].lower(), x.get("norm", "")}]
        if exact:
            return sorted(exact, key=lambda i: KIND[self.nodes[i]["kind"]])[:n]
        found = self.concepts(text)
        if found:
            return sorted(found, key=lambda c: (-found[c] * self.idf[c], c))[:n]
        sub = [i for i, x in self.nodes.items() if x["kind"] != "folder" and t and t in x["label"].lower()]
        return sorted(sub, key=lambda i: (KIND[self.nodes[i]["kind"]], len(self.nodes[i]["label"]), i))[:n]

    def node_line(self, i: str) -> str:
        x = self.nodes[i]
        where = " ".join(filter(None, (x.get("source_file"), x.get("source_location"))))
        extra = f", in {x['doc_count']} documents" if x["kind"] == "concept" else ""
        return f"{x['label']} [{x['kind']}{extra}, community: {x.get('community_name', '-')}] {where}"

    def edge_line(self, a: str, b: str) -> str:
        e = self.g.edges[a, b]["e"]
        rel = e["relation"] + (f" ({e['relation_label']})" if e.get("relation_label") else "")
        where = " ".join(filter(None, (e.get("source_file"), e.get("source_location"))))
        return (f"{self.nodes[e['source']]['label']} --{rel}--> {self.nodes[e['target']]['label']} "
                f"[{e['confidence']} {e['confidence_score']}] {where}")

    def strongest(self, i: str) -> list[str]:
        return sorted(self.g[i], key=lambda m: (self.g.edges[i, m]["cost"], m))

    def query(self, question: str, budget: int = 2000, dfs: bool = False) -> str:
        start = self.resolve(question)
        if not start:
            return f"no node in the graph matches: {question}"
        lines = [f"START {self.node_line(i)}" for i in start]
        used, seen, done = sum(map(tokens, lines)), set(start), set()
        todo = [(i, 0) for i in reversed(start)] if dfs else [(i, 0) for i in start]
        while todo:
            i, depth = todo.pop() if dfs else todo.pop(0)
            nxt = []
            for m in self.strongest(i)[:PER_NODE]:
                if frozenset((i, m)) in done:
                    continue
                line = self.edge_line(i, m)
                if used + tokens(line) > budget:
                    return "\n".join(lines + [f"(stopped at the {budget}-token budget)"])
                lines.append(line)
                used += tokens(line)
                done.add(frozenset((i, m)))
                if m not in seen and depth < 6:  # at most 6 hops out, as graphify's DFS
                    seen.add(m)
                    nxt.append((m, depth + 1))
            todo += reversed(nxt) if dfs else nxt
        return "\n".join(lines)

    def path(self, a: str, b: str) -> str:
        ends = [self.resolve(x, 1) for x in (a, b)]
        if not all(ends):
            return "no node in the graph matches: " + " / ".join(x for x, r in zip((a, b), ends) if not r)
        try:  # cheapest path: few hops, EXTRACTED over INFERRED over AMBIGUOUS, stronger edges first
            p = nx.shortest_path(self.g, ends[0][0], ends[1][0], weight="cost")
        except nx.NetworkXNoPath:
            return f"no path between {self.nodes[ends[0][0]]['label']} and {self.nodes[ends[1][0]]['label']}"
        return "\n".join([f"{len(p) - 1} hops"] + [self.edge_line(x, y) for x, y in zip(p, p[1:])])

    def explain(self, name: str, per_relation: int = 10) -> str:
        found = self.resolve(name)
        if not found:
            return f"no node in the graph matches: {name}"
        i = found[0]
        lines = [self.node_line(i)] + [f"(also matches: {self.nodes[o]['label']} [{self.nodes[o]['kind']}], "
                                       f"explain it by its file or id: {o})" for o in found[1:]]
        by_rel = {}
        for m in self.strongest(i):
            by_rel.setdefault(self.g.edges[i, m]["e"]["relation"], []).append(m)
        for rel, ms in sorted(by_rel.items()):
            lines.append(f"\n{rel} ({len(ms)}):")
            lines += [f"  {self.edge_line(i, m)}" for m in ms[:per_relation]]
            if len(ms) > per_relation:
                lines.append(f"  ... {len(ms) - per_relation} more")
        return "\n".join(lines)
