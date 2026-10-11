import json
import socket
from pathlib import Path

import pytest

from pensieve_docs import config
from pensieve_docs.build import build
from pensieve_docs.edges import RUBRIC, Graph

SAMPLE = Path(__file__).parent.parent / "examples" / "sample"


@pytest.fixture(scope="module")
def graph(tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    build(SAMPLE, config.load(), out)
    return json.loads((out / "graph.json").read_text())


def _edges(g, relation):
    return {(e["source"], e["target"]): e for e in g["links"] if e["relation"] == relation}


def test_graphify_compatible_shape(graph):
    assert list(graph) == ["directed", "multigraph", "graph", "nodes", "links", "hyperedges", "built_at_commit"]
    assert graph["graph"]["hyperedges"] == graph["hyperedges"]
    import networkx as nx

    G = nx.node_link_graph(graph, edges="links")  # list-valued `evidence` must not get in the way
    assert G.number_of_edges() == len(graph["links"])
    for n in graph["nodes"]:
        assert {"id", "label", "norm_label", "file_type", "kind", "community", "community_name", "source_file"} <= n.keys()
    docs = [n for n in graph["nodes"] if n["kind"] == "document"]
    for d in docs:  # pointer to the file, never its content
        assert {"hash", "mtime", "doc_type"} <= d.keys() and "text" not in d
        assert (SAMPLE / d["source_file"]).is_file()


def test_every_edge_has_type_weight_confidence_tier_provenance(graph):
    for e in graph["links"]:
        assert {"source", "target", "relation", "confidence", "confidence_score", "weight", "source_file",
                "source_location", "chunk"} <= e.keys()
        tier, score = e["confidence"], e["confidence_score"]
        if tier == "EXTRACTED":
            assert score == 1.0
        elif tier == "INFERRED":
            assert score in RUBRIC
        else:
            assert tier == "AMBIGUOUS" and 0.1 <= score <= 0.3
        assert (SAMPLE / e["source_file"]).exists()
    assert {e["relation"] for e in graph["links"]} == {
        "contains", "has_section", "links_to", "cites", "mentions", "co_occurs", "similar_to", "conceptually_related_to"}


def test_extracted_structure_links_and_citations(graph):
    contains, links, cites = _edges(graph, "contains"), _edges(graph, "links_to"), _edges(graph, "cites")
    assert ("folder_root", "folder_notes") in contains and ("folder_notes", "notes_solar_basics") in contains
    assert ("index", "notes_battery_storage") in links  # markdown link
    assert ("meetings_2024_05_kickoff", "notes_heat_pumps") in links  # [[heat-pumps]] -> .txt
    assert ("papers_battery_economics", "papers_grid_integration") in cites  # via DOI hyperlink
    assert ("notes_battery_storage", "papers_grid_integration") in cites  # reference-list title
    assert ("notes_battery_storage", "papers_battery_economics") in cites  # longest title wins over "Battery Storage"
    e = links[("index", "notes_battery_storage")]
    assert e["confidence"] == "EXTRACTED" and e["source_file"] == "index.md" and e["source_location"].startswith("L")
    assert all(e["confidence"] == "EXTRACTED" for e in _edges(graph, "has_section").values())


def test_inferred_mentions_cooccur_hyperedges(graph):
    mentions = _edges(graph, "mentions")
    assert ("meetings_2024_06_review", "concept_home_battery") in mentions
    cfg = config.load()
    for e in _edges(graph, "co_occurs").values():
        assert e["count"] >= cfg["cooccur"]["min_count"]
        strong = e["npmi"] >= cfg["cooccur"]["npmi_threshold"] and e["n_docs"] >= cfg["cooccur"]["min_docs_inferred"]
        assert strong == (e["confidence"] == "INFERRED")
        assert 0 < e["max_doc_share"] <= 1
    concepts = {n["id"]: n for n in graph["nodes"] if n["kind"] == "concept"}
    assert graph["hyperedges"]
    for h in graph["hyperedges"]:
        assert concepts[h["nodes"][0]]["doc_count"] >= 3 and len(h["nodes"]) >= 4
    assert any(h["nodes"][0] == "concept_home_battery" for h in graph["hyperedges"])


def test_similarity_threshold_and_top_k(tmp_path):
    cfg = config.load()
    cfg["similarity"].update(top_k=1, threshold=0.2, ambiguous_min=0.1)
    build(SAMPLE, cfg, tmp_path)
    g = json.loads((tmp_path / "graph.json").read_text())
    sims = list(_edges(g, "similar_to").values())
    assert sims and len(sims) <= sum(1 for n in g["nodes"] if n["kind"] == "document")
    for e in sims:
        assert (e["weight"] >= 0.2) == (e["confidence"] == "INFERRED")
        assert e["weight"] >= 0.1


def test_stronger_tier_wins_on_same_pair():
    g = Graph()
    g.edge("a", "b", "similar_to", "AMBIGUOUS", 0.2, 0.1, "x")
    g.edge("b", "a", "links_to", "EXTRACTED", 1.0, 1.0, "x")
    g.edge("a", "b", "similar_to", "INFERRED", 0.75, 0.5, "x")
    assert [e["relation"] for e in g.edges.values()] == ["links_to"]
    (e,) = g.edges.values()
    assert (e["source"], e["target"]) == ("b", "a")
    assert [(x["relation"], x["confidence"]) for x in e["evidence"]] == [
        ("links_to", "EXTRACTED"), ("similar_to", "INFERRED"), ("similar_to", "AMBIGUOUS")]


def test_evidence_is_capped_and_ties_keep_the_first():
    g = Graph(max_evidence=3)
    for i in range(5):
        g.edge("a", "b", "co_occurs", "INFERRED", 0.75, 1.0, f"f{i}")
    g.edge("a", "b", "co_occurs", "INFERRED", 0.75, 1.0, "f0")  # same evidence again: ignored
    (e,) = g.edges.values()
    assert e["source_file"] == "f0" and [x["source_file"] for x in e["evidence"]] == ["f0", "f1", "f2"]


def test_build_makes_no_network_calls(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network call attempted")

    monkeypatch.setattr(socket.socket, "connect", boom)
    monkeypatch.setattr(socket, "create_connection", boom)
    build(SAMPLE, config.load(), tmp_path)
    assert (tmp_path / "graph.json").exists() and (tmp_path / "GRAPH_REPORT.md").exists()


def test_concepts_backfilled_wherever_their_text_appears(graph):
    # spaCy only chunks "lithium-ion battery" in 2 docs; the text is in 4
    n = next(n for n in graph["nodes"] if n["id"] == "concept_lithium_ion_battery")
    assert n["doc_count"] == 4
    assert ("notes_battery_storage", n["id"]) in _edges(graph, "mentions")


def test_nested_concepts_related(graph):
    rel = _edges(graph, "conceptually_related_to")
    assert ("concept_net_metering_export_rate", "concept_net_metering") in rel


def test_pagerank_matches_networkx():
    import random

    import networkx as nx
    from networkx.algorithms.link_analysis.pagerank_alg import _pagerank_python

    from pensieve_docs.analyze import pagerank

    G = nx.karate_club_graph()
    G.add_node("isolated")
    random.seed(1)
    for u, v in G.edges:
        G[u][v]["weight"] = random.random()
    ours, ref = pagerank(G), _pagerank_python(G, weight="weight", tol=1e-12, max_iter=500)
    assert max(abs(ours[n] - ref[n]) for n in G) < 1e-8


def test_reference_lists_are_not_mined_for_concepts(graph):
    # "[1] Battery Storage Economics, internal memo." becomes a cites edge, not a concept
    labels = {n["label"].lower() for n in graph["nodes"] if n["kind"] == "concept"}
    assert "internal memo" not in labels and "residential solar" not in labels
    assert ("notes_battery_storage", "papers_battery_economics") in _edges(graph, "cites")


def test_heading_words_and_own_titles_are_not_concepts(graph):
    labels = {n["label"].lower() for n in graph["nodes"] if n["kind"] == "concept"}
    assert "economics" not in labels  # "## Economics" heading only
    assert "battery storage economics" not in labels  # only the memo's own title
    assert "battery storage" in labels  # a note's title too, but used across 4 docs


def test_node_ids_never_collide(tmp_path):
    # section "Control" of mosquito.md once took the id of mosquito-control.md and overwrote it
    (tmp_path / "mosquito.md").write_text("# Mosquito\n\n## Control\n\nNets and repellent.\n")
    (tmp_path / "mosquito-control.md").write_text("# Mosquito control\n\nNets.\n")
    (tmp_path / "concept-notes.md").write_text("# Notes\n\nNets.\n")
    build(tmp_path, config.load(), tmp_path / "out")
    g = json.loads((tmp_path / "out" / "graph.json").read_text())
    docs = [n for n in g["nodes"] if n["kind"] == "document"]
    assert len(docs) == 3
    assert len({n["id"] for n in g["nodes"]}) == len(g["nodes"])


def _build_dir(tmp_path, files: dict, cfg=None):
    for name, text in files.items():
        (tmp_path / name).write_text(text)
    build(tmp_path, cfg or config.load(), tmp_path / "out")
    return json.loads((tmp_path / "out" / "graph.json").read_text())


_FILLER = "## Filler {0}\n\nThe weather was mild that week. Nobody wrote much down.\n\n"


def test_nested_pair_keeps_cooccurrence_evidence(tmp_path):
    # "solar inverter" contains "inverter" (conceptually_related_to) and they also share chunks (co_occurs)
    body = "## Site {0}\n\nThe solar inverter failed again. We replaced the inverter twice.\n\n"
    files = {f"d{d}.md": f"# Doc {d}\n\n" + body.format(d) + "".join(_FILLER.format(k) for k in range(4)) for d in range(3)}
    g = _build_dir(tmp_path, files)
    (e,) = [e for e in g["links"] if {e["source"], e["target"]} == {"concept_solar_inverter", "concept_inverter"}]
    by_rel = {x["relation"]: x for x in e["evidence"]}
    assert by_rel.keys() == {"conceptually_related_to", "co_occurs"}
    co = by_rel["co_occurs"]
    assert co["count"] == 3 and co["npmi"] > 0 and len(co["chunks"]) == 3
    assert {c[0] for c in co["chunks"]} == {"d0.md", "d1.md", "d2.md"}


def test_pair_repeated_in_one_document_is_not_inferred(tmp_path):
    # one doc says "turbine ... gearbox" in every section; the others mention each alone
    pair = "".join(f"## Part {k}\n\nThe turbine drives a gearbox.\n\n" for k in range(6))
    files = {"a.md": "# A\n\n" + pair,
             "b.md": "# B\n\nThe turbine was tall.\n\n" + "".join(_FILLER.format(k) for k in range(4)),
             "c.md": "# C\n\nThe gearbox was oiled.\n\n" + "".join(_FILLER.format(k) for k in range(4))}
    g = _build_dir(tmp_path, files)
    (e,) = [e for e in g["links"] if {e["source"], e["target"]} == {"concept_turbine", "concept_gearbox"}]
    assert e["relation"] == "co_occurs" and e["count"] == 6 and e["n_docs"] == 1 and e["max_doc_share"] == 1
    assert e["confidence"] == "AMBIGUOUS"
