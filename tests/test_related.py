import pytest

from pensieve_docs import config
from pensieve_docs.build import build
from pensieve_docs.related import Index

from test_edges import SAMPLE


@pytest.fixture(scope="module")
def idx(tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    build(SAMPLE, config.load(), out)
    return Index(out / "graph.json")


def test_related_ranks_by_shared_concepts_with_reasons(idx):
    hits = idx.related("Should we add a home battery? Backup power during a grid outage, and the payback period.")
    assert hits[0]["source_file"] == "meetings/2024-06-review.md"
    assert {r["concept"] for r in hits[0]["reasons"]} >= {"grid outage", "backup power"}
    assert all(r["where"].startswith(hits[0]["source_file"]) for r in hits[0]["reasons"])
    assert all(h["score"] >= 0.2 and h["bm25"] > 0 for h in hits)  # gated by concepts, ordered with BM25


def test_related_matches_lemmas_and_hyphen_variants(idx):
    assert "concept_heat_pump" in idx.concepts("two heat pumps")
    assert "concept_lithium_ion_battery" in idx.concepts("a lithium ion battery")


def test_related_stays_quiet_and_excludes_self(idx):
    assert idx.related("Notes on saxophone improvisation in bebop.") == []
    text = (SAMPLE / "notes/heat-pumps.txt").read_text()
    assert "notes/heat-pumps.txt" not in {h["source_file"] for h in idx.related(text, exclude="notes/heat-pumps.txt")}


def test_related_works_without_bow_json(tmp_path):
    # graphs built before bow.json existed: rank by concepts alone
    import shutil

    build(SAMPLE, config.load(), tmp_path / "out")
    shutil.copy(tmp_path / "out" / "graph.json", tmp_path / "graph.json")
    hits = Index(tmp_path / "graph.json").related("home battery backup power during a grid outage")
    assert hits and all(h["bm25"] == 0 for h in hits)
    assert [h["score"] for h in hits] == sorted((h["score"] for h in hits), reverse=True)
