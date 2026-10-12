import pytest

from pensieve_docs import config
from pensieve_docs.build import build
from pensieve_docs.query import Graph

from test_edges import SAMPLE


@pytest.fixture(scope="module")
def g(tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    build(SAMPLE, config.load(), out)
    return Graph(out / "graph.json")


def test_query_starts_at_named_concepts_and_keeps_budget(g):
    text = g.query("should we add a home battery for backup power", budget=300)
    assert text.startswith("START backup power")
    assert "--co_occurs--> grid outage" in text and "meetings/2024-06-review.md L5" in text
    assert len(text) // 4 <= 300 + 20 and text.endswith("budget)")
    assert g.query("zzqq").startswith("no node")


def test_path_and_explain(g):
    p = g.path("heat pump", "net metering").splitlines()
    assert p[0] == "2 hops" and all(" --" in line for line in p[1:])
    e = g.explain("net metering")
    assert e.startswith("net metering [concept, in 3 documents") and "mentions (" in e
    assert "also matches: Battery Storage [document]" in g.explain("Battery Storage")
    assert g.explain("notes/battery-storage.md").startswith("Battery Storage [document")
