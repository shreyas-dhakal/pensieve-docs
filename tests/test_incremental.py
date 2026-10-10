import json
import shutil
from pathlib import Path

from pensieve_docs import config
from pensieve_docs.build import build

SAMPLE = Path(__file__).parent.parent / "examples" / "sample"


def test_rebuild_only_reprocesses_changed_files(tmp_path):
    src, out = tmp_path / "corpus", tmp_path / "out"
    shutil.copytree(SAMPLE, src)
    cfg = config.load()
    n = len([p for p in src.rglob("*") if p.suffix in (".md", ".txt", ".pdf", ".docx")])

    first = build(src, cfg, out)
    assert (first["parsed"], first["cached"]) == (n, 0)

    second = build(src, cfg, out)
    assert (second["parsed"], second["cached"]) == (0, n)
    assert second["nodes"] == first["nodes"] and second["edges"] == first["edges"]

    (src / "notes" / "solar-basics.md").write_text("# Solar Basics\n\nGeothermal boreholes replace the solar array.\n")
    third = build(src, cfg, out)
    assert (third["parsed"], third["cached"]) == (1, n - 1)
    g = json.loads((out / "graph.json").read_text())
    doc = next(x for x in g["nodes"] if x["id"] == "notes_solar_basics")
    assert doc["hash"] != "" and doc["n_chunks"] == 1

    (src / "notes" / "heat-pumps.txt").unlink()
    fourth = build(src, cfg, out)
    assert (fourth["parsed"], fourth["cached"]) == (0, n - 1)
    g = json.loads((out / "graph.json").read_text())
    assert "notes_heat_pumps" not in {x["id"] for x in g["nodes"]}
    assert len(list((out / ".cache").glob("*.json"))) == n - 1  # stale cache entries pruned


def test_config_change_invalidates_cache(tmp_path):
    out = tmp_path / "out"
    cfg = config.load()
    build(SAMPLE, cfg, out)
    cfg["chunk"]["max_chars"] = 1000
    assert build(SAMPLE, cfg, out)["cached"] == 0
