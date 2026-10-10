import json

from pensieve_docs import config
from pensieve_docs.build import build

SECRETS = ["hunter2", "sk-ant-api03-abcdefghijklmnopqrstuvwxyz", "+61 412 345 678", "jane.doe@example.com",
           "4111 1111 1111 1111"]


def test_secrets_never_reach_cache_graph_or_report(tmp_path):
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "setup.md").write_text(
        "# Server setup\n\nThe database password: hunter2 and the api_key = sk-ant-api03-abcdefghijklmnopqrstuvwxyz.\n"
        "Call Jane Doe on +61 412 345 678 or email jane.doe@example.com about the heat pump.\n"
        "Card on file 4111 1111 1111 1111.\n")
    (notes / "pump.md").write_text("# Heat pump\n\nThe heat pump quote from Jane Doe.\n")
    out = tmp_path / "out"
    build(notes, config.load(), out)
    written = "".join(f.read_text(errors="ignore") for f in out.rglob("*") if f.is_file())
    for s in SECRETS:
        assert s not in written, s
    assert "heat pump" in written  # the rest of the note still indexes
    doc = next(n for n in json.loads((out / "graph.json").read_text())["nodes"] if n["id"] == "setup")
    assert doc["redactions"] == {"password": 1, "api key": 1, "phone": 1, "email": 1, "card number": 1}
    assert "5 found in 1 file(s)" in (out / "GRAPH_REPORT.md").read_text()
