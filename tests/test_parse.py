import pytest

from pensieve_docs.detect import detect
from pensieve_docs.parse import SkipFile, location, parse_file
from helpers import make_docx, make_pdf

MD = """---
title: Front Title
---
# Top

Intro with [a link](other.md#part), [[Wiki Page|alias]], https://example.com/x. and doi 10.1000/xyz123.

```
# not a heading
```

## References

[1] Some Paper Title, 2020.
- Another Ref
"""


def test_markdown_sections_links_citations(tmp_path):
    f = tmp_path / "a.md"
    f.write_text(MD)
    p = parse_file(f)
    assert p["title"] == "Front Title"
    assert [(c["heading"], c["level"]) for c in p["chunks"]] == [("Top", 1), ("References", 2)]
    assert "# not a heading" in p["chunks"][0]["text"]  # fenced code is not a heading
    # provenance: line and char offset point at the heading in the file
    text, lines = f.read_text(), f.read_text().splitlines()
    for c in p["chunks"]:
        assert lines[c["line"] - 1].lstrip("# ").strip() == c["heading"]
        assert text[c["offset"]:].startswith("#")
    refs = {(r["kind"], r["target"]) for r in p["refs"]}
    assert ("link", "other.md#part") in refs
    assert ("wikilink", "Wiki Page") in refs
    assert ("url", "https://example.com/x") in refs
    assert ("doi", "10.1000/xyz123") in refs
    assert ("reference", "[1] Some Paper Title, 2020.") in refs
    assert ("reference", "Another Ref") in refs
    assert location(p["chunks"][1], "md") == f"L{p['chunks'][1]['line']}"


def test_txt_setext_headings_and_long_section_split(tmp_path):
    f = tmp_path / "b.txt"
    body = "\n\n".join(f"Paragraph {i} " + "word " * 40 for i in range(20))
    f.write_text(f"Title Here\n==========\n\n{body}\n\nSecond\n------\nshort\n")
    p = parse_file(f, max_chars=500)
    heads = [c["heading"] for c in p["chunks"] if not c["cont"]]
    assert heads == ["Title Here", "Second"]
    long = [c for c in p["chunks"] if c["heading"] == "Title Here"]
    assert len(long) > 1 and all(len(c["text"]) <= 500 for c in long)
    assert [c["cont"] for c in long] == [False] + [True] * (len(long) - 1)
    text = f.read_text()
    for c in long[1:]:
        assert text[c["offset"]:].startswith(c["text"][:30])  # sub-chunk offsets stay exact


def test_pdf_outline_pages_links_doi(tmp_path):
    f = tmp_path / "p.pdf"
    make_pdf(f, ["Intro words\nDOI 10.5555/self.1", "Method words", "more method", "[1] Cited Thing"],
             outline=[("Intro", 0), ("Method", 1), ("References", 3)], links=[(2, "https://ex.org/q")], title="Paper T")
    p = parse_file(f)
    assert p["title"] == "Paper T" and p["doc_type"] == "pdf" and p["doi"] == "10.5555/self.1"
    assert [(c["heading"], c["page"], c["cont"]) for c in p["chunks"]] == [
        ("Intro", 1, False), ("Method", 2, False), ("Method", 3, True), ("References", 4, False)]
    assert location(p["chunks"][2], "pdf") == "p3"
    assert {"kind": "url", "target": "https://ex.org/q", "chunk": 2} in p["refs"]  # annotation on page 3
    assert any(r["kind"] == "reference" and r["chunk"] == 3 for r in p["refs"])


def test_scanned_pdf_is_skipped(tmp_path):
    f = tmp_path / "scan.pdf"
    make_pdf(f, ["", ""])
    with pytest.raises(SkipFile, match="no text layer"):
        parse_file(f)


def test_docx_headings_and_hyperlinks(tmp_path):
    f = tmp_path / "d.docx"
    make_docx(f, "Doc T", [("One", "alpha text\n\nbeta text"), ("Two", "gamma")], links=["https://doi.org/10.1234/abc"])
    p = parse_file(f)
    assert p["title"] == "Doc T"
    assert [(c["heading"], c["level"]) for c in p["chunks"]] == [("Doc T", 0), ("One", 1), ("Two", 1)]
    assert p["chunks"][1]["text"] == "alpha text\nbeta text\n"
    assert location(p["chunks"][1], "docx") == "¶2"
    refs = {(r["kind"], r["target"], r["chunk"]) for r in p["refs"]}
    assert ("url", "https://doi.org/10.1234/abc", 2) in refs and ("doi", "10.1234/abc", 2) in refs


def test_detect_classifies_and_logs_skips(tmp_path):
    for name in ["a.md", "b.txt", "c.pdf", "d.docx", "e.png", "f.mp3", ".hidden.md"]:
        (tmp_path / name).write_text("x")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "g.md").write_text("x")
    files, skipped = detect(tmp_path, [])
    assert sorted(f.name for f in files) == ["a.md", "b.txt", "c.pdf", "d.docx"]
    assert sorted(f.name for f, _ in skipped) == ["e.png", "f.mp3"]
