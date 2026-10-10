"""Deterministic per-file parsing: the "AST for documents".

parse_file(path) -> {
  "title", "doc_type", "doi",
  "chunks": [{"heading", "level", "page", "line", "offset", "text"}],   # one per section (split if long)
  "refs":   [{"kind": link|wikilink|url|doi|reference, "target", "chunk"}],
}
page is 1-based for PDFs, line is 1-based (paragraph number for docx), offset is a char offset (text files).
"""
import re
from pathlib import Path

VERSION = 2  # bump to invalidate cached parses

HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
SETEXT = re.compile(r"^(=+|-+)\s*$")
MD_LINK = re.compile(r"(?<!!)\[[^\]]*\]\(\s*<?([^)\s>]+)")
WIKI = re.compile(r"\[\[([^\]|#]+)")
URL = re.compile(r"https?://[^\s)<>\]\"']+")
DOI = re.compile(r"\b10\.\d{4,9}/[^\s\"<>)\]]+")
REFLINE = re.compile(r"^\s*\[(\d+)\]\s+\S")
REF_HEADINGS = {"references", "bibliography", "works cited", "citations", "sources", "further reading", "see also", "external links"}


class SkipFile(Exception):
    pass


def parse_file(path: Path, max_chars: int = 4000) -> dict:
    ext = path.suffix.lower()
    if ext == ".pdf":
        title, sections, extra = _pdf(path)
    elif ext == ".docx":
        title, sections, extra = _docx(path)
    else:
        title, sections = _text(path.read_text(encoding="utf-8", errors="replace"), ext != ".txt")
        extra = []
    chunks = []
    for si, s in enumerate(sections):
        chunks += [{**c, "sec": si} for c in _split(s, max_chars) if c["heading"] or c["text"].strip()]
    if not any(c["text"].strip() for c in chunks):
        raise SkipFile("no text")
    refs = _refs(chunks)
    for where, key, url in extra:  # hyperlinks from docx/pdf: map section/page -> chunk index
        i = next((i for i, c in enumerate(chunks) if c["sec" if where == "sec" else "page"] == key), 0)
        refs.append({"kind": "url" if url.startswith(("http://", "https://")) else "link", "target": url, "chunk": i})
    for c in chunks:
        del c["sec"]
    refs = [dict(t) for t in dict.fromkeys(tuple(r.items()) for r in refs)]  # dedupe, keep order
    m = DOI.search(chunks[0]["text"])
    return {
        "title": title or (chunks[0]["heading"] if chunks[0]["heading"] else path.stem),
        "doc_type": ext.lstrip("."),
        "doi": _clean_doi(m.group(0)) if m else None,
        "chunks": chunks,
        "refs": refs,
    }


def location(chunk: dict, doc_type: str) -> str:
    if chunk["page"]:
        return f"p{chunk['page']}"
    return f"¶{chunk['line']}" if doc_type == "docx" else f"L{chunk['line']}"


def _section(heading=None, level=0, page=None, line=1, offset=0, text=""):
    return {"heading": heading, "level": level, "page": page, "line": line, "offset": offset, "text": text}


def _text(text: str, markdown: bool):
    title = None
    lines = text.splitlines(keepends=True)
    start = 0
    if markdown and lines and lines[0].strip() == "---":  # YAML frontmatter
        for j in range(1, len(lines)):
            if lines[j].strip() == "---":
                for fm in lines[1:j]:
                    if fm.lower().startswith("title:"):
                        title = fm.split(":", 1)[1].strip().strip("\"'")
                start = j + 1
                break
    offset = sum(len(l) for l in lines[:start])
    sections, cur, fence = [], _section(line=start + 1, offset=offset), False
    i = start
    while i < len(lines):
        line = lines[i]
        if markdown and line.lstrip().startswith(("```", "~~~")):
            fence = not fence
        m = HEADING.match(line) if markdown and not fence else None
        setext = (not fence and line.strip() and len(line.strip()) < 100 and i + 1 < len(lines)
                  and SETEXT.match(lines[i + 1]) and not HEADING.match(line))
        if m or setext:
            sections.append(cur)
            if m:
                cur = _section(m.group(2).strip(), len(m.group(1)), None, i + 1, offset)
            else:
                cur = _section(line.strip(), 1 if lines[i + 1].startswith("=") else 2, None, i + 1, offset)
                offset += len(line)
                i += 1
                line = lines[i]
            cur["tline"], cur["toff"] = i + 2, offset + len(line)
            title = title or (cur["heading"] if cur["level"] == 1 else None)
        else:
            cur["text"] += line
        offset += len(line)
        i += 1
    sections.append(cur)
    return title, sections


def _docx(path: Path):
    import docx

    d = docx.Document(str(path))
    sections, cur, links, title = [], _section(), [], d.core_properties.title or None
    for i, para in enumerate(d.paragraphs, 1):
        style = para.style.name if para.style is not None else ""
        if style == "Title" or style.startswith("Heading"):
            sections.append(cur)
            digits = "".join(ch for ch in style if ch.isdigit())
            cur = _section(para.text.strip(), int(digits) if digits else (0 if style == "Title" else 1), None, i)
            if title is None and (style == "Title" or cur["level"] == 1):
                title = cur["heading"]
            continue
        if not cur["text"]:
            cur["tline"] = i
            if not cur["heading"]:
                cur["line"] = i
        cur["text"] += para.text + "\n"
        links += [(len(sections), h.address) for h in para.hyperlinks if h.address]
    sections.append(cur)
    return title, sections, [("sec", si, u) for si, u in links]


def _pdf(path: Path):
    from pypdf import PdfReader

    r = PdfReader(str(path))
    pages = [(p.extract_text() or "") for p in r.pages]
    if not any(t.strip() for t in pages):
        raise SkipFile("no text layer (scanned PDF?) - OCR is out of scope")
    by_page: dict[int, list] = {}

    def walk(items, level):
        for it in items:
            if isinstance(it, list):
                walk(it, level + 1)
                continue
            try:
                by_page.setdefault(r.get_destination_page_number(it), []).append((level, str(it.title).strip()))
            except Exception:
                pass

    walk(r.outline or [], 1)
    # one chunk per page (exact page provenance); heading = last outline entry starting on or before it
    sections, cur = [], (None, 0)
    for pno, text in enumerate(pages):
        entries = by_page.get(pno, [])
        for level, t in entries[:-1]:  # several entries on one page: earlier ones are heading-only
            sections.append(_section(t, level, pno + 1))
        if entries:
            cur = (entries[-1][1], entries[-1][0])
        s = _section(cur[0], cur[1], pno + 1, 1, 0, text)
        s["cont"] = not entries and pno > 0
        sections.append(s)
    links = []
    for pno, page in enumerate(r.pages, 1):
        for a in page.get("/Annots") or []:
            a = a.get_object()
            uri = (a.get("/A") or {}).get("/URI") if a.get("/Subtype") == "/Link" else None
            if uri:
                links.append(("page", pno, str(uri)))
    first = next((v[0][1] for _, v in sorted(by_page.items())), None)
    return (r.metadata.title if r.metadata else None) or first, sections, links


def _split(s: dict, max_chars: int) -> list[dict]:
    """Split a long section on paragraph (or line) boundaries; children keep the heading."""
    # first chunk points at the heading; continuations point at where their own text starts
    line, offset = s.pop("tline", s["line"]), s.pop("toff", s["offset"])
    text, out, first = s["text"], [], True
    while True:
        if len(text) <= max_chars:
            cut = len(text)
        else:
            cut = text.rfind("\n\n", 0, max_chars)
            cut = cut + 2 if cut > 0 else (text.rfind("\n", 0, max_chars) + 1 or max_chars)
        out.append({**s, "text": text[:cut], "cont": s.get("cont", False) or not first,
                    **({} if first else {"line": line, "offset": offset})})
        line += text[:cut].count("\n")
        offset += cut
        text, first = text[cut:], False
        if not text:
            return out


def _refs(chunks: list[dict]) -> list[dict]:
    refs = []
    for i, c in enumerate(chunks):
        t = c["text"]
        for m in MD_LINK.finditer(t):
            target = m.group(1)
            refs.append({"kind": "url" if target.startswith(("http://", "https://")) else "link", "target": target, "chunk": i})
        for m in WIKI.finditer(t):
            refs.append({"kind": "wikilink", "target": m.group(1).strip(), "chunk": i})
        for m in URL.finditer(t):
            refs.append({"kind": "url", "target": m.group(0).rstrip(".,;:"), "chunk": i})
        for m in DOI.finditer(t):
            refs.append({"kind": "doi", "target": _clean_doi(m.group(0)), "chunk": i})
        in_refs = is_ref_heading(c["heading"])
        for line in t.splitlines():
            if (in_refs and line.strip()) or REFLINE.match(line):
                refs.append({"kind": "reference", "target": line.strip().lstrip("-*• ").strip(), "chunk": i})
    return refs


def is_ref_heading(heading: str | None) -> bool:
    return (heading or "").strip().lower().lstrip("0123456789. ") in REF_HEADINGS


def _clean_doi(d: str) -> str:
    return d.rstrip(".,;:").lower()
