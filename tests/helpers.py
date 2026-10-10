"""Build tiny real PDF / DOCX files for tests and the sample corpus (no binary fixtures to maintain)."""
from pathlib import Path


def make_pdf(path: Path, pages: list[str], outline: list[tuple[str, int]] = (), links: list[tuple[int, str]] = (),
             title: str | None = None):
    """pages: text per page (lines split on \\n). outline: (title, 0-based page). links: (0-based page, uri)."""
    from pypdf import PdfWriter
    from pypdf.annotations import Link
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    w = PdfWriter()
    font = w._add_object(DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                                           NameObject("/Subtype"): NameObject("/Type1"),
                                           NameObject("/BaseFont"): NameObject("/Helvetica")}))
    for text in pages:
        page = w.add_blank_page(612, 792)
        if text:
            esc = lambda s: s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            body = " T* ".join(f"({esc(line)}) Tj" for line in text.split("\n"))
            stream = DecodedStreamObject()
            stream.set_data(f"BT /F1 11 Tf 14 TL 50 740 Td {body} ET".encode("latin-1"))
            page[NameObject("/Contents")] = w._add_object(stream)
            page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})})
    for t, pno in outline:
        w.add_outline_item(t, pno)
    for pno, uri in links:
        w.add_annotation(pno, Link(rect=(50, 50, 200, 70), url=uri))
    if title:
        w.add_metadata({"/Title": title})
    with open(path, "wb") as f:
        w.write(f)


def make_docx(path: Path, title: str, sections: list[tuple[str, str]], links: list[str] = ()):
    """sections: (heading, body). links: URLs added as real hyperlinks in a final paragraph."""
    import docx
    from docx.opc.constants import RELATIONSHIP_TYPE as RT
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    d = docx.Document()
    d.add_heading(title, level=0)
    for heading, body in sections:
        d.add_heading(heading, level=1)
        for para in body.split("\n\n"):
            d.add_paragraph(para)
    for url in links:
        p = d.add_paragraph("See ")
        rid = p.part.relate_to(url, RT.HYPERLINK, is_external=True)
        h = OxmlElement("w:hyperlink")
        h.set(qn("r:id"), rid)
        r = OxmlElement("w:r")
        t = OxmlElement("w:t")
        t.text = url
        r.append(t)
        h.append(r)
        p._p.append(h)
    d.save(str(path))
