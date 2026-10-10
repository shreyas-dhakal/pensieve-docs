import argparse
import logging
from pathlib import Path

from . import config
from .build import build


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pensieve-docs", description="Turn a folder of documents into a knowledge graph.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build graph.json + GRAPH_REPORT.md for a folder")
    b.add_argument("folder")
    b.add_argument("--config", help="TOML config (default: <folder>/pensieve.toml if present)")
    b.add_argument("--out", help="output directory (default: ./pensieve-out)")
    b.add_argument("-v", "--verbose", action="store_true")
    v = sub.add_parser("ui", help="open the local viewer for a built graph (needs the [ui] extra)")
    v.add_argument("--out", default="pensieve-out", help="directory holding graph.json (default: ./pensieve-out)")
    v.add_argument("--port", type=int, default=8080)
    v.add_argument("--no-show", action="store_true", help="don't open a browser")
    r = sub.add_parser("related", help="documents related to a file or to text on stdin")
    r.add_argument("input", help="a file (md/txt/pdf/docx), or - to read text from stdin")
    r.add_argument("--out", default="pensieve-out", help="directory holding graph.json (default: ./pensieve-out)")
    r.add_argument("-k", type=int, default=5, help="how many documents (default 5)")
    r.add_argument("--min-score", type=float, default=None, help="hide weaker matches (default 0.2)")
    r.add_argument("--json", action="store_true", help="print JSON instead of text")
    args = ap.parse_args(argv)

    if args.cmd == "related":
        return _related(ap, args)

    if args.cmd == "ui":
        out = Path(args.out)
        if not (out / "graph.json").exists():
            ap.error(f"no graph.json in {out}; run `pensieve-docs build <folder>` first")
        from .ui import run

        return run(out, port=args.port, show=not args.no_show)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    folder = Path(args.folder)
    if not folder.is_dir():
        ap.error(f"not a directory: {folder}")
    s = build(folder, config.load(args.config, folder), args.out)
    print(f"{s['detected']} files ({s['parsed']} parsed, {s['cached']} cached, {len(s['skipped'])} skipped) -> "
          f"{s['nodes']} nodes, {s['edges']} edges, {s['hyperedges']} hyperedges, {s['communities']} communities "
          f"in {s['seconds']:.1f}s")
    print(f"wrote {s['out']}/graph.json and GRAPH_REPORT.md")


def _related(ap, args):
    import json
    import sys

    from .parse import SkipFile, parse_file
    from .related import MIN_SCORE, Index

    out = Path(args.out)
    if not (out / "graph.json").exists():
        ap.error(f"no graph.json in {out}; run `pensieve-docs build <folder>` first")
    idx = Index(out / "graph.json")
    exclude = None
    if args.input == "-":
        text = sys.stdin.read()
    else:
        path = Path(args.input).resolve()
        try:
            text = "\n".join(c["text"] for c in parse_file(path)["chunks"])
        except (SkipFile, OSError) as e:
            ap.error(f"cannot read {args.input}: {e}")
        if path.is_relative_to(idx.root):
            exclude = path.relative_to(idx.root).as_posix()
    hits = idx.related(text, k=args.k, min_score=MIN_SCORE if args.min_score is None else args.min_score,
                       exclude=exclude)
    if args.json:
        print(json.dumps(hits, indent=2, ensure_ascii=False))
        return
    if not hits:
        print("nothing related above the threshold")
    for h in hits:
        print(f"{h['score']:.2f}  {h['label']}  ({h['source_file']})")
        for r in h["reasons"]:
            print(f"        {r['concept']}  [{r['where']}]")


if __name__ == "__main__":
    main()
