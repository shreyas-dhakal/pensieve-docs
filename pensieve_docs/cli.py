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
    args = ap.parse_args(argv)

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


if __name__ == "__main__":
    main()
