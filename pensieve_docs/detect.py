import logging
import os
from pathlib import Path

log = logging.getLogger("pensieve-docs")

KINDS = {".md": "text", ".markdown": "text", ".txt": "text", ".pdf": "document", ".docx": "document"}


def detect(root: Path, exclude: list[str]) -> tuple[list[Path], list[tuple[Path, str]]]:
    """Return (supported files, [(skipped file, reason)]). Hidden and excluded dirs are not walked."""
    files, skipped = [], []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and d not in exclude)
        for name in sorted(filenames):
            p = Path(dirpath) / name
            if name.startswith("."):
                continue
            if p.suffix.lower() in KINDS:
                files.append(p)
            else:
                skipped.append((p, f"unsupported type {p.suffix or '(none)'}"))
                log.info("skip %s: unsupported type", p.relative_to(root))
    return files, skipped
