"""Canonical, atomic artefact writes.

`MS-C-704`: both reproducers wrote artefacts in text mode, so on Windows Python
translated `\n` to `\r\n` while `.gitattributes` (`* -text`) told git never to
normalise it back. Re-running two maths tools rewrote four committed artefacts as
**22,667 insertions + 22,667 deletions** whose content, after stripping line endings
and the `seconds` field, was byte-identical. That is the EOL defect for the fourth
time in this project, now in the reproducers themselves.

`MS-C-705`: and the JSONL evidence files were opened `"w"` at the START of a scan, so
an interrupted run left a truncated file in place of the canonical evidence -- a
partial artefact that still looks like a complete one.

Both are fixed the same way: build the bytes, write to a temporary file beside the
target, then `os.replace` -- atomic on Windows and POSIX. A run that dies leaves the
previous artefact untouched.
"""
from __future__ import annotations

import json
import os
from pathlib import Path


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_bytes(payload)
    os.replace(tmp, path)                      # atomic; no truncated artefact survives


def write_text_atomic(path: Path, text: str) -> None:
    """Canonical LF, one trailing newline, atomic replace."""
    body = text.replace("\r\n", "\n")
    if body and not body.endswith("\n"):
        body += "\n"
    write_bytes_atomic(path, body.encode("utf-8"))


def write_json_atomic(path: Path, obj, indent: int = 1) -> None:
    write_text_atomic(path, json.dumps(obj, indent=indent))


def write_jsonl_atomic(path: Path, records) -> None:
    """Whole file at once, at the END of the scan (MS-C-705)."""
    write_text_atomic(path, "\n".join(json.dumps(r) for r in records))
