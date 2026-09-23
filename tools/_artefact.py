"""MS-C-875: forty producers overwrite a canonical artefact unconditionally.

WHAT THE AUDIT FOUND (2026-08-29)
----------------------------------
Every producer writes its result with a temp file and `os.replace`. That is ATOMIC, and
atomic is not the same as non-destructive: `os.replace` overwrites whatever was there,
and the previous artefact is gone. **Measured: 40 tools write a canonical `runs/*.json`
and 0 of them look at whether one already exists.**

This mechanism has fired for real twice. `MS-C-223`: `runs/a1_v3_calibration.json` was
overwritten by the v3b run, so A1 v3's numbers are permanently uncheckable -- and that
row is still in `BRANCH_STATUS_001` as a caveat nobody could remove. And again during
this audit round.

The practical shape of the danger is not exotic. Re-run a long sweep, have it die early
this time, and the good record is replaced by a worse one. `MS-C-849` fixed resume so an
INTERRUPTED run keeps its progress; nothing protected a COMPLETED artefact from being
replaced by a worse re-run.

THE RULE
--------
`write_result` is atomic AND non-destructive. If the target exists and its bytes differ,
the existing file is first copied to `runs/.superseded/<stem>.<sha16><suffix>` -- named by
the CONTENT it holds, so archiving the same bytes twice is a no-op and no timestamp is
needed. Only then does the replace happen, and what was archived is printed.

Nothing is ever refused and no `--force` exists: a producer that wants to re-run should
re-run. What it must not do is destroy the evidence of the previous run in the process.

`runs/.superseded/` is git-tracked, unlike the local `runs/.snapshot/` safety net, because
a superseded artefact is part of the record of what was run.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUPERSEDED = ROOT / "runs" / ".superseded"


def archive_path(path) -> Path:
    """Where the CURRENT contents of `path` would be archived. Content-addressed."""
    path = Path(path)
    h = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    return SUPERSEDED / f"{path.stem}.{h}{path.suffix}"


def write_result(path, text: str, announce: bool = True) -> dict:
    """Atomic and non-destructive. Returns what happened, so a caller can report it.

    `{"wrote": bool, "unchanged": bool, "archived": str | None}`.
    """
    path = Path(path)
    payload = text.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)

    archived = None
    if path.exists():
        if path.read_bytes() == payload:
            if announce:
                print(f"   {path.name}: byte-identical to the existing artefact, "
                      "nothing replaced")
            return {"wrote": False, "unchanged": True, "archived": None}
        dest = archive_path(path)
        if not dest.exists():
            SUPERSEDED.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(path.read_bytes())
        archived = dest.relative_to(ROOT).as_posix()
        if announce:
            print(f"   {path.name}: the previous artefact differs and is archived at "
                  f"{archived}")

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(payload)
    os.replace(tmp, path)
    return {"wrote": True, "unchanged": False, "archived": archived}
