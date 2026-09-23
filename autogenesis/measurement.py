"""Identity of the MEASUREMENT layer, kept separate from the integrator's.

D-025, as amended by CHECKPOINT_032 §4. `code_hash` in `simulator.py` identifies
the integrator -- "every file whose bytes can change a trajectory" -- and that is
the right meaning for it. The defect it was asked to fix is different: a
`runs/*.json` record stamps `code_hash` as though it identified the MEASUREMENT,
when `rules.py` (the band enumerator every sweep writes into its genome) and
`symmetry.py` (the C3 decision function) can change without moving it.

WHY THIS IS A NEW FILE AND NOT A FUNCTION IN simulator.py.
`simulator.py` is itself the sixth entry in `simulator._TRAJECTORY_FILES`, so
adding these functions there would change `simulator.py`'s bytes and therefore
change `code_hash` -- invalidating every existing stamp, which is the exact cost
D-025 was chosen to avoid (MS-C-285). Adding a NEW file changes no existing
file's bytes, so `code_hash` is untouched by construction.

IMPORT DISCIPLINE. This module imports only the standard library at import time.
`stamp()` needs `code_hash`, and imports `simulator` LAZILY, inside the function
body, so that this module can never participate in an import cycle even if
`simulator` later wants to import it.
"""
from __future__ import annotations
import ast
import hashlib
import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parent
_ROOT = _PKG.parent

# every file whose bytes can change a MEASUREMENT without changing a trajectory.
#
# MS-C-735: `measurement.py` was in NEITHER list and reached NO producer closure, so
# **the file that defines every identity in this project had none of its own.** A
# tool imports it as `from autogenesis import measurement`, which resolves to the
# PACKAGE, so `_first_party_deps` walked `autogenesis/__init__.py` and stopped. All
# four hashes could be edited to stand still while this file changed underneath them.
# It is the measurement layer's own code, so it belongs here; including itself is not
# circular, since the digest is over bytes on disk. Cost: `measurement_hash` moves
# once, which is the same one-time cost D-025 accepted when it was created.
_MEASUREMENT_FILES = ["detector.py", "measurement.py", "metrics.py", "rules.py",
                      "symmetry.py"]


def measurement_hash() -> str:
    """Identity of the measurement layer.

    Orthogonal to `code_hash` by construction: the two file lists are disjoint.
    A change here moves no trajectory, but it can change what a trajectory is
    SCORED as, which is why it needs its own identity.

    MS-C-858: framed by name and length, for the reason spelled out in
    `simulator.code_hash` -- an unframed concatenation identifies the concatenation and
    not the files, and a byte moved across a boundary collides.
    """
    h = hashlib.sha256()
    for name in sorted(_MEASUREMENT_FILES):
        data = (_PKG / name).read_bytes()
        h.update(f"{name}\x00{len(data)}\x00".encode("utf-8"))
        h.update(data)
    return h.hexdigest()[:16]


def tool_hash(path) -> str:
    """Identity of one tool's own bytes. Call as `tool_hash(__file__)`.

    A record stamped with this says which PRODUCER made it, not merely which
    engine ran underneath. `tools/` is deliberately not folded into
    `measurement_hash`: 62 files, most unrelated to any given measurement, and a
    hash over all of them would move on every unrelated edit and be ignored
    within a week (D-025).
    """
    return hashlib.sha256(Path(path).resolve().read_bytes()).hexdigest()[:16]


def _covered_elsewhere() -> set[Path]:
    """The files whose bytes `code_hash` and `measurement_hash` already identify."""
    from . import simulator                      # lazy: never an import cycle
    return {(_PKG / n).resolve()
            for n in simulator._TRAJECTORY_FILES + _MEASUREMENT_FILES}


def _first_party_deps(path: Path, seen: set[Path] | None = None) -> set[Path]:
    """The transitive closure of FIRST-PARTY modules `path` imports.

    MS-C-724: `tool_hash` hashes one file. Fifteen stamped tools import their result
    logic from other `tools.*` modules, so editing `tools/storage_score.py` moved no
    hash in any record `tools/calibrate_storage_score.py` had written -- the stamp
    said which producer ran and was wrong about what the producer WAS.

    Resolution mirrors how the tools actually import: a dotted name `a.b` against the
    repo root (`autogenesis.metrics`, `tools.storage_score`), and a bare name against
    the importing file's own directory, because every tool does
    `sys.path.insert(0, str(Path(__file__).parent))` before importing its siblings
    (`from _canonical_io import ...`). Names that resolve to neither are third-party
    or stdlib and are identified by `env()` instead, not by bytes.
    """
    path = Path(path).resolve()
    seen = set() if seen is None else seen
    if path in seen or not path.is_file():
        return seen
    seen.add(path)
    try:
        tree = ast.parse(path.read_bytes())
    except SyntaxError:                          # not ours to interpret; bytes still count
        return seen
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            # MS-C-735: `from autogenesis import measurement` names the PACKAGE in
            # `node.module` and the MODULE in `node.names`. Resolving only the former
            # walked `autogenesis/__init__.py` and never reached `measurement.py`.
            # Each imported name is tried as a submodule too; names that are classes
            # or functions simply fail to resolve to a file and are dropped.
            names |= {f"{node.module}.{a.name}" for a in node.names}
    for name in sorted(names):
        parts = name.split(".")
        for base in (_ROOT, path.parent):
            cand = base.joinpath(*parts)
            for f in (cand.with_suffix(".py"), cand / "__init__.py"):
                if f.is_file():
                    _first_party_deps(f, seen)
    return seen


def producer_hash(path) -> dict:
    """Identity of a tool INCLUDING the first-party modules it imports.

    Returns `{"hash": ..., "files": n}`: the digest, and how many files it spans, so
    a record says whether the closure was found at all rather than looking identical
    to `tool_hash` when resolution silently failed.

    Orthogonal to `code_hash` and `measurement_hash` by construction: files those two
    already cover are excluded here, so the three digests partition the bytes that
    can change a record. `tests/test_provenance.py` fails if that partition breaks.
    """
    covered = _covered_elsewhere()
    files = sorted(f for f in _first_party_deps(Path(path)) if f not in covered)
    h = hashlib.sha256()
    for f in files:
        try:
            rel = f.relative_to(_ROOT).as_posix()
        except ValueError:
            rel = f.name
        # MS-C-858: this already framed by name, which is why it was NOT vulnerable to
        # the boundary move that collides `code_hash`. The length is added so all three
        # digests frame the same way and none is the weak one by accident.
        data = f.read_bytes()
        h.update(f"{rel}\x00{len(data)}\x00".encode("utf-8"))
        h.update(data)
        h.update(b"\x00")
    return {"hash": h.hexdigest()[:16], "files": len(files)}


def env() -> dict:
    """The interpreter and the compiled libraries the numbers were produced with.

    `OB-107` asks for it explicitly: no hash over source bytes can distinguish a
    result computed under numpy 2.5.2 from the same source under 2.3, and the
    difference is capable of moving a floating-point trajectory.
    """
    out = {"python": ".".join(str(v) for v in sys.version_info[:3])}
    for mod in ("numpy", "numba"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:                        # noqa: BLE001 -- absence IS the record
            out[mod] = None
    return out


def stamp(tool_path=None) -> dict:
    """The identities a run record should carry.

    `tool_path` is a tool's `__file__`; omit it for records not produced by a
    tool. `simulator` is imported here rather than at module scope -- see the
    import-discipline note at the top of this file.

    `tool_hash` keeps its old meaning -- the bytes of that one file -- so stamps
    written before MS-C-724 stay comparable with stamps written after. The closure
    is a NEW field beside it rather than a redefinition of the old one; that is the
    same move, and for the same reason, as D-025 adding `measurement_hash` instead
    of widening `code_hash`.
    """
    from . import simulator                      # lazy: never an import cycle
    out = {"code_hash": simulator.code_hash(),
           "measurement_hash": measurement_hash(),
           "env": env()}
    if tool_path is not None:
        ph = producer_hash(tool_path)
        out["tool_hash"] = tool_hash(tool_path)
        out["producer_hash"] = ph["hash"]
        out["producer_files"] = ph["files"]
    return out
