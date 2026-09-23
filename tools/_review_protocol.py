"""The sealed follow-up protocol, and the two things `tools/preflight.py` demands.

`MS-C-733`: three reproduction tools written this session failed the preflight ratchet
on the same two mechanical checks, and the ratchet was right both times.

  * **protocol hash** -- `D-015`: pre-registration is enforced by a hash, not by
    intent. A reproduction has a natural pre-registration: the report it reproduces
    named every number before this project measured any of them. Freezing that report
    as `protocols/REVIEW_FOLLOWUP_20260828.json` and sealing it is what makes an
    agreement checkable instead of asserted. `verify_seal()` refuses to run against an
    unsealed or altered statement.

  * **streams to disk** -- a result that lives only in memory is lost when one worker
    raises, and two of these tools run for five to seven minutes. `Stream` appends one
    JSON object per line and flushes on every record, per the HANDOFF standing rule.
    This is NOT the artefact-truncation defect `MS-C-705` fixed: that was a canonical
    RESULT file opened `"w"` at the start of a scan and left truncated. These are
    working streams beside the canonical result, which is still written atomically at
    the end, and they carry `.stream.jsonl` in the name so the two cannot be confused.

`compare()` is the decision rule the protocol names: CONFIRMED, EXTENDED or CONTESTED,
with both numbers printed and neither silently preferred.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols" / "REVIEW_FOLLOWUP_20260828.json"


def check_protocol() -> str:
    """The frozen protocol's sha256, or a refusal to run. Never a warning.

    Named `check_protocol` because that is this repo's name for the shared checker
    and `tools/preflight.py` recognises it; several tools already import one.
    """
    side = PROTOCOL.with_suffix(".sha256")
    want = side.read_text(encoding="utf-8").split()[0]
    got = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    if got != want:
        raise SystemExit(
            f"PROTOCOL SEAL BROKEN\n  frozen: {want}\n  now:    {got}\n"
            "  Refusing to run: a reproduction measured against an unsealed\n"
            "  statement of the claims is not a reproduction (D-015).")
    return got


def claims() -> dict:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))["claims"]


class Stream:
    """One JSON object per line, flushed on every record.

    `MS-C-849`, external audit Bulgu 05. This used to open `"w"` unconditionally, and
    every one of the fourteen tools constructs it AFTER calling `resume()`. So a resumed
    run read the completed work into memory and then **truncated the file that held it**,
    while the loops that reuse a completed key `continue` past it without writing it
    back. Measured on the real `Stream` and the real `resume()`: 80 completed bands on
    disk, resume, five more, crash -- and the file then holds **five**. The eighty that
    were re-read are destroyed, silently, by the feature whose whole purpose is not to
    lose work. The next run recomputes them, which for `outcome_entropy` is about
    4,000 s.

    Passing `identity` makes the stream APPEND when the file on disk belongs to the same
    experiment, by the same test `resume()` uses, so nothing already computed is ever
    unlinked. Without `identity` the old truncating behaviour is kept; when the identity
    does NOT match, the file is truncated on purpose, because discarding a partial from a
    different run is `resume()`'s existing and correct decision.

    A second `start` record would make `resume()` treat the file as mixed and discard all
    of it, so on the append path `start` is rewritten as `resumed`, an event `resume()`
    ignores. Which mode was chosen is printed.
    """

    def __init__(self, path: Path, identity: dict | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        kept = _resumable_records(self.path, identity) if identity else 0
        self.appending = kept > 0
        self._fh = self.path.open("a" if self.appending else "w",
                                  encoding="utf-8", newline="\n")
        self.n = 0
        if self.appending:
            print(f"   stream: APPENDING to {self.path.name} -- "
                  f"{kept} record(s) already on disk are kept, not truncated")

    def write(self, rec: dict) -> None:
        if self.appending and rec.get("event") == "start":
            rec = {**rec, "event": "resumed"}
        self._fh.write(json.dumps(rec) + "\n")
        self._fh.flush()
        self.n += 1

    def close(self) -> None:
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


_INT_KEY = re.compile(r"-?\d+")


def _restore_int_keys(obj, restored: list):
    """JSON has no integer keys, so `{0: n}` comes back as `{"0": n}`.

    `MS-C-848`, external audit Bulgu 08. A producer builds `by_passive_internal_edges`
    with integer keys; the comparison reads it with `.get(0)`; after a resume that is
    `None`. Measured end to end: the same claim reads **CONFIRMED on a fresh run and
    CONTESTED with `measured: None` on a resumed one**. The earlier repair spelled
    `{int(k): v for ...}` at the one call site that was noticed, which leaves the next
    consumer to rediscover it -- and two sites in the very same function were left.

    So the inverse is applied ONCE, at the boundary where the round trip happened, for
    all fourteen tools that resume. A dict is restored only when EVERY key is a decimal
    integer string, which is the only case where the conversion is unambiguous, and what
    was restored is printed rather than done quietly.
    """
    if isinstance(obj, list):
        return [_restore_int_keys(v, restored) for v in obj]
    if not isinstance(obj, dict):
        return obj
    out = {k: _restore_int_keys(v, restored) for k, v in obj.items()}
    if out and all(isinstance(k, str) and _INT_KEY.fullmatch(k) for k in out):
        restored.append(len(out))
        return {int(k): v for k, v in out.items()}
    return out


def _scan(path: Path):
    """Read one stream. Returns `(head, arms, runs, restored, status)` with status one of
    `ok`, `mixed`, `unreadable`.

    `MS-C-849`: factored out so `resume()` and `Stream()` decide from the SAME reading.
    Two functions answering "is this file mine?" by separate logic is how a stream gets
    appended to when it should have been discarded.

    A restart re-opens the stream from scratch, so a `start` record after other records
    cannot happen; if it does, the file is mixed and the earlier content is not this
    run's. An appending restart writes `resumed` instead, which is ignored here.
    """
    head, arms, runs, restored = None, {}, {}, []
    try:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            ev = rec.get("event")
            if ev == "start":
                if head is not None:
                    return head, arms, runs, restored, "mixed"
                head = rec
            elif ev == "arm":
                arms[rec["arm"]] = _restore_int_keys(rec["result"], restored)
            elif ev == "run":
                runs[rec["key"]] = _restore_int_keys(rec["row"], restored)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None, {}, {}, [], "unreadable"
    return head, arms, runs, restored, "ok"


def _resumable_records(path: Path, identity: dict) -> int:
    """How many completed records on disk belong to THIS experiment.

    Zero means the file must not be appended to: absent, unreadable, mixed, headless, or
    written by a different run. `Stream` uses this to choose append over truncate, so the
    only file it ever appends to is one `resume()` would itself have reused.
    """
    path = Path(path)
    if not path.exists():
        return 0
    head, arms, runs, _restored, status = _scan(path)
    if status != "ok" or head is None:
        return 0
    if any(head.get(k) != v for k, v in identity.items()):
        return 0
    return len(arms) + len(runs)


def resume(path: Path, identity: dict) -> dict:
    """Completed arms from an interrupted run, but ONLY under an identical identity.

    A partial file is reusable exactly when the run that wrote it was the same
    experiment: same protocol seal, same `code_hash`, `measurement_hash`, `tool_hash`
    and `producer_hash`. Anything else and the partial is DISCARDED rather than
    mixed in -- reusing results computed by different bytes is the defect class this
    project keeps finding, and a resume feature is a very comfortable place for it to
    hide. What was resumed is printed, never assumed.
    """
    path = Path(path)
    if not path.exists():
        return {}
    head, arms, runs, restored, status = _scan(path)
    if status == "unreadable":
        print(f"   resume: {path.name} is not readable as JSONL -- starting fresh")
        return {}
    if status != "ok" or head is None:
        return {}
    mismatch = [k for k, v in identity.items() if head.get(k) != v]
    if mismatch:
        print(f"   resume: DISCARDING {len(arms)} completed arm(s) from "
              f"{path.name} -- identity differs in {', '.join(sorted(mismatch))}")
        return {}
    if arms or runs:
        parts = []
        if arms:
            parts.append(f"{len(arms)} arm(s) ({', '.join(sorted(arms))})")
        if runs:
            parts.append(f"{len(runs)} run(s)")
        print(f"   resume: reusing {' and '.join(parts)} from {path.name} "
              "-- identity matches exactly")
        if restored:
            # MS-C-848: never silent. A resumed run that reads a key the fresh run
            # wrote is the whole point, and the reader should see it happen.
            print(f"   resume: restored integer keys in {len(restored)} dict(s) "
                  f"({sum(restored)} keys) that JSON had turned into strings")
    out = dict(arms)
    if runs:
        out["runs"] = runs
    return out


def compare(label: str, claimed, measured, wider_scope: bool = False) -> dict:
    """The protocol's decision rule, applied and printed.

    CONFIRMED  measured == claimed.
    EXTENDED   the measurement covers a strictly wider scope and agrees where they
               overlap -- the caller asserts the overlap with `wider_scope=True`.
    CONTESTED  anything else. Both numbers are printed and neither is preferred.
    """
    # Sets and lists are compared as SETS so an ordering or container difference is
    # never mistaken for a disagreement (MS-C-734: a string vs a one-element list
    # produced a CONTESTED verdict on a claim that agreed).
    if isinstance(claimed, (set, frozenset)) or isinstance(measured, (set, frozenset)):
        claimed, measured = set(claimed), set(measured)
    if claimed == measured:
        verdict = "CONFIRMED"
    elif wider_scope:
        verdict = "EXTENDED"
    else:
        verdict = "CONTESTED"
    mark = {"CONFIRMED": "OK ", "EXTENDED": "EXT", "CONTESTED": "***"}[verdict]
    print(f"     {mark} {label:<46} claimed {claimed!s:<28} measured {measured!s}")

    def jsonable(v):
        # A set is not JSON serializable and the record must survive the write --
        # the first version compared correctly and then crashed the artefact.
        return sorted(v) if isinstance(v, (set, frozenset)) else v

    return {"label": label, "claimed": jsonable(claimed),
            "measured": jsonable(measured), "verdict": verdict}
