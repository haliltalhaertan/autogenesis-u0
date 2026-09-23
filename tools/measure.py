"""MS-C-880: the harness for ad-hoc measurements, built around the four ways they fail.

WHY THIS EXISTS
---------------
The lead's diagnosis, and it is sharper than mine was: *the gates look fine; the problem
is wrong measurement, incomplete measurement, misreading the result, and coding errors.*
Four classes, four different cures, and every one of them fired in a single session:

  WRONG        a `grep` for `refuse` matched the word in COMMENTS and reported five
               guards where there were none; `!= EXTINCT` counted `INIT_FAIL` as a live
               world; a search for `dt` matched `dtype=`.
  INCOMPLETE   a write-pattern list saw 17 of 40 canonical writers; a `ci.yml` parser
               never saw the `for t in ...` loop and silently dropped ten commands.
  MISREAD      `P2` was reported FALSIFIED when the observable fired before the
               phenomenon could occur, so the correct verdict was VOID -- the numbers
               were right and the sentence about them was wrong. That is the dangerous
               one, because nothing about the number looks suspicious.
  CODING       a heredoc turned `\\b` into a literal BACKSPACE byte and made a repair
               inert for four attempts, while the source listing looked correct.

The project already knows three of these -- `D-020` item 5 is the WRONG cure, `M4` is
the INCOMPLETE cure -- and applies them to gates written in code. It applied none of them
to a number typed into a conversation, which is where most reported numbers live.

WHAT A MEASUREMENT MUST DECLARE
--------------------------------
A module under `measurements/` defines:

  CLAIM            the sentence the number is meant to support
  WOULD_OVERTURN   what an observation would have to look like for the claim to be
                   false. If nothing could, the claim is not a measurement -- cures
                   MISREAD
  universe()       -> (n, how) the size of the population and HOW it was counted, by a
                   route independent of the criterion -- cures INCOMPLETE
  measure()        -> (value, detail) the number itself
  CONTROLS         [(label, input, must_match)] with at least one `False` -- an input
                   the criterion must REFUSE. Cures WRONG

The runner refuses a measurement that omits any of them, and records the result next to
the script so a number always has a path. It is a file, run by a runner, never a heredoc
-- which cures CODING by construction.

WHAT IT DOES NOT DO
-------------------
It cannot tell whether the claim is true. A measurement whose criterion is wrong in a way
its own controls do not probe will pass here and be wrong. This raises the floor; it does
not replace an independent reading.

Run:  ./.venv/Scripts/python.exe tools/measure.py                 # run all, record
      ./.venv/Scripts/python.exe tools/measure.py --check         # CI: declarations +
                                                                  # controls only, no
                                                                  # measuring, no recording
      ./.venv/Scripts/python.exe tools/measure.py <name>          # one measurement
"""
from __future__ import annotations

import importlib.util
import io
import sys
import traceback
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "measurements"
REQUIRED = ("CLAIM", "WOULD_OVERTURN", "CONTROLS", "universe", "measure")


def load(path: Path):
    spec = importlib.util.spec_from_file_location(f"m_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def check_declarations(mod) -> list[str]:
    missing = [r for r in REQUIRED if not hasattr(mod, r)]
    if missing:
        return [f"missing {', '.join(missing)}"]
    problems = []
    if not str(getattr(mod, "CLAIM", "")).strip():
        problems.append("CLAIM is empty")
    if not str(getattr(mod, "WOULD_OVERTURN", "")).strip():
        problems.append("WOULD_OVERTURN is empty -- a claim nothing could refute is "
                        "not a measurement")
    controls = list(getattr(mod, "CONTROLS", []))
    if not controls:
        problems.append("CONTROLS is empty")
    elif not any(not must for _, _, must in controls):
        problems.append("no NEGATIVE control -- every control expects a match, so the "
                        "criterion has never been shown to refuse anything")
    return problems


def run_controls(mod) -> list[str]:
    """Every control must come out as declared. A negative control that matches is the
    WRONG-measurement class caught before the number is believed."""
    bad = []
    for label, probe, must in mod.CONTROLS:
        try:
            got = bool(mod.criterion(probe))
        except Exception as exc:                                # noqa: BLE001
            bad.append(f"control {label!r} raised {type(exc).__name__}: {exc}")
            continue
        if got != must:
            bad.append(f"control {label!r}: criterion returned {got}, declared {must}")
    return bad


def run_one(path: Path, record: bool, controls_only: bool = False) -> tuple[bool, str]:
    out = io.StringIO()
    try:
        mod = load(path)
    except Exception:                                           # noqa: BLE001
        return False, f"import failed:\n{traceback.format_exc()}"

    problems = check_declarations(mod)
    if problems:
        return False, "declaration: " + "; ".join(problems)
    if not hasattr(mod, "criterion"):
        return False, "declaration: CONTROLS given but no `criterion` to run them against"

    bad = run_controls(mod)
    if bad:
        return False, "CONTROLS FAILED -- the criterion does not do what it claims:\n  " \
                      + "\n  ".join(bad)

    if controls_only:
        # The CI mode. Measuring is expensive -- `mechanical_axes_inert` screens 24 seeds
        # across 13 genomes -- and CI is not here to reproduce numbers, it is here to keep
        # the DECLARATIONS from rotting: every measurement still names its claim, still
        # says what would overturn it, and its controls still hold against the criterion
        # as the code stands today. That is the part that fails silently.
        return True, (f"declarations present, {len(mod.CONTROLS)} controls hold "
                      f"({sum(1 for _, _, m in mod.CONTROLS if not m)} negative)")

    with redirect_stdout(out):
        n, how = mod.universe()
        value, detail = mod.measure()
    body = (f"CLAIM           {mod.CLAIM}\n"
            f"WOULD OVERTURN  {mod.WOULD_OVERTURN}\n"
            f"UNIVERSE        {n}   ({how})\n"
            f"CONTROLS        {len(mod.CONTROLS)} declared, "
            f"{sum(1 for _, _, m in mod.CONTROLS if not m)} negative, all as declared\n"
            f"VALUE           {value}\n"
            f"DETAIL          {detail}\n")
    printed = out.getvalue().strip()
    if printed:
        body += "\n" + printed + "\n"

    if record:
        sys.path.insert(0, str(ROOT / "tools"))
        from _artefact import write_result                       # noqa: E402
        write_result(path.with_suffix(".out"), body, announce=False)
    return True, body


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    check = "--check" in sys.argv
    record = not check
    DIR.mkdir(exist_ok=True)
    paths = sorted(p for p in DIR.glob("*.py") if not p.name.startswith("_"))
    if args:
        paths = [p for p in paths if p.stem in args or p.name in args]
        if not paths:
            print(f"no measurement matching {args}")
            return 1

    print(f"MEASUREMENTS -- {len(paths)} in {DIR.relative_to(ROOT).as_posix()}/")
    print("  every one declares its universe, a negative control, and what would "
          "overturn it\n")
    failed = []
    for p in paths:
        ok, body = run_one(p, record, controls_only=check)
        print(f"  {'ok  ' if ok else 'FAIL'} {p.stem}")
        for line in body.splitlines():
            print(f"        {line}")
        print()
        if not ok:
            failed.append(p.stem)

    if failed:
        print(f"*** {len(failed)} of {len(paths)} FAILED: {', '.join(failed)} ***")
        return 1
    if check:
        print(f"OK: {len(paths)} of {len(paths)} measurements still declare a claim, "
              "something that\nwould overturn it, and controls that hold. NOTHING WAS "
              "MEASURED -- run without\n--check to reproduce the numbers.")
        return 0
    print(f"OK: {len(paths)} of {len(paths)} measurements ran with their controls "
          "holding.")
    print("This does NOT mean the claims are true -- a criterion wrong in a way its own")
    print("controls do not probe passes here. It means each number has a path, a "
          "denominator,\nand something it was shown to refuse.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
