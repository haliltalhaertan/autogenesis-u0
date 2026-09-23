"""A theorem candidate, tested by exhaustive enumeration rather than by correlation.

THE OBSERVATION. The exact census (`measurements/scaffold_census_exact.py`) finds object
attractors -- cycles with a persistent bond and a cycling one -- on `TRI_2x3` (10) and
`HEX_7` (36), and **zero** on `SQ_2x2`, `SQ_2x3` and `SQ_2x4`.

THE PROPOSED REASON, read off the rule rather than guessed. On a rest scaffold every
candidate pair sits at exactly `r0`, so `r <= R_break` always holds and retention reduces
to *bonded and an active endpoint*. Formation, though, requires
`support[i,j] >= formation_support_min`, and `support[i,j]` counts vertices `k` with
`cand[i,k]`, `cand[j,k]` and `active[k]` (`kernel.py`, the support loop). Together with
the `cand[i,j]` that formation also requires, `{i, j, k}` is a **triangle of the candidate
graph**. U0 has `formation_support_min = 1`.

    Theorem (candidate). On a rest scaffold whose candidate graph is triangle-free, and
    for any band, no bond can ever form. Hence B(t+1) is a subset of B(t) at every state,
    the bond set is non-increasing, and no attractor can have a non-constant bond set.

    Proof sketch. Formation of (i,j) needs cand[i,j] and some k with cand[i,k], cand[j,k]
    and active[k]; that k makes {i,j,k} a triangle in the candidate graph. If there are no
    triangles, support is 0 on every candidate pair and formation is impossible for every
    activity word. So B(t+1) is a subset of B(t). Along a cycle a non-increasing set
    returns to itself, so it is constant.

WHY THIS FILE DOES NOT MEASURE A CORRELATION. "Triangle-free scaffolds happened to have no
object" is three data points. What the proof actually asserts is stronger and is checkable
at every state: **no bond ever forms**. So the whole state space of each triangle-free
scaffold is enumerated and `B(t+1) subset of B(t)` is required at every one -- and on a
scaffold WITH triangles, a state where a bond does form must be exhibited, or the
mechanism named here is not the one doing the work.

The theorem is conditional on `formation_support_min >= 1`. At 0 the support term imposes
nothing and the argument collapses; that is stated rather than left implicit.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from autogenesis.genome import U0                              # noqa: E402
from rest_scaffold_orbits import (Scaffold, candidate_edges,   # noqa: E402
                                  check_lemma, scaffold_hex, scaffold_sq,
                                  scaffold_tri)

CASES = (("SQ_2x2", lambda: scaffold_sq(2, 2)),
         ("SQ_2x3", lambda: scaffold_sq(2, 3)),
         ("SQ_2x4", lambda: scaffold_sq(2, 4)),
         ("TRI_2x3", lambda: scaffold_tri((0, 1), 3)),
         ("HEX_7", scaffold_hex))

CLAIM = ("On every rest scaffold whose candidate graph is triangle-free, no bond forms at "
         "ANY of its states: B(t+1) is a subset of B(t) everywhere, exhaustively. On the "
         "scaffolds that do have triangles, states where a bond forms exist. The zero "
         "object count on the square scaffolds is this, not a coincidence.")

WOULD_OVERTURN = ("A single state on a triangle-free scaffold where a bond forms -- that "
                  "kills the theorem outright. Or no forming state anywhere on a "
                  "triangled scaffold, which would mean triangles are not what "
                  "distinguishes the two families and the mechanism named here is the "
                  "wrong one.")


def criterion(pair) -> bool:
    """True when a bond FORMED -- the new bond set is not a subset of the old."""
    before, after = pair
    return not set(after) <= set(before)


CONTROLS = [
    ("a bond appeared -> formed", ({(0, 1)}, {(0, 1), (1, 2)}), True),
    ("the set is unchanged -> not formed", ({(0, 1)}, {(0, 1)}), False),
    ("a bond was lost -> not formed", ({(0, 1), (1, 2)}, {(0, 1)}), False),
    ("everything lost -> not formed", ({(0, 1)}, set()), False),
    # the case a subset test must not confuse with a loss:
    ("one lost and one gained -> formed", ({(0, 1)}, {(1, 2)}), True),
]


def _sc(X):
    g = U0()
    return g, Scaffold(g, X, candidate_edges(g, X))


def universe() -> tuple[int, str]:
    n = 0
    for _, f in CASES:
        _, sc = _sc(f())
        n += 1 << (sc.n + sc.E)
    return n, ("every state of every scaffold listed, each visited once; complete "
               "enumeration, no sampling and no horizon")


def measure() -> tuple[str, str]:
    g = U0()
    if g.topology.formation_support_min < 1:
        return "VOID", ("formation_support_min is 0, so the support term imposes nothing "
                        "and the theorem's hypothesis does not apply to this genome")

    rows, violations, silent = [], [], []
    for name, f in CASES:
        X = f()
        lem = check_lemma(g, X)
        _, sc = _sc(X)
        total = 1 << (sc.n + sc.E)
        formed, first = 0, None
        for code in range(total):
            bb = code >> sc.n
            nb = sc.step_code(code) >> sc.n
            before = {k for k in range(sc.E) if bb >> k & 1}
            after = {k for k in range(sc.E) if nb >> k & 1}
            if criterion((before, after)):
                formed += 1
                if first is None:
                    first = code
        tri = lem["triangles"]
        rows.append((name, sc.n, sc.E, tri, total, formed, first))
        if tri == 0 and formed:
            violations.append(name)
        if tri > 0 and formed == 0:
            silent.append(name)

    print(f"   {'scaffold':<9}{'n':>3}{'E':>4}{'triangles':>11}{'states':>9}"
          f"{'states forming a bond':>23}")
    for name, n, E, tri, total, formed, first in rows:
        note = "" if formed == 0 else f"  (first at code {first})"
        print(f"   {name:<9}{n:>3}{E:>4}{tri:>11}{total:>9}{formed:>23}{note}")

    if violations:
        return (f"THEOREM FALSE on {violations}",
                "a triangle-free scaffold formed a bond; the hypothesis does not hold")
    if silent:
        return (f"MECHANISM NOT SHOWN on {silent}",
                "a triangled scaffold never formed a bond, so triangles are not what "
                "separates the families")
    free = [r for r in rows if r[3] == 0]
    tri = [r for r in rows if r[3] > 0]
    print(f"\n   {len(free)} triangle-free scaffold(s), "
          f"{sum(r[4] for r in free):,} states, 0 formed a bond -- exhaustive")
    print(f"   {len(tri)} triangled scaffold(s), every one has forming states")
    return (f"0 of {sum(r[4] for r in free):,} triangle-free states form a bond; "
            f"all {len(tri)} triangled scaffolds do",
            "the claim holds iff no triangle-free state forms and every triangled "
            "scaffold has one (HOLDS)")
