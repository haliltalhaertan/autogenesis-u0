"""The edge-support theorem: which edges can move at all, decided per edge.

`L-490` proved that a **triangle-free** rest scaffold can never form a bond. That is a
whole-graph statement and it is the weakest case of something sharper, pointed out in an
external review and stated here as a candidate theorem:

    Let C be the candidate graph of a rest scaffold and m = formation_support_min. For an
    edge e = (i,j) of C write

        s_C(e) = |N_C(i) INTERSECT N_C(j)|

    -- the number of triangles of C containing e. If s_C(e) < m then e can NEVER form.
    Hence in any periodic orbit the bond bit of e is constant, and

        E_dynamic  SUBSET OF  { e in C : s_C(e) >= m }.

    Proof. Formation of (i,j) requires support[i,j] >= m, and support[i,j] counts k with
    cand[i,k], cand[j,k] and active[k] -- so it is at most |N_C(i) INTERSECT N_C(j)| for
    every activity word. If that is below m, the test fails always. An edge that never
    forms has a non-increasing bond bit; along a cycle it returns to itself and is
    therefore constant. QED

`L-490` is the case m = 1 with s_C(e) = 0 for every e. This is strictly stronger: it names
WHICH edges can move in a graph that has triangles somewhere but not everywhere, and it is
the step that makes the one-triangle case tractable -- with only three edges of positive
support, everything else in such a scaffold is frozen by the theorem and what remains is a
three-edge subsystem.

VERIFIED THE WAY THE PROOF ASSERTS IT, not by correlation. Two exhaustive checks over
every state of every scaffold class:

  * no edge with s_C(e) < m ever goes from absent to present, at any state;
  * the edges that actually flicker inside an attractor all have s_C(e) >= m.

And a non-vacuity arm, because a bound nothing reaches proves nothing: some edge with
s_C(e) >= m must actually form somewhere, or the containment is empty on both sides.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from autogenesis.genome import U0                              # noqa: E402
from rest_scaffold_orbits import (Scaffold, candidate_edges,   # noqa: E402
                                  check_lemma, scaffold_tri)

MAX_N = 6

CLAIM = ("For every rest scaffold class of 3 to 6 nodes: no edge whose candidate-graph "
         "support is below formation_support_min ever forms, at any state; and every "
         "edge that flickers inside an attractor has support at or above it. The set of "
         "movable edges is bounded per edge, not per graph.")

WOULD_OVERTURN = ("One edge of support below m forming at one state -- that kills the "
                  "theorem. Or every edge of support at or above m being frozen too, "
                  "which would make the containment vacuous and the bound useless even "
                  "though technically true.")


def criterion(triple) -> bool:
    """True when edge k FORMED across this transition: absent before, present after."""
    before, after, k = triple
    return (k not in before) and (k in after)


CONTROLS = [
    ("absent then present -> formed", ({0}, {0, 1}, 1), True),
    ("present then present -> not formed", ({0, 1}, {0, 1}, 1), False),
    ("present then absent -> not formed, that is a break", ({0, 1}, {0}, 1), False),
    ("absent then absent -> not formed", ({0}, {0}, 1), False),
    ("another edge forming does not count for this one", ({0}, {0, 2}, 1), False),
]


def _classes(maxn: int):
    g = U0()
    P = scaffold_tri((-2, -1, 0, 1, 2), 5)
    check_lemma(g, P)
    E = candidate_edges(g, P)
    adj: dict[int, set] = {i: set() for i in range(len(P))}
    for i, j in E:
        adj[i].add(j)
        adj[j].add(i)
    lvl = {1: {frozenset({v}) for v in range(len(P))}}
    for s in range(2, maxn + 1):
        lvl[s] = {S | {v} for S in lvl[s - 1]
                  for v in {w for u in S for w in adj[u]} - S}
    out = {}
    for size in range(3, maxn + 1):
        for S in sorted(lvl[size], key=sorted):
            idx = sorted(S)
            pos = {v: k for k, v in enumerate(idx)}
            e = sorted((pos[i], pos[j]) for i, j in E if i in pos and j in pos)
            can = min(tuple(sorted(tuple(sorted((p[i], p[j]))) for i, j in e))
                      for p in itertools.permutations(range(size)))
            out.setdefault(can, idx)
    return g, P, out


def _support(sc) -> list[int]:
    """s_C(e) for each candidate edge, from the candidate graph alone."""
    nb: dict[int, set] = {v: set() for v in range(sc.n)}
    for i, j in sc.edges:
        nb[i].add(j)
        nb[j].add(i)
    return [len(nb[i] & nb[j]) for i, j in sc.edges]


def universe() -> tuple[int, str]:
    g, P, cls = _classes(MAX_N)
    n = 0
    for idx in cls.values():
        sc = Scaffold(g, P[idx], candidate_edges(g, P[idx]))
        n += (1 << (sc.n + sc.E)) * sc.E
    return n, ("every (state, edge) pair of every rest scaffold class of 3..6 nodes; the "
               "theorem is per edge, so the universe is per edge too")


def measure() -> tuple[str, str]:
    g, P, cls = _classes(MAX_N)
    m = g.topology.formation_support_min
    violations, formed_low, formed_high = [], 0, 0
    flicker_low, flicker_high = 0, 0
    edges_low = edges_high = 0

    for key, idx in cls.items():
        X = P[idx]
        sc = Scaffold(g, X, candidate_edges(g, X))
        sup = _support(sc)
        edges_low += sum(1 for s in sup if s < m)
        edges_high += sum(1 for s in sup if s >= m)
        total = 1 << (sc.n + sc.E)
        nxt = np.empty(total, dtype=np.int64)
        for code in range(total):
            nxt[code] = sc.step_code(code)

        # arm 1 -- formation, over every state
        for code in range(total):
            bb = code >> sc.n
            nb = int(nxt[code]) >> sc.n
            before = {k for k in range(sc.E) if bb >> k & 1}
            after = {k for k in range(sc.E) if nb >> k & 1}
            for k in range(sc.E):
                if criterion((before, after, k)):
                    if sup[k] < m:
                        violations.append((len(idx), k, sup[k], code))
                        formed_low += 1
                    else:
                        formed_high += 1

        # arm 2 -- flickering inside attractors
        colour = np.zeros(total, dtype=np.int8)
        for start in range(total):
            if colour[start]:
                continue
            path, s = [], start
            while colour[s] == 0:
                colour[s] = 1
                path.append(s)
                s = int(nxt[s])
            if colour[s] == 1:
                cyc = path[path.index(s):]
                seen = [set(k for k in range(sc.E) if (c >> sc.n) >> k & 1) for c in cyc]
                always = set.intersection(*seen)
                ever = set().union(*seen)
                for k in ever - always:
                    if sup[k] < m:
                        flicker_low += 1
                        violations.append((len(idx), k, sup[k], "in a cycle"))
                    else:
                        flicker_high += 1
            for c in path:
                colour[c] = 2

    print(f"   formation_support_min m = {m}")
    print(f"   candidate edges with s_C(e) <  m : {edges_low:>6}")
    print(f"   candidate edges with s_C(e) >= m : {edges_high:>6}")
    print(f"   formation events on LOW-support edges  : {formed_low:>6}   "
          "(the theorem says 0)")
    print(f"   formation events on HIGH-support edges : {formed_high:>6}   "
          "(non-vacuity: must be > 0)")
    print(f"   flickering edges in cycles, LOW support : {flicker_low:>6}   "
          "(the theorem says 0)")
    print(f"   flickering edges in cycles, HIGH support: {flicker_high:>6}")
    if violations:
        print(f"   VIOLATIONS: {violations[:5]}")

    ok = not violations and formed_high > 0
    return (f"{formed_low} formations and {flicker_low} flickers on edges below the "
            f"support bound; {formed_high} formations above it",
            f"the claim needs both zeros and a non-vacuous positive arm "
            f"({'HOLDS' if ok else 'DOES NOT HOLD'})")
