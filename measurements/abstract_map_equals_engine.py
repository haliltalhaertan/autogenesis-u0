"""At rest the discrete map depends only on the candidate GRAPH -- proved by agreement.

WHY THIS IS THE GATE FOR EVERYTHING NEXT. `L-497` reduces the one-triangle case to three
edges, which makes *one triangle is never enough* a finite question rather than a
statement about whole graphs. But the family enumerated so far is only what the exact
triangular lattice can REALIZE, and a theorem about one-triangle scaffolds needs arbitrary
graphs -- trees hanging off the triangle, larger `n`, shapes no lattice patch produces.

The way to get there is the observation that at rest the map is a function of the
candidate graph alone: every candidate pair sits at exactly `r0`, so `r <= R_break` always
(retention's distance test never fires), `r0 > repulsion_cutoff` so no pair repels, and
the spring force is `k*(r0 - r0) = 0`. Positions then enter the step only by deciding which
pairs are candidates. Everything else -- the node rule, retention, eligibility, `support`,
the `B_support_orbit` allocator -- reads `cand`, `bonds` and `active` and nothing else.

**That observation is not assumed here, it is gated.** A map re-implemented from the
kernel is a new place for defects (`D-020` item 8), so this file re-implements it and then
requires it to agree with the production engine **bit for bit, on every state of every one
of the 36 realizable scaffold classes**. If it agrees everywhere, the abstract map may be
used on graphs no lattice realizes. If it disagrees anywhere, it may not be used at all,
and the disagreement is the finding.
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

CLAIM = ("A map re-implemented from the rule and reading only the candidate graph agrees "
         "with the production engine bit for bit on every state of every realizable rest "
         "scaffold class. At rest the discrete dynamics are a function of the graph "
         "alone, so the abstract map may be used on graphs no lattice realizes.")

WOULD_OVERTURN = ("One state anywhere where the two maps differ. The re-implementation "
                  "would then be wrong, or positions would enter the step by some route "
                  "this reasoning misses, and either way nothing may be built on the "
                  "abstract map.")


def abstract_step(n: int, cand: list[set], bonds: int, active: int, g) -> tuple[int, int]:
    """One step from (bond bits, activity bits), reading only the candidate graph.

    Transcribed from `kernel._step` with the rest-scaffold simplifications applied and
    named: `r <= R_break` always holds so retention's distance test is dropped, and no
    pair is inside the repulsion cutoff so no force term survives.
    """
    t = g.topology
    edges = [(i, j) for i in range(n) for j in cand[i] if i < j]
    idx = {e: k for k, e in enumerate(edges)}
    B = {e for e, k in idx.items() if bonds >> k & 1}
    A = {i for i in range(n) if active >> i & 1}

    nb_of = [set() for _ in range(n)]
    for i, j in B:
        nb_of[i].add(j)
        nb_of[j].add(i)

    # --- node rule: OLD bonds, OLD activity, synchronous
    new_A = set()
    for i in range(n):
        d = len(nb_of[i])
        if d == 0:
            continue
        a = len(nb_of[i] & A)
        lo, hi = ((t.survival_lo, t.survival_hi) if i in A
                  else (t.birth_lo, t.birth_hi))
        if lo <= a / d <= hi:
            new_A.add(i)

    def support(i, j):
        return len(cand[i] & cand[j] & A)

    # --- retention: the distance test cannot fire at rest
    kept = {(i, j) for (i, j) in B
            if (not t.retention_requires_active_endpoint or i in A or j in A)
            and support(i, j) >= t.retention_support_min}
    kdeg = [0] * n
    for i, j in kept:
        kdeg[i] += 1
        kdeg[j] += 1
    olddeg = [len(nb_of[i]) for i in range(n)]

    elig = {(i, j) for (i, j) in edges
            if (i, j) not in B
            and (not t.formation_requires_active_endpoint or i in A or j in A)
            and olddeg[i] < t.valence and olddeg[j] < t.valence
            and support(i, j) >= t.formation_support_min}

    formed = set()
    if t.allocator == "B_support_orbit":
        acc = set()
        for p in range(n):
            rem = t.valence - kdeg[p]
            for s in range(n, -1, -1):
                tier = [q for q in range(n)
                        if (min(p, q), max(p, q)) in elig and p != q
                        and support(p, q) == s]
                if not tier:
                    continue
                if len(tier) <= rem:
                    acc |= {(p, q) for q in tier}
                    rem -= len(tier)
                else:
                    break
        formed = {(i, j) for (i, j) in elig if (i, j) in acc and (j, i) in acc}
    else:
        raise NotImplementedError(f"allocator {t.allocator!r} is not transcribed here")

    nb_bits = 0
    for e in kept | formed:
        nb_bits |= 1 << idx[e]
    return nb_bits, sum(1 << i for i in new_A)


def criterion(pair) -> bool:
    """True when the two maps agree on this state."""
    a, b = pair
    return a == b


CONTROLS = [
    ("identical outputs -> agree", ((3, 5), (3, 5)), True),
    ("bond bits differ -> disagree", ((3, 5), (2, 5)), False),
    ("activity bits differ -> disagree", ((3, 5), (3, 4)), False),
    ("both differ -> disagree", ((3, 5), (2, 4)), False),
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


def universe() -> tuple[int, str]:
    g, P, cls = _classes(MAX_N)
    n = 0
    for idx in cls.values():
        sc = Scaffold(g, P[idx], candidate_edges(g, P[idx]))
        n += 1 << (sc.n + sc.E)
    return n, ("every state of every realizable rest scaffold class of 3..6 nodes; both "
               "maps are run on each and compared bit for bit")


def measure() -> tuple[str, str]:
    g, P, cls = _classes(MAX_N)
    checked, bad, examples = 0, 0, []
    for key, idx in cls.items():
        X = P[idx]
        sc = Scaffold(g, X, candidate_edges(g, X))
        cand = [set() for _ in range(sc.n)]
        for i, j in sc.edges:
            cand[i].add(j)
            cand[j].add(i)
        for code in range(1 << (sc.n + sc.E)):
            bb, ab = sc.split(code)
            mine = abstract_step(sc.n, cand, bb, ab, g)
            engine = sc.split(sc.step_code(code))
            checked += 1
            if not criterion((mine, engine)):
                bad += 1
                if len(examples) < 3:
                    examples.append((sc.n, sc.E, code, mine, engine))

    print(f"   scaffold classes compared    {len(cls):>10}")
    print(f"   states compared              {checked:>10}")
    print(f"   states where the maps differ {bad:>10}")
    for n, E, code, mine, eng in examples:
        print(f"     n={n} E={E} code={code}: mine {mine}, engine {eng}")
    return (f"{bad} of {checked} states disagree",
            f"the claim needs zero "
            f"({'HOLDS -- the abstract map may be used on unrealizable graphs' if bad == 0 else 'DOES NOT HOLD -- it may not be used at all'})")
