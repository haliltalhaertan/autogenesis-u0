"""The census recounted after a canonicalization bug: every count in L-491/L-492 was wrong.

THE BUG, found by an external review and reproduced here before anything was changed.
`scaffold_family_census.py` and `scaffold_writability.py` deduped scaffolds with

    can = min(frozenset(...) for p in itertools.permutations(range(size)))

and **`frozenset` comparison in Python is the SUBSET relation, not a lexicographic
order**. `min` over a partial order returns whichever element survived its pairwise
comparisons, which depends on generation order, so two labellings of the same graph can
produce different "canonical" values. Demonstrated on `P4`: `[(0,1),(1,2),(2,3)]` and
`[(1,2),(2,0),(0,3)]` are the same graph and came out different.

The independent arithmetic check is decisive on its own: connected unlabelled simple
graphs on 3, 4, 5 and 6 nodes number 2, 6, 21 and 112, so **at most 141** isomorphism
classes exist in that range. `L-491` reported **443**. A count above the ceiling is not a
count of isomorphism classes.

WHAT THIS DOES AND DOES NOT TOUCH. The duplicates were real scaffolds, correctly
enumerated and correctly censused, so nothing that is a statement about states survives
or falls with them: `L-490`'s theorem, its exhaustive verification, `L-493`'s
`0 of 2,318,880` reduction check and `L-494`'s `0` robust bits are all unaffected -- a
duplicated zero is still zero. **Every COUNT and every RATE in `L-491` and `L-492` is
wrong**, and the existence results are the only part of them that stands.

The canonical form here is `min` over **sorted tuples**, which are totally ordered, and
the count is checked against the 141 ceiling rather than trusted.
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
CEILING = 2 + 6 + 21 + 112          # connected unlabelled simple graphs, n = 3..6


def canon(edges, n):
    """A real canonical form: min over SORTED TUPLES, which are totally ordered."""
    return min(tuple(sorted(tuple(sorted((p[i], p[j]))) for i, j in edges))
               for p in itertools.permutations(range(n)))


def bad_canon(edges, n):
    """The form L-491 used. Kept so the control can show it failing."""
    return min(frozenset(tuple(sorted((p[i], p[j]))) for i, j in edges)
               for p in itertools.permutations(range(n)))


CLAIM = (f"Deduped by a real canonical form, the connected rest scaffolds of 3 to "
         f"{MAX_N} nodes number far fewer than the 443 reported in L-491, and at most "
         f"{CEILING} -- the count of connected unlabelled simple graphs in that range. "
         "The existence of writable topological bits survives; every count does not.")

WOULD_OVERTURN = ("A recount still above the 141 ceiling, which would mean the new "
                  "canonical form is broken too; or the corrected census finding NO "
                  "writable topological bit, which would make L-492 an artefact of the "
                  "duplication rather than merely miscounted.")


def criterion(pair) -> bool:
    """True when two edge lists of the same size get the same canonical form."""
    (e1, n1), (e2, n2) = pair
    return n1 == n2 and canon(e1, n1) == canon(e2, n2)


CONTROLS = [
    ("two labellings of P4 -> same class",
     (([(0, 1), (1, 2), (2, 3)], 4), ([(1, 2), (2, 0), (0, 3)], 4)), True),
    ("a path and a star on 4 nodes -> different classes",
     (([(0, 1), (1, 2), (2, 3)], 4), ([(0, 1), (0, 2), (0, 3)], 4)), False),
    ("a triangle and a path on 3 nodes -> different classes",
     (([(0, 1), (1, 2), (0, 2)], 3), ([(0, 1), (1, 2)], 3)), False),
    ("identical input -> same class",
     (([(0, 1), (1, 2)], 3), ([(0, 1), (1, 2)], 3)), True),
]


def _patch():
    g = U0()
    P = scaffold_tri((-2, -1, 0, 1, 2), 5)
    check_lemma(g, P)
    E = candidate_edges(g, P)
    adj: dict[int, set] = {i: set() for i in range(len(P))}
    for i, j in E:
        adj[i].add(j)
        adj[j].add(i)
    return g, P, E, adj


def _classes(maxn: int):
    g, P, E, adj = _patch()
    lvl = {1: {frozenset({v}) for v in range(len(P))}}
    for s in range(2, maxn + 1):
        lvl[s] = {S | {v} for S in lvl[s - 1]
                  for v in {w for u in S for w in adj[u]} - S}
    good, bad = {}, set()
    for size in range(3, maxn + 1):
        for S in sorted(lvl[size], key=sorted):
            idx = sorted(S)
            pos = {v: k for k, v in enumerate(idx)}
            e = sorted((pos[i], pos[j]) for i, j in E if i in pos and j in pos)
            good.setdefault(canon(e, size), idx)
            bad.add(bad_canon(e, size))
    return g, P, good, len(bad)


def _analyse(g, X):
    sc = Scaffold(g, X, candidate_edges(g, X))
    total = 1 << (sc.n + sc.E)
    nxt = np.empty(total, dtype=np.int64)
    for code in range(total):
        nxt[code] = sc.step_code(code)
    colour = np.zeros(total, dtype=np.int8)
    which = np.full(total, -1, dtype=np.int32)
    cycles: list[list[int]] = []
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
            cid = len(cycles)
            cycles.append(cyc)
            for c in cyc:
                which[c] = cid
        for c in path:
            colour[c] = 2
    for start in range(total):
        if which[start] >= 0:
            continue
        path, s = [], start
        while which[s] < 0:
            path.append(s)
            s = int(nxt[s])
        for c in path:
            which[c] = which[s]

    live, obj = set(), set()
    for cid, cyc in enumerate(cycles):
        sets, act = [], False
        for c in cyc:
            bb, ab = sc.split(c)
            sets.append(frozenset(k for k in range(sc.E) if bb >> k & 1))
            act = act or ab != 0
        if act:
            live.add(cid)
        if act and len(set(sets)) > 1:
            always = set.intersection(*map(set, sets))
            if always and (set().union(*map(set, sets)) - always):
                obj.add(cid)

    fwd: dict[int, set] = {cid: set() for cid in range(len(cycles))}
    for cid, cyc in enumerate(cycles):
        for code in cyc:
            bb, ab = sc.split(code)
            for v in range(sc.n):
                cur = bool(ab >> v & 1)
                for val in (True, False):
                    if val == cur:
                        continue
                    nab = (ab | (1 << v)) if val else (ab & ~(1 << v))
                    dst = int(which[(bb << sc.n) | nab])
                    if dst != cid:
                        fwd[cid].add(dst)
    rev_live = {(a, b) for a in fwd for b in fwd[a]
                if a < b and a in fwd.get(b, set()) and a in live and b in live}
    rev_obj = {(a, b) for a, b in rev_live if a in obj and b in obj}
    deg = np.zeros(sc.n, dtype=int)
    for i, j in sc.edges:
        deg[i] += 1
        deg[j] += 1
    return len(cycles), len(live), len(obj), len(rev_live), len(rev_obj), int(deg.max())


def universe() -> tuple[int, str]:
    _, _, good, _ = _classes(MAX_N)
    return len(good), (f"isomorphism classes of connected rest scaffolds of 3..{MAX_N} "
                       "nodes on the exact triangular patch, deduped by a canonical form "
                       "over sorted tuples")


def measure() -> tuple[str, str]:
    g, P, good, n_bad = _classes(MAX_N)
    rows = {}
    tot_obj = tot_rev = tot_revobj = 0
    carriers = writable = 0
    for key, idx in good.items():
        X = P[idx]
        tri = check_lemma(g, X)["triangles"]
        natt, nlive, nobj, nrev, nrevobj, dmax = _analyse(g, X)
        rows.setdefault(tri, []).append((nobj, nrevobj, dmax, len(idx)))
        tot_obj += nobj
        tot_rev += nrev
        tot_revobj += nrevobj
        carriers += nobj > 0
        writable += nrevobj > 0

    print(f"   isomorphism classes, correct canonical form   {len(good):>5}")
    print(f"   the broken form's count (L-491 reported 443)  {n_bad:>5}")
    print(f"   ceiling: connected unlabelled graphs n=3..6   {CEILING:>5}")
    print()
    print(f"   {'triangles':>10}{'classes':>9}{'carrying':>10}{'writable':>10}")
    for tri in sorted(rows):
        v = rows[tri]
        print(f"   {tri:>10}{len(v):>9}{sum(1 for r in v if r[0]):>10}"
              f"{sum(1 for r in v if r[1]):>10}")
    print(f"\n   object attractors, total        {tot_obj:>5}")
    print(f"   reversible live pairs, total    {tot_rev:>5}")
    print(f"   reversible OBJECT pairs, total  {tot_revobj:>5}")
    print(f"   classes carrying an object      {carriers:>5}")
    print(f"   classes with a writable bit     {writable:>5}")

    # the pattern the external review proposed, checked rather than accepted
    ok_pattern = all((r[0] > 0) == (tri >= 2 and r[2] >= 4)
                     for tri, v in rows.items() for r in v)
    print(f"\n   proposed rule  OBJECT <=> (triangles >= 2 and max degree >= 4): "
          f"{'holds on this family' if ok_pattern else 'DOES NOT hold'}")

    return (f"{len(good)} classes, {carriers} carry an object, {writable} hold a "
            f"writable topological bit",
            f"the claim needs the count under the {CEILING} ceiling and the writable bit "
            f"to survive "
            f"({'HOLDS' if len(good) <= CEILING and writable > 0 else 'DOES NOT HOLD'})")
