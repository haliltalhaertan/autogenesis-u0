"""Do the writable bits survive being touched, or does any nudge destroy them?

`L-492` found 13 scaffolds holding a reversible pair of object attractors -- a bit that
can be written and cleared by single-node operations. That is writability. It says nothing
about whether the stored value SURVIVES anything, and on seed 63 the answer was as bad as
it can be: of the four nodes that could reach the bit, **all four killed the structure**
(`L-487`). A value that the next touch destroys is not stored.

So for every one of those bits, this asks the complementary question exactly: from each
state of the pair, over every single-node forced-value operation on every phase, where
does it land?

    RETAINED   back in the same attractor -- the value survived the touch
    WRITTEN    in the other member of the pair -- the intended operation
    CORRUPTED  in some third live attractor -- the value is gone but the structure lives
    KILLED     in a dead attractor

`RETAINED` is the number that matters. No-ops are excluded throughout: forcing a node to
the value it already holds at that phase changes nothing, and counting those as retention
would manufacture exactly the robustness this file is trying to measure (`L-489`).

THE PREDICTION, WRITTEN BEFORE THE RUN, and it is not the comfortable one. Seed 63 gave
0 of 4. If that generalises, these bits are writable and completely fragile, and the
honest headline becomes *writable but unprotectable*. The claim below says the opposite --
that at least one bit retains its value under a majority of operations -- so that a
failure is recorded as a failure rather than absorbed into a softer sentence.
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

CLAIM = ("At least one of the writable topological bits retains its value under a "
         "MAJORITY of single-node operations: more than half of the non-no-op "
         "perturbations of each of its two states leave that state's attractor unchanged.")

WOULD_OVERTURN = ("Every bit losing its value under more than half of the operations. "
                  "The bits would then be writable and unprotectable -- seed 63's 0 of 4 "
                  "generalised -- and L-492 must be read as writability alone, with no "
                  "claim of storage that survives contact.")


def criterion(counts) -> bool:
    """True when this state retains its attractor under a majority of operations."""
    retained, total = counts
    return total > 0 and retained * 2 > total


CONTROLS = [
    ("all operations retained -> robust", (10, 10), True),
    ("a bare majority retained -> robust", (6, 10), True),
    ("exactly half -> NOT a majority", (5, 10), False),
    ("none retained, seed 63's case -> not robust", (0, 4), False),
    ("no operations at all -> not robust, and not a division by zero", (0, 0), False),
]


def _scaffolds(maxn: int):
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
    seen, out = set(), []
    for size in range(3, maxn + 1):
        for S in sorted(lvl[size], key=sorted):
            idx = sorted(S)
            pos = {v: k for k, v in enumerate(idx)}
            e = {tuple(sorted((pos[i], pos[j]))) for i, j in E if i in pos and j in pos}
            # MS-C-896: `min` over FROZENSETS is not a canonical form -- frozenset
            # comparison is the SUBSET relation. Sorted tuples are totally ordered.
            can = min(tuple(sorted(tuple(sorted((p[i], p[j]))) for i, j in e))
                      for p in itertools.permutations(range(size)))
            if can in seen:
                continue
            seen.add(can)
            out.append(idx)
    return g, P, out


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
    tally: dict[int, dict] = {}
    for cid, cyc in enumerate(cycles):
        t = {"retained": 0, "total": 0, "killed": 0, "elsewhere": {}}
        for code in cyc:
            bb, ab = sc.split(code)
            for v in range(sc.n):
                cur = bool(ab >> v & 1)
                for val in (True, False):
                    if val == cur:
                        continue
                    nab = (ab | (1 << v)) if val else (ab & ~(1 << v))
                    dst = int(which[(bb << sc.n) | nab])
                    t["total"] += 1
                    if dst == cid:
                        t["retained"] += 1
                    else:
                        fwd[cid].add(dst)
                        if dst not in live:
                            t["killed"] += 1
                        t["elsewhere"][dst] = t["elsewhere"].get(dst, 0) + 1
        tally[cid] = t

    pairs = {(a, b) for a in fwd for b in fwd[a]
             if a < b and a in fwd.get(b, set()) and a in obj and b in obj}
    return tally, pairs


def universe() -> tuple[int, str]:
    return 22, ("the reversible object pairs the CORRECTED census finds -- MS-C-896: the "
                "76 in L-492 counted duplicated scaffolds, because `min` over frozensets "
                "is not a canonical form. Every pair is examined, and for each, every "
                "phase, node and forced value, with no-ops excluded")


def measure() -> tuple[str, str]:
    g, P, subs = _scaffolds(MAX_N)
    rows, robust, seen_pairs = [], 0, 0
    hist = {}
    for idx in subs:
        tally, pairs = _analyse(g, P[idx])
        for a, b in sorted(pairs):
            seen_pairs += 1
            ta, tb = tally[a], tally[b]
            ra = criterion((ta["retained"], ta["total"]))
            rb = criterion((tb["retained"], tb["total"]))
            if ra and rb:
                robust += 1
            fa = ta["retained"] / ta["total"] if ta["total"] else 0.0
            fb = tb["retained"] / tb["total"] if tb["total"] else 0.0
            key = round(min(fa, fb), 1)
            hist[key] = hist.get(key, 0) + 1
            if len(rows) < 8:
                rows.append((len(idx), a, b, ta["retained"], ta["total"],
                             tb["retained"], tb["total"], ta["killed"], tb["killed"]))

    print(f"   {'n':>3}{'A':>5}{'B':>5}{'A retained':>13}{'B retained':>13}"
          f"{'A killed':>11}{'B killed':>10}")
    for n, a, b, ra, ta, rb, tb, ka, kb in rows:
        print(f"   {n:>3}{a:>5}{b:>5}{f'{ra}/{ta}':>13}{f'{rb}/{tb}':>13}"
              f"{ka:>11}{kb:>10}")
    print(f"\n   reversible object pairs examined      {seen_pairs:>5}")
    print(f"   pairs where BOTH states hold a majority {robust:>5}")
    print(f"   weaker retention fraction, histogram   {dict(sorted(hist.items()))}")
    return (f"{robust} of {seen_pairs} bits retain their value under a majority of "
            "single-node operations",
            f"the claim asks for at least one "
            f"({'HOLDS' if robust else 'DOES NOT HOLD -- writable but unprotectable, as seed 63 was'})")
