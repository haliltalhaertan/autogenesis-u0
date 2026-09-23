"""The foundation everything else in this line rests on, verified exhaustively.

`L-490`, `L-491` and `L-492` all enumerate a rest scaffold as a finite map on
`(activity, bonds)`. That reduction is only legitimate if stepping the production engine
**never moves a position or a velocity**, because `Scaffold.step_code` encodes only the
activity and bond bits and silently discards `x` and `u`. If a step did move anything,
every one of those results would be computed on a fiction and nothing in the enumeration
would say what it claims.

WHAT WAS ACTUALLY IN PLACE, AND WHY IT IS NOT ENOUGH. `check_lemma` verifies the GEOMETRY
-- every candidate pair at exactly `r0`, no pair below the repulsion cutoff -- and
`MS-C-741` argues from that to zero force. `rest_scaffold_orbits.assert_frozen` checks the
consequence empirically, with tolerance exactly 0.0, but on **randomly sampled** states,
and none of the measurements in `L-490`-`L-492` call it at all. The freezing was inherited
from a lemma rather than verified on the states actually enumerated.

So it is verified here on **every state of every scaffold in the family** -- the same
`sim.step` calls the census already makes, with a comparison added. Any motion at all is a
failure; the claim is exactly 0.0, so the tolerance is exactly 0.0.

A SECOND CONTROL, folded in rather than left in conversation. `L-492`'s transition graph
builds a perturbed code by hand, `(bb << n) | nab`. Its identity control verified the
LOOKUP but not that construction: if bond bits and activity bits were transposed, that
control would still have passed. Here the construction is checked directly -- bonds
untouched, the forced bit set to the demanded value, every other activity bit unchanged.
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
from autogenesis import simulator as sim                       # noqa: E402
from rest_scaffold_orbits import (Scaffold, candidate_edges,   # noqa: E402
                                  check_lemma, scaffold_tri)

MAX_N = 6

CLAIM = (f"On every state of every connected rest scaffold of 3 to {MAX_N} nodes, one "
         "step of the production engine changes no position and no velocity, bit for "
         "bit. The reduction to a finite map on (activity, bonds) that L-490, L-491 and "
         "L-492 rest on is exact, not inherited.")

WOULD_OVERTURN = ("A single state anywhere with a non-zero |dx| or |du|. Scaffold.step_code "
                  "discards x and u, so any motion at all means the enumerations were "
                  "computed on a fiction and every result in L-490 through L-492 is void. "
                  "Also: the perturbation construction failing its own control, which "
                  "would void L-492 specifically.")


def criterion(pair) -> bool:
    """True when a step moved something. The claim is that this is never true."""
    before, after = pair
    return not (np.array_equal(before[0], after[0]) and np.array_equal(before[1], after[1]))


_Z = np.zeros((2, 3))
_E = np.zeros((2, 3))
_E[1, 2] = np.spacing(1.0)
CONTROLS = [
    ("nothing moved -> not moved", ((_Z, _Z), (_Z.copy(), _Z.copy())), False),
    ("one ULP of position -> MOVED", ((_Z, _Z), (_E, _Z.copy())), True),
    ("one ULP of velocity -> MOVED", ((_Z, _Z), (_Z.copy(), _E)), True),
    ("a large displacement -> MOVED", ((_Z, _Z), (_Z + 1.0, _Z.copy())), True),
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
            # MS-C-896: this used `min` over FROZENSETS, and frozenset comparison in
            # Python is the SUBSET relation, not a lexicographic order. `min` over a
            # partial order returns whatever survived its pairwise comparisons, so two
            # labellings of one graph could come out different -- reproduced on P4. The
            # census then reported 443 classes where at most 141 connected unlabelled
            # graphs exist on 3..6 nodes. Sorted TUPLES are totally ordered.
            can = min(tuple(sorted(tuple(sorted((p[i], p[j]))) for i, j in e))
                      for p in itertools.permutations(range(size)))
            if can in seen:
                continue
            seen.add(can)
            out.append(idx)
    return g, P, out


def universe() -> tuple[int, str]:
    g, P, subs = _scaffolds(MAX_N)
    n = 0
    for idx in subs:
        sc = Scaffold(g, P[idx], candidate_edges(g, P[idx]))
        n += 1 << (sc.n + sc.E)
    return n, (f"every state of every connected rest scaffold of 3..{MAX_N} nodes; each "
               "stepped once and compared bit for bit, none sampled")


def measure() -> tuple[str, str]:
    g, P, subs = _scaffolds(MAX_N)
    moved, worst_dx, worst_du, checked = 0, 0.0, 0.0, 0
    perturb_bad, perturb_checked = 0, 0

    for idx in subs:
        X = P[idx]
        sc = Scaffold(g, X, candidate_edges(g, X))
        total = 1 << (sc.n + sc.E)
        for code in range(total):
            s = sc.state(code)
            x0, u0 = s.x.copy(), s.u.copy()
            out = sim.step(g, s)
            checked += 1
            if criterion(((x0, u0), (out.x, out.u))):
                moved += 1
                worst_dx = max(worst_dx, float(np.abs(out.x - x0).max()))
                worst_du = max(worst_du, float(np.abs(out.u - u0).max()))

        # L-492's construction, checked on a stride through this scaffold's states
        for code in range(0, total, max(1, total // 64)):
            bb, ab = sc.split(code)
            if ((bb << sc.n) | ab) != code:
                perturb_bad += 1
                continue
            for v in range(sc.n):
                cur = bool(ab >> v & 1)
                for val in (True, False):
                    if val == cur:
                        continue
                    nab = (ab | (1 << v)) if val else (ab & ~(1 << v))
                    st = sc.state((bb << sc.n) | nab)
                    perturb_checked += 1
                    if sc.split((bb << sc.n) | nab)[0] != bb:
                        perturb_bad += 1
                    if bool(st.active[v]) != val:
                        perturb_bad += 1
                    if int(st.bonds[sc.ei, sc.ej] @ sc.pow_e) != bb:
                        perturb_bad += 1
                    if any(bool(st.active[u]) != bool(ab >> u & 1)
                           for u in range(sc.n) if u != v):
                        perturb_bad += 1

    print(f"   scaffolds                    {len(subs):>10}")
    print(f"   states stepped and compared  {checked:>10}")
    print(f"   states that MOVED            {moved:>10}   "
          f"worst |dx| {worst_dx!r}, |du| {worst_du!r}")
    print(f"   perturbation constructions   {perturb_checked:>10}   "
          f"failures {perturb_bad}")
    ok = moved == 0 and perturb_bad == 0
    return (f"{moved} of {checked} states moved; {perturb_bad} of {perturb_checked} "
            "perturbation constructions wrong",
            f"the claim is that both are zero "
            f"({'HOLDS -- the reduction is exact' if ok else 'DOES NOT HOLD -- the enumerations are void'})")
