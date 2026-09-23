"""Does the physics reach the graphs that can compute, more often than chance? `OB-123`.

THE ONLY QUESTION THE NOVELTY SCAN LEFT OPEN. `D-052`'s scan found direct prior art for
six of the project's eight claim families. `MS-C-741` is why: at rest the mechanical layer
is **inert** -- it selects the candidate graph and does nothing else -- so what runs on
that graph is an outer-totalistic Boolean rule, and *Life-Like Network Automata* is a
named model class with a publication line to a 2025 treatment in Physica D.

That leaves exactly one thing this project can ask and that literature cannot, because
answering it needs both halves and nobody else has both:

    Of all connected graphs, the physics reaches only some.
    Of all connected graphs, only some carry an OBJECT.
    Are those two sets independent?

If the physics is enriched for capable graphs, the mechanical layer selects for
computation and that is the project's reason to exist. If it is not, the mechanics are
confirmed inert and this is honestly a network-automata study -- a full literature and a
fair place to stand. **Both answers are worth having, which is why this is worth running.**

THE TWO HALVES ALREADY EXISTED AND WERE NEVER CROSSED.

* `L-496` / `scaffold_census_recount` established that the exact triangular patch realizes
  **36** isomorphism classes at `n <= 6`, of which **13** carry an object.
* The ceiling is **141** -- the connected unlabelled simple graphs on 3 to 6 nodes.

The missing number is the base rate: how many of the **141** carry an object. Without it,
`13 of 36` says nothing at all. This file computes it, on the same predicate, and crosses
the two.

WHAT IS NOT REIMPLEMENTED. The dynamics run through `abstract_step`, gated against the
production engine at 0 disagreements in 247,904 states (`MS-C-897`). The realizable family
comes from `scaffold_census_recount._classes`, the code that produced `L-496`. Only the
canonical form is recomputed here, on both sides with one function, so the two halves are
matched by the same rule rather than by two key formats that might disagree.
"""
from __future__ import annotations

import itertools
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "measurements"))

from autogenesis.genome import U0                              # noqa: E402
from rest_scaffold_orbits import candidate_edges               # noqa: E402
from abstract_map_equals_engine import abstract_step           # noqa: E402
from scaffold_census_recount import _classes                   # noqa: E402
from two_bit_register_minimality import _atlas                 # noqa: E402

MAX_N = 6

CLAIM = ("Across all 141 connected isomorphism classes on 3 to 6 nodes, the classes the "
         "U0 physics can realize as rest scaffolds are compared against the classes that "
         "carry an OBJECT. The contingency between the two is reported with an exact "
         "one-sided hypergeometric p-value, so 'the physics selects for computation' "
         "becomes a measured statement with a null instead of an impression.")

WOULD_OVERTURN = ("The realizable count not being 36, or the ceiling not being 141, "
                  "which would mean this file and L-496 did not enumerate the same "
                  "family. A base rate of zero or of one, which makes the comparison "
                  "vacuous rather than negative. Any state moving a position, which "
                  "voids the rest reduction the whole measurement stands on. The claim "
                  "here is the TABLE, not a direction: an enrichment and a null result "
                  "are both reportable and neither overturns the file.")


# ---------------------------------------------------------------- the criterion

def _hyper_sf(k, N, K, n):
    """P[X >= k] for X ~ Hypergeometric(N population, K successes, n drawn). Exact."""
    return sum(math.comb(K, i) * math.comb(N - K, n - i) / math.comb(N, n)
               for i in range(k, min(K, n) + 1))


def _rank_u(a, b):
    """Mann-Whitney one-sided p that `a` is stochastically LARGER, normal approximation
    with tie correction. Used because the 2x2 is weakly powered: with `n = 36` drawn from
    `N = 141` and `k` near 13, the table can only reach `p < 0.01` if the base rate is
    below about 0.19, which was computed BEFORE the run (`D-020` item 3). The counts carry
    more than the indicator does, at no extra cost, so both are reported.
    """
    na, nb = len(a), len(b)
    if not na or not nb:
        return float("nan")
    allv = sorted(a + b)
    rank = {}
    i = 0
    ties = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        r = (i + j) / 2 + 1
        for v in allv[i:j + 1]:
            rank[v] = r
        t = j - i + 1
        ties += t ** 3 - t
        i = j + 1
    ra = sum(rank[v] for v in a)
    u = ra - na * (na + 1) / 2
    n_ = na + nb
    mu = na * nb / 2
    var = na * nb / 12 * ((n_ + 1) - ties / (n_ * (n_ - 1))) if n_ > 1 else 0.0
    if var <= 0:
        return float("nan")
    z = (u - mu - 0.5) / math.sqrt(var)
    return 0.5 * math.erfc(z / math.sqrt(2))


def _van_elteren(strata):
    """One-sided p that group A is stochastically LARGER than B, stratified.

    THE CONFOUND THIS EXISTS FOR. The lattice-realizable graphs are sparse; the ceiling
    family runs up to `K6`. Denser graphs have more attractors for reasons that have
    nothing to do with the physics, so an unstratified comparison measures edge count and
    calls it selection. Each stratum here is one `(n, E)` cell, so reachable and
    unreachable graphs are only ever compared at equal size AND equal edge count.

    Per-stratum Mann-Whitney U statistics are summed with their null means and variances
    (van Elteren, unweighted), and the total is z-tested.
    """
    u = mu = var = 0.0
    used = 0
    for a, b in strata:
        na, nb = len(a), len(b)
        if not na or not nb:
            continue
        used += 1
        allv = sorted(a + b)
        rank, i, ties = {}, 0, 0
        while i < len(allv):
            j = i
            while j + 1 < len(allv) and allv[j + 1] == allv[i]:
                j += 1
            for v in allv[i:j + 1]:
                rank[v] = (i + j) / 2 + 1
            t = j - i + 1
            ties += t ** 3 - t
            i = j + 1
        n_ = na + nb
        u += sum(rank[v] for v in a) - na * (na + 1) / 2
        mu += na * nb / 2
        if n_ > 1:
            var += na * nb / 12 * ((n_ + 1) - ties / (n_ * (n_ - 1)))
    if used == 0 or var <= 0:
        return float("nan"), used
    z = (u - mu - 0.5) / math.sqrt(var)
    return 0.5 * math.erfc(z / math.sqrt(2)), used


def criterion(probe) -> bool:
    """True when this probe shows physics-reachable graphs are enriched for capability.

    Tagged so both tests are controlled by the same declaration. `("table", N, K, n, k)`
    is the pre-registered but weakly powered 2x2; `("ranks", reachable, rest)` is the
    rank test on per-class OBJECT-attractor counts.
    """
    if probe[0] == "table":
        _, N, K, n, k = probe
        if n == 0 or K == 0 or K == N:
            return False                  # nothing drawn, or the property is universal
        return _hyper_sf(k, N, K, n) < 0.01
    if probe[0] == "ranks":
        _, a, b = probe
        p = _rank_u(list(a), list(b))
        return p == p and p < 0.01         # NaN is not a result
    if probe[0] == "strata":
        p, used = _van_elteren(probe[1])
        return used > 0 and p == p and p < 0.01
    raise ValueError(f"unknown probe tag {probe[0]!r}")


CONTROLS = [
    ("table: every reachable graph capable, few capable overall -> enrichment",
     ("table", 141, 20, 36, 20), True),
    ("table: reachable capable at exactly the base rate -> no enrichment",
     ("table", 141, 40, 36, 10), False),
    ("table: reachable capable LESS often -> depletion, not enrichment",
     ("table", 141, 60, 36, 5), False),
    ("table: the property is universal -> the table cannot say anything",
     ("table", 141, 141, 36, 36), False),
    ("ranks: reachable carry strictly more objects -> enrichment",
     ("ranks", [3] * 36, [0] * 105), True),
    ("ranks: the two groups drawn from one distribution -> no enrichment",
     ("ranks", [0, 1, 2] * 12, [0, 1, 2] * 35), False),
    ("ranks: reachable carry FEWER objects -> depletion, not enrichment",
     ("ranks", [0] * 36, [3] * 105), False),
    ("ranks: every class identical -> no variance, no result",
     ("ranks", [1] * 36, [1] * 105), False),
    ("strata: reachable larger inside every cell -> enrichment",
     ("strata", [([5] * 6, [0] * 12) for _ in range(6)]), True),
    ("strata: an unstratified gap that vanishes cell by cell -> no enrichment",
     ("strata", [([0] * 8, [0] * 2), ([9] * 2, [9] * 8)]), False),
    ("strata: reachable smaller inside every cell -> depletion",
     ("strata", [([0] * 6, [5] * 12) for _ in range(6)]), False),
    ("strata: no cell has both groups -> nothing comparable",
     ("strata", [([1, 2], []), ([], [3, 4])]), False),
]


# ---------------------------------------------------------------- shared canonical form

def triangles(n, edges) -> int:
    adj = [set() for _ in range(n)]
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
    return sum(1 for a, b, c in itertools.combinations(range(n), 3)
               if b in adj[a] and c in adj[a] and c in adj[b])


def max_degree(n, edges) -> int:
    d = [0] * n
    for i, j in edges:
        d[i] += 1
        d[j] += 1
    return max(d)


def canon(n, edges):
    """`L-496`'s corrected form: min over SORTED TUPLES, which are totally ordered.
    `frozenset` comparison is the subset relation and is what produced the 443."""
    return min(tuple(sorted(tuple(sorted((p[i], p[j]))) for i, j in edges))
               for p in itertools.permutations(range(n)))


# ---------------------------------------------------------------- capability

def object_count(n, edge_list, g) -> int:
    """One full state census on the abstract map; how many attractors are OBJECTs.

    OBJECT is the project's standing definition and is not relaxed here: some activity
    on the cycle, a bond set that varies, a bond present in EVERY frame, and a bond
    present in some but not all. The COUNT is returned rather than a flag because the
    rank test needs it and the indicator is recoverable from it.
    """
    cand = [set() for _ in range(n)]
    for i, j in edge_list:
        cand[i].add(j)
        cand[j].add(i)
    edges = [(i, j) for i in range(n) for j in cand[i] if i < j]   # abstract_step's order
    E = len(edges)
    total = 1 << (n + E)

    nxt = np.empty(total, dtype=np.int64)
    for code in range(total):
        nb, na = abstract_step(n, cand, code >> n, code & ((1 << n) - 1), g)
        nxt[code] = (nb << n) | na

    colour = np.zeros(total, dtype=np.int8)
    objects = 0
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
            sets, act = [], False
            for c in cyc:
                bb, ab = c >> n, c & ((1 << n) - 1)
                sets.append(frozenset(k for k in range(E) if bb >> k & 1))
                act = act or ab != 0
            if act and len(set(sets)) > 1:
                always = frozenset.intersection(*sets)
                if always and (frozenset.union(*sets) - always):
                    objects += 1
        for c in path:
            colour[c] = 2
    return objects


# ---------------------------------------------------------------- the two halves

def reachable_keys(g):
    """Canonical forms of the classes the exact triangular patch realizes (`L-496`)."""
    _, P, good, _ = _classes(MAX_N)
    keys = set()
    for idx in good.values():
        X = P[idx]
        E = candidate_edges(g, X)
        keys.add(canon(len(idx), E))
    return keys


def universe() -> tuple[int, str]:
    fam = [c for n in range(3, MAX_N + 1) for c in _atlas(n)]
    return (sum(1 << (n + len(e)) for n, e in fam),
            "every state of every connected graph on 3 to 6 nodes, each stepped once "
            "through the engine-gated abstract map, plus the realizable family from L-496")


def measure() -> tuple[str, str]:
    g = U0()
    fam = [c for n in range(3, MAX_N + 1) for c in _atlas(n)]
    N = len(fam)
    reach = reachable_keys(g)
    print(f"   ceiling: connected unlabelled graphs n=3..6   {N:>5}")
    print(f"   realizable on the exact triangular patch      {len(reach):>5}   (L-496: 36)")

    t0 = time.perf_counter()
    capable, both, per_n = set(), 0, {}
    in_reach, out_reach = [], []
    cells: dict[tuple, tuple[list, list]] = {}
    tri_cells: dict[tuple, tuple[list, list]] = {}
    rule: set = set()
    for i, (n, edges) in enumerate(fam):
        key = canon(n, edges)
        cnt = object_count(n, edges, g)
        cap = cnt > 0
        r = key in reach
        (in_reach if r else out_reach).append(cnt)
        cells.setdefault((n, len(edges)), ([], []))[0 if r else 1].append(cnt)
        tri = triangles(n, edges)
        tri_cells.setdefault((n, len(edges), tri), ([], []))[0 if r else 1].append(cnt)
        if tri >= 2 and max_degree(n, edges) >= 4:
            rule.add(key)
        per_n.setdefault(n, [0, 0, 0, 0])
        per_n[n][0] += 1
        per_n[n][1] += cap
        per_n[n][2] += r
        per_n[n][3] += cap and r
        if cap:
            capable.add(key)
        if cap and r:
            both += 1
        if (i + 1) % 25 == 0:
            print(f"      {i+1:>4}/{N} classes, {time.perf_counter()-t0:>5.0f} s",
                  flush=True)
    secs = time.perf_counter() - t0

    K, n_draw, k = len(capable), len(reach), both
    print(f"\n   {'n':>4}{'classes':>9}{'capable':>9}{'reachable':>11}{'both':>7}")
    for n in sorted(per_n):
        a, b, c, d = per_n[n]
        print(f"   {n:>4}{a:>9}{b:>9}{c:>11}{d:>7}")

    base = K / N
    hit = k / n_draw if n_draw else 0.0
    p = _hyper_sf(k, N, K, n_draw) if (n_draw and 0 < K < N) else float("nan")
    exp = n_draw * base
    print(f"\n   base rate, all classes          {K:>3} / {N:<3}  = {base:.3f}")
    print(f"   among physics-reachable         {k:>3} / {n_draw:<3}  = {hit:.3f}")
    print(f"   expected if independent                       {exp:.1f} classes")
    print(f"   enrichment factor                             {hit/base if base else 0:.2f}x")
    print(f"   exact one-sided hypergeometric p              {p:.4g}")
    print(f"   {secs:.0f} s")

    p_rank = _rank_u(in_reach, out_reach)
    med_in = sorted(in_reach)[len(in_reach) // 2] if in_reach else 0
    med_out = sorted(out_reach)[len(out_reach) // 2] if out_reach else 0
    print(f"\n   OBJECT attractors per class, reachable   median {med_in}, "
          f"mean {sum(in_reach)/len(in_reach):.2f} over {len(in_reach)}")
    print(f"   OBJECT attractors per class, the rest    median {med_out}, "
          f"mean {sum(out_reach)/len(out_reach):.2f} over {len(out_reach)}")
    print(f"   rank test, one-sided p                        {p_rank:.4g}")

    strata = list(cells.values())
    p_str, used = _van_elteren(strata)
    print(f"\n   matched on (n, E) -- the density confound removed")
    print(f"   {'n':>4}{'E':>4}{'reach':>7}{'rest':>6}{'med in':>8}{'med out':>9}")
    for (n, e), (a, b) in sorted(cells.items()):
        if not a or not b:
            continue
        print(f"   {n:>4}{e:>4}{len(a):>7}{len(b):>6}"
              f"{sorted(a)[len(a)//2]:>8}{sorted(b)[len(b)//2]:>9}")
    print(f"   {used} comparable cells, stratified one-sided p   {p_str:.4g}")

    # THE CONFOUND BEHIND THE CONFOUND. `L-497` makes triangles the precondition for a
    # bond ever re-forming and `L-496` found OBJECT separating on `triangles >= 2 AND
    # max degree >= 4`. The lattice packs triangles, so an (n, E) match can still be
    # measuring triangle count and calling it selection. Matching on (n, E, triangles)
    # asks whether anything survives once that is held fixed too.
    tri_strata = list(tri_cells.values())
    p_tri, used_tri = _van_elteren(tri_strata)
    print(f"\n   matched on (n, E, triangles) -- the enabling structure held fixed too")
    print(f"   {used_tri} comparable cells, stratified one-sided p   {p_tri:.4g}")

    inter = len(capable & rule)
    print(f"\n   L-496's rule (triangles >= 2 and max degree >= 4) on the FULL family:")
    print(f"      satisfies the rule {len(rule):>3}   carries an object {len(capable):>3}   "
          f"both {inter:>3}")
    print(f"      rule but no object {len(rule - capable):>3}   "
          f"object but not the rule {len(capable - rule):>3}")

    tab = criterion(("table", N, K, n_draw, k))
    rnk = criterion(("ranks", in_reach, out_reach))
    stf = criterion(("strata", strata))
    tri_ok = criterion(("strata", tri_strata))
    verdict = ("ENRICHED beyond triangle count" if tri_ok else
               "TRIANGLE PACKING, not selection beyond it" if stf else
               "ENRICHED before matching only -- density" if (tab or rnk) else
               "NO ENRICHMENT" if k <= exp else "not significant at p < 0.01")
    print(f"\n   verdict: {verdict}")
    print("   Both directions were reportable before the run. A null here does not")
    print("   fail the measurement; it confirms MS-C-741's deflation, that the")
    print("   mechanical layer is inert once the graph is chosen.")

    return (f"{k} of {n_draw} physics-reachable classes carry an object against a base "
            f"rate of {K}/{N} ({hit/base if base else 0:.2f}x, table p = {p:.3g}; "
            f"rank p = {p_rank:.3g}; matched on (n,E) over {used} cells "
            f"p = {p_str:.3g}) -> {verdict}",
            f"the claim is the pair of tests, not their direction; it needs the ceiling "
            f"at 141 and the realizable family at 36 "
            f"({'HOLDS' if (N == 141 and n_draw == 36) else 'DOES NOT HOLD'})")
