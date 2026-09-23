"""Does the graph the settled geometry INDUCES support objects beyond degrees and triangles?

`OB-152`, on the graphs the engine's own settled geometry produces rather than on
hand-built lattice patches.

**A SCOPE CORRECTION THAT WAS NEARLY PUBLISHED WITHOUT BEING CHECKED.** An earlier draft
of this file called itself *`OB-152` asked of the dynamics rather than the geometry*. That
is false and the check that caught it is cheap: **real settled 32-node scaffolds do not
satisfy the rest-scaffold condition.** `check_lemma` passes on **0 of 10** of them, and
**0 of 619** candidate pairs sit at exactly `r0` -- median `|d - r0| = 0.148`, worst
`0.346`. `MS-C-741`'s reduction, and therefore `abstract_step` as a model of the ENGINE,
holds on geometries where every candidate pair is at `r0`; the hand-built patches are, and
what the engine settles into is not. Driving a settled scaffold's own geometry from
arbitrary bond states moves positions in **400 of 400** trials, exactly as that predicts.

So what follows is a statement about **graphs**, not about engine trajectories: the
candidate graph the settled geometry induces is compared against rewirings of itself, both
evaluated as abstract rest scaffolds under the same map. That is the same standing as
`L-514`, which compared lattice-*admissible* classes. It is a real question and a real
comparison; it is not the dynamics, and the earlier wording claimed it was.

WHAT `L-514` SETTLED AND WHAT IT LEFT. Matched on `(n, E)`, the graphs the exact
triangular patch *admits* carry more `OBJECT` attractors than other graphs -- and the whole
effect is triangle packing: matched on `(n, E, triangles)` it vanishes, `p = 0.9638`. Two
gaps were named there and this file closes the first. *Admissible* is not *built*: `L-514`
compared what the lattice allows against all graphs, and never asked what the engine
actually settles onto from initial conditions.

THE NULL IS THE DESIGN, AND IT IS WHERE TODAY'S LESSON LIVES. `L-512` failed because a
confound was closed one layer at a time and the layer underneath was left open -- density
was matched, the structure density stands in for was not. Here the control is not a
stratum added after the fact; it is **the construction of the null**. Each settled
candidate graph is compared against rewirings of itself holding four invariants fixed:

    node count · edge count · degree sequence · triangle count

Degree-preserving double-edge swaps preserve the first three by construction; swaps that
change the triangle count are rejected. So the comparison starts with everything `L-496`
and `L-514` identified as explanatory already held equal, and can only report what is left
over. There is no layer underneath to forget, because the two known layers ARE the null.

THE UNIT IS THE SEED, WHICH IS WHY THIS IS NOT `L-512` AGAIN. An earlier draft decomposed
each scaffold into its connected induced subgraphs and treated those as observations.
Thousands of overlapping subgraphs cut from one 32-node graph are not independent, and a
p-value computed over them is pseudo-replication -- inflated by construction and worth
nothing. Withdrawn before it was written. Each seed contributes exactly one paired
difference, and the test is over seeds.

WHY `q = 1.0` AND NOT U0's `0.75`. `COUPLING_RATIO_001` swept `q = repulsion_cutoff / r0`
over eight sealed cells and `L-481` scored it. Two of its numbers were never used: at
`q = 1.0` the engine reaches geometric rest in **96 of 96** seeds against U0's 92, and
yields **11** `OBJECT_strong` against U0's 7 -- while `gain_at_rest` is still exactly `0`,
because repulsion applies on the strict `r < r_rep` and candidate pairs sit at `r0`. So
`q = 1.0` is the better point on U0's own axis, at zero cost, and the lead directed this
run there. The margin is thinner: at `0.75` the rest state sits `0.25` clear of the
repulsion threshold, at `1.0` it sits exactly on it, and that is recorded rather than
smoothed over. Measured here: `q = 1.0` builds sparser, less triangle-rich graphs --
`tri/E` `0.32` against `0.53`, mean `E` `51.6` against `60.4`, max `s_C` `4` against `5`.

MEASURED FIRST, NOT PROJECTED (`D-020` item 1). At the real scaffold size, `n = 32` and
`E = 65`, `abstract_step` costs **4.96 ms**, eighty times its cost at `n = 7`. Transients
are short: over ten random discrete initial conditions the longest was **17 steps** and all
ten closed a cycle within 18, at period 1 or 2. So a cap of 128 is roughly seven times the
observed worst case and a sample costs about 36 ms.

Run:  ./.venv/Scripts/python.exe tools/measure.py built_graph_beyond_structure
"""
from __future__ import annotations

import itertools
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "measurements"))

from autogenesis.genome import U0                              # noqa: E402
from autogenesis.init import make_initial_state                # noqa: E402
from autogenesis import simulator as sim                       # noqa: E402
from lo_zero_period_audit import NAMESPACE                     # noqa: E402
from rest_scaffold_orbits import candidate_edges               # noqa: E402
from abstract_map_equals_engine import abstract_step           # noqa: E402

Q = 1.00               # repulsion_cutoff / r0. NOT U0's 0.75 -- see below.
SEEDS = 200            # paired observations
NULLS = 5              # rewirings per seed
SAMPLES = 60           # shared discrete initial conditions per graph
SETTLE = 2048          # engine steps to the rest scaffold
CAP = 128              # cycle-detection cap; measured worst transient was 17
SWAP_FACTOR = 4        # accepted swaps per edge, to randomise the null properly

CLAIM = ("Each of 200 settled U0 scaffolds' candidate graphs is compared against 5 "
         "rewirings of itself "
         "holding node count, edge count, degree sequence and triangle count fixed. Every "
         "graph is driven from the same 60 random discrete initial conditions to its "
         "attractor through the engine-gated abstract map and scored by the fraction "
         "reaching an OBJECT. A paired one-sided sign test over the seeds says whether the "
         "graph the physics built supports objects BEYOND what its degrees and triangles "
         "already explain.")

WOULD_OVERTURN = ("A null that fails to match on any of the four invariants, or that comes "
                  "back identical to the original, which would make the comparison "
                  "vacuous. A seed whose geometry is not at positional rest, which is the "
                  "only rest this file claims -- it does NOT claim the settled geometry "
                  "is a valid rest scaffold, and it is not: check_lemma passes on 0 of "
                  "10. Trajectories not closing a "
                  "cycle within the cap. An OBJECT rate of zero on both arms, which is no "
                  "result rather than a negative one. The claim is the paired test, not a "
                  "direction: a null outcome closes OB-152 in the negative and is "
                  "reportable exactly as an enrichment would be.")


# ---------------------------------------------------------------- the criterion

def _binom_sf(k, n):
    """Exact P[X >= k] for X ~ Binomial(n, 1/2). No approximation anywhere."""
    return sum(math.comb(n, i) for i in range(k, n + 1)) / (1 << n)


def criterion(probe) -> bool:
    """True when the built graphs beat their matched nulls more often than a coin.

    `probe` is (wins, losses): pairs where the real graph scored strictly higher, and
    strictly lower. Ties carry no sign and are excluded, which is what makes this a sign
    test rather than a proportion.
    """
    wins, losses = probe
    n = wins + losses
    if n == 0:
        return False                      # every pair tied: no information, not a result
    return _binom_sf(wins, n) < 0.01


CONTROLS = [
    ("the built graph wins 45 of 50 signed pairs -> beyond structure", (45, 5), True),
    ("wins 25 of 50 -> a coin, nothing beyond degrees and triangles", (25, 25), False),
    ("wins 5 of 50 -> the null is better, still not enrichment", (5, 45), False),
    ("every pair ties -> the test has no information to report", (0, 0), False),
    ("wins 3 of 3 -> a real direction but too few pairs to pass", (3, 0), False),
]


# ---------------------------------------------------------------- graph plumbing

def triangles(adj, n) -> int:
    return sum(1 for a, b, c in itertools.combinations(range(n), 3)
               if b in adj[a] and c in adj[a] and c in adj[b])


def degseq(adj, n):
    return sorted(len(adj[i]) for i in range(n))


def rewire(adj, n, rng, factor=SWAP_FACTOR):
    """Double-edge swaps preserving degree sequence AND triangle count.

    Degree preservation is a property of the swap; triangle preservation is enforced by
    rejecting swaps that change the count. Returns None if the graph is too rigid to
    randomise, which is a refusal rather than a near-copy quietly passed off as a null.
    """
    adj = [set(x) for x in adj]
    edges = {tuple(sorted((i, j))) for i in range(n) for j in adj[i] if i < j}
    target = triangles(adj, n)
    want = factor * len(edges)
    done = tries = 0
    while done < want and tries < 200 * want:
        tries += 1
        (a, b), (c, d) = rng.sample(sorted(edges), 2)
        if len({a, b, c, d}) < 4:
            continue
        n1, n2 = tuple(sorted((a, c))), tuple(sorted((b, d)))
        if n1 in edges or n2 in edges:
            continue
        for x, y in ((a, b), (c, d)):
            adj[x].discard(y)
            adj[y].discard(x)
        for x, y in (n1, n2):
            adj[x].add(y)
            adj[y].add(x)
        if triangles(adj, n) == target:
            edges.discard((a, b))
            edges.discard((c, d))
            edges.add(n1)
            edges.add(n2)
            done += 1
        else:
            for x, y in (n1, n2):
                adj[x].discard(y)
                adj[y].discard(x)
            for x, y in ((a, b), (c, d)):
                adj[x].add(y)
                adj[y].add(x)
    return adj if done >= want // 2 else None


# ---------------------------------------------------------------- dynamics

def cycle_of(n, cand, bonds, act, g):
    """Iterate to the attractor; return the list of states on the cycle, or None."""
    seen, order = {}, []
    for t in range(CAP):
        key = (bonds, act)
        if key in seen:
            return order[seen[key]:]
        seen[key] = t
        order.append(key)
        bonds, act = abstract_step(n, cand, bonds, act, g)
    return None


def is_object(cycle, n, E) -> bool:
    """The project's standing OBJECT: activity somewhere, a bond set that varies, a bond
    in EVERY frame, and a bond in some but not all."""
    if len(cycle) < 2:
        return False
    sets, act = [], False
    for bonds, a in cycle:
        sets.append(frozenset(k for k in range(E) if bonds >> k & 1))
        act = act or a != 0
    if not act or len(set(sets)) < 2:
        return False
    always = frozenset.intersection(*sets)
    return bool(always and (frozenset.union(*sets) - always))


def score(n, adj, g, inits) -> tuple[float, int]:
    """Fraction of shared initial conditions that reach an OBJECT. Also how many failed
    to close a cycle within the cap, which the verdict refuses to ignore."""
    cand = [set(x) for x in adj]
    E = sum(len(x) for x in cand) // 2
    hits = open_ = 0
    for bonds, act in inits:
        cyc = cycle_of(n, cand, bonds & ((1 << E) - 1), act, g)
        if cyc is None:
            open_ += 1
        elif is_object(cyc, n, E):
            hits += 1
    return hits / len(inits), open_


# ---------------------------------------------------------------- the run

def settle(g, seed):
    """One engine run to a rest scaffold. Returns (adj, n) or None if it never rests."""
    s = make_initial_state(g, NAMESPACE, seed)
    for _ in range(SETTLE):
        s = sim.step(g, s)
    prev = s.x.copy()
    s = sim.step(g, s)
    if not np.array_equal(s.x, prev):
        return None
    n = len(s.x)
    adj = [set() for _ in range(n)]
    for i, j in candidate_edges(g, s.x):
        adj[i].add(j)
        adj[j].add(i)
    return adj, n


def universe() -> tuple[int, str]:
    return (SEEDS * (1 + NULLS) * SAMPLES,
            f"{SEEDS} settled U0 rest scaffolds, each against {NULLS} rewirings matched on "
            f"node count, edge count, degree sequence and triangle count, every graph "
            f"driven from the same {SAMPLES} random discrete initial conditions")


def measure() -> tuple[str, str]:
    g = U0().replace(**{"physics.repulsion_cutoff": Q})
    g.validate()
    print(f"   genome: q = repulsion_cutoff / r0 = {Q}"
          + ("   (U0's own value)" if Q == 0.75 else "   (NOT U0's 0.75)"))
    rng = random.Random(20260830)
    t0 = time.perf_counter()

    wins = losses = ties = 0
    no_rest = rigid = open_total = 0
    diffs, real_rates, null_rates = [], [], []
    for seed in range(SEEDS):
        got = settle(g, seed)
        if got is None:
            no_rest += 1
            continue
        adj, n = got
        E = sum(len(x) for x in adj) // 2
        inits = [(rng.getrandbits(E), rng.getrandbits(n)) for _ in range(SAMPLES)]

        r_rate, r_open = score(n, adj, g, inits)
        open_total += r_open

        nulls = []
        for _ in range(NULLS):
            nadj = rewire(adj, n, rng)
            if nadj is None:
                continue
            if degseq(nadj, n) != degseq(adj, n) or triangles(nadj, n) != triangles(adj, n):
                raise RuntimeError("rewiring broke an invariant it is required to hold")
            rate, o = score(n, nadj, g, inits)
            open_total += o
            nulls.append(rate)
        if not nulls:
            rigid += 1
            continue

        n_rate = sum(nulls) / len(nulls)
        real_rates.append(r_rate)
        null_rates.append(n_rate)
        diffs.append(r_rate - n_rate)
        if r_rate > n_rate:
            wins += 1
        elif r_rate < n_rate:
            losses += 1
        else:
            ties += 1
        if (seed + 1) % 20 == 0:
            print(f"      {seed+1:>4}/{SEEDS} seeds, {wins}W {losses}L {ties}T, "
                  f"{time.perf_counter()-t0:>5.0f} s", flush=True)

    secs = time.perf_counter() - t0
    paired = wins + losses
    p = _binom_sf(wins, paired) if paired else float("nan")
    mr = sum(real_rates) / len(real_rates) if real_rates else 0.0
    mn = sum(null_rates) / len(null_rates) if null_rates else 0.0

    print(f"\n   seeds run {SEEDS}   not at rest {no_rest}   too rigid to rewire {rigid}")
    print(f"   trajectories not closing within the cap  {open_total}")
    print(f"   mean OBJECT rate, built graph            {mr:.4f}")
    print(f"   mean OBJECT rate, matched nulls          {mn:.4f}")
    print(f"   paired: {wins} wins, {losses} losses, {ties} ties")
    print(f"   exact one-sided sign test p              {p:.4g}")
    print(f"   {secs:.0f} s")

    if open_total:
        return (f"{open_total} trajectories did not close a cycle within {CAP}",
                "REFUSED -- a state whose attractor was never reached was scored as "
                "carrying no object, which biases both arms by an unknown amount")
    if mr == 0 and mn == 0:
        return ("0 objects on either arm",
                "REFUSED -- nothing to compare; this is no result rather than a negative "
                "one, and the sampler or the horizon is what needs fixing")

    ok = criterion((wins, losses))
    verdict = ("BEYOND degrees and triangles" if ok else
               "nothing beyond degrees and triangles")
    print(f"\n   verdict: {verdict}")
    print("   Both directions were reportable before the run. A null closes OB-152 in")
    print("   the negative: the physics would then contribute exactly the degree")
    print("   sequence and the triangle count, and nothing else.")

    return (f"built {mr:.4f} against matched-null {mn:.4f}; {wins}W/{losses}L/{ties}T over "
            f"{paired} signed pairs, sign-test p = {p:.3g} -> {verdict}",
            f"the claim is the paired test, not its direction; it needs every null to "
            f"match on all four invariants and every trajectory to close "
            f"({'HOLDS' if (paired and not open_total) else 'DOES NOT HOLD'})")
