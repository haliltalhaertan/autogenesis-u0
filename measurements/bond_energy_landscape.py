"""Is the energy landscape over bond configurations flat? `L-518`..`L-520`, made runnable.

WHY THIS FILE EXISTS. The three derivations were produced in throwaway scripts and their
numbers went into the ledger. That is exactly what `D-019` forbids -- a claim anchored to a
number someone once printed rather than to code that reproduces it. This runs them.

THE ONE LINE THE WHOLE THING RESTS ON. On a valid rest scaffold every candidate pair sits
at exactly `r0`, so a bonded pair contributes `0.5*k_s*(r-r0)^2 = 0` and an unbonded pair
sits beyond `r_rep` and contributes `0`. So `E(B) = 0` for **every** bond configuration:
**no barrier separates any two bond sets.** `L-494`/`L-504`'s zero robustness, `L-514`'s
collapse to triangle count and `L-516`'s coin flip then stop being four independent facts
and become consequences of one line of algebra.

THE POTENTIAL IS READ OFF THE FORCE LAW, NOT INVENTED. `reference.py` applies
`-k_rep*(r_rep - r)` to nonbonded pairs inside the cutoff and the bonded spring
`-k_s*(r - r0)`, so the potentials are `0.5*k_rep*(r_rep-r)^2` and `0.5*k_s*(r-r0)^2`.
Nothing here is a new physics; it is the integral of the physics already shipped.

WHAT IS NOT CLAIMED. That adding a binding depth would help. It would not: `dE/dr` drops a
constant, and U0's bond rule reads support counts and activity, never the energy. The
landscape this file draws is one the system cannot see, which is `OB-153`.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from autogenesis.genome import U0                              # noqa: E402
from autogenesis.init import make_initial_state                # noqa: E402
from autogenesis import simulator as sim                       # noqa: E402
from lo_zero_period_audit import NAMESPACE                     # noqa: E402
from rest_scaffold_orbits import (candidate_edges, check_lemma,  # noqa: E402
                                  scaffold_tri)

SEEDS = 16
SETTLE = 2048
DRAWS = 200

CLAIM = ("On a valid rest scaffold the potential energy is IDENTICAL for every bond "
         "configuration -- exactly zero, not approximately -- so no barrier separates any "
         "two bond sets and no memory can be stored in the topology. On the scaffolds the "
         "engine actually settles into the landscape is not flat, but forming a bond is "
         "uphill in every measured candidate pair while breaking one at r0 is free, which "
         "is a one-way slope rather than a memory.")

WOULD_OVERTURN = ("Any nonzero energy spread on the hand-built patch, which would mean "
                  "the rest-scaffold condition does not do what MS-C-741 says. A settled "
                  "scaffold whose landscape IS flat, which would make the second half "
                  "vacuous. Any candidate pair where forming a bond is downhill, which "
                  "would break the one-way reading. Or check_lemma accepting a settled "
                  "scaffold, which would collapse the distinction the file is built on.")


def criterion(probe) -> bool:
    """True when this set of energies is flat -- one value, bit for bit."""
    vals = list(probe)
    if not vals:
        return False                      # nothing measured is not flatness
    return max(vals) == min(vals)


CONTROLS = [
    ("every configuration the same energy -> flat", [0.0, 0.0, 0.0, 0.0], True),
    ("one configuration differs -> not flat", [0.0, 0.0, 1e-9, 0.0], False),
    ("a spread of energies -> not flat", [0.6, 0.9, 1.4], False),
    ("no configurations at all -> nothing to call flat", [], False),
]


def potential(g, X, bonded) -> float:
    """Integral of the shipped force law. Bonded: the spring. Nonbonded inside the
    cutoff: the linear repulsion. Everything else: zero."""
    p = g.physics
    n = len(X)
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = float(np.linalg.norm(X[i] - X[j]))
            if (i, j) in bonded or (j, i) in bonded:
                total += 0.5 * p.spring_k * (r - p.r0) ** 2
            elif r < p.repulsion_cutoff:
                total += 0.5 * p.repulsion_strength * (p.repulsion_cutoff - r) ** 2
    return total


def settled(g, seed):
    s = make_initial_state(g, NAMESPACE, seed)
    for _ in range(SETTLE):
        s = sim.step(g, s)
    prev = s.x.copy()
    s = sim.step(g, s)
    return s.x if np.array_equal(s.x, prev) else None


def universe() -> tuple[int, str]:
    return (DRAWS + SEEDS * DRAWS,
            f"{DRAWS} random bond configurations on the hand-built rest scaffold and "
            f"{DRAWS} on each of up to {SEEDS} settled U0 scaffolds")


def measure() -> tuple[str, str]:
    g = U0()
    rng = random.Random(3)

    # --- the ideal rest scaffold: the regime every exhaustive result here lives in
    P = scaffold_tri((-2, -1, 0, 1, 2), 5)
    check_lemma(g, P)
    X = P[[0, 1, 2, 3, 4, 5]]
    cand = candidate_edges(g, X)
    ideal = [potential(g, X, {e for e in cand if rng.random() < 0.5})
             for _ in range(DRAWS)]
    flat = criterion(ideal)
    print(f"   IDEAL rest scaffold  n={len(X)} candE={len(cand)}")
    print(f"      {DRAWS} random bond sets: min {min(ideal):.3e}  max {max(ideal):.3e}"
          f"  spread {max(ideal)-min(ideal):.3e}   FLAT={flat}")

    # --- what the engine settles into, which is a different object
    spreads, flips, uphill, tot, lemma_ok, rested = [], [], 0, 0, 0, 0
    for seed in range(SEEDS):
        Xs = settled(g, seed)
        if Xs is None:
            continue
        rested += 1
        try:
            check_lemma(g, Xs)
            lemma_ok += 1
        except Exception:
            pass
        E = candidate_edges(g, Xs)
        vals = [potential(g, Xs, {e for e in E if rng.random() < 0.5})
                for _ in range(DRAWS)]
        spreads.append(max(vals) - min(vals))
        base = {e for e in E if rng.random() < 0.5}
        Eb = potential(g, Xs, base)
        for k in range(min(40, len(E))):
            e = E[k]
            B = set(base)
            B.discard(e) if e in B else B.add(e)
            flips.append(abs(potential(g, Xs, B) - Eb))
        for i, j in E:
            r = float(np.linalg.norm(Xs[i] - Xs[j]))
            form = 0.5 * g.physics.spring_k * (r - g.physics.r0) ** 2
            rep = (0.5 * g.physics.repulsion_strength
                   * (g.physics.repulsion_cutoff - r) ** 2
                   if r < g.physics.repulsion_cutoff else 0.0)
            tot += 1
            uphill += (form - rep) > 0

    print(f"\n   SETTLED scaffolds    {rested} of {SEEDS} reached positional rest")
    print(f"      check_lemma accepts them: {lemma_ok} of {rested}")
    print(f"      energy spread over bond sets: median {np.median(spreads):.3f}")
    print(f"      one bond flipped: |dE| median {np.median(flips):.4f}  "
          f"max {max(flips):.4f}")
    print(f"      forming a bond is UPHILL in {uphill} of {tot} candidate pairs "
          f"({100*uphill/tot:.0f}%)")
    print(f"      breaking a bond once relaxed to r0 costs 0 exactly, by the same algebra")

    print(f"\n   L-376 puts the noise death threshold near 3e-2; the settled barrier is")
    print(f"   {np.median(flips):.4f}, so memory here is marginal by roughly a factor of one.")

    ok = flat and lemma_ok == 0 and uphill == tot and np.median(spreads) > 0
    return (f"ideal landscape spread {max(ideal)-min(ideal):.3e} (flat={flat}); settled "
            f"spread {np.median(spreads):.3f}, one-flip barrier {np.median(flips):.4f}, "
            f"formation uphill {100*uphill/tot:.0f}%",
            f"the claim needs the ideal patch exactly flat, settled scaffolds NOT flat and "
            f"NOT accepted by check_lemma, and formation uphill everywhere "
            f"({'HOLDS' if ok else 'DOES NOT HOLD'})")
