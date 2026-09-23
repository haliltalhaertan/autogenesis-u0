"""How many seeds at U0 carry the object under a STRICT rest test?

`COUPLING_RATIO_001` reported 7 of 96 at U0's own genome. `L-483` showed that number is an
UPPER BOUND, because the instrument's `geometric_rest` compares exactly one pair of
frames -- `x(T)` against `x(T-2)` -- so a seed still moving through the scored window can
satisfy it, and seed 38 does move, up to step 8148 with the window opening at 8062.

This replaces that test with the one the claim actually needs: **every** position array in
the window identical to its predecessor, all 128 frames, no pair privileged. Everything
else is held exactly as the sealed run had it, so the only thing that changes is the rest
criterion and the difference between the two numbers is attributable to it alone.

`L-484` is a separate narrowing and is NOT folded in here: of the seeds that pass, one
(18) keeps blinking when its graph is frozen at one phase, so it is a shadow rather than a
topological oscillator. That count is reported there. This file answers only: how many are
really at rest.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from autogenesis.genome import U0                              # noqa: E402
from autogenesis.init import make_initial_state, InitFailure   # noqa: E402
from autogenesis import simulator as sim                       # noqa: E402
from lo_zero_period_audit import HORIZON, WINDOW, NAMESPACE    # noqa: E402

FAMILY = {"initial_conditions.N": 24, "initial_conditions.cube_side": 3.6,
          "initial_conditions.bond_init_mode": "mean_degree",
          "initial_conditions.target_mean_degree": 3.0,
          "initial_conditions.initial_degree_cap": 4}
SEEDS = 96
LOOSE_HITS = (8, 18, 19, 35, 38, 63, 68)     # what the one-pair test returned

CLAIM = ("Under a rest test that requires EVERY frame in the scored window to be "
         "bit-identical to its predecessor, fewer than 7 of the 96 seeds at U0 carry "
         "OBJECT_strong -- seed 38 at least should drop, since it is still moving at "
         "step 8148 and the window opens at 8062.")

WOULD_OVERTURN = ("The strict count coming back equal to the loose one, which would mean "
                  "the one-pair test was not actually loose on this population and "
                  "L-483's caveat was theoretical; or the strict count EXCEEDING the "
                  "loose one, which is impossible unless the two tests disagree about "
                  "which seeds they are scoring and the comparison is broken.")


def criterion(frames) -> bool:
    """True when every position array in the window equals the one before it."""
    return all(np.array_equal(frames[i], frames[i - 1]) for i in range(1, len(frames)))


_A, _B = np.zeros((2, 3)), np.zeros((2, 3))
_B[0, 0] = 1e-12
CONTROLS = [
    ("every frame identical -> at rest", [_A, _A.copy(), _A.copy(), _A.copy()], True),
    ("the LAST frame differs -> not at rest", [_A, _A.copy(), _A.copy(), _B], False),
    # THE control: the one-pair test x(T) vs x(T-2) passes this and it must not.
    ("motion in the MIDDLE, first and last equal -> not at rest",
     [_A, _B, _A.copy(), _B.copy(), _A.copy()], False),
    ("a single 1e-12 displacement anywhere -> not at rest",
     [_A, _A.copy(), _B, _A.copy()], False),
]


def _classify(g, seed: int):
    try:
        s = make_initial_state(g, NAMESPACE, seed)
    except InitFailure:
        return None
    xs, sigs, edges = [], [], []
    for t in range(1, HORIZON + 1):
        try:
            s = sim.step(g, s)
        except FloatingPointError:
            return None
        if t > HORIZON - WINDOW - 2:
            xs.append(s.x.copy())
            sigs.append(s.discrete_signature())
            edges.append(frozenset(s.edge_list()))
    loose = np.array_equal(xs[-1], xs[-3])          # the instrument's own test
    strict = criterion(xs)
    always = set.intersection(*map(set, edges))
    sometimes = set().union(*map(set, edges)) - always
    obj = len(set(sigs)) > 1 and bool(always) and bool(sometimes)
    return {"loose": loose and obj, "strict": strict and obj}


def universe() -> tuple[int, str]:
    return SEEDS, (f"every seed of the {SEEDS}-seed pilot family at U0's own genome, "
                   f"the same population COUPLING_RATIO_001 scored; none sampled")


def measure() -> tuple[str, str]:
    g = U0().replace(**FAMILY)
    loose, strict = [], []
    for i in range(SEEDS):
        r = _classify(g, i)
        if r is None:
            continue
        if r["loose"]:
            loose.append(i)
        if r["strict"]:
            strict.append(i)
    lost = sorted(set(loose) - set(strict))
    gained = sorted(set(strict) - set(loose))
    print(f"loose  (x(T) vs x(T-2) only): {len(loose)}  {loose}")
    print(f"strict (all {WINDOW + 2} frames):     {len(strict)}  {strict}")
    print(f"dropped by the strict test: {lost}")
    if gained:
        print(f"*** GAINED under the strict test, which should be impossible: {gained}")
    if tuple(loose) != LOOSE_HITS:
        print(f"NOTE: the loose arm reproduced {tuple(loose)}, and the sealed run "
              f"recorded {LOOSE_HITS}")
    return (f"{len(strict)} of {SEEDS} strict, {len(loose)} loose",
            f"the claim says the strict count is smaller "
            f"({'HOLDS' if len(strict) < len(loose) else 'DOES NOT HOLD -- the one-pair test was not loose on this population'})")
