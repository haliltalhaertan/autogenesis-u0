"""Is the object topological, or is the bond motion a shadow of a node blinking?

THE DEFLATIONARY READING, WHICH `L-481` DID NOT RULE OUT. At U0 seed 8 the positions are
bit-frozen while node 3's activity and bonds `(1,3)`, `(3,10)` cycle with period 2. That
was reported as a topology oscillating at rest. But there is a cheaper explanation: node 3
may be an ordinary Boolean blinker that would oscillate on the graph as it stands, with
its bonds merely FOLLOWING it because retention requires an active endpoint. If so the
object is a plain Life-Like Network Automaton 2-cycle -- known since Miranda 2016, and
the analogue of a Conway blinker -- and nothing about it is topological.

`MS-C-730b` does NOT settle this. That theorem forbids a period->=2 orbit all of whose
states are vertex covers; here 3 of 24 nodes are active over 12 edges, which is almost
certainly not a cover, so oscillation on a fixed graph is not excluded.

THE DISCRIMINATING TEST. Freeze the bond set and run the node rule alone from the settled
state. If node 3 still oscillates, the bond motion is a shadow and the deflationary
reading wins. If the activity falls to a fixed point, the bond motion is load-bearing and
the oscillation genuinely needs the topology to move.

HOW THE BONDS ARE FROZEN WITHOUT TOUCHING THE NODE RULE. `candidate_radius` is shrunk so
no pair is a candidate, which makes formation impossible; `retention_requires_active_
endpoint` is turned off so nothing is dropped for lack of an active endpoint, and
`retention_support_min` is already 0 at U0. The node rule reads `bonds`, never `cand`
(`kernel.py`, the node update loop), so it is bit-identical under this change. That the
graph really does freeze is not assumed -- it is the first control below.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from autogenesis.genome import U0                              # noqa: E402
from autogenesis.init import make_initial_state                # noqa: E402
from autogenesis import simulator as sim                       # noqa: E402
from lo_zero_period_audit import HORIZON, NAMESPACE            # noqa: E402

FAMILY = {"initial_conditions.N": 24, "initial_conditions.cube_side": 3.6,
          "initial_conditions.bond_init_mode": "mean_degree",
          "initial_conditions.target_mean_degree": 3.0,
          "initial_conditions.initial_degree_cap": 4}
HITS = (8, 18, 19, 35, 38, 63, 68)
PROBE = 64                      # steps to run after freezing the bonds

CLAIM = ("On the settled state of each seed that carries the object, freezing the bond "
         "set kills the activity oscillation. The node rule alone, on the graph as it "
         "stands, goes to a fixed point -- so the bond motion is load-bearing and the "
         "oscillation is genuinely topological, not a blinker with bonds in tow.")

WOULD_OVERTURN = ("Activity still oscillating with the bonds frozen, on any seed. That "
                  "would make the object an ordinary LLNA 2-cycle whose bond changes "
                  "merely follow it, known since Miranda 2016, and `L-481` would have to "
                  "be narrowed to say so. Also: the frozen-bond genome failing to freeze "
                  "the bonds, in which case the test measured nothing.")


def _live():
    return U0().replace(**FAMILY)


def _frozen_bonds():
    """Same node rule, bonds that can neither form nor break.

    THE FIRST VERSION OF THIS FUNCTION WAS WRONG AND IT MATTERED. It also set
    `break_radius = 0.01`. Retention requires `r <= R_break` and every bonded pair sits at
    `r0 = 1.0`, so nothing was frozen -- every bond was CUT on the first step. The setup
    control passed anyway, because it asked whether the edge set was CONSTANT and "empty
    from step one onwards" is constant. With every degree at zero the node rule makes
    every node passive, the activity went (old set) -> (empty) and stayed there, and the
    oscillation test counted that single transition as blinking. Every seed came back
    SHADOW and the deflationary reading looked confirmed. It measured nothing.

    `break_radius` therefore stays at U0's 1.60. Only `candidate_radius` shrinks, which
    blocks FORMATION (`elig` requires `cand`), and `retention_requires_active_endpoint`
    goes off, which blocks the activity-driven DROP. `retention_support_min` is 0 at U0,
    so the support term cannot drop anything either.
    """
    g = U0().replace(**FAMILY, **{
        "topology.candidate_radius": 0.01,          # nothing is a candidate -> no formation
        "topology.retention_requires_active_endpoint": False})
    if g.topology.break_radius != U0().topology.break_radius:
        raise ValueError("break_radius must stay at U0's value or retention cuts every "
                         "bond instead of freezing it")
    return g


def _settle(seed: int):
    g = _live()
    s = make_initial_state(g, NAMESPACE, seed)
    for _ in range(HORIZON):
        s = sim.step(g, s)
    return s


def _run(g, s0, steps: int):
    """(activity signatures, edge-set signatures) over `steps` steps from `s0`."""
    s, acts, edges = s0.copy(), [], []
    for _ in range(steps):
        s = sim.step(g, s)
        acts.append(tuple(int(i) for i in np.flatnonzero(s.active)))
        edges.append(frozenset(s.edge_list()))
    return acts, edges


SUSTAIN = 16                    # the tail that must still be moving


def criterion(seq) -> bool:
    """True when the sequence is STILL oscillating at the end, not merely different once.

    The first version returned `len(set(seq)) > 1` over the whole probe, which counts a
    one-time decay -- alive, then dead forever -- as an oscillation. That is what made a
    broken setup look like a confirmed result.
    """
    return len(set(seq[-SUSTAIN:])) > 1


CONTROLS = [
    ("a constant sequence does NOT oscillate", [("a",)] * 20, False),
    ("an alternating sequence DOES oscillate", [("a",), ("b",)] * 10, True),
    # THE control the first version got backwards:
    ("alive once and then dead forever does NOT oscillate",
     [("a",), ("b",)] + [("c",)] * 18, False),
    ("still moving only at the very end DOES oscillate",
     [("a",)] * 18 + [("b",), ("a",)], True),
]


def universe() -> tuple[int, str]:
    return len(HITS), ("every seed COUPLING_RATIO_001 scored as OBJECT_strong at U0's "
                       "own genome; all seven are tested, none sampled")


def measure() -> tuple[str, str]:
    live, frozen = _live(), _frozen_bonds()
    print(f"{'seed':>6}{'bonds froze?':>14}{'live: act':>11}{'edges':>7}"
          f"{'frozen: act':>13}{'edges':>7}   reading")
    shadow, topological, broken = [], [], []
    for seed in HITS:
        s0 = _settle(seed)
        la, le = _run(live, s0, PROBE)

        # BOTH PHASES. The settled trajectory alternates between two edge sets, and
        # freezing only the one that happens to sit at step HORIZON would leave the
        # answer phase-dependent and untested. `s1` is the other phase.
        s1 = sim.step(live, s0)
        blinks, froze_ok = [], True
        for phase, s in (("A", s0), ("B", s1)):
            fa, fe = _run(frozen, s, PROBE)
            # The control the whole test rests on, and the one the first version got
            # wrong: the frozen arm must hold THAT PHASE'S edge set, not merely a
            # constant one. "empty from step one and empty ever after" is constant too,
            # and that is a cut graph rather than a frozen one.
            start = frozenset(s.edge_list())
            if not (len(set(fe)) == 1 and fe[0] == start and len(start) > 0):
                froze_ok = False
            blinks.append(criterion(fa))

        if not froze_ok:
            broken.append(seed)
        still = any(blinks)
        (shadow if still else topological).append(seed)
        print(f"{seed:>6}{('yes' if froze_ok else 'NO'):>14}"
              f"{len(set(la)):>11}{len(set(le)):>7}"
              f"{str(blinks):>13}{len(set(fe)):>7}   "
              f"{'SHADOW -- blinks anyway' if still else 'topological -- dies'}")

    if broken:
        return "SETUP FAILED", (f"the frozen-bond genome did not freeze the bonds on "
                                f"seeds {broken}; nothing was measured")
    print(f"\n{len(shadow)} seed(s) keep blinking with the bonds frozen: {shadow}")
    print(f"{len(topological)} seed(s) fall to a fixed point: {topological}")
    return (f"{len(topological)} of {len(HITS)} topological, {len(shadow)} shadow",
            f"the claim says ALL seven are topological "
            f"({'HOLDS' if not shadow else 'DOES NOT HOLD -- ' + str(len(shadow)) + ' are ordinary LLNA blinkers'})")
