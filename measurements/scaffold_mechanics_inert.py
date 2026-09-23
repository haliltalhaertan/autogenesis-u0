"""In the scaffold regime, does the continuous layer move at all?

Reported in conversation as *537 sampled states, 0 with any position or velocity change*,
with no artefact and -- more importantly -- **with no control showing the detector could
see movement if there were any**. A null result from an instrument never shown to respond
is not a null result; it is silence.

That control is the negative one below: a state with one node pushed off `r0` MUST come
back as moving. If it does not, every zero above means nothing.

WHY THE QUESTION MATTERS. `CHECKPOINT_063`-`077` -- `G3`, `P4` and the `T0` column, the
project's headline results -- all run on scaffolds. If positions never move there, those
results come from a discrete system on a fixed graph, and the entire continuous engine
(mass, springs, damping, repulsion, integrator, timestep) is switched off while they are
computed. `MS-C-741` predicts exactly that: with every candidate pair at exactly `r0` the
force is bit-exactly zero.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from autogenesis.genome import U0                              # noqa: E402
from autogenesis import simulator as sim                       # noqa: E402
from rest_scaffold_orbits import (Scaffold, candidate_edges,   # noqa: E402
                                  scaffold_hex)

STRIDE = 977            # sample every 977th state; coprime with the space size

CLAIM = ("On HEX_7 at rest, stepping the production engine changes no position and no "
         "velocity in any sampled state. The mechanical layer contributes nothing in the "
         "regime where CHECKPOINT_063-077 computed G3, P4 and the T0 column.")

WOULD_OVERTURN = ("Any sampled state showing a non-zero |dx| or |du|; or the negative "
                  "control failing -- if a deliberately perturbed state ALSO comes back "
                  "as unmoved, the detector is blind and every zero is meaningless.")


def _scaffold():
    g = U0()
    X = scaffold_hex()
    return g, Scaffold(g, X, candidate_edges(g, X))


def criterion(state) -> bool:
    """Did one engine step move anything continuous? True means it moved."""
    g = U0()
    out = sim.step(g, state)
    return bool(np.abs(out.x - state.x).max() or np.abs(out.u - state.u).max())


def _rest_state(code: int):
    _, sc = _scaffold()
    return sc.state(code)


def _perturbed_state(code: int, push: float):
    """A rest state with one node displaced. It MUST move: the springs pull it back."""
    s = _rest_state(code).copy()
    s.x[0, 0] += push
    return s


ALL_BONDED_ACTIVE = None          # filled in by universe(), which builds the scaffold


CONTROLS = [
    # positive: the detector must report movement when there is movement
    ("a node pushed 0.20 off r0 MUST be seen to move",
     None, True),
    ("a node pushed 0.02 off r0 MUST also be seen to move",
     None, True),
    # negative: an untouched rest state must come back unmoved
    ("an untouched rest state does NOT move", None, False),
]


def _fill_controls():
    """Controls need real states, and the scaffold has to exist first."""
    _, sc = _scaffold()
    code = ((1 << sc.E) - 1) << sc.n | ((1 << sc.n) - 1)      # all bonded, all active
    CONTROLS[0] = (CONTROLS[0][0], _perturbed_state(code, 0.20), True)
    CONTROLS[1] = (CONTROLS[1][0], _perturbed_state(code, 0.02), True)
    CONTROLS[2] = (CONTROLS[2][0], _rest_state(code), False)


_fill_controls()


def universe() -> tuple[int, str]:
    _, sc = _scaffold()
    total = 1 << (sc.E + sc.n)
    return total, (f"every state of HEX_7: 2^{sc.E} bond words x 2^{sc.n} activity "
                   f"words; this measurement samples every {STRIDE}th")


def measure() -> tuple[str, str]:
    _, sc = _scaffold()
    total = 1 << (sc.E + sc.n)
    codes = range(0, total, STRIDE)
    moved, worst = 0, 0.0
    for code in codes:
        s = sc.state(code)
        out = sim.step(U0(), s)
        d = max(float(np.abs(out.x - s.x).max()), float(np.abs(out.u - s.u).max()))
        worst = max(worst, d)
        moved += d != 0.0
    print(f"scaffold: n={sc.n} nodes, E={sc.E} candidate edges, {total:,} states")
    print(f"sampled {len(codes)}, moved {moved}, largest |dx| or |du| = {worst!r}")
    return (f"{moved} of {len(codes)} moved",
            f"largest displacement {worst!r}; the claim holds iff this is 0.0 "
            f"({'HOLDS' if moved == 0 else 'DOES NOT HOLD'})")
