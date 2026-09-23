"""Stage-A metrics: cheap enough to run on every universe, integer/hash only.

Two design decisions worth stating, because they shape what the search can see.

1. GRADED, NOT BINARY. "survived: yes/no" is a ~2% rare event under U0 and gives
   a search algorithm nothing to climb. `log_persistence` is defined for dead
   runs too, so the dead region still carries a gradient toward the living one.

2. SURVIVING IS NOT INTERESTING. A frozen crystal survives forever and is inert.
   The tail classification separates dead / frozen / periodic / rich at
   essentially zero cost, by counting DISTINCT discrete signatures in a trailing
   window. Dead=0, frozen=1, period-p=p, rich=many.

Nothing here measures memory, computation or autonomy. Those need interventional
definitions and a measured false-positive rate (MEASUREMENT_SPEC_V1) and are
deliberately absent.
"""
from __future__ import annotations
import math
import numpy as np
from .genome import Genome
from .state import State
from .init import make_initial_state, achieved_mean_degree, InitFailure
from . import simulator as sim

TAIL_WINDOW = 128
RICH_THRESHOLD = 16          # distinct tail signatures above which we say "rich"
RUNAWAY_RADIUS = 1.0e3

CLASSES = ("EXTINCT", "RUNAWAY", "FROZEN", "PERIODIC", "RICH",
           "INIT_FAIL", "NUMERICAL_FAIL")


def extinction_is_absorbing(g) -> bool:
    """Is a zero-activity state a DEAD END under this genome? (F-01, MS-C-586)

    The old code assumed it always was -- `if not s.active.any(): break` with the
    comment "absorbing under U0's rule". True for U0, false in general, and the
    sweep it fed ranged over bands where it is false: if 0 lies in the BIRTH band,
    a passive node with no active neighbours sees fraction 0 and is BORN. Zero
    activity is a transient, not an end.

    THE ONLY ROUTE BACK IS BIRTH. From all-passive, no node is active, so the
    survival band is vacuous; a node can only become active by BIRTH, which needs
    its active-neighbour fraction -- necessarily 0 -- to lie in `[birth_lo,
    birth_hi]`. Bond formation cannot help: it changes degrees, and the fraction is
    0 at every positive degree. So the condition is about the birth band ALONE.

    MS-C-720: the first version of this predicate read

        return not (zero_is_born or can_form_without_activity)

    which is `(not zero_is_born) AND formation_requires_active_endpoint` -- it
    demanded BOTH. A genome whose birth band excludes 0 (U0's own `[1/6, 1/2]`) but
    whose formation is unconditional was reported NOT absorbing, so a world that is
    permanently and provably dead ran to the horizon and was labelled FROZEN instead
    of EXTINCT. Reproduced: U0 band, `active_fraction = 0`, `bond_probability = 0`,
    `formation_requires_active_endpoint = False` -- 0 active at every step, never
    revives, predicate said False.

    Returning False costs a full-horizon run; returning True wrongly labels a live
    trajectory EXTINCT. The asymmetry is deliberate: this must never say True when
    the state can revive. The converse -- saying False for a world that is in fact
    dead -- is only a cost, and is what this function got wrong.
    """
    t = g.topology
    zero_is_born = t.birth_lo <= 0.0 <= t.birth_hi
    return not zero_is_born


def screen(g: Genome, namespace: str, index: int, horizon: int,
           tail_window: int = TAIL_WINDOW) -> dict:
    """One cheap screening run. Early-exits on extinction and runaway."""
    rec = {"genome_hash": g.short_hash(), "namespace": namespace,
           "index": index, "horizon": horizon}
    try:
        s = make_initial_state(g, namespace, index)
    except InitFailure as e:
        rec.update(klass="INIT_FAIL", detail=str(e)[:70])
        return rec

    rec["k0_achieved"] = round(achieved_mean_degree(s), 4)
    rec["k0_requested"] = g.initial_conditions.target_mean_degree

    tail: list[str] = []
    peak_deg = int(s.degrees().max()) if s.N else 0
    max_r = 0.0
    ext = -1

    absorbing = extinction_is_absorbing(g)
    try:
        for t in range(1, horizon + 1):
            s = sim.step(g, s)
            d = s.degrees()
            if d.size:
                peak_deg = max(peak_deg, int(d.max()))
            r = float(np.linalg.norm(s.x, axis=1).max())
            if r > max_r:
                max_r = r

            if not s.active.any():
                if absorbing:                      # F-01: not always true -- MS-C-586
                    ext = t
                    break
                # otherwise zero activity is a transient; keep integrating
            if r > RUNAWAY_RADIUS:
                rec.update(klass="RUNAWAY", extinction_step=-1, last_step=t,
                           peak_degree=peak_deg, max_radius=max_r,
                           tail_distinct=-1, final_edges=len(s.edge_list()))
                rec["log_persistence"] = math.log(t)
                return rec

            if t > horizon - tail_window:
                tail.append(s.discrete_signature())
    except (FloatingPointError, AssertionError) as e:
        rec.update(klass="NUMERICAL_FAIL",
                   detail=f"{type(e).__name__}: {str(e)[:60]}")
        return rec

    survived = ext < 0
    last = horizon if survived else ext
    distinct = len(set(tail)) if survived else 0

    if not survived:
        klass = "EXTINCT"
    elif distinct <= 1:
        klass = "FROZEN"
    elif distinct <= RICH_THRESHOLD:
        klass = "PERIODIC"
    else:
        klass = "RICH"

    rec.update(klass=klass,
               extinction_step=ext,
               last_step=last,
               survived=survived,
               peak_degree=peak_deg,
               max_radius=round(max_r, 4),
               tail_distinct=distinct,
               final_edges=len(s.edge_list()),
               final_active=int(s.active.sum()),
               # graded fitness: defined for dead runs too, so the dead region
               # still has a gradient pointing toward the living one
               log_persistence=round(math.log(max(1, last)), 6))
    return rec
