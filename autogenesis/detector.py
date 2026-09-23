"""BISTABILITY DETECTOR v0 -- implements MEASUREMENT_SPEC_V1_DRAFT.

NOT FROZEN. Every threshold here is provisional and must be set by calibration
against `calibration/labelled.json`, never chosen to make a candidate pass.

The core claim this file makes, and the one most likely to be wrong:

    memory is ATTRACTOR MULTIPLICITY, not trajectory change.

Perturbing a chaotic soup changes its trajectory forever; that is not storage.
The discriminator is that a memory reaches FEW distinct attractors from MANY
distinct perturbations, and that the count stops growing when you try more
perturbations. Both are measured here, and a verdict that reports k without
reporting k-growth is not permitted.

Attractor identity (V1 choice, flagged for audit):
  - cyclic-rotation-canonical hash of the discrete (activity, topology) sequence
  - AND a rigid-motion-invariant geometric fingerprint: the sorted vector of
    pairwise distances, compared with tolerance.
  The pair is used because topology alone MERGES geometrically distinct
  attractors (a trap recorded in the original project's own audit), while
  geometry alone splits the same attractor seen at different phases.
"""
from __future__ import annotations
import hashlib
from dataclasses import dataclass
import numpy as np

from .genome import Genome
from .state import State
from . import simulator as sim
from .metrics import extinction_is_absorbing   # F-01 / MS-C-586

# --- provisional thresholds (CALIBRATE, do not tune) -----------------------
SETTLE = 256          # steps after an intervention before observing
WINDOW = 96           # observation window
MAX_PERIOD = 24       # longest cycle we attempt to canonicalise
N_PERTURB = 16        # N in k(N)
GEOM_TOL = 1e-3       # rigid-motion-invariant distance-fingerprint tolerance
K_MAX_FRACTION = 0.5  # k must be <= this fraction of N, else "chaotic"
K_GROWTH_MAX = 1.35   # k(2N)/k(N) above this => growing => chaotic


@dataclass
class Attractor:
    period: int | None
    disc_hash: str          # canonical discrete signature, phase-invariant
    geom: np.ndarray | None  # sorted pairwise distances, rigid-motion invariant
    dead: bool

    def same_as(self, other: "Attractor", tol=GEOM_TOL) -> bool:
        if self.dead or other.dead:
            return self.dead and other.dead
        if self.disc_hash != other.disc_hash:
            return False
        if self.geom is None or other.geom is None:
            return True
        if self.geom.shape != other.geom.shape:
            return False
        return float(np.abs(self.geom - other.geom).max()) <= tol


def _geom_fingerprint(s: State) -> np.ndarray:
    """Invariant under translation, rotation AND particle relabelling."""
    d = s.x[:, None, :] - s.x[None, :, :]
    r = np.sqrt((d * d).sum(-1))
    return np.sort(r[np.triu_indices(s.N, 1)])


def observe(g: Genome, s: State, settle=SETTLE, window=WINDOW) -> Attractor:
    """Settle, then canonicalise the trailing behaviour into an identity."""
    absorbing = extinction_is_absorbing(g)                     # F-01 / MS-C-586
    for _ in range(settle):
        s = sim.step(g, s)
        if not s.active.any() and absorbing:   # F-01 / MS-C-586
            return Attractor(None, "DEAD", None, True)
    sigs, geoms = [], []
    for _ in range(window):
        s = sim.step(g, s)
        if not s.active.any() and absorbing:   # F-01 / MS-C-586
            return Attractor(None, "DEAD", None, True)
        sigs.append(s.discrete_signature())
        geoms.append(_geom_fingerprint(s))

    period = None
    for p in range(1, MAX_PERIOD + 1):
        if all(sigs[i] == sigs[i - p] for i in range(p, len(sigs))):
            period = p
            break

    if period is None:
        # Non-recurrent within MAX_PERIOD. Canonicalise over the whole window so
        # two such outcomes can still be compared, but flag the period as
        # unknown -- callers must not read this as "one attractor".
        h = hashlib.sha256("||".join(sigs).encode()).hexdigest()[:16]
        return Attractor(None, h, np.mean(geoms, axis=0), False)

    cyc = sigs[-period:]
    canon = min(tuple(cyc[k:] + cyc[:k]) for k in range(period))
    h = hashlib.sha256("||".join(canon).encode()).hexdigest()[:16]
    return Attractor(period, h, np.mean(geoms[-period:], axis=0), False)


def _flip(s: State, i: int) -> State:
    out = s.copy()
    out.active[i] = not out.active[i]
    return out


def _distinct(atts: list[Attractor]) -> list[Attractor]:
    reps: list[Attractor] = []
    for a in atts:
        if not any(a.same_as(r) for r in reps):
            reps.append(a)
    return reps


def classify(g: Genome, s0: State, n_perturb: int = N_PERTURB,
             warmup: int = 0) -> dict:
    """Verdict plus every control that could expose it. Controls are not
    optional and are returned whether or not they look good."""
    absorbing = extinction_is_absorbing(g)                     # F-01 / MS-C-586
    s = s0.copy()
    for _ in range(warmup):
        s = sim.step(g, s)
        if not s.active.any() and absorbing:   # F-01 / MS-C-586
            break

    base = observe(g, s)
    if base.dead:
        return {"verdict": "NOT_BISTABLE", "reason": "extinct before observation",
                "k": 0, "k_growth": None, "base_period": None,
                "null_control_k": None, "n_perturb": 0}

    # --- control A: null intervention. Flip a node and flip it straight back.
    # The system must land in exactly the base attractor. If it does not, we are
    # measuring integrator sensitivity, not memory, and nothing below is valid.
    null_atts = []
    for i in range(min(8, s.N)):
        t = _flip(_flip(s, i), i)          # identity operation, by construction
        null_atts.append(observe(g, t))
    null_k = len(_distinct(null_atts + [base]))

    # --- M3: k(N)
    order = np.argsort([hash((int(a), i)) for i, a in enumerate(s.active)])
    idx = [int(i) for i in order[:n_perturb]]
    atts = [observe(g, _flip(s, i)) for i in idx]
    reps = _distinct(atts + [base])
    k = len(reps)

    # --- control B: k(2N). Growth means chaos, not capacity.
    idx2 = [int(i) for i in order[:min(2 * n_perturb, s.N)]]
    atts2 = [observe(g, _flip(s, i)) for i in idx2]
    k2 = len(_distinct(atts2 + [base]))
    growth = k2 / k if k else None

    live = [a for a in atts if not a.dead]
    n_live = len(live)

    if null_k != 1:
        verdict, reason = "UNDECIDABLE", (
            f"null-intervention control produced {null_k} attractors; the "
            f"measurement is sensitive to an operation that changes nothing")
    elif k <= 1:
        verdict, reason = "NOT_BISTABLE", "single attractor under all perturbations"
    elif n_live == 0:
        verdict, reason = "NOT_BISTABLE", (
            "every perturbation killed the structure; fragility, not storage")
    elif k > max(2, K_MAX_FRACTION * len(idx)):
        verdict, reason = "NOT_BISTABLE", (
            f"k={k} of {len(idx)} perturbations -- near-unique outcomes, chaotic")
    elif growth is not None and growth > K_GROWTH_MAX:
        verdict, reason = "NOT_BISTABLE", (
            f"k grew {k}->{k2} when perturbations doubled; chaotic, not finite "
            f"storage")
    else:
        verdict, reason = "BISTABLE", (
            f"k={k} distinct attractors from {len(idx)} perturbations, stable "
            f"under doubling (k2={k2})")

    return {"verdict": verdict, "reason": reason,
            "k": k, "k2": k2, "k_growth": growth,
            "n_perturb": len(idx), "n_perturb_2": len(idx2),
            "n_lethal": len(idx) - n_live,
            "base_period": base.period,
            "null_control_k": null_k,
            "periods": sorted({a.period for a in reps if a.period is not None})}
