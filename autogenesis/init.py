"""Initial-condition generator.

Faithful reimplementation of the procedure described in
PHASE_B_FROZEN_PROTOCOL.json -> initial_conditions.

HONESTY NOTE (docs/DECISIONS.md D-003): this reproduces the described
PROCEDURE, not the original implementation's exact PRNG consumption order.
Initial states are therefore NOT expected to be bit-identical to the original
U0 production runs.  Bit-equality is claimed only for `step()` -- see
tests/test_u0_equivalence.py, which feeds BOTH engines the same state.
"""
from __future__ import annotations
import numpy as np
from .genome import Genome
from .state import State
from .rng import streams


class InitFailure(RuntimeError):
    """Rejection sampling could not place N particles. Fail loud, never relax
    the separation floor to 'save' a configuration."""


def _sample_positions(rng, N, D, side, min_sep, max_tries_per_point=10_000):
    pts = np.empty((N, D), dtype=np.float64)
    half = side / 2.0
    m2 = min_sep * min_sep
    for i in range(N):
        for _ in range(max_tries_per_point):
            p = rng.uniform(-half, half, size=D)
            if i == 0:
                pts[0] = p
                break
            d = pts[:i] - p
            if float((d * d).sum(1).min()) >= m2:
                pts[i] = p
                break
        else:
            raise InitFailure(
                f"could not place particle {i}/{N} in side={side} with "
                f"min_separation={min_sep} (density too high)")
    pts -= pts.mean(0)          # arithmetic COM removal, no relaxation
    return pts


def make_initial_state(g: Genome, namespace: str, index: int) -> State:
    g.validate()
    ic, t = g.initial_conditions, g.topology
    D = g.space.dimensions
    N = ic.N
    st = streams(namespace, index)

    x = _sample_positions(st["coordinates"], N, D, ic.cube_side, ic.min_separation)
    u = np.zeros((N, D), dtype=np.float64)

    n_active = int(round(N * ic.active_fraction))
    active = np.zeros(N, dtype=bool)
    active[st["activity"].choice(N, size=n_active, replace=False)] = True

    bonds = np.zeros((N, N), dtype=bool)
    want_bonds = (ic.bond_init_mode == "mean_degree" and ic.target_mean_degree > 0)         or (ic.bond_init_mode == "probability" and ic.bond_probability > 0.0)
    if want_bonds:
        d = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)
        ii, jj = np.nonzero(np.triu(d <= t.candidate_radius, 1))
        rb = st["bonds"]
        keys = rb.random(len(ii))            # one key per candidate edge
        order = np.argsort(keys, kind="stable")
        deg = np.zeros(N, dtype=np.int32)

        if ic.bond_init_mode == "probability":
            succ = rb.random(len(ii)) < ic.bond_probability
            for k in order:
                if not succ[k]:
                    continue
                i, j = int(ii[k]), int(jj[k])
                if deg[i] < ic.initial_degree_cap and deg[j] < ic.initial_degree_cap:
                    bonds[i, j] = bonds[j, i] = True
                    deg[i] += 1
                    deg[j] += 1
        else:
            # Add candidate edges in random-key order until the target mean
            # degree is met. Stops early if the candidate graph or the degree
            # cap runs out -- the shortfall is real information, not an error,
            # so it is reported by achieved_mean_degree() rather than raised.
            target_edges = int(round(N * ic.target_mean_degree / 2.0))
            placed = 0
            for k in order:
                if placed >= target_edges:
                    break
                i, j = int(ii[k]), int(jj[k])
                if deg[i] < ic.initial_degree_cap and deg[j] < ic.initial_degree_cap:
                    bonds[i, j] = bonds[j, i] = True
                    deg[i] += 1
                    deg[j] += 1
                    placed += 1

    s = State(x, u, active, bonds)
    s.check(t.valence)
    return s


def achieved_mean_degree(s: State) -> float:
    """What <k0> the generator actually produced. Requested != achieved
    whenever density or the degree cap binds; always report this, never the
    requested value."""
    return float(s.degrees().mean())
