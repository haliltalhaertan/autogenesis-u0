"""REFERENCE ENGINE -- naive, slow, obviously correct.

Its only job is to be the oracle the fast kernel is diffed against.
Optimise NOTHING here.  Clarity beats speed; every line should map onto a
sentence of PHASE_B_FROZEN_PROTOCOL.json.

Step order (fixed, do not reorder):
  1. candidate pairs from x_t                    (dist <= R_candidate)
  2. common-active-support scores from x_t, a_t
  3. node update       -- uses OLD active, OLD bonds, synchronous
  4. retention (kept)  -- OLD bonds surviving: endpoint-active AND dist <= R_break
                          AND common-active-support >= retention_support_min
  5. eligibility       -- OLD active, OLD degrees, support >= min, not bonded
  6. allocation        -- valence-capped, deterministic, order-independent
  7. forces            -- from x_t, u_t and the NEW bond set
  8. semi-implicit Euler: u_{t+1} = u_t + dt*F/m ; x_{t+1} = x_t + dt*u_{t+1}
"""
from __future__ import annotations
from collections import Counter, defaultdict
import numpy as np
from .genome import Genome
from .state import State


# ---------------------------------------------------------------- geometry --
def pair_distances(x: np.ndarray) -> np.ndarray:
    d = x[:, None, :] - x[None, :, :]
    r = np.sqrt((d * d).sum(-1))
    np.fill_diagonal(r, np.inf)
    return r


def candidate_edges(r: np.ndarray, R_candidate: float) -> list[tuple[int, int]]:
    ii, jj = np.nonzero(np.triu(r <= R_candidate, 1))
    return sorted(zip(ii.tolist(), jj.tolist()))


def support_scores(cand, N, active) -> dict[tuple[int, int], int]:
    """# of active k (k != i,j) that are candidate-neighbours of BOTH i and j."""
    adj = [set() for _ in range(N)]
    for i, j in cand:
        adj[i].add(j)
        adj[j].add(i)
    out = {}
    for i, j in cand:
        small, other = (adj[i], adj[j]) if len(adj[i]) <= len(adj[j]) else (adj[j], adj[i])
        out[(i, j)] = sum(1 for k in small if k in other and bool(active[k]))
    return out


# ------------------------------------------------------------------- rules --
def node_update(g: Genome, active, bonds) -> np.ndarray:
    """Synchronous. Uses OLD active flags and the OLD bond graph.

    Band selection depends on the node's CURRENT state (D-010):
      passive now -> birth band ;  active now -> survival band.
    """
    t = g.topology
    N = len(active)
    new = np.zeros(N, dtype=bool)
    for i in range(N):
        nb = np.nonzero(bonds[i])[0]
        if nb.size == 0:                       # degree_zero -> passive
            continue
        frac = float(active[nb].sum()) / float(nb.size)
        lo, hi = ((t.survival_lo, t.survival_hi) if active[i]
                  else (t.birth_lo, t.birth_hi))
        if lo <= frac <= hi:
            new[i] = True
    return new


def retained_edges(g: Genome, s: State, r, support) -> set[tuple[int, int]]:
    t = g.topology
    kept = set()
    for i, j in s.edge_list():
        if t.retention_requires_active_endpoint and not (s.active[i] or s.active[j]):
            continue
        if r[i, j] > t.break_radius:
            continue
        if support.get((i, j), 0) < t.retention_support_min:
            continue
        kept.add((i, j))
    return kept


def eligible_proposals(g: Genome, s: State, cand, support) -> set[tuple[int, int]]:
    t = g.topology
    olddeg = s.degrees()                       # OLD degrees, per protocol
    out = set()
    for e in cand:
        i, j = e
        if s.bonds[i, j]:
            continue
        if t.formation_requires_active_endpoint and not (s.active[i] or s.active[j]):
            continue
        if olddeg[i] >= t.valence or olddeg[j] >= t.valence:
            continue
        if support[e] < t.formation_support_min:
            continue
        out.add(e)
    return out


# -------------------------------------------------------------- allocators --
def _kept_degrees(kept, N):
    d = np.zeros(N, dtype=np.int32)
    for i, j in kept:
        d[i] += 1
        d[j] += 1
    return d


def allocate_A_batch_all(elig, kept, N, v, support):
    kd = _kept_degrees(kept, N)
    inc = Counter()
    for i, j in elig:
        inc[i] += 1
        inc[j] += 1
    ok = {p: (c <= v - int(kd[p])) for p, c in inc.items()}
    return {e for e in elig if ok.get(e[0], False) and ok.get(e[1], False)}


def allocate_B_support_orbit(elig, kept, N, v, support):
    kd = _kept_degrees(kept, N)
    incident = defaultdict(list)
    for e in elig:
        incident[e[0]].append(e)
        incident[e[1]].append(e)
    accepted = {}
    for p, es in incident.items():
        rem = v - int(kd[p])
        groups = defaultdict(list)
        for e in es:
            groups[support[e]].append(e)
        acc = set()
        for s in sorted(groups, reverse=True):      # high support first
            g_ = groups[s]
            if len(g_) <= rem:                      # whole tied class or nothing
                acc.update(g_)
                rem -= len(g_)
            else:
                break
        accepted[p] = acc
    return {e for e in elig
            if e in accepted.get(e[0], set()) and e in accepted.get(e[1], set())}


def allocate_C_overload_prune(elig, kept, N, v, support):
    kd = _kept_degrees(kept, N)
    T = set(elig)
    for _ in range(2):                              # exactly two bounded rounds
        inc = Counter()
        for i, j in T:
            inc[i] += 1
            inc[j] += 1
        over = {p for p, c in inc.items() if int(kd[p]) + c > v}
        if not over:
            break
        rem = {e for e in T if e[0] in over and e[1] in over}
        if not rem:
            break
        T -= rem
    inc = Counter()
    for i, j in T:
        inc[i] += 1
        inc[j] += 1
    over = {p for p, c in inc.items() if int(kd[p]) + c > v}
    return {e for e in T if e[0] not in over and e[1] not in over}


ALLOCATORS = {
    "A_batch_all": allocate_A_batch_all,
    "B_support_orbit": allocate_B_support_orbit,
    "C_overload_prune": allocate_C_overload_prune,
}


# ------------------------------------------------------------------ forces --
def forces(g: Genome, x, u, bonds) -> np.ndarray:
    p = g.physics
    N, D = x.shape
    F = -p.gamma * u.copy()
    for i in range(N):
        for j in range(i + 1, N):
            dr = x[j] - x[i]
            r = float(np.sqrt((dr * dr).sum()))
            if r < 1e-12:
                raise FloatingPointError(f"coincident pair {i},{j}")
            rh = dr / r
            if bonds[i, j]:
                fij = p.spring_k * (r - p.r0) * rh          # attractive if r>r0
            elif r < p.repulsion_cutoff:
                fij = -p.repulsion_strength * (p.repulsion_cutoff - r) * rh
            else:
                continue
            F[i] += fij
            F[j] -= fij                                     # antisymmetry by construction
    if not np.isfinite(F).all():
        raise FloatingPointError("nonfinite force")
    return F


# -------------------------------------------------------------------- step --
def step(g: Genome, s: State) -> State:
    t, nm, ph = g.topology, g.numerics, g.physics
    N = s.N
    r = pair_distances(s.x)
    cand = candidate_edges(r, t.candidate_radius)
    sup = support_scores(cand, N, s.active)

    new_active = node_update(g, s.active, s.bonds)
    kept = retained_edges(g, s, r, sup)
    elig = eligible_proposals(g, s, cand, sup)
    formed = ALLOCATORS[t.allocator](elig, kept, N, t.valence, sup)

    nb = np.zeros((N, N), dtype=bool)
    for i, j in (kept | formed):
        nb[i, j] = nb[j, i] = True

    F = forces(g, s.x, s.u, nb)
    nu = s.u + nm.dt * F / ph.mass
    nx = s.x + nm.dt * nu

    out = State(nx, nu, new_active, nb)
    out.check(t.valence)
    return out


def run(g: Genome, s: State, steps: int) -> State:
    for _ in range(steps):
        s = step(g, s)
    return s
