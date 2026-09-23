"""FAST KERNEL -- numba-compiled, array-only, no Python objects.

CONTRACT: bit-for-bit identical to autogenesis.reference.  Not "close".
Identical.  tests/test_differential.py enforces it.

To keep floating-point results identical, this file reproduces the REFERENCE's
arithmetic *order*, not merely its formulas:
  - pairs are visited in lexicographic (i, j) order, i < j
  - r2 accumulates as ((dx*dx + dy*dy) + dz*dz)
  - the unit vector rh = dr/r is formed BEFORE scaling, so the product is
    coef * (dr_c / r), never (coef * dr_c) / r -- these round differently
  - damping is applied as the initial value of F, then pair terms are added
Any "simplification" of these expressions is a bug, however equivalent it
looks in exact arithmetic.
"""
from __future__ import annotations
import numpy as np
from numba import njit

ALLOC_A, ALLOC_B, ALLOC_C = 0, 1, 2
ALLOC_CODES = {"A_batch_all": ALLOC_A,
               "B_support_orbit": ALLOC_B,
               "C_overload_prune": ALLOC_C}


@njit(cache=True, fastmath=False, boundscheck=False)
def _step(x, u, active, bonds,
          r0, mass, spring_k, gamma, rep_cut, rep_str,
          R_cand, R_break, valence, alloc_code,
          birth_lo, birth_hi, surv_lo, surv_hi,
          ret_needs_active, form_needs_active,
          form_sup_min, ret_sup_min, dt):
    N = x.shape[0]
    D = x.shape[1]

    # ---- pair distances (upper triangle computed, lower mirrored) ---------
    r = np.zeros((N, N), np.float64)
    for i in range(N):
        for j in range(i + 1, N):
            acc = 0.0
            for c in range(D):
                d = x[j, c] - x[i, c]
                acc = acc + d * d
            rr = np.sqrt(acc)
            r[i, j] = rr
            r[j, i] = rr

    # ---- candidate neighbours ---------------------------------------------
    cand = np.zeros((N, N), np.bool_)
    for i in range(N):
        for j in range(i + 1, N):
            if r[i, j] <= R_cand:
                cand[i, j] = True
                cand[j, i] = True

    # ---- common-active-support (integer; order independent) ---------------
    support = np.zeros((N, N), np.int32)
    for i in range(N):
        for j in range(i + 1, N):
            if not cand[i, j]:
                continue
            s = 0
            for k in range(N):
                if cand[i, k] and cand[j, k] and active[k]:
                    s += 1
            support[i, j] = s
            support[j, i] = s

    # ---- node update: OLD active, OLD bonds, synchronous -------------------
    new_active = np.zeros(N, np.bool_)
    for i in range(N):
        deg = 0
        act = 0
        for j in range(N):
            if bonds[i, j]:
                deg += 1
                if active[j]:
                    act += 1
        if deg == 0:
            continue                      # degree_zero -> passive
        frac = act / deg
        if active[i]:
            lo = surv_lo
            hi = surv_hi
        else:
            lo = birth_lo
            hi = birth_hi
        if lo <= frac and frac <= hi:
            new_active[i] = True

    # ---- retention --------------------------------------------------------
    kept = np.zeros((N, N), np.bool_)
    kdeg = np.zeros(N, np.int32)
    for i in range(N):
        for j in range(i + 1, N):
            if not bonds[i, j]:
                continue
            if ret_needs_active and not (active[i] or active[j]):
                continue
            if r[i, j] > R_break:
                continue
            if support[i, j] < ret_sup_min:
                continue
            kept[i, j] = True
            kept[j, i] = True
            kdeg[i] += 1
            kdeg[j] += 1

    # ---- eligibility (OLD degrees, per protocol) --------------------------
    olddeg = np.zeros(N, np.int32)
    for i in range(N):
        s = 0
        for j in range(N):
            if bonds[i, j]:
                s += 1
        olddeg[i] = s

    elig = np.zeros((N, N), np.bool_)
    for i in range(N):
        for j in range(i + 1, N):
            if not cand[i, j]:
                continue
            if bonds[i, j]:
                continue
            if form_needs_active and not (active[i] or active[j]):
                continue
            if olddeg[i] >= valence or olddeg[j] >= valence:
                continue
            if support[i, j] < form_sup_min:
                continue
            elig[i, j] = True
            elig[j, i] = True

    # ---- allocation -------------------------------------------------------
    formed = np.zeros((N, N), np.bool_)

    if alloc_code == 0:                                   # A_batch_all
        inc = np.zeros(N, np.int32)
        for i in range(N):
            s = 0
            for j in range(N):
                if elig[i, j]:
                    s += 1
            inc[i] = s
        accept = np.zeros(N, np.bool_)
        for p in range(N):
            if inc[p] > 0 and inc[p] <= valence - kdeg[p]:
                accept[p] = True
        for i in range(N):
            for j in range(i + 1, N):
                if elig[i, j] and accept[i] and accept[j]:
                    formed[i, j] = True
                    formed[j, i] = True

    elif alloc_code == 1:                                 # B_support_orbit
        # acc[p, q] means: endpoint p accepts edge (p, q)
        acc = np.zeros((N, N), np.bool_)
        for p in range(N):
            rem = valence - kdeg[p]
            # walk support classes high -> low; accept a whole tied class or stop
            for s in range(N, -1, -1):
                cnt = 0
                for q in range(N):
                    if elig[p, q] and support[p, q] == s:
                        cnt += 1
                if cnt == 0:
                    continue
                if cnt <= rem:
                    for q in range(N):
                        if elig[p, q] and support[p, q] == s:
                            acc[p, q] = True
                    rem -= cnt
                else:
                    break
        for i in range(N):
            for j in range(i + 1, N):
                if elig[i, j] and acc[i, j] and acc[j, i]:
                    formed[i, j] = True
                    formed[j, i] = True

    else:                                                 # C_overload_prune
        T = elig.copy()
        for _round in range(2):
            inc = np.zeros(N, np.int32)
            for i in range(N):
                s = 0
                for j in range(N):
                    if T[i, j]:
                        s += 1
                inc[i] = s
            over = np.zeros(N, np.bool_)
            n_over = 0
            for p in range(N):
                if kdeg[p] + inc[p] > valence:
                    over[p] = True
                    n_over += 1
            if n_over == 0:
                break
            n_rem = 0
            for i in range(N):
                for j in range(i + 1, N):
                    if T[i, j] and over[i] and over[j]:
                        T[i, j] = False
                        T[j, i] = False
                        n_rem += 1
            if n_rem == 0:
                break
        inc = np.zeros(N, np.int32)
        for i in range(N):
            s = 0
            for j in range(N):
                if T[i, j]:
                    s += 1
            inc[i] = s
        over = np.zeros(N, np.bool_)
        for p in range(N):
            if kdeg[p] + inc[p] > valence:
                over[p] = True
        for i in range(N):
            for j in range(i + 1, N):
                if T[i, j] and (not over[i]) and (not over[j]):
                    formed[i, j] = True
                    formed[j, i] = True

    # ---- new bond set -----------------------------------------------------
    nb = np.zeros((N, N), np.bool_)
    for i in range(N):
        for j in range(i + 1, N):
            if kept[i, j] or formed[i, j]:
                nb[i, j] = True
                nb[j, i] = True

    # ---- forces (arithmetic order is load-bearing) ------------------------
    F = np.empty((N, D), np.float64)
    for i in range(N):
        for c in range(D):
            F[i, c] = -gamma * u[i, c]

    coincident = False
    for i in range(N):
        for j in range(i + 1, N):
            rr = r[i, j]
            if rr < 1e-12:
                coincident = True
                continue
            if nb[i, j]:
                coef = spring_k * (rr - r0)
            elif rr < rep_cut:
                coef = -rep_str * (rep_cut - rr)
            else:
                continue
            for c in range(D):
                rh = (x[j, c] - x[i, c]) / rr
                f = coef * rh
                F[i, c] += f
                F[j, c] -= f

    # ---- semi-implicit Euler ---------------------------------------------
    nu = np.empty((N, D), np.float64)
    nx = np.empty((N, D), np.float64)
    for i in range(N):
        for c in range(D):
            nu[i, c] = u[i, c] + dt * F[i, c] / mass
            nx[i, c] = x[i, c] + dt * nu[i, c]

    return nx, nu, new_active, nb, coincident


def step_args(g):
    """Flatten a Genome into the kernel's scalar argument tuple."""
    p, t, nm = g.physics, g.topology, g.numerics
    return (p.r0, p.mass, p.spring_k, p.gamma, p.repulsion_cutoff,
            p.repulsion_strength,
            t.candidate_radius, t.break_radius, t.valence,
            ALLOC_CODES[t.allocator],
            t.birth_lo, t.birth_hi, t.survival_lo, t.survival_hi,
            t.retention_requires_active_endpoint,
            t.formation_requires_active_endpoint,
            t.formation_support_min, t.retention_support_min, nm.dt)
