"""Differential test: fast numba kernel vs naive reference engine.

Exact equality, no tolerance. A tolerance here would let a real divergence
hide as "rounding".

Covers, deliberately:
  - the sparse regime (nothing binds)
  - the DENSE regime (valence saturates, allocators actually differ)
  - all three allocators, all five protocol valences
  - 2D as well as 3D, so the dimension-generic code path is exercised
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from autogenesis.genome import U0                 # noqa: E402
from autogenesis.init import make_initial_state   # noqa: E402
from autogenesis import reference as ref          # noqa: E402
from autogenesis import simulator as fast         # noqa: E402

DENSE = {"initial_conditions.N": 24,
         "initial_conditions.cube_side": 2.4,
         "initial_conditions.bond_probability": 0.35,
         "initial_conditions.initial_degree_cap": 2}


def compare(g, s0, steps, label):
    a, b = s0.copy(), s0.copy()
    for t in range(steps):
        a = ref.step(g, a)
        b = fast.step(g, b)
        if not np.array_equal(a.x, b.x):
            raise AssertionError(f"[{label}] x divergence at step {t+1}: "
                                 f"max|dx|={np.abs(a.x - b.x).max():.3e}")
        if not np.array_equal(a.u, b.u):
            raise AssertionError(f"[{label}] u divergence at step {t+1}: "
                                 f"max|du|={np.abs(a.u - b.u).max():.3e}")
        if not np.array_equal(a.active, b.active):
            raise AssertionError(f"[{label}] active divergence at step {t+1}")
        if not np.array_equal(a.bonds, b.bonds):
            raise AssertionError(f"[{label}] bonds divergence at step {t+1}")
    return {"label": label, "steps": steps, "hash": a.hash()[:16]}


def cases():
    out = []
    # sparse, U0 defaults
    for k in range(3):
        g = U0()
        out.append((g, make_initial_state(g, "differential_v1", k), 96,
                    f"U0_sparse[{k}]"))
    # dense, every allocator
    for alloc in ("A_batch_all", "B_support_orbit", "C_overload_prune"):
        g = U0().replace(**DENSE, **{"topology.allocator": alloc})
        out.append((g, make_initial_state(g, "differential_v1", 50), 96,
                    f"dense_alloc={alloc}"))
    # dense, every protocol valence
    for v in (4, 5, 6, 8, 12):
        g = U0().replace(**DENSE, **{"topology.valence": v})
        out.append((g, make_initial_state(g, "differential_v1", 60), 96,
                    f"dense_valence={v}"))
    # 2D path
    for k in range(2):
        # NOTE: 2D needs a much larger box for the same N -- area grows as L^2
        # while volume grows as L^3, so "density" is not transferable between
        # dimensions. See docs/DECISIONS.md D-001.
        g = U0().replace(**{"space.dimensions": 2,
                            "initial_conditions.N": 20,
                            "initial_conditions.cube_side": 6.0,
                            "initial_conditions.bond_probability": 0.2})
        out.append((g, make_initial_state(g, "differential_v1", 700 + k), 96,
                    f"2D[{k}]"))
    # --- the retention_support_min guard (OB-066) --------------------------
    # D-024 removed the kernel's retention-support gate, which was the ONLY
    # behavioural difference between kernel and reference. MS-C-246 went
    # undetected for the life of the repository because every case above runs at
    # retention_support_min == 0, the one value at which the two engines agreed
    # even while the gate was present. These cases vary the field, so a
    # reintroduced gate diverges here instead of silently.
    for rsm in (1, 2):
        for k in range(2):
            g = U0().replace(**{"topology.retention_support_min": rsm})
            out.append((g, make_initial_state(g, "differential_v1", k), 96,
                        f"rsm={rsm}_sparse[{k}]"))
        g = U0().replace(**DENSE, **{"topology.retention_support_min": rsm})
        out.append((g, make_initial_state(g, "differential_v1", 50), 96,
                    f"rsm={rsm}_dense"))
    return out


def gate_would_have_fired(g, s0):
    """Non-vacuity for the guard cases: does at least one edge that retention
    KEEPS have support < retention_support_min at step 0?

    If nothing satisfies that, the removed gate could not have changed anything on
    this input and the case proves nothing -- so the guard asserts it, rather than
    trusting it. This re-derives the condition from `reference`, never from the
    kernel, so it stays honest if the kernel changes again.
    """
    t = g.topology
    r = ref.pair_distances(s0.x)
    cand = ref.candidate_edges(r, t.candidate_radius)
    sup = ref.support_scores(cand, s0.N, s0.active)
    for i, j in s0.edge_list():
        if t.retention_requires_active_endpoint and not (s0.active[i] or s0.active[j]):
            continue
        if r[i, j] > t.break_radius:
            continue
        if sup.get((i, j), 0) < t.retention_support_min:
            return True
    return False


if __name__ == "__main__":
    # PATCH (OB-030): an uncaught AssertionError already exits non-zero, but the
    # exit path was implicit. Make success and failure both explicit, so the
    # process contract is visible in the source and cannot be lost in a refactor.
    try:
        results = [compare(*c) for c in cases()]
        hs = {r["label"]: r["hash"] for r in results}
        for r in results:
            print(f"  PASS  {r['label']:<24} {r['steps']:>4} steps  hash={r['hash']}")
        # non-vacuity: dense allocator variants must differ from one another
        dense_alloc = {k: v for k, v in hs.items() if k.startswith("dense_alloc")}
        if len(set(dense_alloc.values())) <= 1:                    # MS-C-683: not `assert`
            raise AssertionError(
                "VACUOUS: dense allocator variants identical; allocator code untested")
        dense_val = {k: v for k, v in hs.items() if k.startswith("dense_valence")}
        if len(set(dense_val.values())) <= 1:                      # MS-C-683
            raise AssertionError(
                "VACUOUS: dense valence variants identical; valence cap untested")
        # OB-066: the guard cases must actually reach the removed gate's
        # condition, or they guard nothing.
        fired = [lab for g, s0, _st, lab in cases()
                 if lab.startswith("rsm=") and gate_would_have_fired(g, s0)]
        if not fired:                                              # MS-C-683
            raise AssertionError(
                "VACUOUS: no retention_support_min case has a retained edge "
                "with support below the threshold; the guard cannot detect "
                "a reintroduced gate")
        print(f"  INFO  guard non-vacuity: {len(fired)} of "
              f"{len([l for l in hs if l.startswith('rsm=')])} rsm cases reach the "
              f"gate condition at step 0 -> {fired}")
    except AssertionError as exc:
        print(f"\n*** DIFFERENTIAL FAILED ***\n  {exc}")
        sys.exit(1)
    print(f"\nDIFFERENTIAL: {len(results)}/{len(results)} kernel==reference, "
          f"bit-identical")
    sys.exit(0)

# NOTE (OB-066): the note that stood here said the varying case belonged with the
# OB-029 decision. That decision was taken -- D-024 removed the gate from
# kernel.py:102-103 -- so the case is now added above. It is a REGRESSION guard,
# not a probe: the two engines are expected to agree at every value of
# retention_support_min, and the guard fails if a support gate returns to either.
# Demonstrated to fail against the pre-D-024 kernel and pass against the current
# one: runs/compute_003, T3.
