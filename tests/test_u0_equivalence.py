"""THE load-bearing test.

Claim under test:
    Our genome-driven engine, configured as U0, reproduces the ORIGINAL
    audited Project Autogenesis engine BIT-FOR-BIT.

If this passes, every U0 result computed here is comparable to the existing
Project Autogenesis record.  If it fails, nothing else in this repo means
anything.

MS-C-906. That last sentence used to read "Do not weaken the tolerance -- it is exact
equality on purpose", and it is now split in two, because the two channels are not the
same claim (`D-040`):

  * The DISCRETE channel -- activity and bond set -- is exact equality and must never be
    weakened. A single differing bit branches the trajectory, and `D-040`'s void clause
    fires if it ever diverges. It is also checked FIRST, so it cannot be made unreachable
    by a continuous difference (`L-495`).
  * The CONTINUOUS channel carries the `OB-109` tolerance, derived in
    `docs/OB_109_BRIEF.md` from the arithmetic and computed at runtime from the genome.
    The smallest defect that could change a trajectory is 2.0e+13 times that bound.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "u0_original" / "src"))

from autogenesis.genome import U0                      # noqa: E402
from autogenesis.state import State                    # noqa: E402
from autogenesis import reference as ours              # noqa: E402
import phaseB_reference_engine as orig                 # noqa: E402


def _load_m006():
    d = json.loads((ROOT / "vendor" / "u0_original" / "data" / "B4X_M006.json").read_text())
    s = d["isolated_state"]
    x = np.array(s["x"], float)
    u = np.array(s["u"], float)
    a = np.array(s["active"], bool)
    N = len(a)
    b = np.zeros((N, N), bool)
    for i, j in s["bonds"]:
        b[i, j] = b[j, i] = True
    return State(x, u, a, b), d["source"]


def _to_orig(s: State):
    return orig.State(s.x.copy(), s.u.copy(), s.active.copy(),
                      orig.canonical_bonds(s.edge_list()))


def _bonds_equal(ours_state: State, orig_state) -> bool:
    return set(ours_state.edge_list()) == set(orig_state.bonds)


def continuous_tolerance(g, n: int, step: int) -> float:
    """The `OB-109` bound, derived in `docs/OB_109_BRIEF.md` and computed here.

    `D-019`: anchored to CODE, never to a stored number. Change `dt`, `spring_k`, the
    dimension or `N` and the bound moves with them; nothing here was fitted to an observed
    divergence, and the observation landed 450x inside it.

        |dr|/r   <= (D + 2) * eps          D-term accumulation, then sqrt
        |dcoef|  <= k_s * r0 * (D+2) * eps ABSOLUTE: r - r0 is exactly 0 at rest
        |df|     <= 3 * |dcoef|            coefficient, division, multiply
        |du'|    <= (N-1) * |df| * dt / m  at most N-1 pairs, then the integrator

    Scaled linearly by the step index: a first-order accumulation bound, declared as such.
    It is not a proof that the error cannot grow faster -- that protection is the discrete
    channel above, which stays exact and whose divergence fires `D-040`'s void clause.
    """
    eps = float(np.finfo(np.float64).eps)
    rel_r = (g.space.dimensions + 2) * eps
    per_pair = 3.0 * g.physics.spring_k * g.physics.r0 * rel_r
    return max(1, n - 1) * per_pair * g.numerics.dt / g.physics.mass * max(1, step)


def compare(g, s0: State, steps: int, label: str) -> dict:
    a = s0.copy()
    b = _to_orig(s0)
    v, alloc = g.topology.valence, g.topology.allocator
    worst_du = worst_dx = 0.0
    for t in range(steps):
        a = ours.step(g, a)
        b = orig.step(b, v, alloc)
        # MS-C-895. THE DISCRETE CHANNEL IS CHECKED FIRST, and the order is the point.
        # `D-040` adopted "exact equality on the discrete channel, a declared tolerance
        # on the continuous one", and recorded that the last bit depends on the
        # (numpy, BLAS build) pair. The continuous checks used to run FIRST and at exact
        # equality, so on a platform where the last bit differs they raised at step 1 and
        # **the discrete channel -- the one D-040 says stays exact forever -- was never
        # reached.** On ubuntu that is exactly what happened: `max|du| = 4.445e-18`, about
        # 5 ULP at `max|u| = 7.7e-03`, and the two assertions that actually carry the
        # claim never ran. The most important test in this file was unreachable in the
        # only environment where the difference appears.
        #
        # Do not weaken THESE two. On the discrete channel a single differing bit
        # branches the trajectory, and `D-040`'s void clause fires if it ever diverges.
        if not np.array_equal(a.active, b.active):
            raise AssertionError(f"[{label}] activity divergence at step {t+1} -- "
                                 "D-040's void clause: the discrete channel diverged, "
                                 "so the two-channel framing itself falls")
        if not _bonds_equal(a, b):
            raise AssertionError(f"[{label}] bond-set divergence at step {t+1} -- "
                                 "D-040's void clause: the discrete channel diverged, "
                                 "so the two-channel framing itself falls")
        # The continuous channel, at the tolerance `OB-109` landed. MS-C-906, derived in
        # docs/OB_109_BRIEF.md from the arithmetic and NEVER from an observation: the two
        # implementations compute the same distance by scalar accumulation and by BLAS
        # `ddot`, which differ by at most `(D+2)*eps` relatively; the spring coefficient
        # carries that as an ABSOLUTE error `k_s*r*(D+2)*eps`, because a relative bound is
        # meaningless where `r - r0` is exactly zero; a node sums at most `N-1` pairs, and
        # the integrator applies `dt/m`.
        #
        # It is computed from the genome at runtime, never stored (D-019). At U0 with
        # N = 16 the bound is 1.998e-15, the observed platform divergence is 4.445e-18 --
        # 450x inside it -- and the smallest defect that could change a trajectory,
        # `dt*k_s*r0/m = 4.0e-02`, is 2.0e+13 times the bound. There is no defect this can
        # hide. The DISCRETE channel above stays at exact equality and is not weakened.
        tol_u = continuous_tolerance(g, a.N, t + 1)
        tol_x = tol_u * g.numerics.dt
        du = float(np.abs(a.u - b.u).max())
        dx = float(np.abs(a.x - b.x).max())
        worst_du = max(worst_du, du)
        worst_dx = max(worst_dx, dx)
        if dx > tol_x:
            raise AssertionError(f"[{label}] position divergence at step {t+1}: "
                                 f"max|dx|={dx:.3e} exceeds the derived bound "
                                 f"{tol_x:.3e} (OB_109_BRIEF)")
        if du > tol_u:
            raise AssertionError(f"[{label}] velocity divergence at step {t+1}: "
                                 f"max|du|={du:.3e} exceeds the derived bound "
                                 f"{tol_u:.3e} (OB_109_BRIEF)")
    return {"label": label, "steps": steps, "state_hash": a.hash()[:16],
            "max_du": worst_du, "max_dx": worst_dx,
            "continuous_exact": worst_du == 0.0 and worst_dx == 0.0}


def test_m006_anchor(steps=512):
    s0, src = _load_m006()
    g = U0().replace(**{"initial_conditions.N": s0.N})
    if g.topology.valence != src["valence"]:                       # MS-C-683
        raise AssertionError("valence mismatch vs M006 source")
    if g.topology.allocator != src["allocator"]:                   # MS-C-683
        raise AssertionError("allocator mismatch vs M006 source")
    return compare(g, s0, steps, "M006_anchor")


def test_random_states(n_states=6, steps=64, N=20):
    """M006 is one special state. Random states exercise branches M006 never
    hits (valence saturation, retention failure, support ties)."""
    from autogenesis.init import make_initial_state
    out = []
    for k in range(n_states):
        g = U0().replace(**{"initial_conditions.N": N})
        s0 = make_initial_state(g, "engine_equivalence_v1", k)
        out.append(compare(g, s0, steps, f"random[{k}]"))
    return out


# --- dense regime: the ONLY regime where allocator/valence actually bind -----
# A sparse world never saturates valence, so every allocator returns the same
# edge set and the comparison passes VACUOUSLY.  These tests therefore assert
# non-degeneracy: the variants must produce DIFFERENT trajectories from each
# other, otherwise the test declares itself vacuous and fails.
DENSE = {"initial_conditions.N": 24,
         "initial_conditions.cube_side": 2.4,     # ~compact family C, tighter
         "initial_conditions.bond_probability": 0.35,
         "initial_conditions.initial_degree_cap": 2}


def _assert_discriminating(hashes: dict, what: str):
    uniq = set(hashes.values())
    if len(uniq) == 1:
        raise AssertionError(
            f"VACUOUS TEST: all {what} variants produced identical trajectories "
            f"({uniq.pop()}). The configuration never exercises {what}; both "
            f"engines agreeing here proves nothing. Increase density.")


def test_all_allocators(steps=64):
    from autogenesis.init import make_initial_state
    out, hashes = [], {}
    for alloc in ("A_batch_all", "B_support_orbit", "C_overload_prune"):
        g = U0().replace(**DENSE, **{"topology.allocator": alloc})
        s0 = make_initial_state(g, "engine_equivalence_v1", 100)
        r = compare(g, s0, steps, f"alloc={alloc}")
        out.append(r); hashes[alloc] = r["state_hash"]
    _assert_discriminating(hashes, "allocator")
    return out


def test_all_valences(steps=64):
    from autogenesis.init import make_initial_state
    out, hashes = [], {}
    for v in (4, 5, 6, 8, 12):
        g = U0().replace(**DENSE, **{"topology.valence": v})
        s0 = make_initial_state(g, "engine_equivalence_v1", 200)
        r = compare(g, s0, steps, f"valence={v}")
        out.append(r); hashes[v] = r["state_hash"]
    _assert_discriminating(hashes, "valence")
    return out


def test_saturation_report(steps=64):
    """Prove the dense regime really does saturate valence -- print the numbers
    rather than trusting that it does."""
    from autogenesis.init import make_initial_state
    import numpy as np
    g = U0().replace(**DENSE)
    s = make_initial_state(g, "engine_equivalence_v1", 100)
    peak, sat_steps = 0, 0
    for _ in range(steps):
        s = ours.step(g, s)
        d = s.degrees()
        peak = max(peak, int(d.max()))
        sat_steps += int((d >= g.topology.valence).any())
    print(f"  INFO  dense regime: peak degree {peak}/{g.topology.valence}, "
          f"valence-saturated on {sat_steps}/{steps} steps")
    if peak < g.topology.valence:                                  # MS-C-683
        raise AssertionError(
            f"dense regime never reached valence cap (peak {peak}); tests are vacuous")
    return []


if __name__ == "__main__":
    # PATCH (OB-030): make the process contract explicit in both directions.
    try:
        results = []
        results.append(test_m006_anchor())
        results += test_random_states()
        results += test_saturation_report()
        results += test_all_allocators()
        results += test_all_valences()
    except AssertionError as exc:
        print(f"\n*** U0 EQUIVALENCE FAILED ***\n  {exc}")
        sys.exit(1)
    for r in results:
        print(f"  PASS  {r['label']:<24} {r['steps']:>5} steps  hash={r['state_hash']}")
    exact = sum(1 for r in results if r.get("continuous_exact"))
    worst = max((r.get("max_du", 0.0) for r in results), default=0.0)
    # MS-C-906: this line said "bit-identical" unconditionally. It was true on a
    # platform where the comparison happens to be bit-exact and false on one where
    # it is not -- a summary describing something other than what ran. It reports
    # both channels for what they are now.
    print(f"\nU0 EQUIVALENCE: {len(results)}/{len(results)} comparisons -- "
          "discrete channel EXACT in all; continuous bit-exact in "
          f"{exact}/{len(results)}, worst |du| {worst:.3e} within the OB-109 "
          "bound (docs/OB_109_BRIEF.md)")
    sys.exit(0)

# NOTE (measurement, not a patch): line 22 binds `ours` to autogenesis.reference,
# NOT to the numba kernel in autogenesis.kernel that simulator.step actually runs.
# This file therefore compares reference.py against the vendored original; the
# kernel is covered only transitively, through tests/test_differential.py.
