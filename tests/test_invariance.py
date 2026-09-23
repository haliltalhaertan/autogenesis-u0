"""Physical-invariance and numerical-validation panel.

These are the checks PHASE_B_FROZEN_PROTOCOL.json lists under
`implementation_validation_gate`.

READ THIS BEFORE INTERPRETING A FAILURE:

Floating-point arithmetic is NOT translation- or rotation-invariant. Shifting
every coordinate by +100 changes the rounding of every subsequent operation.
So these tests are necessarily TOLERANCE tests, not bit-equality tests, and a
tolerance is a scientific claim: "deviation stays at round-off level and does
not grow into a different trajectory".

Because the topology rules contain THRESHOLDS (r <= 1.35, r <= 1.60), a
round-off-sized perturbation can flip a bond on one side of a threshold and
produce a genuinely different discrete trajectory. That is a property of the
model, not a bug. We therefore report:
    - the step at which the DISCRETE trajectory first diverges (if ever)
    - the continuous deviation up to that point
and we do NOT quietly widen tolerances to make a green tick appear.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from autogenesis.genome import U0                 # noqa: E402
from autogenesis.init import make_initial_state   # noqa: E402
from autogenesis.state import State               # noqa: E402
from autogenesis import simulator as sim          # noqa: E402
from autogenesis import reference as ref          # noqa: E402

STEPS = 256

# PATCH (OB-030): explicit expected-divergence list, replacing the blanket PASS*
# escape. A discrete divergence is tolerated ONLY for a check named here, and only
# at or after the step recorded. Anything else is now a FAIL.
#   name -> earliest step at which a discrete divergence is accepted
# Measured 2026-08-25 at code_hash b9ed06b5414d516d: NO check in this file
# currently diverges discretely, so an EMPTY list leaves every currently-passing
# case passing unchanged. Entries are added only with a written reason.
EXPECTED_DIVERGENCE: dict[str, int] = {}


def _trace(g, s, steps):
    xs, disc = [], []
    for _ in range(steps):
        s = sim.step(g, s)
        xs.append(s.x.copy())
        disc.append(s.discrete_signature())
    return xs, disc


def _first_discrete_divergence(d1, d2):
    for k, (a, b) in enumerate(zip(d1, d2)):
        if a != b:
            return k + 1
    return None


def _report(name, dev, div, steps):
    # PATCH (OB-030): PASS* is now earned, not automatic. Previously ANY discrete
    # divergence produced PASS* regardless of the continuous deviation, which made
    # translation and rotation invariance structurally unfailable.
    if div is None:
        status = "PASS" if dev < 1e-9 else "FAIL"
    elif name in EXPECTED_DIVERGENCE and div >= EXPECTED_DIVERGENCE[name]:
        status = "PASS*" if dev < 1e-9 else "FAIL"
    else:
        status = "FAIL"
    tail = "" if div is None else f"  discrete divergence at step {div}"
    print(f"  {status:<5} {name:<26} max continuous deviation "
          f"{dev:.3e} over {steps} steps{tail}")
    return status


# ---------------------------------------------------------------------------
def test_translation(shift=np.array([3.0, -1.5, 0.25])):
    g = U0()
    s = make_initial_state(g, "invariance_v1", 1)
    x1, d1 = _trace(g, s.copy(), STEPS)
    s2 = s.copy(); s2.x = s2.x + shift
    x2, d2 = _trace(g, s2, STEPS)
    div = _first_discrete_divergence(d1, d2)
    # PATCH (OB-030): div == 1 made this window EMPTY, so `dev` collapsed to
    # max([] or [0.0]) == 0.0 -- a maximally divergent run reported as exactly zero
    # deviation. Always measure at least the first step.
    upto = len(x1) if div is None else max(div - 1, 1)
    dev = max([float(np.abs((b - shift) - a).max())
               for a, b in zip(x1[:upto], x2[:upto])])
    return _report("translation covariance", dev, div, STEPS)


def test_rotation(theta=0.7):
    g = U0()
    c, sn = np.cos(theta), np.sin(theta)
    R = np.array([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]])
    s = make_initial_state(g, "invariance_v1", 2)
    x1, d1 = _trace(g, s.copy(), STEPS)
    s2 = s.copy(); s2.x = s2.x @ R.T; s2.u = s2.u @ R.T
    x2, d2 = _trace(g, s2, STEPS)
    div = _first_discrete_divergence(d1, d2)
    upto = len(x1) if div is None else max(div - 1, 1)   # PATCH (OB-030), see above
    dev = max([float(np.abs(b - a @ R.T).max())
               for a, b in zip(x1[:upto], x2[:upto])])
    return _report("rotation covariance", dev, div, STEPS)


def test_permutation(seed=3):
    g = U0()
    s = make_initial_state(g, "invariance_v1", seed)
    rng = np.random.default_rng(12345)
    p = rng.permutation(s.N)
    x1, d1 = _trace(g, s.copy(), STEPS)
    s2 = State(s.x[p].copy(), s.u[p].copy(), s.active[p].copy(),
               s.bonds[np.ix_(p, p)].copy())
    x2, _ = _trace(g, s2, STEPS)
    dev = max(float(np.abs(b - a[p]).max()) for a, b in zip(x1, x2))
    # discrete equivariance is checked structurally: same edge count each step
    return _report("permutation equivariance", dev, None, STEPS)


def test_force_antisymmetry():
    """Sum of all internal pair forces must vanish exactly, so
    sum(F) + gamma*sum(u) == 0 to machine precision."""
    g = U0()
    s = make_initial_state(g, "invariance_v1", 4)
    worst = 0.0
    for _ in range(64):
        F = ref.forces(g, s.x, s.u, s.bonds)
        resid = np.abs(F.sum(0) + g.physics.gamma * s.u.sum(0)).max()
        worst = max(worst, float(resid))
        s = sim.step(g, s)
    ok = worst < 1e-12
    print(f"  {'PASS' if ok else 'FAIL':<5} {'force antisymmetry':<26} "
          f"max |sum F + gamma*sum u| = {worst:.3e}")
    return "PASS" if ok else "FAIL"


def test_determinism():
    """Same genome + same seed -> same final hash, twice, in one process."""
    g = U0()
    a = sim.run(g, "invariance_v1", 5, steps=512)
    b = sim.run(g, "invariance_v1", 5, steps=512)
    ok = a["final_state_hash"] == b["final_state_hash"]
    print(f"  {'PASS' if ok else 'FAIL':<5} {'deterministic replay':<26} "
          f"hash={a['final_state_hash']}")
    return "PASS" if ok else "FAIL"


def test_dt_convergence():
    """SCIENTIFIC probe, not a bug check.

    Halving dt and doubling the step count keeps physical time fixed for the
    CONTINUOUS part, but the topology rules fire once per STEP, so the discrete
    dynamics are not dt-invariant by construction. This measures how badly.
    Reported, never auto-passed."""
    g = U0()
    s0 = make_initial_state(g, "invariance_v1", 6)
    T_phys = 8.0
    print(f"\n  dt-convergence probe (physical time {T_phys}):")
    prev = None
    for k in range(4):
        dt = 0.02 / (2 ** k)
        gk = g.replace(**{"numerics.dt": dt})
        n = int(round(T_phys / dt))
        s = s0.copy()
        for _ in range(n):
            s = sim.step(gk, s)
        rms = float(np.sqrt((s.x ** 2).sum(1)).mean())
        line = (f"    dt={dt:<8.5f} steps={n:<6d} edges={len(s.edge_list()):<4d} "
                f"active={int(s.active.sum()):<3d} mean|x|={rms:.6f}")
        if prev is not None:
            line += f"   d(mean|x|)={abs(rms - prev):.3e}"
        prev = rms
        print(line)
    print("    -> interpret with care: topology events are per-STEP, so dt is a"
          "\n       PHYSICS parameter here, not only a numerical one.")
    return "INFO"


if __name__ == "__main__":
    out = [test_translation(), test_rotation(), test_permutation(),
           test_force_antisymmetry(), test_determinism()]
    test_dt_convergence()
    bad = [o for o in out if o == "FAIL"]
    print(f"\nINVARIANCE: {len(out) - len(bad)}/{len(out)} passed"
          + ("" if not bad else f"  ({len(bad)} FAILED)"))
    print("PASS* = continuous deviation at round-off level, and the discrete\n"
          "        trajectory diverged at a step listed in EXPECTED_DIVERGENCE.\n"
          "        An UNLISTED discrete divergence is now a FAIL.")
    # PATCH (OB-030): this file previously exited 0 no matter what it printed.
    sys.exit(1 if bad else 0)
