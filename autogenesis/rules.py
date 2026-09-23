"""The node-rule band space is FINITE and small. Enumerate it, do not sample it.

A node's activation depends on the fraction (active bonded neighbours)/(degree).
With degree bounded by the valence cap v, only finitely many fractions are
reachable: all k/d with 1 <= d <= v, 0 <= k <= d. That is the Farey sequence
F_v, and a band [lo, hi] can only ever select a CONTIGUOUS run of it.

    v <= 4  ->   7 reachable fractions ->    29 distinct rules
    v <= 5  ->  11                     ->    67
    v <= 6  ->  13                     ->    92
    v <= 8  ->  23                     ->   277
    v <= 12 ->  47                     ->  1129

So "rule search", which the original planning document deferred to Tier 3 as
too expensive, is exhaustively enumerable at v <= 6: 92 rules x 200 seeds is
about two minutes of compute.

Two members of the set are the designated CONTROLS, and they come for free
because they are ordinary members rather than bolted-on machinery:

  EMPTY  band selecting nothing  -> no node can ever be active. Everything dies
                                    at step 1. The floor.
  FULL   band [0, 1]             -> every node with degree >= 1 is active,
                                    regardless of its neighbours. The state
                                    dynamics are switched off; whatever survives
                                    here survives on MECHANICS alone. This is
                                    the null-rule arm: any persistence a real
                                    rule shows must beat this to mean anything.
"""
from __future__ import annotations
from fractions import Fraction


def reachable_fractions(valence: int) -> list[Fraction]:
    """Every activation fraction reachable at degrees 1..valence, ascending."""
    return sorted({Fraction(k, d)
                   for d in range(1, valence + 1)
                   for k in range(0, d + 1)})


def enumerate_bands(valence: int) -> list[dict]:
    """All distinct contiguous bands, plus the empty band.

    Each entry gives representative float bounds usable directly as
    (birth_lo, birth_hi) / (survival_lo, survival_hi), and the exact set of
    fractions selected, which is the rule's true identity.
    """
    vals = reachable_fractions(valence)
    out = [{
        "rule_id": "R000_EMPTY",
        "lo": 1.0, "hi": 0.0,            # selects nothing (lo > hi)
        "selected": [],
        "n_selected": 0,
        "is_control": "floor",
    }]
    n = 0
    for i in range(len(vals)):
        for j in range(i, len(vals)):
            n += 1
            sel = vals[i:j + 1]
            full = (sel[0] == 0 and sel[-1] == 1)
            out.append({
                "rule_id": f"R{n:03d}",
                "lo": float(sel[0]),
                "hi": float(sel[-1]),
                "selected": [str(f) for f in sel],
                "n_selected": len(sel),
                "is_control": "null_rule" if full else None,
            })
    return out


def u0_band(valence: int = 6) -> dict:
    """The band U0 uses: [1/6, 1/2]."""
    lo, hi = Fraction(1, 6), Fraction(1, 2)
    for b in enumerate_bands(valence):
        if b["selected"] and Fraction(b["selected"][0]) == lo \
                and Fraction(b["selected"][-1]) == hi:
            return b
    raise LookupError("U0 band not found in enumeration")


def activation_table(lo: float, hi: float, valence: int) -> dict[int, list[int]]:
    """degree -> which active-neighbour counts activate. The rule's fingerprint.

    Reading this table is often more informative than any simulation: U0's
    shows that a degree-1 node can never activate, which is the whole reason
    sparse worlds die instantly."""
    return {d: [k for k in range(0, d + 1) if lo <= k / d <= hi]
            for d in range(1, valence + 1)}


if __name__ == "__main__":
    for v in (4, 5, 6, 8, 12):
        b = enumerate_bands(v)
        print(f"valence<={v:3d}:  {len(reachable_fractions(v)):3d} fractions  "
              f"->  {len(b)} rules (incl. EMPTY)")
    print()
    u = u0_band(6)
    print(f"U0 band = {u['rule_id']}  [{u['lo']:.4f}, {u['hi']:.4f}]  "
          f"selects {u['selected']}")
    print("U0 activation table:")
    for d, ks in activation_table(u["lo"], u["hi"], 6).items():
        print(f"   degree {d}: {ks if ks else 'NONE (always passive)'}")
    ctl = [b for b in enumerate_bands(6) if b["is_control"]]
    print("\ncontrols in the enumeration:")
    for c in ctl:
        print(f"   {c['rule_id']:<10} {c['is_control']:<10} "
              f"[{c['lo']}, {c['hi']}]  {c['n_selected']} fractions")
