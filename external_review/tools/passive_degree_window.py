"""PASSIVE DEGREE WINDOW -- a closed-form constraint on every live memory state,
and the graph-theoretic obstruction it implies.

Supporting commentary. `docs/` is the project's record; see
`external_review/README.md`. Nothing outside `external_review/` is touched.

THE THEOREM. Let `A` be a live cover fixed point of band `[lo, hi]` on a graph
with `m` edges -- that is, `A` is non-empty, `V\\A` is independent (so the bond
set is preserved), and every node's activity is stationary. Write `D_P` for the
total degree of the PASSIVE set `V\\A`. Then

        (1 - hi) / (2 - hi)   <=   D_P / (2m)   <=   (1 - lo) / (2 - lo)

Proof. `V\\A` independent means every one of its edges lands in `A`, so
`m = e(A) + D_P` where `e(A)` counts edges inside `A`. Summing the band
condition `lo*d_i <= |N(i) cap A| <= hi*d_i` over `i in A` gives
`lo*D_A <= 2e(A) <= hi*D_A` with `D_A = 2e(A) + D_P`. Eliminate `e(A)`. QED

For U0's `[1/6, 1/2]` the window is `[1/3, 5/11]` -- width 0.1212.

THE COROLLARY, which is pure graph theory and mentions no dynamics: a graph
admits a live cover fixed point only if it has an INDEPENDENT SET carrying at
least `(1-hi)/(2-hi)` of its total degree. For U0, at least one third. On a
`D`-regular graph that reads `alpha(G) >= n/3`.
"""
from __future__ import annotations
import itertools, json, sys, time
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from autogenesis.rules import enumerate_bands
sys.path.insert(0, str(Path(__file__).resolve().parent))   # _canonical_io lives here
from _canonical_io import write_json_atomic, write_jsonl_atomic   # noqa: E402
from autogenesis import measurement                                # noqa: E402

OUT = HERE / "runs" / "passive_degree_window"


def window(lo: Fraction, hi: Fraction) -> tuple[Fraction, Fraction]:
    return (1 - hi) / (2 - hi), (1 - lo) / (2 - lo)


def live_cover_fixed_points(n, adj, deg, edges, lo, hi):
    """Every non-empty stationary activity set whose complement is independent."""
    for s in range(1, 1 << n):
        A = {i for i in range(n) if s >> i & 1}
        if any(i not in A and j not in A for i, j in edges):
            continue
        good = True
        for i in range(n):
            active = deg[i] >= 1 and lo * deg[i] <= len(adj[i] & A) <= hi * deg[i]
            if active != (i in A):
                good = False
                break
        if good:
            yield A


def all_labelled(n):
    pairs = list(itertools.combinations(range(n), 2))
    for mask in range(1, 1 << len(pairs)):
        edges = [pairs[k] for k in range(len(pairs)) if mask >> k & 1]
        adj = [set() for _ in range(n)]
        for i, j in edges:
            adj[i].add(j)
            adj[j].add(i)
        yield edges, adj, [len(a) for a in adj]


def max_independent_degree_share(n, adj, deg, m) -> Fraction:
    best = 0
    for s in range(1 << n):
        S = [i for i in range(n) if s >> i & 1]
        if any(j in adj[i] for i, j in itertools.combinations(S, 2)):
            continue
        best = max(best, sum(deg[i] for i in S))
    return Fraction(best, 2 * m)


def main(nmax: int = 6) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = measurement.stamp(__file__)
    print("PASSIVE DEGREE WINDOW")
    print("   " + json.dumps(stamp))
    t0 = time.perf_counter()
    result = {**stamp, "nmax": nmax, "arms": {}}

    # --- arm 1: U0's band, every labelled graph, every live cover fixed point
    lo, hi = Fraction(1, 6), Fraction(1, 2)
    wlo, whi = window(lo, hi)
    print(f"\n   U0 band [1/6, 1/2] -> window [{wlo}, {whi}] "
          f"= [{float(wlo):.4f}, {float(whi):.4f}]")
    tot = viol = at_lo = at_hi = graphs = 0
    rec_lines = []                       # MS-C-705: buffered, written atomically at the end
    for n in range(2, nmax + 1):
        for edges, adj, deg in all_labelled(n):
            graphs += 1
            m = len(edges)
            for A in live_cover_fixed_points(n, adj, deg, edges, lo, hi):
                tot += 1
                share = Fraction(sum(deg[i] for i in range(n) if i not in A), 2 * m)
                inside = wlo <= share <= whi
                viol += not inside
                at_lo += share == wlo
                at_hi += share == whi
                # OB-132 item 5 / MS-C-598: a line carrying only (n, m, A, share)
                # has NO graph identity, so 22,567 lines collapsed to 556 distinct
                # tuples and no line was auditable back to the graph that produced
                # it. The edge list is that identity.
                rec_lines.append({"n": n, "m": m, "edges": [list(e) for e in edges],
                                  "A": sorted(A), "share": str(share),
                                  "inside": inside})
    write_jsonl_atomic(OUT / "u0_band.jsonl", rec_lines)
    result["arms"]["u0_band"] = {"graphs": graphs, "fixed_points": tot,
                                 "violations": viol, "at_lower_edge": at_lo,
                                 "at_upper_edge": at_hi}
    print(f"   {graphs} labelled graphs, {tot} live cover fixed points, "
          f"violations {viol}")
    print(f"   at the lower edge {at_lo} ({100*at_lo/max(1,tot):.1f} %), "
          f"at the upper edge {at_hi}")

    # --- arm 2: the general form, across the band family
    tot2 = viol2 = 0
    bands = [b for b in enumerate_bands(6) if b["selected"]][::6]
    for b in bands:
        blo, bhi = Fraction(b["selected"][0]), Fraction(b["selected"][-1])
        w0, w1 = window(blo, bhi)
        for n in range(2, nmax + 1):
            for edges, adj, deg in all_labelled(n):
                m = len(edges)
                for A in live_cover_fixed_points(n, adj, deg, edges, blo, bhi):
                    tot2 += 1
                    share = Fraction(sum(deg[i] for i in range(n) if i not in A), 2 * m)
                    viol2 += not (w0 <= share <= w1)
    result["arms"]["general"] = {"bands": len(bands), "fixed_points": tot2,
                                 "violations": viol2}
    print(f"\n   general form: {len(bands)} bands, {tot2} fixed points, "
          f"violations {viol2}")

    # --- arm 3: where the ring sits
    rings = {}
    for n in (6, 9, 12, 15):
        adj = [{(i - 1) % n, (i + 1) % n} for i in range(n)]
        A = {i for i in range(n) if i % 3 != 2}
        share = Fraction(sum(2 for i in range(n) if i not in A), 2 * n)
        rings[f"C{n}"] = {"share": str(share), "at_lower_edge": share == wlo}
    result["arms"]["rings"] = rings
    print("   rings: " + ", ".join(f"C{k[1:]} {v['share']}"
                                   f"{' (edge)' if v['at_lower_edge'] else ''}"
                                   for k, v in rings.items()))
    # --- arm 4: both edges are TIGHT, and the extremal structures are named
    # LOWER 1/3: every active node has EVEN degree with exactly half its
    #            neighbours active. Minimal case: a ring, degree 2, 1 of 2.
    # UPPER 5/11: every active node has degree exactly 6 -- the valence cap, the
    #            only degree with `lo*d` an integer -- and exactly ONE active
    #            neighbour, so `A` induces a perfect matching. Minimal case n = 7:
    #            two joined active nodes sharing five passive neighbours.
    n = 7
    A = {0, 1}
    edges = [(0, 1)] + [(a, p) for a in (0, 1) for p in range(2, 7)]
    adj = [set() for _ in range(n)]
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
    deg = [len(a) for a in adj]
    valid = A in list(live_cover_fixed_points(n, adj, deg, edges, lo, hi))
    share = Fraction(sum(deg[i] for i in range(n) if i not in A), 2 * len(edges))
    result["arms"]["upper_edge_witness"] = {
        "n": n, "m": len(edges), "degrees": deg, "active": sorted(A),
        "share": str(share), "is_valid_fixed_point": valid,
        "attains_ceiling": share == whi}
    print(f"\n   upper-edge witness n=7: share {share}, valid {valid}, "
          f"attains the ceiling {share == whi}")
    print(f"   (the n<={nmax} scan cannot see it -- a degree-6 node needs 7 vertices)")

    result["seconds"] = round(time.perf_counter() - t0, 1)
    write_json_atomic(OUT / "RESULTS.json", result)   # MS-C-704: canonical LF, atomic
    print(f"\n   -> {OUT / 'RESULTS.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
