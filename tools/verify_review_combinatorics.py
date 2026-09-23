"""Independent reproduction of three external combinatorial results (MS-C-729..731).

Each is re-derived from scratch here rather than checked against the reviewer's
numbers: the Boolean fixed-point definition, the edge enumeration and the influence
measurement are all rebuilt from `autogenesis.rules`' band and the node rule, so an
agreement is two implementations agreeing rather than one being read twice.

  A. `MS-C-729` -- SINGLE-EDGE ROBUSTNESS, n <= 6. Claim: over every labelled graph
     and every live U0 Boolean fixed point at `n = 2..6`, the fixed points that
     survive EVERY single-edge addition or deletion are exactly those with
     `n = 6`, `|A| = 3`, `G[A] = K3`, all nine `A-P` edges present, and `G[P]` a
     matching (empty matching included). Predicted count `C(6,3) * (1 + 3) = 80`.

     NOTE ON SCOPE, which the claim does not state and which matters: these are NOT
     the `live cover fixed points` of the window theorem (`MS-C-591`). That object
     requires `V\\A` INDEPENDENT so the bond set is preserved; here `G[P]` may carry
     a matching edge, so the two families overlap without either containing the
     other. Both are reported so nobody can conflate them later.

  B. `MS-C-730` -- THE IMPOSSIBILITY THEOREM IS ALREADY OURS. The reported result --
     fixed graph, birth band excluding `1`, passive set independent at every phase,
     therefore no nontrivial activity cycle -- is `CHECKPOINT_046`'s Lemma and
     Theorem, statement and proof. "`A(t)` is a vertex cover" IS "`P_t` is
     independent", and for a contiguous band "excludes 1" IS `hi < 1`. It was proved
     there, verified exhaustively as `MS-C-481` (432,037 cover states, 0 violations)
     and `MS-C-482` (27,475 connected graphs, 0 qualifying orbits), given a
     non-vacuity arm (3,249 orbits at `hi = 1`), and upgraded to an iff by `MS-C-509`.
     Recording it as new would be an `L-213`-class record error.

     What IS new is one sentence of it: the survival band never enters. Arm D tests
     that against the production engine. Arm B re-establishes, on an independently
     written implementation, that both hypotheses are load-bearing.

  C. `MS-C-731` -- INFLUENCE DOES NOT CHARACTERISE ADDITIVITY, and the reason is
     sharper than "not necessary". Over all 512 subsets of the 9 cross edges between
     two U0 triangles: 43 graphs carry `3 * 3 = 9` both-live fixed points, split
     `{0 cross edges: 1, 3: 6, 6: 36}`. But only ONE of the 43 -- the disconnected
     one -- has those fixed points actually BE the products, state by state. The
     other 42 match in COUNT and in nothing else.

     That distinction lands directly on `MS-C-716`'s capacity meter, which multiplies
     per-component fixed-point COUNTS and never inspects a state. On these 42 graphs
     it would read 2 bits of independent storage out of two blocks that are not
     independent at all. It does not disturb the published `0.0 bit` -- every live
     component there has `live_fixed_points = 1`, and `1 * 1 = 1` under either
     reading -- but it bounds what a POSITIVE reading could ever be taken to mean.

Run:  ./.venv/Scripts/python.exe tools/verify_review_combinatorics.py

THIS TOOL IS NOT READ-ONLY (`MS-C-876`)
----------------------------------------
The name begins `verify_`, which reads as read-only, and it WRITES a canonical
artefact under `runs/`. Running "just a verification" therefore changes the
record -- which is how `Bulgu 08` was found in the first place. The write is
legitimate (the verification produces a record) but it is now NON-DESTRUCTIVE:
`_artefact.write_result` archives the previous artefact under
`runs/.superseded/` before replacing it. Check `git status` before and after
anyway; this note exists so the surprise is not the discovery.
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from fractions import Fraction as F
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(ROOT / "tools"))

from autogenesis import measurement                            # noqa: E402
from _review_protocol import (Stream, check_protocol, claims,      # noqa: E402
                              compare, resume)

OUT = ROOT / "runs" / "review_combinatorics.json"
STREAM = ROOT / "runs" / "review_combinatorics.stream.jsonl"
LO, HI = F(1, 6), F(1, 2)          # U0's band, as exact rationals


def node_next(active_mask: int, nbr_mask: list[int], deg: list[int],
              lo: F, hi: F) -> int:
    """One Boolean step on a fixed graph. Degree-zero is passive (kernel.py:79-80)."""
    out = 0
    for i, (m, d) in enumerate(zip(nbr_mask, deg)):
        if d == 0:
            continue
        k = bin(active_mask & m).count("1")
        if lo <= F(k, d) <= hi:
            out |= 1 << i
    return out


def graph_masks(n: int, edges):
    nbr = [0] * n
    for i, j in edges:
        nbr[i] |= 1 << j
        nbr[j] |= 1 << i
    return nbr, [bin(m).count("1") for m in nbr]


def all_labelled(n: int):
    pairs = list(itertools.combinations(range(n), 2))
    for mask in range(1 << len(pairs)):
        yield [pairs[k] for k in range(len(pairs)) if mask >> k & 1], pairs


def live_fixed_points(n, nbr, deg):
    """Non-empty activity sets fixed by the Boolean rule."""
    return [a for a in range(1, 1 << n) if node_next(a, nbr, deg, LO, HI) == a]


# ---------------------------------------------------------------- arm A ----
def arm_a(nmax: int = 6) -> dict:
    total_fp, robust = 0, []
    for n in range(2, nmax + 1):
        for edges, pairs in all_labelled(n):
            nbr, deg = graph_masks(n, edges)
            eset = set(edges)
            for a in live_fixed_points(n, nbr, deg):
                total_fp += 1
                ok = True
                for e in pairs:                     # every single-edge toggle
                    ne = sorted(eset ^ {e})
                    nnbr, ndeg = graph_masks(n, ne)
                    if node_next(a, nnbr, ndeg, LO, HI) != a:
                        ok = False
                        break
                if ok:
                    robust.append((n, tuple(sorted(eset)), a))

    # does every robust case have the predicted shape?
    shape_ok, shapes = 0, []
    for n, edges, a in robust:
        A = [i for i in range(n) if a >> i & 1]
        P = [i for i in range(n) if not (a >> i & 1)]
        eset = set(edges)
        inA = [e for e in eset if e[0] in A and e[1] in A]
        inP = [e for e in eset if e[0] in P and e[1] in P]
        cross = [e for e in eset if (e[0] in A) != (e[1] in A)]
        pdeg = {}
        for i, j in inP:
            pdeg[i] = pdeg.get(i, 0) + 1
            pdeg[j] = pdeg.get(j, 0) + 1
        is_matching = all(v <= 1 for v in pdeg.values())
        good = (n == 6 and len(A) == 3 and len(inA) == 3
                and len(cross) == len(A) * len(P) and is_matching)
        shape_ok += good
        shapes.append({"n": n, "A": len(A), "inA": len(inA), "cross": len(cross),
                       "inP": len(inP), "matches_shape": good,
                       "P_independent": len(inP) == 0})
    predicted = len(list(itertools.combinations(range(6), 3))) * (1 + 3)
    by_inP = {}
    for s in shapes:
        by_inP[s["inP"]] = by_inP.get(s["inP"], 0) + 1
    return {"fixed_points_scanned": total_fp, "robust": len(robust),
            "predicted_count": predicted, "all_match_shape": shape_ok == len(robust),
            "shape_matches": shape_ok,
            "by_passive_internal_edges": by_inP,
            "P_independent_count": sum(1 for s in shapes if s["P_independent"])}


# ---------------------------------------------------------------- arm B ----
def arm_b(nmax: int = 6) -> dict:
    """Both hypotheses of the impossibility theorem must be load-bearing."""
    one_in_band = {"U0_[1/6,1/2]": (LO, HI), "admits_1_[1/6,1]": (LO, F(1)),
                   "admits_1_[0,1]": (F(0), F(1))}
    out = {}
    for label, (lo, hi) in one_in_band.items():
        cycles_with_independent_P = 0
        nontrivial_cycles = 0
        example = None
        for n in range(2, nmax + 1):
            for edges, _ in all_labelled(n):
                nbr, deg = graph_masks(n, edges)
                adj = {i: {j for j in range(n) if nbr[i] >> j & 1} for i in range(n)}
                for a0 in range(1 << n):
                    seen, a, path = {}, a0, []
                    for t in range(64):
                        if a in seen:
                            break
                        seen[a] = t
                        path.append(a)
                        a = node_next(a, nbr, deg, lo, hi)
                    if a not in seen:
                        continue
                    cyc = path[seen[a]:]
                    if len(cyc) < 2:
                        continue
                    nontrivial_cycles += 1
                    # is the PASSIVE set independent at every phase?
                    indep = True
                    for st in cyc:
                        P = [i for i in range(n) if not (st >> i & 1)]
                        if any(j in adj[i] for i, j in itertools.combinations(P, 2)):
                            indep = False
                            break
                    if indep:
                        cycles_with_independent_P += 1
                        if example is None:
                            example = {"n": n, "edges": [list(e) for e in edges],
                                       "cycle": [f"{s:0{n}b}" for s in cyc]}
        out[label] = {"nontrivial_cycles": nontrivial_cycles,
                      "with_independent_passive_set": cycles_with_independent_P,
                      "one_in_band": lo <= F(1) <= hi,
                      "example": example}
    return out


# ---------------------------------------------------------------- arm C ----
def arm_c() -> dict:
    """Two U0 triangles, all 512 subsets of the 9 cross edges."""
    n = 6
    left, right = [0, 1, 2], [3, 4, 5]
    base = [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)]
    cross_all = [(i, j) for i in left for j in right]
    solo_nbr, solo_deg = graph_masks(3, [(0, 1), (0, 2), (1, 2)])
    solo_fp = live_fixed_points(3, solo_nbr, solo_deg)

    additive, influence_free, both, connected_nonadditive = [], [], [], []
    set_additive = []
    for mask in range(1 << 9):
        cross = [cross_all[k] for k in range(9) if mask >> k & 1]
        nbr, deg = graph_masks(n, base + cross)
        fps = live_fixed_points(n, nbr, deg)
        # ADDITIVITY IN TWO SENSES, and keeping them apart is the whole result.
        #
        #   COUNT-additive : |both-live fixed points| == 3 * 3. This is what a
        #                    capacity meter measures -- it multiplies per-component
        #                    counts and never inspects the states.
        #   SET-additive   : the both-live fixed points ARE the products, state by
        #                    state. This is what "the two blocks are independent"
        #                    actually means.
        #
        # MS-C-731: my first version tested SET equality against ALL fixed points and
        # scored 0 -- even the disconnected graph, because `{0,1}` with the right
        # triangle entirely passive is a perfectly good fixed point and is not a
        # product of two LIVE ones. Repairing that to both-live still gave 1, and the
        # reviewer's 43 only appears under the COUNT reading. That gap is not a
        # bookkeeping detail: it is the finding.
        prod = {a | (b << 3) for a in solo_fp for b in solo_fp}
        both_live = {a for a in fps if (a & 0b000111) and (a & 0b111000)}
        is_add = len(both_live) == len(solo_fp) ** 2          # COUNT-additive
        is_set_add = both_live == prod                        # SET-additive
        # one-step cross influence: does the left block's next state ever depend on
        # the right block's bits, or vice versa? Measured on all 64 states.
        infl = False
        for a in range(1 << n):
            nxt = node_next(a, nbr, deg, LO, HI)
            for flip in range(3, 6):                       # flip a RIGHT bit
                if (node_next(a ^ (1 << flip), nbr, deg, LO, HI) & 0b000111) != (nxt & 0b000111):
                    infl = True
                    break
            for flip in range(0, 3):                       # flip a LEFT bit
                if (node_next(a ^ (1 << flip), nbr, deg, LO, HI) & 0b111000) != (nxt & 0b111000):
                    infl = True
                    break
            if infl:
                break
        if is_add:
            additive.append((mask, len(cross), infl))
        if is_set_add:
            set_additive.append(mask)
        if not infl:
            influence_free.append(mask)
        if is_add and not infl:
            both.append(mask)
        if is_add and infl:
            connected_nonadditive.append((mask, len(cross)))

    by_cross = {}
    for _, c, _ in additive:
        by_cross[c] = by_cross.get(c, 0) + 1
    return {"solo_live_fixed_points": len(solo_fp),
            "graphs": 1 << 9,
            "count_additive": len(additive),
            "set_additive": len(set_additive),
            "count_additive_but_NOT_set_additive":
                len(additive) - len(set_additive),
            "additive": len(additive),
            "influence_free": len(influence_free),
            "additive_and_influence_free": len(both),
            "additive_WITH_nonzero_influence": len(connected_nonadditive),
            "influence_zero_implies_additive":
                all(m in {a[0] for a in additive} for m in influence_free),
            "additive_by_cross_edge_count": dict(sorted(by_cross.items()))}


# ---------------------------------------------------------------- arm D ----
def arm_d(nmax_full_pairs: int = 4, nmax_sampled: int = 5) -> dict:
    """Does the cover lemma survive `birth != survival`? Production engine only.

    `CHECKPOINT_046` states the node rule as ONE band, and `PILOT_001`'s protocol
    scopes `birth != survival` out in as many words ("a separate, larger axis... out
    of scope for this pilot"). So the lemma stands verified on the DIAGONAL of the
    92 x 92 band-pair space. The proof says the survival band cannot enter -- the
    only nodes at issue are those in `A(t+1) \\ A(t)`, and those are BIRTHS -- but
    that is an argument, and this project's rule is that the engine gets asked.

    Anchored to `autogenesis.reference.node_update`, the production update, never a
    re-implementation of it (D-019, D-044).
    """
    import numpy as np                                          # noqa: PLC0415
    from autogenesis.genome import Genome                       # noqa: PLC0415
    from autogenesis.reference import node_update               # noqa: PLC0415
    from autogenesis.rules import enumerate_bands               # noqa: PLC0415

    bands = enumerate_bands(6)
    sampled = bands[::max(1, len(bands) // 8)][:8]

    def connected(n, edges):
        adj = {i: set() for i in range(n)}
        for i, j in edges:
            adj[i].add(j)
            adj[j].add(i)
        seen, st = {0}, [0]
        while st:
            v = st.pop()
            for w in adj[v] - seen:
                seen.add(w)
                st.append(w)
        return len(seen) == n

    def cover_states(nmax):
        """(bonds, active) for every vertex-cover state of every connected graph."""
        out = []
        for n in range(2, nmax + 1):
            for edges, _ in all_labelled(n):
                if not edges or not connected(n, edges):
                    continue
                bonds = np.zeros((n, n), np.bool_)
                for i, j in edges:
                    bonds[i, j] = bonds[j, i] = True
                for a in range(1 << n):
                    A = {i for i in range(n) if a >> i & 1}
                    if all(i in A or j in A for i, j in edges):
                        out.append((bonds, np.array(
                            [(a >> i & 1) == 1 for i in range(n)], np.bool_)))
        return out

    def scan(nmax, birth_bands, survival_bands):
        # The band pair is the OUTER loop so one Genome is built per pair rather than
        # one per (state, pair) -- measured 45 us/call the other way round, 15 us this
        # way, and the difference is entirely `Genome().replace()`.
        states = cover_states(nmax)
        viol_lt = births_eq = checked = 0
        for bb in birth_bands:
            b_lt_1 = float(bb["hi"]) < 1.0
            for sb in survival_bands:
                g = Genome().replace(**{
                    "topology.birth_lo": float(bb["lo"]),
                    "topology.birth_hi": float(bb["hi"]),
                    "topology.survival_lo": float(sb["lo"]),
                    "topology.survival_hi": float(sb["hi"])})
                for bonds, act in states:
                    checked += 1
                    grew = bool((node_update(g, act, bonds) & ~act).any())
                    if b_lt_1:
                        viol_lt += grew
                    else:
                        births_eq += grew
        return {"cover_states": len(states), "engine_calls": checked,
                "violations_birth_hi_lt_1": viol_lt,
                "births_when_birth_hi_eq_1": births_eq}

    return {"full_pair_space": {"nmax": nmax_full_pairs,
                                "band_pairs": len(bands) ** 2,
                                **scan(nmax_full_pairs, bands, bands)},
            "sampled_survival": {"nmax": nmax_sampled,
                                 "band_pairs": len(bands) * len(sampled),
                                 **scan(nmax_sampled, bands, sampled)}}


def main() -> int:
    sha = check_protocol()
    C = claims()
    stamp = measurement.stamp(__file__)
    print("REVIEW COMBINATORICS -- independent reproduction")
    print("   protocol " + sha[:16] + " verified")
    print("   " + json.dumps(stamp))
    identity = {"protocol_sha256": sha, **stamp}
    # Resume only under an identical identity; STREAM.exists() is checked there and
    # a mismatched partial is discarded rather than mixed in.
    done = resume(STREAM, identity)
    print()
    t0 = time.perf_counter()
    # MS-C-849: identity makes the stream APPEND to a partial from this same run
    # instead of truncating the work resume() just read out of it.
    stream = Stream(STREAM, identity)
    stream.write({"event": "start", **identity})
    verdicts = []

    a = done.get("A") or arm_a()
    stream.write({"event": "arm", "arm": "A", "result": a})
    print(f"A  single-edge robustness, n<=6")
    print(f"     live Boolean fixed points scanned : {a['fixed_points_scanned']}")
    print(f"     robust under EVERY single toggle  : {a['robust']}   "
          f"predicted C(6,3)*(1+3) = {a['predicted_count']}   "
          f"{'MATCH' if a['robust'] == a['predicted_count'] else '*** DIFFERS ***'}")
    print(f"     every robust case has the claimed shape: {a['all_match_shape']}")
    print(f"     by passive-internal edges: {a['by_passive_internal_edges']}")
    print(f"     of which P is INDEPENDENT (window-theorem family): "
          f"{a['P_independent_count']}\n")

    b = done.get("B") or arm_b()
    stream.write({"event": "arm", "arm": "B", "result": b})
    print("B  impossibility theorem -- are the hypotheses load-bearing?")
    for label, v in b.items():
        print(f"     {label:<20} 1 in band {str(v['one_in_band']):<5} "
              f"nontrivial cycles {v['nontrivial_cycles']:>7}   "
              f"with independent passive set {v['with_independent_passive_set']}")
    print()

    c = done.get("C") or arm_c()
    stream.write({"event": "arm", "arm": "C", "result": c})
    print("C  cross influence vs additivity, two U0 triangles, 512 cross-edge subsets")
    print(f"     one triangle's live fixed points  : {c['solo_live_fixed_points']}")
    print(f"     COUNT-additive graphs (|fp| == 3*3): {c['count_additive']}")
    print(f"     SET-additive graphs (fp ARE products): {c['set_additive']}")
    print(f"     count-additive but NOT set-additive : "
          f"{c['count_additive_but_NOT_set_additive']}"
          "   <- a capacity meter cannot tell these from independence")
    print(f"     influence-free graphs             : {c['influence_free']}")
    print(f"     influence zero => additive        : {c['influence_zero_implies_additive']}")
    print(f"     additive WITH nonzero influence   : {c['additive_WITH_nonzero_influence']}"
          f"   (so the condition is NOT necessary)")
    print(f"     additive by cross-edge count      : {c['additive_by_cross_edge_count']}")

    d = done.get("D") or arm_d()
    stream.write({"event": "arm", "arm": "D", "result": d})
    print("\nD  does the cover lemma survive birth != survival? (production engine)")
    for k, v in d.items():
        print(f"     {k:<18} n<={v['nmax']}  {v['cover_states']} cover states x "
              f"{v['band_pairs']} band pairs = {v['engine_calls']} engine calls")
        print(f"     {'':<18}   violations with birth_hi < 1 : "
              f"{v['violations_birth_hi_lt_1']}"
              f"   {'OK' if v['violations_birth_hi_lt_1'] == 0 else '*** LEMMA FAILS ***'}")
        print(f"     {'':<18}   births with birth_hi == 1    : "
              f"{v['births_when_birth_hi_eq_1']}"
              f"   {'(non-vacuity OK)' if v['births_when_birth_hi_eq_1'] else '*** VACUOUS ***'}")

    # --- the protocol's decision rule, applied to the sealed claims ------------
    print("\nVERDICTS against the sealed pre-registration "
          "(protocols/REVIEW_FOLLOWUP_20260828.json)")
    n2, n5 = C["S2_single_edge_robustness"]["reviewer_numbers"], \
        C["S5_influence_vs_additivity"]["reviewer_numbers"]
    n3 = C["S3_impossibility_theorem"]["reviewer_numbers"]
    verdicts += [
        compare("S2 fixed points scanned", n2["fixed_points_scanned"],
                a["fixed_points_scanned"]),
        compare("S2 robust under every single toggle", n2["robust"], a["robust"]),
        compare("S2 labellings, no passive internal edge",
                n2["labellings_no_passive_internal_edge"],
                a["by_passive_internal_edges"].get(0)),
        compare("S2 labellings, one passive internal edge",
                n2["labellings_one_passive_internal_edge"],
                a["by_passive_internal_edges"].get(1)),
        # S3: the cycle COUNT is scope-dependent (their prior scan vs all labelled
        # graphs n<=6 here), so only the number that carries the claim is compared.
        compare("S3 U0-band cycles with an independent passive set",
                n3["prior_scan_with_independent_passive_set"],
                b["U0_[1/6,1/2]"]["with_independent_passive_set"]),
        compare("S5 influence-free graphs", n5["influence_free"],
                c["influence_free"]),
        compare("S5 additive total (COUNT reading)", n5["additive_total"],
                c["count_additive"]),
        compare("S5 additive, connected, with influence",
                n5["additive_connected_with_influence"],
                c["additive_WITH_nonzero_influence"]),
        # MS-C-848. The two sides are NOT symmetric and it matters. The protocol side
        # `n5` is read from `REVIEW_FOLLOWUP_20260828.json` on every run and is measured
        # to carry keys `['0', '3', '6']` -- strings, always -- so it still needs the
        # conversion. The measured side `c` is int-keyed when fresh and, since the
        # round-trip inverse moved into `resume()`, int-keyed when resumed too. Dropping
        # the conversion on the protocol side as well would have turned this claim
        # CONTESTED on EVERY run; I nearly did, and checked the file instead of assuming.
        compare("S5 additive by cross-edge count",
                {int(k): v for k, v in n5["additive_by_cross_edge_count"].items()},
                c["additive_by_cross_edge_count"]),
    ]
    contested = [v for v in verdicts if v["verdict"] == "CONTESTED"]
    print(f"\n   {len(verdicts)} claims compared: "
          f"{sum(1 for v in verdicts if v['verdict'] == 'CONFIRMED')} confirmed, "
          f"{sum(1 for v in verdicts if v['verdict'] == 'EXTENDED')} extended, "
          f"{len(contested)} CONTESTED")
    stream.write({"event": "verdicts", "verdicts": verdicts})
    stream.close()

    result = {**stamp, "band": [str(LO), str(HI)],
              "protocol_sha256": sha, "verdicts": verdicts,
              "contested": len(contested),
              "D_cover_lemma_asymmetric": d,
              "A_single_edge_robustness": a,
              "B_impossibility_hypotheses": b,
              "C_influence_vs_additivity": c,
              "seconds": round(time.perf_counter() - t0, 1)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    # MS-C-876: `verify_*` reads as read-only and this one WRITES. See the header note.
    from _artefact import write_result                         # noqa: E402
    write_result(OUT, json.dumps(result, indent=1) + "\n")
    print(f"\n   -> {OUT}   ({result['seconds']}s)")

    ok = (a["robust"] == a["predicted_count"] and a["all_match_shape"]
          and c["influence_zero_implies_additive"]
          and c["additive_WITH_nonzero_influence"] > 0
          and all(v["violations_birth_hi_lt_1"] == 0
                  and v["births_when_birth_hi_eq_1"] > 0 for v in d.values())
          and not contested)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
