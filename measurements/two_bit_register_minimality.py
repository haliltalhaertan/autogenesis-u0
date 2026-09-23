"""Is `n = 7, E = 10` really the smallest clocked 2-bit register? An exhaustive recount.

`L-509` verified the reported `n = 7, E = 10` register WORKS -- all sixteen closed-form
equalities, through the production engine, on a geometry `check_lemma` accepts. It did
**not** verify that nothing smaller works. That is a different and much larger claim: it
quantifies over every connected candidate graph below the reported one.

An external computation reported the minimality census (491 classes, 36,080,480 states,
exactly one positive class). This file recounts it here, and the two halves are recounted
by different means on purpose:

* **The universe** is rebuilt from scratch -- orbits of edge subsets under `S_n`, found by
  pointer-jumping with the `n-1` adjacent transpositions as generators, so the group is
  generated rather than enumerated. No graph library is imported, so the counts are not
  read out of the same atlas twice.
* **The dynamics** are NOT re-implemented. They run through `abstract_step`, which
  `MS-C-897` gated against the production engine at **0 disagreements in 247,904 states**.
  A second transcription would be a second place for defects (`D-020` item 8) and would
  make agreement with the external number evidence of nothing.

THE FIRST RUN OF THIS FILE WAS WRONG, AND THE DEFECT IS THE POINT OF THE `a != b` LINE.
It reported **nine** positive classes and a smallest at `n = 6, E = 9`, which would have
refuted the minimality claim outright. It was refuted by its own criterion instead: the
search allowed the two control nodes to be **the same node**, writing bit `x` at phase 0
and bit `y` at phase 1. Every other equality holds for such a witness, and it is not a
two-bit register -- it is one control line used twice. Requiring `a != b` kills eight of
the nine and leaves exactly the reported graph.

The warning was in hand and was read backwards. The calibration graph gave 72 witnesses
here against 12 reported externally, and that gap was written down as evidence the
criterion was *looser*, with the note that looseness was "the safe direction for
minimality". **It is the dangerous direction.** A looser criterion can only ADD positives,
and a positive below the claimed minimum destroys the claim. Recorded as `L-510`.

THE CRITERION IS THE REPORTED CLOSED FORM, NOT A PARAPHRASE. Four period-2 `OBJECT`
attractors labelled `(x, y)`, phase-aligned as `F(S0(x,y)) = S1(x,y)` and
`F(S1(x,y)) = S0(x,y)`, plus two nodes `a, b` with

    flip_a(S0(x,y)) = S0(x^1, y)        flip_b(S1(x,y)) = S1(x, y^1)

for all four `(x,y)`. Equality means the state itself, so the write is zero-transient: the
flipped state IS the target attractor state, not a point in its basin.

WHAT THIS DOES NOT CLAIM. The census runs on the abstract rest map, which reaches graphs
no lattice realizes. Minimality is therefore minimality *within the abstract family*; that
the WINNER is physically realizable is a separate fact, and it is the one `L-509` already
established through `check_lemma`. Robustness, reachability and self-assembly are
untouched here, as `L-494`/`L-504` require them to be.

Run:  ./.venv/Scripts/python.exe tools/measure.py measurements/two_bit_register_minimality.py
"""
from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "measurements"))

from autogenesis.genome import U0                              # noqa: E402
from abstract_map_equals_engine import abstract_step           # noqa: E402

MAX_N = 7
E_CAP = 10                    # the reported minimum; nothing above it can be smaller

CLAIM = ("Over every connected candidate graph at n <= 6, and at n = 7 with at most 10 "
         "candidate edges -- 491 isomorphism classes, 36,080,480 states, each stepped "
         "through the engine-gated abstract map -- exactly ONE class carries a clocked "
         "2-bit register under the reported closed form, and it is the reported graph. "
         "n_min = 7 and E_min = 10 within the abstract rest-scaffold family.")

WOULD_OVERTURN = ("Any positive class at n <= 6, or at n = 7 with E < 10, which would "
                  "break minimality outright. A positive count of zero, which would mean "
                  "the detector cannot see the register it is calibrated on and the "
                  "census measured nothing. More than one class at n = 7, E = 10, or a "
                  "winner not isomorphic to the reported graph. Or the rebuilt universe "
                  "not being 491 classes and 36,080,480 states, which would mean this "
                  "file and the external computation did not search the same thing.")


# ---------------------------------------------------------------- the criterion

def criterion(probe) -> bool:
    """True when this labelled system satisfies all sixteen register equalities.

    `S[(p, x, y)]` is a state id, `F` the next-state map, `flip[(v, s)]` the state with
    node `v`'s activity bit inverted. `a` writes bit x at phase 0, `b` writes bit y at
    phase 1.
    """
    S, F, flip, a, b = probe
    quad = [(0, 0), (0, 1), (1, 0), (1, 1)]
    if a == b:
        return False                      # one node writing both bits is ONE control line
    if len({S[(0, x, y)] for x, y in quad}) != 4:
        return False                      # four labels naming fewer than four states
    for x, y in quad:
        if F[S[(0, x, y)]] != S[(1, x, y)] or F[S[(1, x, y)]] != S[(0, x, y)]:
            return False                  # not a phase-aligned period-2 clock
        if flip[(a, S[(0, x, y)])] != S[(0, x ^ 1, y)]:
            return False                  # bit x not written at phase 0
        if flip[(b, S[(1, x, y)])] != S[(1, x, y ^ 1)]:
            return False                  # bit y not written at phase 1
    return True


def _toy(broken=None):
    """A synthetic 8-state register, optionally sabotaged in one named way."""
    S = {(p, x, y): p * 4 + x * 2 + y for p in (0, 1) for x in (0, 1) for y in (0, 1)}
    if broken == "degenerate":
        S[(0, 1, 1)] = S[(0, 0, 0)]
    F = {}
    for x in (0, 1):
        for y in (0, 1):
            F[S[(0, x, y)]] = S[(1, x, y)]
            F[S[(1, x, y)]] = S[(0, x, y)]
    if broken == "not_period_2":
        F[S[(1, 0, 0)]] = S[(0, 1, 1)]
    flip = {}
    for x in (0, 1):
        for y in (0, 1):
            flip[("a", S[(0, x, y)])] = S[(0, x ^ 1, y)]
            flip[("b", S[(1, x, y)])] = S[(1, x, y ^ 1)]
    if broken == "control_leaks":
        flip[("a", S[(0, 0, 1)])] = S[(0, 1, 0)]      # writing x also flips y
    return (S, F, flip, "a", "b")


def _toy_one_node():
    """The defect that made the first run report nine positives: ONE node writing both
    bits, at the two different phases. It satisfies every other equality, so only an
    explicit `a != b` rejects it."""
    S, F, flip, _, _ = _toy()
    for x in (0, 1):
        for y in (0, 1):
            flip[("a", S[(1, x, y)])] = S[(1, x, y ^ 1)]      # node a also writes y
    return (S, F, flip, "a", "a")


CONTROLS = [
    ("a perfect 8-state register -> register", _toy(), True),
    ("control a also flips bit y -> not independent", _toy("control_leaks"), False),
    ("one attractor not period 2 -> not a clock", _toy("not_period_2"), False),
    ("two labels naming one state -> only three bits' worth", _toy("degenerate"), False),
    ("one node writing both bits -> ONE control line, not two", _toy_one_node(), False),
]


# ---------------------------------------------------------------- the universe

def _atlas(n, e_cap=None):
    """Connected unlabeled graphs on n nodes as edge lists, by orbit of edge subsets."""
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    idx = {p: k for k, p in enumerate(pairs)}
    E, N = len(pairs), 1 << len(pairs)

    gens = []
    for a in range(n - 1):
        p = list(range(n))
        p[a], p[a + 1] = p[a + 1], p[a]
        bit = [idx[tuple(sorted((p[i], p[j])))] for (i, j) in pairs]
        lo_w = min(11, E)
        hi_w = E - lo_w
        T0 = np.zeros(1 << lo_w, dtype=np.int64)
        for v in range(1 << lo_w):
            m = 0
            for k in range(lo_w):
                if v >> k & 1:
                    m |= 1 << bit[k]
            T0[v] = m
        T1 = np.zeros(1 << hi_w, dtype=np.int64)
        for v in range(1 << hi_w):
            m = 0
            for k in range(hi_w):
                if v >> k & 1:
                    m |= 1 << bit[lo_w + k]
            T1[v] = m
        masks = np.arange(N, dtype=np.int64)
        gens.append((T0[masks & ((1 << lo_w) - 1)] | T1[masks >> lo_w]).astype(np.int64))

    c = np.arange(N, dtype=np.int64)
    while True:
        before = c.copy()
        for P in gens:
            np.minimum(c, c[P], out=c)
        c = c[c]
        if np.array_equal(c, before):
            break

    out = []
    for m in np.flatnonzero(c == np.arange(N, dtype=np.int64)).tolist():
        edges = [pairs[k] for k in range(E) if m >> k & 1]
        if e_cap is not None and len(edges) > e_cap:
            continue
        adj = {i: set() for i in range(n)}
        for i, j in edges:
            adj[i].add(j)
            adj[j].add(i)
        seen, stack = {0}, [0]
        while stack:
            v = stack.pop()
            for w in adj[v] - seen:
                seen.add(w)
                stack.append(w)
        if len(seen) == n:
            out.append((n, edges))
    return out


def family():
    fam = []
    for n in range(3, MAX_N):
        fam += _atlas(n)
    fam += _atlas(MAX_N, E_CAP)
    return fam


def universe():
    return (sum(1 << (n + len(e)) for n, e in family()),
            "every state of every connected candidate graph at n <= 6 and at n = 7 with "
            "E <= 10, each stepped once through the engine-gated abstract map")


# ---------------------------------------------------------------- the census

def graph6(n, edges):
    """graph6 of the CANONICAL labelling. The string is labelling-dependent, so encoding
    the atlas representative directly produced `FtrE?` for the same graph an earlier check
    encoded as `FjrE?` -- two names for one class, which is exactly the confusion `L-496`
    was about. `min` over sorted tuples, which are totally ordered."""
    edges = min(tuple(sorted(tuple(sorted((p[i], p[j]))) for i, j in edges))
                for p in itertools.permutations(range(n)))
    have = {tuple(sorted(e)) for e in edges}
    bits = [1 if (i, j) in have else 0 for j in range(1, n) for i in range(j)]
    bits += [0] * (-len(bits) % 6)
    return chr(n + 63) + "".join(
        chr(63 + int("".join(map(str, bits[k:k + 6])), 2)) for k in range(0, len(bits), 6))


def scan(job):
    """One graph class: full state census, then the register search. Returns a summary."""
    n, edge_list = job
    g = U0()
    cand = [set() for _ in range(n)]
    for i, j in edge_list:
        cand[i].add(j)
        cand[j].add(i)
    edges = [(i, j) for i in range(n) for j in cand[i] if i < j]   # abstract_step's order
    E = len(edges)
    total = 1 << (n + E)

    nxt = np.empty(total, dtype=np.int64)
    for code in range(total):
        nb, na = abstract_step(n, cand, code >> n, code & ((1 << n) - 1), g)
        nxt[code] = (nb << n) | na

    colour = np.zeros(total, dtype=np.int8)
    which = np.full(total, -1, dtype=np.int32)
    cycles = []
    for start in range(total):
        if colour[start]:
            continue
        path, s = [], start
        while colour[s] == 0:
            colour[s] = 1
            path.append(s)
            s = int(nxt[s])
        if colour[s] == 1:
            cyc = path[path.index(s):]
            for c in cyc:
                which[c] = len(cycles)
            cycles.append(cyc)
        for c in path:
            colour[c] = 2

    # period-2 OBJECT states, indexed by the state itself
    obj_states = set()
    for cyc in cycles:
        if len(cyc) != 2:
            continue
        sets, act = [], False
        for c in cyc:
            bb, ab = c >> n, c & ((1 << n) - 1)
            sets.append(frozenset(k for k in range(E) if bb >> k & 1))
            act = act or ab != 0
        if not act or sets[0] == sets[1]:
            continue
        always = sets[0] & sets[1]
        if always and (sets[0] | sets[1]) - always:
            obj_states.update(cyc)

    F = {s: int(nxt[s]) for s in obj_states}
    flip = {(v, s): s ^ (1 << v) for v in range(n) for s in obj_states}
    hits = []
    for s00 in obj_states:
        for a in range(n):
            s10 = flip[(a, s00)]
            if s10 not in obj_states:
                continue
            for b in range(n):
                s01 = F.get(flip[(b, F[s00])])
                s11 = F.get(flip[(b, F[s10])])
                if s01 not in obj_states or s11 not in obj_states:
                    continue
                S = {(0, 0, 0): s00, (0, 1, 0): s10, (0, 0, 1): s01, (0, 1, 1): s11}
                S.update({(1, x, y): F[S[(0, x, y)]] for x in (0, 1) for y in (0, 1)})
                if criterion((S, F, flip, a, b)):
                    hits.append((a, b))
    return {"n": n, "E": E, "states": total, "attractors": len(cycles),
            "p2_object": len(obj_states) // 2, "witnesses": len(hits),
            "g6": graph6(n, edge_list), "edges": edge_list}


def measure():
    # SERIAL ON PURPOSE. `multiprocessing` spawn does not work in this environment:
    # every child dies in `spawn_main` with `OSError: [WinError 87]` opening the parent
    # handle, and `Pool.map` then hangs with idle workers rather than raising. Measured
    # cost of the serial run is 61.5 us/state -> ~37 min for 36,080,480 states, which is
    # affordable, so the fix is to pay it rather than to fight the sandbox.
    fam = family()
    states = sum(1 << (n + len(e)) for n, e in fam)
    print(f"   universe rebuilt from scratch: {len(fam)} classes, {states:,} states")
    print("   external computation reported: 491 classes, 36,080,480 states")
    universe_ok = len(fam) == 491 and states == 36080480
    print(f"   universe matches: {universe_ok}\n")

    t0 = time.perf_counter()
    rows, done = [], 0
    for job in fam:
        rows.append(scan(job))
        done += 1 << (job[0] + len(job[1]))
        if len(rows) % 25 == 0:
            print(f"      {len(rows):>3}/{len(fam)} classes, {done:>10,}/{states:,} "
                  f"states, {time.perf_counter()-t0:>6.0f} s", flush=True)
    secs = time.perf_counter() - t0

    strata = {}
    for r in rows:
        lab = ("E<=9" if r["n"] == 7 and r["E"] <= 9 else
               ("E=10" if r["n"] == 7 else "all E"))
        strata.setdefault((r["n"], lab), [0, 0])
        strata[(r["n"], lab)][0] += 1
        strata[(r["n"], lab)][1] += r["witnesses"] > 0

    print(f"   {'stratum':<14}{'classes':>9}{'positive':>10}")
    for (n, lab), (cnt, pos) in sorted(strata.items()):
        print(f"   n={n} {lab:<9}{cnt:>9}{pos:>10}")

    pos = [r for r in rows if r["witnesses"]]
    print(f"\n   {len(rows)} classes, {sum(r['states'] for r in rows):,} states, "
          f"{secs:.1f} s on 16 workers")
    for r in pos:
        print(f"   POSITIVE  n={r['n']} E={r['E']}  graph6={r['g6']}  "
              f"attractors={r['attractors']}  period-2 OBJECT={r['p2_object']}  "
              f"witnesses={r['witnesses']}")
        print(f"             edges={r['edges']}")
    print("\n   witnesses are (a, b) control-node pairs and phase orientations inside ONE")
    print("   graph, not distinct graphs. The class count is what minimality is about.")

    below = [r for r in pos if r["n"] < 7 or r["E"] < 10]
    ok = universe_ok and len(pos) == 1 and not below
    if not pos:
        return ("0 positive classes", "REFUSED -- a detector that finds nothing on a "
                "family containing its own calibration case has measured nothing, not "
                "established minimality (L-484)")
    return (f"{len(pos)} positive class of {len(rows)}, at n={pos[0]['n']} "
            f"E={pos[0]['E']}, graph6={pos[0]['g6']}",
            f"the claim needs the rebuilt universe to be 491/36,080,480, exactly one "
            f"positive class, and none below n=7,E=10 "
            f"({'HOLDS' if ok else 'DOES NOT HOLD'})")
