"""OB-104 / OB-122 -- the founding object made EXACT, and its orbits enumerated.

WHAT THIS IS FOR
----------------
`OB-104` asks for a topology that oscillates while the system is at kinematic rest.
`MS-C-445`/`MS-C-465` found it in random U0 worlds: 164 worlds end with
`max|v| <= 1e-9` and an edge set that changes on every step. `MS-C-469` gave the
mechanism -- `spring_k*(r - r0)` is zero at `r0`, repulsion is zero for
`r >= repulsion_cutoff`, and U0 has `r0 = 1.0 > 0.75 = repulsion_cutoff`, so a pair
sitting at `r0` exerts **exactly zero force in both bond states** and the discrete
layer can switch it forever without the continuous layer noticing.

Those worlds are at rest to `4.9e-15`, not to `0`, so every statement about them is a
statement about a tolerance. `HANDOFF_002` sec 5 reading (a) -- `x(t+2) == x(t)`
EXACTLY -- has never been exhibited. This file exhibits it, and then uses it.

THE REST SCAFFOLD LEMMA (proved here; verified against the production kernel below)
-----------------------------------------------------------------------------------
Let `X` be a point set with, for every pair `i != j`,

    (L1)  r_ij == r0            whenever  r_ij <= R_candidate
    (L2)  r_ij >= repulsion_cutoff
    (L3)  r_ij >= 1e-12

and let `u == 0`. Then for EVERY bond set `B` and EVERY activity `A`,
`kernel._step` returns `nx == x` and `nu == u` bit-for-bit.

Proof. A bond can only ever exist between candidate pairs: `formed` requires
`cand[i,j]` (`kernel.py:118`) and `kept` requires `bonds[i,j]` (`kernel.py:93`), so
`B` stays inside the candidate set for all time. Take any pair. If it is bonded then
by (L1) `rr == r0`, so `coef = spring_k*(rr - r0) == 0.0`, every component
`f = 0.0 * rh` is a signed zero, and `F[i,c] += f` / `F[j,c] -= f` leave `F` unchanged
because `z + 0.0 == z` and `z - (-0.0) == z` for every finite `z`. If it is not bonded
then by (L2) `rr >= repulsion_cutoff`, so the `elif rr < rep_cut` branch is not taken
and `continue` skips the pair. By (L3) no pair is coincident. So the only contribution
to `F` is the damping initialiser `-gamma*u == -0.0`, hence
`nu = u + dt*(-0.0)/mass == 0.0` and `nx = x + dt*0.0 == x`, exactly. []

Two consequences, and they are the point:

  * `x` and `u` are a fixed point of the continuous layer under EVERY discrete state,
    so one verified step verifies all of them by induction. Reading (a) is realised
    with residual 0.0, not 1e-15.
  * On such a scaffold U0 IS the finite deterministic map of `OB-122`,
    `(B, A) -> (B', A')`, with no approximation anywhere. Its orbit structure is
    exactly decidable and, on small scaffolds, exhaustively enumerable -- which is
    what `OB-122` asks for and what no horizon-bounded sweep can give.

`R_break` never enters: `B` stays inside the candidate set and
`R_candidate < R_break`, so the retention distance test cannot bind on a scaffold.

THE SCAFFOLDS
-------------
`TRI` -- triangular-lattice patches, rows at `y in {-2h, -h, 0, h, 2h}` with
`h = 0.8660254037844387`, ONE ULP above `fl(sqrt(3)/2)`. That ULP is load-bearing: at
`fl(sqrt(3)/2)` the diagonal pair computes `r = 0.9999999999999999` and the scaffold
leaks force at 1e-16 per step; at `h`, `0.25 + h*h == 1.0` and `sqrt(1.0) == 1.0`. Row
heights are built only by negation and doubling of `h`, so every consecutive row gap is
`h` to the bit. Candidate degree reaches exactly 6 = U0's valence, and the candidate
graph has triangles.

`HEX_7` -- one hexagonal cell: a centre at candidate degree 6, the maximal-valence
unit of the triangular lattice, and the smallest patch that saturates U0's valence.

`SQ` -- the integer square lattice, spacing 1. Also exact (`1.0*1.0 == 1.0`), also a
scaffold, and TRIANGLE-FREE: its second distance is `sqrt(2) = 1.414 > 1.35`.

CONTROLS, all three of which can fail
-------------------------------------
  * `MS-C-730b` (on a fixed graph, under any birth band excluding 1 and any survival
    band, no oscillating activity preserves its own bond set): **no cycle of period
    >= 2 here may have a constant bond set.** Any such cycle refutes it, or refutes
    this scaffold. Reported per scaffold, never as a union.
  * `MS-C-550` (a formed edge is a triangle in `C` with an active apex, so a
    triangle-free `C` admits no formation, the bond chain is non-increasing, and
    `CHECKPOINT_046` then freezes the activity): **every orbit on `SQ` must have
    period 1.** One oscillating `SQ` orbit refutes it. `triangles` is MEASURED per
    scaffold, not assumed, because the whole control rests on it.
  * detector: `cycle_of` is run against a synthetic map with known transient and
    known period, against a map with no cycle inside the cap, and against a map whose
    period differs by one -- the failing cases are constructed BEFORE any physics is
    measured (`_selftest_detector`).

CAPS. `EXHAUSTIVE_MAX_STATES` bounds which scaffolds get a complete enumeration;
`MAX_ITER` bounds the sampled arm. Both are reported with what exceeded them. A
sampled world that exceeds `MAX_ITER` is recorded as such and is NOT counted as any
period.

Run:  ./.venv/Scripts/python.exe tools/rest_scaffold_orbits.py
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from autogenesis.genome import U0                              # noqa: E402
from autogenesis.state import State                            # noqa: E402
from autogenesis import measurement, rng, simulator as sim     # noqa: E402
from _review_protocol import Stream, resume                    # noqa: E402

PROTOCOL = ROOT / "protocols" / "REST_SCAFFOLD_001.json"
OUT = ROOT / "runs" / "rest_scaffold_orbits.json"
STREAM = ROOT / "runs" / "rest_scaffold_orbits.stream.jsonl"

NAMESPACE = "rest_scaffold_v1"        # fresh: cannot collide with any prior stream
H = 0.8660254037844387                # one ULP above fl(sqrt(3)/2); see docstring

EXHAUSTIVE_MAX_STATES = 1 << 20       # complete-enumeration budget, per scaffold
MAX_ITER = 4096                       # sampled arm: iterations before giving up
FREEZE_TRIALS, FREEZE_STEPS = 64, 64  # empirical arm of the Rest Scaffold Lemma
P_BOND = (0.2, 0.4, 0.6, 0.8, 1.0)
Q_ACTIVE = (0.2, 0.4, 0.6, 0.8)
SAMPLE_SEEDS = 50                     # 5 x 4 x 50 = 1000 worlds per sampled scaffold


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def scaffold_tri(rows: tuple[int, ...], cols: int) -> np.ndarray:
    ys = {-2: -(2.0 * H), -1: -H, 0: 0.0, 1: H, 2: 2.0 * H}
    pts = [(float(c) + (0.5 if (rw % 2) else 0.0), ys[rw], 0.0)
           for rw in rows for c in range(cols)]
    return np.array(pts, dtype=np.float64)


def scaffold_sq(rows: int, cols: int) -> np.ndarray:
    return np.array([(float(c), float(rw), 0.0) for rw in range(rows)
                     for c in range(cols)], dtype=np.float64)


def scaffold_hex() -> np.ndarray:
    return np.array([(-1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                     (-0.5, H, 0.0), (0.5, H, 0.0),
                     (-0.5, -H, 0.0), (0.5, -H, 0.0)], dtype=np.float64)


def kernel_distances(X: np.ndarray) -> np.ndarray:
    """`kernel._step`'s pair loop, in its arithmetic order: `acc` accumulates
    component by component and `rr = sqrt(acc)`. That this reimplementation agrees
    with the kernel is CHECKED, not asserted -- see `verify_candidate_set`."""
    n = X.shape[0]
    r = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            acc = 0.0
            for c in range(X.shape[1]):
                d = X[j, c] - X[i, c]
                acc = acc + d * d
            r[i, j] = r[j, i] = math.sqrt(acc)
    return r


def check_lemma(g, X: np.ndarray) -> dict:
    """(L1)(L2)(L3). Raises -- never repairs silently (D-004)."""
    r0, rc, rep = g.physics.r0, g.topology.candidate_radius, g.physics.repulsion_cutoff
    r = kernel_distances(X)
    n = X.shape[0]
    d = r[np.triu_indices(n, 1)]
    cand = d <= rc
    if not bool(np.all(d[cand] == r0)):
        bad = d[cand][d[cand] != r0]
        raise AssertionError(f"(L1) {bad.size} candidate pairs are not exactly r0; "
                             f"worst |r-r0| = {float(np.abs(bad - r0).max())!r}")
    if not bool(np.all(d >= rep)):
        raise AssertionError(f"(L2) min pair distance {float(d.min())!r} "
                             f"< repulsion_cutoff {rep}")
    if not bool(np.all(d >= 1e-12)):
        raise AssertionError("(L3) coincident pair")
    C = (r <= rc) & ~np.eye(n, dtype=bool)
    tri = sum(1 for i in range(n) for j in range(i + 1, n) for k in range(j + 1, n)
              if C[i, j] and C[i, k] and C[j, k])
    return {"n": n, "candidate_edges": int(C.sum() // 2), "triangles": tri,
            "max_candidate_degree": int(C.sum(1).max()),
            "min_nonunit_distance": (float(d[d != r0].min())
                                     if bool((d != r0).any()) else None),
            "pairs_exactly_r0": int((d == r0).sum())}


def candidate_edges(g, X: np.ndarray) -> list[tuple[int, int]]:
    r = kernel_distances(X)
    n = X.shape[0]
    return [(i, j) for i in range(n) for j in range(i + 1, n)
            if r[i, j] <= g.topology.candidate_radius]


# --------------------------------------------------------------------------- #
# the discrete map, through the PRODUCTION kernel
# --------------------------------------------------------------------------- #
class Scaffold:
    """Encodes a state as one integer `code = (bond_bits << n) | active_bits`."""

    def __init__(self, g, X: np.ndarray, edges: list[tuple[int, int]]):
        self.g, self.X, self.edges = g, X, edges
        self.n, self.E = X.shape[0], len(edges)
        self.ei = np.array([i for i, _ in edges], dtype=np.intp)
        self.ej = np.array([j for _, j in edges], dtype=np.intp)
        self.pow_e = (1 << np.arange(self.E, dtype=np.int64)) if self.E else \
            np.zeros(0, dtype=np.int64)
        self.pow_n = 1 << np.arange(self.n, dtype=np.int64)
        self._bond_cache: dict[int, np.ndarray] = {}

    def bonds_of(self, bb: int) -> np.ndarray:
        m = self._bond_cache.get(bb)
        if m is None:
            m = np.zeros((self.n, self.n), dtype=bool)
            for k, (i, j) in enumerate(self.edges):
                if bb >> k & 1:
                    m[i, j] = m[j, i] = True
            self._bond_cache[bb] = m
        return m

    def state(self, code: int) -> State:
        bb, ab = code >> self.n, code & ((1 << self.n) - 1)
        active = (ab >> np.arange(self.n) & 1).astype(bool)
        return State(self.X, np.zeros_like(self.X), active, self.bonds_of(bb))

    def encode(self, s: State) -> int:
        bb = int(s.bonds[self.ei, self.ej] @ self.pow_e) if self.E else 0
        ab = int(s.active @ self.pow_n)
        return (bb << self.n) | ab

    def step_code(self, code: int) -> int:
        return self.encode(sim.step(self.g, self.state(code)))

    def split(self, code: int) -> tuple[int, int]:
        return code >> self.n, code & ((1 << self.n) - 1)


def verify_candidate_set(sc: Scaffold) -> None:
    """`kernel_distances` reimplements the kernel's pair loop; if the two disagreed,
    every `cand` set here would describe a different experiment from the one the
    kernel runs. Step the kernel from the all-bonded, all-active state: every edge has
    an active endpoint so all are kept, and nothing is left to form, so the edge set
    can only change if the kernel's `cand` differs from ours."""
    code = (((1 << sc.E) - 1) << sc.n) | ((1 << sc.n) - 1)
    out = sim.step(sc.g, sc.state(code))
    if out.edge_list() != sorted(sc.edges):
        raise AssertionError("kernel candidate set != kernel_distances candidate set: "
                             f"{len(out.edge_list())} kernel edges vs {sc.E} ours")


def assert_frozen(sc: Scaffold, trials: int, steps: int, index: int) -> dict:
    """Empirical arm of the lemma, against the production kernel. The claim is 0.0, so
    the tolerance is 0.0 -- any motion at all is a failure."""
    st = rng.streams(NAMESPACE, index)
    worst_x = worst_u = 0.0
    for _ in range(trials):
        bb = int(st["bonds"].integers(0, 1 << sc.E))
        ab = int(st["activity"].integers(0, 1 << sc.n))
        s = sc.state((bb << sc.n) | ab)
        x0, u0 = s.x.copy(), s.u.copy()
        for _t in range(steps):
            s = sim.step(sc.g, s)
        worst_x = max(worst_x, float(np.abs(s.x - x0).max()))
        worst_u = max(worst_u, float(np.abs(s.u - u0).max()))
    if worst_x != 0.0 or worst_u != 0.0:
        raise AssertionError(f"REST SCAFFOLD LEMMA VIOLATED: max|dx| = {worst_x!r}, "
                             f"max|du| = {worst_u!r} -- the lemma claims exactly 0.0")
    return {"trials": trials, "steps": steps, "max_abs_dx": worst_x,
            "max_abs_du": worst_u, "exactly_frozen": True}


# --------------------------------------------------------------------------- #
# orbits
# --------------------------------------------------------------------------- #
def cycle_of(start, succ, max_iter: int):
    """(transient, period, cycle) by first repeat, for any deterministic `succ`.
    Period is None when `max_iter` is exhausted -- never rounded to a guess."""
    seen, order, s = {}, [], start
    for t in range(max_iter):
        if s in seen:
            first = seen[s]
            return first, t - first, order[first:]
        seen[s] = t
        order.append(s)
        s = succ(s)
    return None, None, None


def _selftest_detector() -> dict:
    """The failing cases first (D-020 item 5)."""
    tr, pd, cyc = cycle_of(0, lambda k: k + 1 if k < 9 else 3, 128)
    if (tr, pd) != (3, 7) or len(cyc) != 7:
        raise AssertionError(f"detector self-test: got {(tr, pd)}, want (3, 7)")
    if cycle_of(0, lambda k: k + 1, 32) != (None, None, None):
        raise AssertionError("detector self-test: a map with no cycle inside max_iter "
                             "must report None, not a period")
    if cycle_of(0, lambda k: k + 1 if k < 8 else 3, 128)[1] == 7:
        raise AssertionError("detector self-test: period 6 read as 7")
    return {"rho_transient_period": [tr, pd], "no_cycle_reports_none": True,
            "period_6_not_read_as_7": True}


def classify_cycle(sc: Scaffold, cycle: list[int]) -> dict:
    bonds = {sc.split(c)[0] for c in cycle}
    act = {sc.split(c)[1] for c in cycle}
    return {"period": len(cycle),
            "bonds_constant": len(bonds) == 1,
            "active_constant": len(act) == 1,
            "dead": len(cycle) == 1 and cycle[0] == 0,
            "distinct_bond_sets": len(bonds),
            "distinct_activities": len(act)}


def enumerate_exhaustive(sc: Scaffold, budget: int) -> dict:
    """Every state, once. Complete: not a sample, and not horizon-bounded."""
    total = 1 << (sc.n + sc.E)
    if total > budget:
        return {"complete": False, "states": total, "budget": budget,
                "exceeded_budget": True}
    nxt = np.empty(total, dtype=np.int64)
    for code in range(total):
        nxt[code] = sc.step_code(code)
    colour = np.zeros(total, dtype=np.int8)         # 0 unvisited, 1 on path, 2 done
    which = np.full(total, -1, dtype=np.int32)      # attractor id
    cycles: list[list[int]] = []
    for start in range(total):
        if colour[start]:
            continue
        path, s = [], start
        while colour[s] == 0:
            colour[s] = 1
            path.append(s)
            s = int(nxt[s])
        if colour[s] == 1:                          # closed a new cycle
            cyc = path[path.index(s):]
            cid = len(cycles)
            cycles.append(cyc)
            for c in cyc:
                which[c] = cid
        for c in path:
            colour[c] = 2
    for start in range(total):                      # basins, each state assigned once
        if which[start] >= 0:
            continue
        path, s = [], start
        while which[s] < 0:
            path.append(s)
            s = int(nxt[s])
        for c in path:
            which[c] = which[s]
    sizes = np.bincount(which, minlength=len(cycles))
    rows = [{**classify_cycle(sc, cyc), "basin": int(sizes[cid])}
            for cid, cyc in enumerate(cycles)]
    return {"complete": True, "states": total, "n_attractors": len(cycles),
            "attractors": rows, "exceeded_budget": False}


def sample_worlds(sc: Scaffold, base_index: int) -> tuple[list, int]:
    rows, over = [], 0
    for pi, p in enumerate(P_BOND):
        for qi, q in enumerate(Q_ACTIVE):
            for k in range(SAMPLE_SEEDS):
                idx = base_index + 1000 * pi + 100 * qi + k
                st = rng.streams(NAMESPACE, idx)
                bb = sum(1 << e for e in range(sc.E) if st["bonds"].random() < p)
                ab = sum(1 << i for i in range(sc.n) if st["activity"].random() < q)
                tr, pd, cyc = cycle_of((bb << sc.n) | ab, sc.step_code, MAX_ITER)
                if pd is None:
                    over += 1
                    rows.append({"p": p, "q": q, "seed_index": idx,
                                 "exceeded_MAX_ITER": True})
                    continue
                rows.append({"p": p, "q": q, "seed_index": idx, "transient": tr,
                             **classify_cycle(sc, cyc)})
    return rows, over


# --------------------------------------------------------------------------- #
def verify_protocol() -> str:
    """D-015: preregistration is enforced by a hash, not by intent."""
    want = PROTOCOL.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
    got = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    if got != want:
        raise SystemExit(f"REST_SCAFFOLD_001 PROTOCOL HASH MISMATCH\n  frozen: {want}\n"
                         f"  now:    {got}")
    return got


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def build_scaffolds() -> list[dict]:
    """`triangles` is measured by `check_lemma`, never taken from these names."""
    return [
        {"name": "TRI_3", "X": scaffold_tri((0, 1), 2)[:3], "arm": "exhaustive"},
        {"name": "TRI_4", "X": scaffold_tri((0, 1), 2), "arm": "exhaustive"},
        {"name": "TRI_2x3", "X": scaffold_tri((0, 1), 3), "arm": "exhaustive"},
        {"name": "HEX_7", "X": scaffold_hex(), "arm": "exhaustive"},
        {"name": "SQ_2x2", "X": scaffold_sq(2, 2), "arm": "exhaustive"},
        {"name": "SQ_2x3", "X": scaffold_sq(2, 3), "arm": "exhaustive"},
        {"name": "SQ_2x4", "X": scaffold_sq(2, 4), "arm": "exhaustive"},
        {"name": "TRI_5x5", "X": scaffold_tri((-2, -1, 0, 1, 2), 5), "arm": "sampled"},
        {"name": "SQ_5x5", "X": scaffold_sq(5, 5), "arm": "sampled"},
    ]


def report(results: list[dict]) -> dict:
    """Per case, never per union (MS-C-738): every scaffold prints its own line and a
    control verdict names the scaffold it came from."""
    print("   PER-SCAFFOLD RESULT (per case, not per union)")
    obj_tri = obj_sq = 0
    const_bond_cycles, tri_free_oscillating = [], []
    for r in results:
        name, tri = r["name"], r["geometry"]["triangles"]
        if r["arm"] == "exhaustive" and r["exhaustive"].get("complete"):
            att = r["exhaustive"]["attractors"]
            total = r["exhaustive"]["states"]
            hist: dict[str, int] = {}
            for a in att:
                hist[str(a["period"])] = hist.get(str(a["period"]), 0) + 1
            obj = [a for a in att if a["period"] >= 2 and not a["bonds_constant"]]
            bad = [a for a in att if a["period"] >= 2 and a["bonds_constant"]]
            share = sum(a["basin"] for a in obj) / total
            print(f"     {name:<9} EXHAUSTIVE {total:>8} states  {len(att):>4} "
                  f"attractors  periods {json.dumps(hist, sort_keys=True)}")
            print(f"               OBJECT (period>=2, bond set MOVES): {len(obj)} "
                  f"attractors, basin share {share:.4f}")
        elif r["arm"] == "sampled":
            rows = [x for x in r["sampled"]["rows"] if "exceeded_MAX_ITER" not in x]
            hist = {}
            for x in rows:
                hist[str(x["period"])] = hist.get(str(x["period"]), 0) + 1
            obj = [x for x in rows if x["period"] >= 2 and not x["bonds_constant"]]
            bad = [x for x in rows if x["period"] >= 2 and x["bonds_constant"]]
            dead = [x for x in rows if x["dead"]]
            med = int(np.median([x["transient"] for x in rows])) if rows else -1
            print(f"     {name:<9} SAMPLED    {len(rows):>8} worlds  "
                  f"periods {json.dumps(hist, sort_keys=True)}")
            print(f"               OBJECT: {len(obj)} of {len(rows)} "
                  f"({len(obj) / max(1, len(rows)):.3f})   dead-state: {len(dead)}   "
                  f"median transient {med}")
        else:
            print(f"     {name:<9} NOT ENUMERATED (exceeded the budget)")
            continue
        if bad:
            const_bond_cycles.append([name, len(bad)])
        if tri == 0 and any(a["period"] >= 2 for a in (att if r["arm"] ==
                            "exhaustive" else rows)):
            tri_free_oscillating.append(name)
        if tri:
            obj_tri += len(obj)
        else:
            obj_sq += len(obj)

    c730b, c550 = not const_bond_cycles, not tri_free_oscillating
    print()
    print("   CONTROL MS-C-730b (no period>=2 cycle with a CONSTANT bond set): "
          + ("PASS" if c730b else f"*** FAILED: {const_bond_cycles} ***"))
    print("   CONTROL MS-C-550  (triangle-free scaffold -> period 1 only): "
          + ("PASS" if c550 else f"*** FAILED: {tri_free_oscillating} ***"))
    print(f"\n   THE OBJECT (exact positional rest + a moving bond set): "
          f"{obj_tri} on triangulated scaffolds, {obj_sq} on triangle-free ones")
    return {"object_triangulated": obj_tri, "object_triangle_free": obj_sq,
            "control_MS_C_730b_pass": c730b, "control_MS_C_550_pass": c550,
            "constant_bond_cycles": const_bond_cycles,
            "triangle_free_oscillating": tri_free_oscillating,
            "controls_pass": bool(c730b and c550)}


def main() -> int:
    sha = verify_protocol()
    stamp = measurement.stamp(__file__)
    g = U0()
    g.validate()
    print("REST SCAFFOLD ORBITS -- OB-104 / OB-122")
    print(f"   protocol REST_SCAFFOLD_001 {sha[:16]} verified")
    print("   " + json.dumps(stamp))
    print(f"   genome {g.short_hash()} -- U0 UNMODIFIED, so the physics is U0's. "
          "initial_conditions are bypassed: states are constructed, not sampled.")
    print(f"   namespace {NAMESPACE}   h = {H!r}")
    print(f"   caps: EXHAUSTIVE_MAX_STATES = {EXHAUSTIVE_MAX_STATES}, "
          f"MAX_ITER = {MAX_ITER}")
    detector = _selftest_detector()
    print(f"   detector self-test: {json.dumps(detector)}")

    identity = {"protocol_sha256": sha, **stamp}
    # STREAM.exists() is checked inside resume(); a completed scaffold is reused ONLY
    # under an identical identity (protocol seal + all four code digests), and a
    # partial written by different bytes is DISCARDED rather than mixed in.
    prior = resume(STREAM, identity)

    t0 = time.perf_counter()
    # MS-C-849: identity makes the stream APPEND to a partial from this same run
    # instead of truncating the work resume() just read out of it.
    stream = Stream(STREAM, identity)
    stream.write({"event": "start", **identity})
    results, exceeded = [], []
    for k, spec in enumerate(build_scaffolds()):
        name, X = spec["name"], spec["X"]
        if name in prior:
            results.append(prior[name])
            continue
        geom = check_lemma(g, X)
        sc = Scaffold(g, X, candidate_edges(g, X))
        verify_candidate_set(sc)
        frozen = assert_frozen(sc, FREEZE_TRIALS, FREEZE_STEPS, index=k)
        row = {"name": name, "arm": spec["arm"], "geometry": geom, "frozen": frozen}
        if spec["arm"] == "exhaustive":
            row["exhaustive"] = enumerate_exhaustive(sc, EXHAUSTIVE_MAX_STATES)
            if row["exhaustive"]["exceeded_budget"]:
                exceeded.append([name + " (EXHAUSTIVE_MAX_STATES)",
                                 row["exhaustive"]["states"]])
        else:
            rows, over = sample_worlds(sc, base_index=100000 + 10000 * k)
            row["sampled"] = {"worlds": len(rows), "exceeded_MAX_ITER": over,
                              "rows": rows}
            if over:
                exceeded.append([name + " (MAX_ITER)", over])
        results.append(row)
        stream.write({"event": "arm", "arm": name, "result": row})
        print(f"   {name:<9} n={geom['n']:<3} E={geom['candidate_edges']:<3} "
              f"triangles={geom['triangles']:<4} maxdeg={geom['max_candidate_degree']} "
              f"frozen=0.0   ({time.perf_counter() - t0:.1f}s)", flush=True)

    print()
    summary = report(results)
    stream.write({"event": "summary", **summary})
    stream.close()

    print("\n   CAPS -- what exceeded them")
    if exceeded:
        for what, howmany in exceeded:
            print(f"     {what}: {howmany}")
    else:
        print(f"     nothing exceeded EXHAUSTIVE_MAX_STATES = {EXHAUSTIVE_MAX_STATES} "
              f"or MAX_ITER = {MAX_ITER}: every enumeration above is COMPLETE and "
              "every sampled world reached an exact cycle.")

    result = {**stamp, "protocol_sha256": sha, "genome_hash": g.short_hash(),
              "namespace": NAMESPACE, "h_constant": repr(H),
              "detector_selftest": detector,
              "caps": {"EXHAUSTIVE_MAX_STATES": EXHAUSTIVE_MAX_STATES,
                       "MAX_ITER": MAX_ITER, "exceeded": exceeded},
              "scaffolds": results, "summary": summary,
              "seconds": round(time.perf_counter() - t0, 1)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(OUT, json.dumps(result, indent=1) + "\n")
    print(f"\n   -> {OUT}   ({result['seconds']}s)")
    return 0 if summary["controls_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
