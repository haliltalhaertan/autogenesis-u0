"""Explicit rest-scaffold coordinates for the minimal two-bit register, verified on the
production engine.

Labelling follows L-509: E = {01,04,05,06,12,13,14,15,16,23}; node 1 is the degree-6 hub,
node 0 (degree 4) is adjacent to 4, 5, 6; nodes 2-3 form a pendant edge pair on the hub.

Geometry (exact reals): hub at the origin, every other node on the unit sphere.
  node 0 = (0, 0, 1)
  nodes 4,5,6 at polar angle 60 deg, azimuths 90, 210, 330 deg
  nodes 2,3   = (+-1/2, 0, -sqrt(3)/2)
Candidate pairs sit at distance 1 exactly; the closest non-candidate pair is at sqrt(2).
Floating point: the free irrational coordinates are nudged by a few ulps until the
kernel's own pair arithmetic returns exactly 1.0 for every candidate pair.

usage: python paper_repro/register_geometry.py [out_json]
"""
import itertools, json, math, sys, time
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[1]   # repository root
out = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "runs" / "register_geometry.json"
for p in (root, root / "tools", root / "measurements"):
    sys.path.insert(0, str(p))

from autogenesis.genome import U0                                   # noqa: E402
from autogenesis import simulator as sim                            # noqa: E402
from rest_scaffold_orbits import (check_lemma, kernel_distances, Scaffold,  # noqa: E402
                                  verify_candidate_set, candidate_edges)
import two_bit_register_minimality as m                             # noqa: E402

g = U0()
EDGES = sorted([(0, 1), (0, 4), (0, 5), (0, 6), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (2, 3)])


def ulps(v, k):
    for _ in range(abs(k)):
        v = math.nextafter(v, math.inf if k > 0 else -math.inf)
    return v


def build(h, x, y):
    X = np.zeros((7, 3))
    X[1] = (0.0, 0.0, 0.0)                 # hub
    X[0] = (0.0, 0.0, 1.0)
    X[4] = (0.0, h, 0.5)
    X[5] = (-x, -y, 0.5)
    X[6] = (x, -y, 0.5)
    X[2] = (0.5, 0.0, -h)
    X[3] = (-0.5, 0.0, -h)
    return X


def exact_ok(X):
    r = kernel_distances(X)
    return all(r[i, j] == 1.0 for i, j in EDGES)


h0 = math.sqrt(3.0) / 2.0
x0, y0 = 0.75, math.sqrt(3.0) / 4.0
found = None
for dh, dx, dy in sorted(itertools.product(range(-6, 7), repeat=3), key=lambda t: sum(map(abs, t))):
    X = build(ulps(h0, dh), ulps(x0, dx), ulps(y0, dy))
    if exact_ok(X):
        found = (dh, dx, dy, X)
        break
if found is None:
    raise SystemExit("no ulp nudge within +-6 makes every candidate pair exactly 1.0")
dh, dx, dy, X = found

lem = check_lemma(g, X)
cand = candidate_edges(g, X)
if cand != EDGES:
    raise SystemExit(f"candidate graph differs from the register graph: {cand}")
sc = Scaffold(g, X, EDGES)
verify_candidate_set(sc)

# full transition table through the PRODUCTION engine
t0 = time.perf_counter()
total = 1 << (sc.n + sc.E)
F = {}
moved = 0
for code in range(total):
    st = sc.state(code)
    nx = sim.step(g, st)
    if not (np.array_equal(nx.x, st.x) and np.array_equal(nx.u, st.u)):
        moved += 1
    F[code] = sc.encode(nx)
secs = time.perf_counter() - t0


def flip(v, code):
    return code ^ (1 << v)


# attractors
seen, cycles = {}, []
for s0 in range(total):
    s, path = s0, []
    while s not in seen:
        seen[s] = None
        path.append(s)
        s = F[s]
    if s in path:
        cycles.append(path[path.index(s):])
p2 = [c for c in cycles if len(c) == 2]
obj = [c for c in p2 if len({sc.split(s)[0] for s in c}) == 2]    # bond set moves
obj_states = {s for c in obj for s in c}

# every witness of the repository's own closed-form criterion
flipd = {(v, s): flip(v, s) for v in range(sc.n) for s in obj_states}
hits = []
for s00 in obj_states:
    for a in range(sc.n):
        s10 = flipd[(a, s00)]
        if s10 not in obj_states:
            continue
        for b in range(sc.n):
            s01 = F.get(flip(b, F[s00])); s11 = F.get(flip(b, F[s10]))
            if s01 not in obj_states or s11 not in obj_states:
                continue
            S = {(0, 0, 0): s00, (0, 1, 0): s10, (0, 0, 1): s01, (0, 1, 1): s11}
            S.update({(1, xx, yy): F[S[(0, xx, yy)]] for xx in (0, 1) for yy in (0, 1)})
            fl = {(v, s): flip(v, s) for v in range(sc.n) for s in S.values()}
            if m.criterion((S, F, fl, a, b)):
                hits.append({"a": a, "b": b,
                             "S": {f"{p}{xx}{yy}": int(v) for (p, xx, yy), v in S.items()}})


def decode(code):
    bb, ab = sc.split(code)
    return {"active": [v for v in range(sc.n) if ab >> v & 1],
            "bonds": [list(EDGES[k]) for k in range(sc.E) if bb >> k & 1]}


w = next((h for h in hits if (h["a"], h["b"]) == (6, 4)), hits[0] if hits else None)
res = {
    "edges": EDGES, "ulp_offsets": {"h": dh, "x": dx, "y": dy},
    "coordinates": [[repr(float(c)) for c in row] for row in X],
    "check_lemma": lem, "states": total, "states_moved": moved, "engine_seconds": round(secs, 1),
    "attractors": len(cycles), "period2": len(p2), "period2_object": len(obj),
    "witnesses": len(hits), "control_pairs": sorted({(h["a"], h["b"]) for h in hits}),
    "example_witness": w,
    "example_states": ({k: decode(v) for k, v in w["S"].items()} if w else None),
}
out.write_text(json.dumps(res, indent=1, default=list))
print(json.dumps({k: res[k] for k in ("ulp_offsets", "states", "states_moved", "attractors",
                                      "period2", "period2_object", "witnesses", "control_pairs")},
                 default=list))
print("check_lemma:", lem)
