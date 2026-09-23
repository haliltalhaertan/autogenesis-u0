"""Single-flip robustness of the minimal two-bit register, on the production engine.

Builds the register scaffold of register_geometry.py, steps all 2^17 states through the
production engine, finds every witness with the repository's own closed-form criterion,
and applies every single-node activity flip and every single-bond flip to all 8 states of
every witness. Reports where each perturbation lands.

usage: python paper_repro/register_robustness.py
"""
import json, sys
from collections import Counter
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[1]
for p in (root, root / "tools", root / "measurements", root / "paper_repro"):
    sys.path.insert(0, str(p))

from autogenesis.genome import U0                                  # noqa: E402
from rest_scaffold_orbits import Scaffold, check_lemma             # noqa: E402
import two_bit_register_minimality as m                            # noqa: E402

H = 0.8660254037844387            # one ulp above fl(sqrt(3)/2); see register_geometry.py
X = np.array([(0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (0.5, 0.0, -H), (-0.5, 0.0, -H),
              (0.0, H, 0.5), (-0.75, -0.4330127018922193, 0.5), (0.75, -0.4330127018922193, 0.5)])
EDGES = [(0, 1), (0, 4), (0, 5), (0, 6), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (2, 3)]

g = U0()
check_lemma(g, X)
sc = Scaffold(g, X, EDGES)
total = 1 << (sc.n + sc.E)
F = np.array([sc.step_code(c) for c in range(total)], dtype=np.int64)

att = np.full(total, -1)
cycles = []
for s0 in range(total):
    if att[s0] >= 0:
        continue
    path, pos, s = [], {}, s0
    while att[s] < 0 and s not in pos:
        pos[s] = len(path); path.append(s); s = int(F[s])
    if att[s] < 0:
        cid = len(cycles); cycles.append(path[pos[s]:])
        for c in cycles[-1]:
            att[c] = cid
    for c in path:
        if att[c] < 0:
            att[c] = att[s]

obj = {i for i, c in enumerate(cycles) if len(c) == 2 and len({sc.split(s)[0] for s in c}) == 2}
obj_states = {s for i in obj for s in cycles[i]}
witnesses = []
for s00 in obj_states:
    for a in range(sc.n):
        s10 = s00 ^ (1 << a)
        if s10 not in obj_states:
            continue
        for b in range(sc.n):
            s01 = int(F[int(F[s00]) ^ (1 << b)]); s11 = int(F[int(F[s10]) ^ (1 << b)])
            if s01 not in obj_states or s11 not in obj_states:
                continue
            S = {(0, 0, 0): s00, (0, 1, 0): s10, (0, 0, 1): s01, (0, 1, 1): s11}
            S.update({(1, x, y): int(F[S[(0, x, y)]]) for x in (0, 1) for y in (0, 1)})
            Fd = {s: int(F[s]) for s in S.values()}
            fl = {(v, s): s ^ (1 << v) for v in range(sc.n) for s in S.values()}
            if m.criterion((S, Fd, fl, a, b)):
                witnesses.append((a, b, S))
if len(witnesses) != 48:
    raise SystemExit(f"expected 48 witnesses, found {len(witnesses)}")


def kind(home, target, reg):
    if target == home:
        return "same attractor"
    if target in reg:
        return "other register state"
    return "extinct" if cycles[target] == [0] else "other attractor"


node, bond, designed_only = Counter(), Counter(), True
for a, b, S in witnesses:
    reg = {att[s] for s in S.values()}
    for (p, x, y), s in S.items():
        into_reg = []
        for v in range(sc.n):
            k = kind(att[s], att[s ^ (1 << v)], reg); node[k] += 1
            if k == "other register state":
                into_reg.append(v)
        designed_only &= into_reg == [a if p == 0 else b]
        for e in range(sc.E):
            bond[kind(att[s], att[s ^ (1 << (sc.n + e))], reg)] += 1

res = {"attractors": len(cycles), "witnesses": len(witnesses), "node_flips": dict(node),
       "bond_flips": dict(bond), "only_designed_write_stays_in_register": designed_only}
(root / "runs").mkdir(exist_ok=True)
(root / "runs" / "register_robustness.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
ok = designed_only and node.get("same attractor", 0) == 0
sys.exit(0 if ok else 1)
