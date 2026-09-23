"""Sharded re-run of measurements/two_bit_register_minimality.py.

Calls the repository's own family() and scan() unchanged; only the loop over the
491 classes is split across K independent processes (no multiprocessing module,
which the repository notes does not work in this sandbox).

usage: python paper_repro/shard_minimality.py <k> <K> <out_dir>   (run k = 0..K-1, then paper_repro/merge_minimality.py)
"""
import json, sys, time
from pathlib import Path

root = Path(__file__).resolve().parents[1]   # repository root
k, K, out = int(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
for p in (root, root / "tools", root / "measurements"):
    sys.path.insert(0, str(p))

import two_bit_register_minimality as m  # noqa: E402

fam = m.family()
# greedy longest-processing-time assignment by state count, deterministic
order = sorted(range(len(fam)), key=lambda i: -(1 << (fam[i][0] + len(fam[i][1]))))
load, assign = [0] * K, {}
for i in order:
    j = min(range(K), key=lambda s: (load[s], s))
    assign[i] = j
    load[j] += 1 << (fam[i][0] + len(fam[i][1]))
mine = [i for i in range(len(fam)) if assign[i] == k]

t0 = time.perf_counter()
rows = []
for i in mine:
    r = m.scan(fam[i])
    r["class_index"] = i
    r["edges"] = [list(e) for e in r["edges"]]
    rows.append(r)
res = {"shard": k, "K": K, "n_family": len(fam), "classes": len(mine),
       "states": sum(r["states"] for r in rows), "seconds": time.perf_counter() - t0, "rows": rows}
out.mkdir(parents=True, exist_ok=True)
(out / f"shard_{k:02d}.json").write_text(json.dumps(res))
print(f"shard {k}/{K}: {len(mine)} classes, {res['states']:,} states, {res['seconds']:.0f} s")
