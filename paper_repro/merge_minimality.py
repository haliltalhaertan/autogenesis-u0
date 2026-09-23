"""Merge the shards written by shard_minimality.py and check the census.

usage: python paper_repro/merge_minimality.py <out_dir>
"""
import collections, glob, json, sys
rows = [r for p in sorted(glob.glob(sys.argv[1] + "/shard_*.json")) for r in json.load(open(p))["rows"]]
if sorted(r["class_index"] for r in rows) != list(range(491)):
    raise SystemExit("shards incomplete")
strata = collections.defaultdict(lambda: [0, 0])
for r in rows:
    lab = "E<=9" if r["n"] == 7 and r["E"] <= 9 else ("E=10" if r["n"] == 7 else "all E")
    strata[(r["n"], lab)][0] += 1
    strata[(r["n"], lab)][1] += r["witnesses"] > 0
for k in sorted(strata):
    print(f"n={k[0]} {k[1]:<6} classes {strata[k][0]:>4}  positive {strata[k][1]}")
pos = [r for r in rows if r["witnesses"]]
print(f"{len(rows)} classes, {sum(r['states'] for r in rows):,} states")
for r in pos:
    print(f"POSITIVE n={r['n']} E={r['E']} graph6={r['g6']} attractors={r['attractors']} "
          f"period-2 OBJECT={r['p2_object']} witnesses={r['witnesses']}")
ok = len(pos) == 1 and pos[0]["n"] == 7 and pos[0]["E"] == 10
print("census reproduces the paper:", ok)
sys.exit(0 if ok else 1)
