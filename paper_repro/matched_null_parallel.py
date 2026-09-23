"""Exact parallel replay of measurements/built_graph_beyond_structure.py, with per-seed output.

The original loop draws everything random from ONE random.Random(20260830) stream, in seed
order, so seeds cannot simply be split. But only two things consume it -- the shared initial
conditions (getrandbits) and rewire() -- while the two expensive steps, settle() and score(),
are deterministic. So:

  settle <k> <K>   parallel: settle every seed in shard k (no randomness)
  plan             serial:   replay the RNG stream in the original order -> every scoring job
  score  <k> <K>   parallel: score the jobs in shard k (no randomness)
  merge            serial:   assemble per-seed rates; must reproduce 98W / 97L / 4T

usage: python paper_repro/matched_null_parallel.py <work_dir> full <k> <K>   (k = 0..K-1), then  ... <work_dir> merge
"""
import json, random, sys, time
from pathlib import Path

root = Path(__file__).resolve().parents[1]   # repository root
work = Path(sys.argv[1]); work.mkdir(parents=True, exist_ok=True)
mode = sys.argv[2]
for p in (root, root / "tools", root / "measurements"):
    sys.path.insert(0, str(p))

import built_graph_beyond_structure as B          # noqa: E402
from autogenesis.genome import U0                 # noqa: E402

g = U0().replace(**{"physics.repulsion_cutoff": B.Q})
g.validate()


def adj_to_json(adj):
    # ITERATION ORDER, not sorted: score() maps the shared random bond bits onto edges in
    # the order it iterates the adjacency sets, so sorting would silently change which
    # initial condition each trial starts from.
    return [[int(v) for v in x] for x in adj]


def adj_key(adj):
    return [sorted(int(v) for v in x) for x in adj]


def adj_from_json(a):
    return [set(x) for x in a]


if mode == "settle":
    k, K = int(sys.argv[3]), int(sys.argv[4])
    out = {}
    for seed in range(k, B.SEEDS, K):
        got = B.settle(g, seed)
        out[seed] = None if got is None else {"adj": adj_to_json(got[0]), "n": got[1]}
    (work / f"settle_{k:02d}.json").write_text(json.dumps(out))

elif mode == "plan":
    settled = {}
    for p in sorted(work.glob("settle_*.json")):
        settled.update({int(s): v for s, v in json.loads(p.read_text()).items()})
    if sorted(settled) != list(range(B.SEEDS)):
        raise SystemExit("settle shards incomplete")
    rng = random.Random(20260830)
    jobs, seeds = [], []
    for seed in range(B.SEEDS):                   # identical order of RNG consumption
        # settle() is re-run IN THIS PROCESS rather than read back from JSON: rewire()
        # picks edges by iterating Python sets, and a set rebuilt from a list can iterate
        # in a different order from the one settle() built, which silently changes which
        # swaps the same random numbers select. The JSON copy is kept only as a check.
        live = B.settle(g, seed)
        if (live is None) != (settled[seed] is None) or (
                live is not None and adj_key(live[0]) != adj_key(settled[seed]["adj"])):
            raise RuntimeError(f"seed {seed}: in-process settle differs from the shard")
        if live is None:
            seeds.append({"seed": seed, "status": "not_at_rest"}); continue
        adj, n = live
        E = sum(len(x) for x in adj) // 2
        inits = [(rng.getrandbits(E), rng.getrandbits(n)) for _ in range(B.SAMPLES)]
        jobs.append({"seed": seed, "role": "built", "n": n, "adj": adj_to_json(adj), "inits": inits})
        k_null = 0
        for _ in range(B.NULLS):
            nadj = B.rewire(adj, n, rng)
            if nadj is None:
                continue
            if B.degseq(nadj, n) != B.degseq(adj, n) or B.triangles(nadj, n) != B.triangles(adj, n):
                raise RuntimeError("rewiring broke an invariant")
            jobs.append({"seed": seed, "role": f"null{k_null}", "n": n, "adj": adj_to_json(nadj),
                         "inits": inits})
            k_null += 1
        seeds.append({"seed": seed, "status": "ok" if k_null else "rigid", "n": n, "E": E})
    (work / "plan.json").write_text(json.dumps({"jobs": jobs, "seeds": seeds}))
    print(f"plan: {len(jobs)} scoring jobs over {sum(s['status']=='ok' for s in seeds)} seeds")

elif mode == "score":
    k, K = int(sys.argv[3]), int(sys.argv[4])
    jobs = json.loads((work / "plan.json").read_text())["jobs"]
    res = []
    for i in range(k, len(jobs), K):
        j = jobs[i]
        rate, open_ = B.score(j["n"], adj_from_json(j["adj"]), g, [tuple(x) for x in j["inits"]])
        res.append({"i": i, "seed": j["seed"], "role": j["role"], "rate": rate, "open": open_})
    (work / f"score_{k:02d}.json").write_text(json.dumps(res))

elif mode == "full":
    # Each shard replays settle() + the RNG stream + rewire() IN-PROCESS, exactly as the
    # original loop does, and scores only its own share of the jobs. No graph ever goes
    # through a file: CPython set iteration order depends on each set's construction
    # history (colliding ints), and score() maps random bond bits onto edges in that
    # order, so a serialized-and-rebuilt adjacency is a different experiment.
    k, K = int(sys.argv[3]), int(sys.argv[4])
    rng = random.Random(20260830)
    job_i, res, seeds = 0, [], []
    for seed in range(B.SEEDS):
        got = B.settle(g, seed)
        if got is None:
            seeds.append({"seed": seed, "status": "not_at_rest"}); continue
        adj, n = got
        E = sum(len(x) for x in adj) // 2
        inits = [(rng.getrandbits(E), rng.getrandbits(n)) for _ in range(B.SAMPLES)]
        if job_i % K == k:
            rate, o = B.score(n, adj, g, inits)
            res.append({"i": job_i, "seed": seed, "role": "built", "rate": rate, "open": o})
        job_i += 1
        k_null = 0
        for _ in range(B.NULLS):
            nadj = B.rewire(adj, n, rng)
            if nadj is None:
                continue
            if job_i % K == k:
                rate, o = B.score(n, nadj, g, inits)
                res.append({"i": job_i, "seed": seed, "role": f"null{k_null}", "rate": rate, "open": o})
            job_i += 1
            k_null += 1
        seeds.append({"seed": seed, "status": "ok" if k_null else "rigid", "n": n, "E": E})
    (work / f"score_{k:02d}.json").write_text(json.dumps(res))
    if k == 0:
        (work / "plan.json").write_text(json.dumps({"jobs": [None] * job_i, "seeds": seeds}))

elif mode == "merge":
    plan = json.loads((work / "plan.json").read_text())
    got = [r for p in sorted(work.glob("score_*.json")) for r in json.loads(p.read_text())]
    if sorted(r["i"] for r in got) != list(range(len(plan["jobs"]))):
        raise SystemExit("score shards incomplete")
    by = {}
    for r in got:
        by.setdefault(r["seed"], {})[r["role"]] = r
    rows, W, L, T, open_total = [], 0, 0, 0, sum(r["open"] for r in got)
    for s in plan["seeds"]:
        if s["status"] != "ok":
            rows.append(s); continue
        d = by[s["seed"]]
        built = d["built"]["rate"]
        nulls = [d[k]["rate"] for k in sorted(d) if k.startswith("null")]
        nm = sum(nulls) / len(nulls)
        W += built > nm; L += built < nm; T += built == nm
        rows.append({**s, "built": built, "null_mean": nm, "nulls": nulls})
    ok = [r for r in rows if r["status"] == "ok"]
    res = {"Q": B.Q, "wins": W, "losses": L, "ties": T, "open_total": open_total,
           "mean_built": sum(r["built"] for r in ok) / len(ok),
           "mean_null": sum(r["null_mean"] for r in ok) / len(ok),
           "reproduces_record": (W, L, T) == (98, 97, 4), "rows": rows}
    (work / "matched_null_perseed.json").write_text(json.dumps(res))
    print(f"merge: {W}W {L}L {T}T  built {res['mean_built']:.4f}  null {res['mean_null']:.4f}  "
          f"open {open_total}  reproduces_record={res['reproduces_record']}")
