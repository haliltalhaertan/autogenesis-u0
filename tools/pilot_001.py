"""PILOT 001 -- node-rule band x initial connectivity.

Runs the sweep described in protocols/PILOT_001_BAND_CONNECTIVITY.json and
applies that file's PREREGISTERED decision rule.

The decision rule is read from the protocol file and its SHA256 is recorded in
the output. The analysis code here does not contain any threshold of its own --
if you want to change the decision rule you must change the protocol file, which
changes its hash, which is visible in every result written afterwards.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, sys, time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from autogenesis.genome import U0, Genome          # noqa: E402
from autogenesis.rules import enumerate_bands, u0_band   # noqa: E402
from autogenesis.metrics import screen             # noqa: E402
from autogenesis import simulator as sim
from autogenesis import measurement   # noqa: E402  D-025           # noqa: E402

PROTOCOL = ROOT / "protocols" / "PILOT_001_BAND_CONNECTIVITY.json"


def protocol_sha256() -> str:
    return hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()


def build_genome(P, band, k0) -> Genome:
    f = P["fixed"]
    return U0().replace(**{
        "universe_id": f"P001_{band['rule_id']}_k{k0}",
        "space.dimensions": f["dimensions"],
        "initial_conditions.N": f["N"],
        "initial_conditions.cube_side": f["cube_side"],
        "initial_conditions.bond_init_mode": "mean_degree",
        "initial_conditions.target_mean_degree": k0,
        "initial_conditions.initial_degree_cap": f["initial_degree_cap"],
        "topology.valence": f["valence"],
        "topology.allocator": f["allocator"],
        "topology.birth_lo": band["lo"], "topology.birth_hi": band["hi"],
        "topology.survival_lo": band["lo"], "topology.survival_hi": band["hi"],
        "numerics.max_steps": f["horizon"],
    })


def task_fn(task):
    gdict, ns, idx, horizon, rule_id, k0 = task
    r = screen(Genome.from_dict(gdict), ns, idx, horizon)
    r["rule_id"] = rule_id
    r["k0"] = k0
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run in a separate seed namespace, for wiring "
                         "checks only; never used for the decision")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()

    P = json.loads(PROTOCOL.read_text())
    sha = protocol_sha256()
    sidecar = PROTOCOL.with_suffix(".sha256")
    if sidecar.exists():
        expect = sidecar.read_text().split()[0]
        if expect != sha:
            raise SystemExit(f"PROTOCOL HASH MISMATCH\n  frozen: {expect}\n"
                             f"  now:    {sha}\nThe preregistered protocol was "
                             f"edited. Refusing to run.")

    f = P["fixed"]
    bands = enumerate_bands(f["valence"])
    k0s = P["axes"]["k0_values"]
    if a.smoke:
        bands = [b for b in bands if b["rule_id"] in
                 ("R000_EMPTY", "R013", u0_band(f["valence"])["rule_id"])]
        k0s = [k0s[0], k0s[len(k0s) // 2], k0s[-1]]
        seeds, ns = 12, P["seeds"]["smoke_namespace"]
    else:
        seeds, ns = P["seeds"]["n_seeds"], P["seeds"]["namespace"]

    tasks = []
    for b in bands:
        for k0 in k0s:
            g = build_genome(P, b, k0)
            for i in range(seeds):
                tasks.append((g.to_dict(), ns, i, f["horizon"], b["rule_id"], k0))

    print(f"PILOT 001{'  [SMOKE]' if a.smoke else ''}")
    print(f"  protocol sha256 : {sha}")
    print(f"  code_hash       : {sim.code_hash()}")
    print(f"  rules           : {len(bands)}   k0 values: {len(k0s)}   "
          f"seeds: {seeds}")
    print(f"  runs            : {len(tasks):,}   horizon: {f['horizon']}")
    print(f"  seed namespace  : {ns}\n")

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(task_fn, tasks, chunksize=16))
    wall = time.perf_counter() - t0
    print(f"done in {wall:.1f}s  ({len(tasks)/wall:.0f} runs/s wall)\n")

    tag = "smoke" if a.smoke else "prod"
    out = ROOT / "runs" / f"pilot_001_{tag}.csv"
    out.parent.mkdir(exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    analyse(P, rows, sha, wall, tag, len(bands), k0s, seeds)
    print(f"\nwrote {out}")


# ---------------------------------------------------------------- analysis --
def rates(rows):
    """(survival, non-trivial) rate per (rule_id, k0)."""
    agg = defaultdict(lambda: [0, 0, 0])
    for r in rows:
        k = (r["rule_id"], r["k0"])
        agg[k][0] += 1
        if r.get("klass") in ("FROZEN", "PERIODIC", "RICH"):
            agg[k][1] += 1
        if r.get("klass") in ("PERIODIC", "RICH"):
            agg[k][2] += 1
    return {k: (v[1] / v[0], v[2] / v[0], v[0]) for k, v in agg.items()}


def analyse(P, rows, sha, wall, tag, n_rules, k0s, seeds):
    R = rates(rows)
    dr = P["decision_rule"]
    excluded = set(dr["exclude_rules_from_dominance"])
    metric_idx = 0 if dr["metric"] == "survival_rate" else 1

    rule_ids = sorted({rid for rid, _ in R} - excluded)
    # band effect: spread across rules, at fixed k0; median over k0
    band_spreads = []
    for k0 in k0s:
        vals = [R[(rid, k0)][metric_idx] for rid in rule_ids if (rid, k0) in R]
        if vals:
            band_spreads.append(max(vals) - min(vals))
    # connectivity effect: spread across k0, at fixed rule; median over rules
    conn_spreads = []
    for rid in rule_ids:
        vals = [R[(rid, k0)][metric_idx] for k0 in k0s if (rid, k0) in R]
        if vals:
            conn_spreads.append(max(vals) - min(vals))

    B = float(np.median(band_spreads)) if band_spreads else 0.0
    K = float(np.median(conn_spreads)) if conn_spreads else 0.0
    ratio = B / K if K > 0 else float("inf")

    if ratio > dr["band_dominates_if_ratio_above"]:
        verdict = "BAND_DOMINATES"
    elif ratio < dr["connectivity_dominates_if_ratio_below"]:
        verdict = "CONNECTIVITY_DOMINATES"
    else:
        verdict = "NEITHER_DOMINATES"

    from collections import Counter
    cls = Counter(r.get("klass") for r in rows)
    print("class distribution:")
    for k, n in cls.most_common():
        print(f"   {k:<16} {n:>8,}  {100*n/len(rows):5.1f}%")

    print(f"\nPREREGISTERED DECISION  (metric: {dr['metric']})")
    print(f"   band effect         B = {B:.4f}   (median spread across rules)")
    print(f"   connectivity effect K = {K:.4f}   (median spread across k0)")
    print(f"   B/K                   = {ratio:.2f}")
    print(f"   thresholds            > {dr['band_dominates_if_ratio_above']} "
          f"-> BAND, < {dr['connectivity_dominates_if_ratio_below']} -> CONNECTIVITY")
    print(f"   VERDICT               = {verdict}")

    # controls
    print("\ncontrols:")
    for cid, label in (("R000_EMPTY", "floor (no rule can activate)"),
                       ("R013", "null rule [0,1] -- mechanics only")):
        vals = [(k0, R[(cid, k0)]) for k0 in k0s if (cid, k0) in R]
        if vals:
            sv = ", ".join(f"k0={k0}:{v[0]:.0%}" for k0, v in vals)
            print(f"   {cid:<12} {label:<34} survival {sv}")

    # U0's own rule, and how it ranks
    u0 = u0_band(P["fixed"]["valence"])["rule_id"]
    ranked = sorted(rule_ids,
                    key=lambda rid: -np.mean([R[(rid, k0)][metric_idx]
                                              for k0 in k0s if (rid, k0) in R]))
    if u0 in ranked:
        print(f"\nU0's band is {u0}; rank {ranked.index(u0)+1} of {len(ranked)} "
              f"by mean {dr['metric']}")
    print("top 8 rules:")
    for rid in ranked[:8]:
        m = np.mean([R[(rid, k0)][metric_idx] for k0 in k0s if (rid, k0) in R])
        nt = np.mean([R[(rid, k0)][1] for k0 in k0s if (rid, k0) in R])
        mark = "  <- U0" if rid == u0 else ("  <- null control" if rid == "R013" else "")
        print(f"   {rid:<10} {dr['metric']}={m:.3f}  non-trivial={nt:.3f}{mark}")

    # MS-C-706 / D-025: code_hash alone says which ENGINE ran, not which measurement
    # layer scored it nor which tool produced the record.
    summary = {"protocol_sha256": sha, **measurement.stamp(__file__),
               "tag": tag, "n_runs": len(rows), "wall_seconds": round(wall, 1),
               "band_effect_B": B, "connectivity_effect_K": K,
               "ratio_B_over_K": ratio, "verdict": verdict,
               "class_distribution": dict(cls),
               "label": "[EXPLORATORY]" if tag == "smoke" else "[LEAD]",
               "track": "DISCOVERY"}
    (ROOT / "runs" / f"pilot_001_{tag}_summary.json").write_text(
        json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
