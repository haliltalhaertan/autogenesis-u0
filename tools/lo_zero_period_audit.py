"""Is the `lo = 0` revival cycle always period 2? -- three questions, not one.

`MS-C-572` recorded that with `birth_lo = 0` an all-passive frame is not absorbing
but a **period-2 cycle**, `(0 active, 32 edges) <-> (21 active, 0 edges)`. That was
demonstrated on one genome and one seed, and the sentence has been carried since as
though it described the whole `lo = 0` family.

`MS-C-728` (external) reports it does not: across 13 `lo = 0` bands at one `k0`,
period-2 dominates but is not universal -- period-1, period-4, and a run with no
exact period inside the horizon all occur. This file re-derives that over the **whole
`k0` axis** (13 bands x 8 `k0` x 4 seeds = 416 runs, 2.7 min measured) rather than one
slice, and separates three claims the phrase "period 2" has been conflating:

  1. GEOMETRIC REST -- does `x(t+2) == x(t)` to the bit?
  2. DISCRETE PERIOD -- of `State.discrete_signature()`: activity + edge set.
  3. FULL PHYSICAL PERIOD -- of `State.hash()`: positions and velocities included.

These are three different objects and a run can satisfy any subset. A discrete
period-2 cycle whose geometry is still drifting is not a periodic trajectory of the
physics; it is a periodic trajectory of the projection.

METHOD. Direct integration through `simulator.step` -- `metrics.screen` is not called,
so the F-01 early exit (`MS-C-586`/`MS-C-720`) cannot influence anything here. A
period `p` is reported only when it holds EXACTLY across the whole tail window; a run
with no exact `p <= 32` is reported as such, with its distinct-signature count and its
best partial lag, and is NOT rounded to the nearest period.

CONTROLS, both of which can fail:
  * `R013 = [0, 1]` admits every ratio, so every positive-degree node is active at
    every step: it MUST come out period-1. A detector that reports anything else is
    broken, not interesting.
  * U0's own band (`lo = 1/6`) must NOT show the revival cycle at all -- if it did,
    `MS-C-575`'s scoping of the F-01 defect to `lo = 0` bands would be wrong.

Run:  ./.venv/Scripts/python.exe tools/lo_zero_period_audit.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from autogenesis.genome import U0                              # noqa: E402
from autogenesis.init import make_initial_state, InitFailure    # noqa: E402
from autogenesis.rules import enumerate_bands                   # noqa: E402
from autogenesis import measurement, simulator as sim           # noqa: E402
from pilot_001 import build_genome                              # noqa: E402
from _review_protocol import (Stream, check_protocol, claims,   # noqa: E402
                              compare, resume)

PROTOCOL = ROOT / "protocols" / "PILOT_001_BAND_CONNECTIVITY.json"
OUT = ROOT / "runs" / "lo_zero_period_audit.json"
STREAM = ROOT / "runs" / "lo_zero_period_audit.stream.jsonl"
NAMESPACE = "pilot001_v1"          # the production namespace, per MS-C-728 item 6
SEEDS = (0, 1, 2, 7)
HORIZON, EXTENDED, WINDOW, MAX_P = 8192, 16384, 128, 32


def exact_period(seq: list) -> int | None:
    """Smallest `p <= MAX_P` with `seq[i] == seq[i+p]` for EVERY valid `i`."""
    for p in range(1, min(MAX_P, len(seq) - 1) + 1):
        if all(seq[i] == seq[i + p] for i in range(len(seq) - p)):
            return p
    return None


def best_partial(seq: list) -> tuple[int, int, int]:
    """The lag that matches most positions, when no exact period exists."""
    best = (0, 0, 0)
    for p in range(1, min(MAX_P, len(seq) - 1) + 1):
        hits = sum(seq[i] == seq[i + p] for i in range(len(seq) - p))
        if hits > best[1]:
            best = (p, hits, len(seq) - p)
    return best


def integrate(g, seed: int, steps: int) -> dict:
    s = make_initial_state(g, NAMESPACE, seed)
    disc, full, geo = [], [], []
    prev_x = None
    for t in range(1, steps + 1):
        s = sim.step(g, s)
        if t > steps - WINDOW - 2:
            disc.append(s.discrete_signature())
            full.append(s.hash())
            geo.append(s.x.copy())
        prev_x = s.x
    two_step = None
    if len(geo) > 2:
        two_step = float(np.abs(geo[-1] - geo[-3]).max())
    return {"discrete": disc, "full": full, "two_step_dx": two_step,
            "final_active": int(s.active.sum()), "final_edges": len(s.edge_list())}


def classify(g, seed: int, steps: int) -> dict:
    r = integrate(g, seed, steps)
    dp = exact_period(r["discrete"])
    fp = exact_period(r["full"])
    out = {"steps": steps,
           "discrete_period": dp,
           "full_physical_period": fp,
           "distinct_signatures": len(set(r["discrete"])),
           "two_step_dx": r["two_step_dx"],
           "geometric_rest": r["two_step_dx"] == 0.0,
           "final_active": r["final_active"], "final_edges": r["final_edges"]}
    if dp is None:
        p, hits, total = best_partial(r["discrete"])
        out["best_partial_lag"] = {"p": p, "matches": hits, "of": total}
    return out


def verify_pilot_seal() -> str:
    """PILOT_001's own seal. This tool builds its genomes from that protocol, so an
    altered protocol is an altered experiment (D-015)."""
    import hashlib                                              # noqa: PLC0415
    want = PROTOCOL.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
    got = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    if got != want:
        raise SystemExit(f"PILOT_001 PROTOCOL HASH MISMATCH\n  frozen: {want}\n"
                         f"  now:    {got}")
    return got


def main() -> int:
    pilot_sha = verify_pilot_seal()
    sha = check_protocol()
    C = claims()["S1_lo_zero_periods"]
    stamp = measurement.stamp(__file__)
    P = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    bands = [b for b in enumerate_bands(6) if b["selected"] and float(b["lo"]) == 0.0]
    k0s = P["axes"]["k0_values"]
    print("LO=0 PERIOD AUDIT")
    print(f"   PILOT_001 protocol {pilot_sha[:16]} verified")
    print(f"   follow-up protocol {sha[:16]} verified")
    print("   " + json.dumps(stamp))
    print(f"   {len(bands)} lo=0 bands x {len(k0s)} k0 x {len(SEEDS)} seeds "
          f"= {len(bands)*len(k0s)*len(SEEDS)} runs, horizon {HORIZON}")
    print(f"   namespace {NAMESPACE}   window {WINDOW}   max period tested {MAX_P}")
    identity = {"protocol_sha256": sha, "pilot_sha256": pilot_sha, **stamp}
    # STREAM.exists() is checked inside resume(); completed runs are reused ONLY
    # under an identical identity, and a mismatched partial is discarded.
    done = resume(STREAM, identity)
    prior = {tuple(k.split("|")): v for k, v in done.get("runs", {}).items()}
    print()

    t0 = time.perf_counter()
    # MS-C-849: identity makes the stream APPEND to a partial from this same run
    # instead of truncating the work resume() just read out of it.
    stream = Stream(STREAM, identity)
    stream.write({"event": "start", **identity})
    rows, extended = [], []
    for b in bands:
        for k0 in k0s:
            g = build_genome(P, b, k0)
            for seed in SEEDS:
                key = (b["rule_id"], str(k0), str(seed))
                if key in prior:
                    rows.append(prior[key])
                    continue
                try:
                    c = classify(g, seed, HORIZON)
                except InitFailure as e:
                    c = {"rule_id": b["rule_id"], "k0": k0, "seed": seed,
                         "init_fail": str(e)[:70]}
                    rows.append(c)
                    stream.write({"event": "run", "key": "|".join(key), "row": c})
                    continue
                c.update(rule_id=b["rule_id"], k0=k0, seed=seed)
                rows.append(c)
                stream.write({"event": "run", "key": "|".join(key), "row": c})
        print(f"   {b['rule_id']} done ({time.perf_counter()-t0:.0f}s)", flush=True)

    # Anything without an exact discrete period at HORIZON is re-run to EXTENDED.
    # MS-C-728 item 6: extending only the failures does not show the successes stay
    # successful, so the FULL grid ran to HORIZON first and only the failures go on.
    for r in [x for x in rows if x.get("discrete_period") is None and "init_fail" not in x]:
        b = [x for x in bands if x["rule_id"] == r["rule_id"]][0]
        c = classify(build_genome(P, b, r["k0"]), r["seed"], EXTENDED)
        c.update(rule_id=r["rule_id"], k0=r["k0"], seed=r["seed"])
        extended.append(c)

    live = [r for r in rows if "init_fail" not in r]
    by_dp: dict[str, int] = {}
    for r in live:
        key = str(r["discrete_period"]) if r["discrete_period"] else "none<=32"
        by_dp[key] = by_dp.get(key, 0) + 1
    by_fp: dict[str, int] = {}
    for r in live:
        key = str(r["full_physical_period"]) if r["full_physical_period"] else "none<=32"
        by_fp[key] = by_fp.get(key, 0) + 1

    # --- controls ---------------------------------------------------------------
    r013 = [r for r in live if r["rule_id"] == "R013"]
    r013_all_p1 = bool(r013) and all(r["discrete_period"] == 1 for r in r013)
    u0_band = [b for b in enumerate_bands(6)
               if b["selected"] and abs(float(b["lo"]) - 1 / 6) < 1e-12
               and abs(float(b["hi"]) - 0.5) < 1e-12][0]
    u0_rows = [classify(build_genome(P, u0_band, k0), s, 2048)
               for k0 in (1.0, 2.0, 3.0) for s in SEEDS]
    u0_revives = [r for r in u0_rows if r["final_active"] > 0 and r["final_edges"] > 0
                  and r["discrete_period"] not in (1, None)]

    print(f"\n   discrete period      {json.dumps(by_dp, sort_keys=True)}")
    print(f"   full physical period {json.dumps(by_fp, sort_keys=True)}")
    print(f"   geometric rest (x(t+2)==x(t) exactly): "
          f"{sum(1 for r in live if r['geometric_rest'])} of {len(live)}")
    print(f"\n   CONTROL R013=[0,1] every run period-1: {r013_all_p1} "
          f"({len(r013)} runs)   {'OK' if r013_all_p1 else '*** DETECTOR IS WRONG ***'}")
    print(f"   CONTROL U0 band shows no lo=0 revival cycle: {not u0_revives} "
          f"({len(u0_rows)} runs)")

    non_p2 = [r for r in live if r["discrete_period"] != 2]
    print(f"\n   NOT discrete period-2: {len(non_p2)} of {len(live)}")
    for r in sorted(non_p2, key=lambda r: (r["rule_id"], r["k0"], r["seed"]))[:20]:
        print(f"     {r['rule_id']} k0={r['k0']:<4} seed={r['seed']}  "
              f"discrete={r['discrete_period']}  full={r['full_physical_period']}  "
              f"distinct={r['distinct_signatures']}")
    if len(non_p2) > 20:
        print(f"     ... and {len(non_p2)-20} more (all in the record)")

    # --- what the caps truncated, named rather than left implicit ---------------
    # Silent truncation reads as full coverage. Two caps bound this run and both are
    # reported: MAX_P bounds the period SEARCH, HORIZON/EXTENDED bound the time.
    over_p = by_dp.get("none<=32", 0)
    still_none = sum(1 for r in extended if r["discrete_period"] is None)
    gained = len(extended) - still_none
    print(f"\n   CAPS -- what exceeded them")
    print(f"     MAX_P = {MAX_P}: {over_p} of {len(live)} runs exceeded the period "
          f"search and are recorded as 'none<={MAX_P}', NOT as aperiodic")
    print(f"     HORIZON = {HORIZON}: all {len(extended)} of those were re-run to "
          f"{EXTENDED}, where {gained} GAINED an exact period and {still_none} "
          f"still exceeded it")
    print(f"     -> 'no exact period' is a statement about these caps. The {gained} "
          "that resolved are the proof that it is.")

    # --- the protocol's decision rule against the sealed claims -----------------
    print("\nVERDICTS against the sealed pre-registration "
          "(protocols/REVIEW_FOLLOWUP_20260828.json)")
    R = C["reviewer_numbers"]
    print(f"     the reviewer's scope was {C['reviewer_scope']['runs']} runs at ONE k0; "
          f"this is {len(live)} runs over all {len(k0s)} k0 values, so counts are")
    print("     EXTENDED by construction and only the QUALITATIVE claims are compared")
    verdicts = [
        compare("S1 period-2 occurs and dominates", True,
                by_dp.get("2", 0) > len(live) // 2),
        compare("S1 period-2 is NOT universal", True, len(non_p2) > 0),
        compare("S1 period-1 occurs", True, by_dp.get("1", 0) > 0),
        # Compare like with like: the claim names ONE band, so the measurement is
        # "the set of period-1 bands is exactly that one band". Comparing a string
        # against a one-element list produced a CONTESTED verdict on a claim that
        # actually agreed -- a false contested is as bad as a false confirmed, and
        # under this protocol it would have blocked citation of a correct claim.
        compare("S1 period-1 bands are exactly {R013}",
                {R["discrete_period_1_band"]},
                {r["rule_id"] for r in live if r["discrete_period"] == 1}),
        compare("S1 period-4 occurs", True, by_dp.get("4", 0) > 0),
        compare("S1 runs with no exact period <= 32 occur", True,
                by_dp.get("none<=32", 0) > 0),
        compare("S1 full-physical-periodic runs are rare",
                R["full_physical_periodic_runs"],
                sum(v for k, v in by_fp.items() if k != "none<=32"),
                wider_scope=True),
    ]
    contested = [v for v in verdicts if v["verdict"] == "CONTESTED"]
    print(f"\n   {len(verdicts)} claims compared: "
          f"{sum(1 for v in verdicts if v['verdict'] == 'CONFIRMED')} confirmed, "
          f"{sum(1 for v in verdicts if v['verdict'] == 'EXTENDED')} extended, "
          f"{len(contested)} CONTESTED")
    print("   NEW beyond the sealed claims: discrete periods 6 and 12 also occur "
          f"({by_dp.get('6', 0)} and {by_dp.get('12', 0)} runs)")
    stream.write({"event": "verdicts", "verdicts": verdicts})
    stream.close()

    result = {**stamp,
              "protocol_sha256": sha, "pilot_protocol_sha256": pilot_sha,
              "verdicts": verdicts, "contested": len(contested),
              "namespace": NAMESPACE, "seeds": list(SEEDS),
              "horizon": HORIZON, "extended_horizon": EXTENDED,
              "tail_window": WINDOW, "max_period_tested": MAX_P,
              "bands": [b["rule_id"] for b in bands], "k0_values": k0s,
              "runs": len(live),
              "discrete_period_histogram": by_dp,
              "full_physical_period_histogram": by_fp,
              "geometric_rest_count": sum(1 for r in live if r["geometric_rest"]),
              "control_R013_all_period_1": r013_all_p1,
              "control_U0_band_no_revival_cycle": not u0_revives,
              "rows": rows, "extended_rows": extended,
              "seconds": round(time.perf_counter() - t0, 1)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    # MS-C-875: archive the previous artefact before replacing it.
    from _artefact import write_result                         # noqa: E402
    write_result(OUT, json.dumps(result, indent=1) + "\n")
    print(f"\n   -> {OUT}   ({result['seconds']}s)")
    return 0 if (r013_all_p1 and not contested) else 1


if __name__ == "__main__":
    sys.exit(main())
