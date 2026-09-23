"""GOLDEN REGRESSION -- the drift alarm.

A fixed set of (genome, seed) pairs and their exact final state hashes are
recorded in tests/golden.json. Any change to the engine that alters ANY of
these hashes trips this test.

That is the whole point. When it trips you have exactly two honest options:

  1. You did not intend to change the physics -> you introduced a bug. Fix it.
  2. You DID intend to change the physics -> bump `protocol_version`, write the
     reason into docs/DECISIONS.md, regenerate with `--accept`, and understand
     that every result produced before this moment is no longer comparable to
     results produced after it.

There is no third option. Do not regenerate golden.json to make a red test
green. (docs/DECISIONS.md D-005)
"""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from autogenesis.genome import U0                 # noqa: E402
from autogenesis import simulator as sim          # noqa: E402

GOLDEN = Path(__file__).resolve().parent / "golden.json"

CASES = [
    ("U0_default_s0",      {},                                                  0, 512),
    ("U0_default_s7",      {},                                                  7, 512),
    ("U0_familyA",         {"initial_conditions.cube_side": 4.5,
                            "initial_conditions.bond_probability": 0.0,
                            "initial_conditions.initial_degree_cap": 0},        3, 512),
    ("U0_familyC",         {"initial_conditions.cube_side": 2.8,
                            "initial_conditions.bond_probability": 0.12},       3, 512),
    ("dense_allocA",       {"initial_conditions.N": 24,
                            "initial_conditions.cube_side": 2.4,
                            "initial_conditions.bond_probability": 0.35,
                            "topology.allocator": "A_batch_all"},               1, 256),
    ("dense_allocC",       {"initial_conditions.N": 24,
                            "initial_conditions.cube_side": 2.4,
                            "initial_conditions.bond_probability": 0.35,
                            "topology.allocator": "C_overload_prune"},          1, 256),
    ("valence12_dense",    {"initial_conditions.N": 24,
                            "initial_conditions.cube_side": 2.4,
                            "initial_conditions.bond_probability": 0.35,
                            "topology.valence": 12},                            1, 256),
    ("two_dimensional",    {"space.dimensions": 2,
                            "initial_conditions.N": 20,
                            "initial_conditions.cube_side": 6.0,
                            "initial_conditions.bond_probability": 0.2},        4, 256),
    ("gamma_low",          {"physics.gamma": 0.3},                              2, 512),
    ("stiff_spring",       {"physics.spring_k": 8.0},                           2, 512),
]

NAMESPACE = "golden_v1"


def compute() -> dict:
    out = {"code_hash": sim.code_hash(), "cases": {}}
    for name, patch, idx, steps in CASES:
        g = U0().replace(**patch) if patch else U0()
        r = sim.run(g, NAMESPACE, idx, steps=steps)
        out["cases"][name] = {"genome_hash": g.short_hash(),
                              "seed_index": idx, "steps": steps,
                              "initial_state_hash": r["initial_state_hash"],
                              "final_state_hash": r["final_state_hash"]}
    return out


if __name__ == "__main__":
    accept = "--accept" in sys.argv
    create = "--create" in sys.argv
    strict_schema = "--strict-schema" in sys.argv
    bless_schema = "--bless-schema" in sys.argv
    now = compute()
    if not GOLDEN.exists():
        # PATCH (OB-030): a missing oracle used to be silently self-written and the
        # run exited 0 -- the drift alarm rearmed itself against whatever the code
        # happened to do at that moment, which is the one thing an oracle must never
        # do. Creating it is now an explicit, opt-in act.
        if not create:
            print(f"*** MISSING ORACLE ***\n  {GOLDEN} does not exist.\n"
                  "  This test compares against a recorded oracle; without one it\n"
                  "  proves nothing. Restore the file from the record, or, if you\n"
                  "  really are creating the oracle for the first time, rerun with\n"
                  "  --create and record why in docs/DECISIONS.md.")
            sys.exit(1)
        GOLDEN.write_text(json.dumps(now, indent=2))
        print(f"golden.json CREATED with {len(now['cases'])} cases "
              f"(code_hash {now['code_hash']})")
        sys.exit(0)

    old = json.loads(GOLDEN.read_text())
    diffs = []
    # PATCH (OB-031/OB-032): the oracle records code_hash, genome_hash, seed_index,
    # steps and initial_state_hash, and this test compared NONE of them -- only
    # final_state_hash. They are now always reported. Whether a mismatch should be
    # FATAL is a decision (the set is non-empty today), so it is gated behind
    # --strict-schema rather than silently changed here.
    schema_diffs = []
    if old.get("code_hash") != now["code_hash"]:
        schema_diffs.append(f"code_hash: {old.get('code_hash')} -> {now['code_hash']}")
    for name in sorted(set(old["cases"]) & set(now["cases"])):
        for fld in ("genome_hash", "seed_index", "steps", "initial_state_hash"):
            a, b = old["cases"][name].get(fld), now["cases"][name].get(fld)
            if a != b:
                schema_diffs.append(f"{name}.{fld}: {a} -> {b}")
    for name in sorted(set(old["cases"]) | set(now["cases"])):
        a, b = old["cases"].get(name), now["cases"].get(name)
        if a is None:
            diffs.append(f"NEW case {name}")
        elif b is None:
            diffs.append(f"REMOVED case {name}")
        elif a["final_state_hash"] != b["final_state_hash"]:
            diffs.append(f"CHANGED {name}: {a['final_state_hash']} -> "
                         f"{b['final_state_hash']}")
        else:
            print(f"  PASS  {name:<20} {b['final_state_hash']}")

    if schema_diffs:
        print(f"\n  --- {len(schema_diffs)} NON-TRAJECTORY field(s) differ from the "
              "oracle (reported, see OB-031/OB-032) ---")
        for d in schema_diffs:
            print(f"    {d}")

    if not diffs:
        print(f"\nGOLDEN: {len(now['cases'])}/{len(now['cases'])} unchanged "
              f"(code_hash {now['code_hash']})")
        # MS-C-726: `--strict-schema` was written (OB-031/OB-032) and then never
        # made a gate, because it could not be green: `code_hash` moves on ANY edit
        # to the seven trajectory files, including one that changes no trajectory,
        # and there was no way to record "the engine changed, the physics did not"
        # short of hand-editing the oracle. So OB-130 asked for strict golden in CI
        # and CI ran the LOOSE form. `--bless-schema` is that missing act: it
        # updates the non-trajectory fields ONLY, and only when all
        # `final_state_hash` values are identical -- which is the condition under
        # which the update is a statement of fact rather than a regeneration.
        if bless_schema:
            if not schema_diffs:
                print("nothing to bless: the schema fields already match")
                sys.exit(0)
            merged = json.loads(GOLDEN.read_text())
            merged["code_hash"] = now["code_hash"]
            for name, case in now["cases"].items():
                if name in merged["cases"]:
                    merged["cases"][name].update(
                        {k: case[k] for k in ("genome_hash", "seed_index", "steps",
                                              "initial_state_hash")})
            GOLDEN.write_text(json.dumps(merged, indent=2), newline="\n")
            print(f"\nBLESSED {len(schema_diffs)} schema field(s). Every "
                  f"final_state_hash is UNCHANGED -- this records that the engine's "
                  f"bytes moved and its physics did not. If you cannot say why the "
                  f"bytes moved, do not commit this.")
            sys.exit(0)
        if schema_diffs and strict_schema:
            print("*** --strict-schema: non-trajectory drift treated as failure ***")
            print("    If the trajectories above are all PASS, the physics did not "
                  "change and\n    the drift is `code_hash` alone. Record why, then "
                  "rerun with --bless-schema.")
            sys.exit(1)
        sys.exit(0)

    print("\n*** GOLDEN REGRESSION TRIPPED ***")
    print(f"  code_hash: {old['code_hash']} -> {now['code_hash']}")
    for d in diffs:
        print(f"  {d}")
    if accept:
        GOLDEN.write_text(json.dumps(now, indent=2))
        print("\nACCEPTED. You have just declared a physics change. Record the "
              "reason in docs/DECISIONS.md and bump protocol_version, or this "
              "repo now silently mixes two different models.")
        sys.exit(0)
    print("\nNot accepted. Fix the bug, or rerun with --accept if the change "
          "was intended (and then document it).")
    sys.exit(1)
