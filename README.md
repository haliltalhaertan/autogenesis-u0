# autogenesis-u0

Code accompanying the preprint

> H. T. Ertan, *Memory without mechanics: exact analysis of a particle–network automaton with rule-driven bonds* (2026).

U0 is a three-layer model: damped particles in 3D, bonds that form and break by distance with
hysteresis, and a synchronous outer-totalistic Boolean rule on the bond graph that gates bond
formation and retention. This repository contains the engine and every script that produces a
number in the paper. It is a subset of a larger private research record; nothing here depends
on files that are not included.

## Setup

```
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Windows
# .venv/bin/python -m pip install -r requirements.txt     # Linux / macOS
```

Pinned versions: Python 3.14, numpy 2.5.2, numba 0.67.0, matplotlib 3.11.1. All results are bit-exact
under these versions; the engine is required to agree bit-for-bit with a pure-Python reference.

## Engine checks (about 30 s)

```
python tests/test_u0_equivalence.py    # 15/15 against the vendored original specification
python tests/test_differential.py      # 19/19 numba kernel == reference, bit-identical
python tests/test_invariance.py        # 5/5
python tests/test_golden.py            # 10/10 golden trajectories
```

## Reproducing the paper

Every number in the paper is listed in `paper_repro/claims_ledger.csv` with the script that produces it.
Times are single-core unless noted.

| paper | command | time | expected |
|---|---|---|---|
| Prop. 1, exact reduction | `python tools/measure.py scaffold_reduction_is_exact` | 1 min | 0 of 247,904 states moved |
| abstract map = engine | `python tools/measure.py abstract_map_equals_engine` | 1 min | 0 of 247,904 disagree |
| Prop. 2, support bound | `python tools/measure.py triangle_free_no_formation` and `edge_support_theorem` | 1 min each | 0 formations below the bound |
| Lemma 1 / Prop. 3 | `python tools/verify_review_combinatorics.py` | 11 min | part D: 0 violations in 8,628,128 engine calls |
| Prop. 4, window | `python external_review/tools/passive_degree_window.py` | 5 min | 22,567 fixed points, 0 violations |
| objects at U0 | `python tools/measure.py object_count_strict` and `is_the_blinker_topological` | 1 min | 6 strict, 1 blinker, so 5 |
| census, n = 3..6 | `python tools/measure.py scaffold_census_recount` and `scaffold_bit_robustness` | 1 min | 36 / 13 / 3; 0 of 22 |
| minimal register | `python paper_repro/shard_minimality.py k 12 runs/shards` for k = 0..11, then `python paper_repro/merge_minimality.py runs/shards` | 13 min on 12 cores | 1 of 491 classes, `F}qCG` |
| register coordinates | `python paper_repro/register_geometry.py` | 10 s | 0 moved; 282 / 112 / 48 |
| register robustness | `python paper_repro/register_robustness.py` | 10 s | only the designed write stays in the register |
| lattice matched null | `python tools/measure.py physics_selected_graphs` | 22 min | p = 0.9638 on (n, E, triangles) |
| settled-graph matched null | `python tools/measure.py built_graph_beyond_structure` (75 min), or `python paper_repro/matched_null_parallel.py runs/mnull full k 14` for k = 0..13, then `... runs/mnull merge` | 18 min on 14 cores | 98W / 97L / 4T, p = 0.5 |
| Prop. 5, flat energy | `python tools/measure.py bond_energy_landscape` | 20 s | spread 0.000 at rest; 967 of 967 uphill |

`tools/measure.py <name>` runs a declared measurement from `measurements/`. Each declaration states its
claim, what would overturn it, its universe, and at least one negative control. The runner writes its
result next to the script.

**A reproducibility pitfall.** `built_graph_beyond_structure.score()` maps the shared random bond bits
onto edges in the iteration order of Python sets. A parallel replay must therefore rebuild every graph
in-process along the original code path; serializing and reloading an adjacency changes the set order
and yields a different, statistically equivalent experiment. `matched_null_parallel.py` does this.

## Layout

```
autogenesis/        engine (numba kernel, reference, genome, state, measurement stamps)
tests/              engine checks and golden trajectories
vendor/u0_original/ the original specification, used as an independent oracle
tools/              verifiers and the measurement runner
measurements/       declared measurements used in the paper
external_review/    window-bound verifier
paper_repro/        scripts written for the paper, and the claims ledger
protocols/          frozen protocols the verifiers check by hash
```
