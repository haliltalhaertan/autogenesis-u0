"""Deterministic driver.

(genome_hash, seed_namespace, seed_index, code_hash) -> exact trajectory.

If code_hash changes, results computed before and after are NOT comparable.
That is not an inconvenience, it is the point: see docs/DECISIONS.md D-005.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
from .genome import Genome
from .state import State
from .init import make_initial_state
from .kernel import _step, step_args

_PKG = Path(__file__).resolve().parent

# every file whose bytes can change a trajectory.
#
# MS-C-735: `__init__.py` was in no identity list either. It carries only a docstring
# and `__version__` today, but it EXECUTES on every import of the package, ahead of
# every trajectory module, so it is capable of changing one. "Harmless today" is not
# an identity argument. `tests/test_provenance.py` now fails if any `autogenesis/*.py`
# is left out of all three digests.
_TRAJECTORY_FILES = ["__init__.py", "genome.py", "init.py", "kernel.py",
                     "reference.py", "rng.py", "simulator.py", "state.py"]


def code_hash() -> str:
    """Identity of the trajectory layer.

    `MS-C-858`, external audit Bulgu 15. This used to concatenate the eight files with
    no name, no length and no separator, so the digest identified only the CONCATENATION
    and not the files. Demonstrated with the engine's own three lines against two copies
    of the tree: move the final `\\n` from `__init__.py` to the front of `genome.py` --
    both files now differ on disk -- and the digest is byte-for-byte the same
    `e884f0e2b2185114`. Any byte moved across a boundary in sorted order collides, and
    the boundary between `__init__.py` and `genome.py` is the one that matters most,
    since `__init__.py` executes ahead of every trajectory module (`MS-C-735`).

    Each file now contributes its NAME and its LENGTH before its bytes, so a boundary
    cannot be moved without moving the digest. `producer_hash` already framed by name;
    it now frames by length too, and `measurement_hash` matches. This changes the value
    of all three digests once -- no physics moves, which is what `--bless-schema` is for.
    """
    h = hashlib.sha256()
    for name in sorted(_TRAJECTORY_FILES):
        data = (_PKG / name).read_bytes()
        h.update(f"{name}\x00{len(data)}\x00".encode("utf-8"))
        h.update(data)
    return h.hexdigest()[:16]


def step(g: Genome, s: State) -> State:
    # MS-C-719: `State.check()` was taught to reject malformed shapes (MS-C-700) but
    # this function checked only its OUTPUT, so the repair never reached the call site
    # it was written for. A state with x/u=(2,3), active=(1,), bonds=(2,2) was rejected
    # by `s.check(6)` and ACCEPTED by `step()`: `_step` runs under `boundscheck=False`,
    # read `active[1]` out of bounds, and returned a (2,) result with no error. The
    # guard has to be on the way IN -- once numba has read past the end, checking what
    # came back proves nothing about what was read.
    s.check(g.topology.valence)
    nx, nu, na, nb, coincident = _step(s.x, s.u, s.active, s.bonds, *step_args(g))
    if coincident:
        # protocol: "any r<1e-12 during execution is an implementation/numerical
        # failure, not resolved with an arbitrary direction"
        raise FloatingPointError("coincident pair (r < 1e-12)")
    out = State(nx, nu, na, nb)
    out.check(g.topology.valence)
    return out


def run(g: Genome, namespace: str, index: int, steps: int | None = None,
        s0: State | None = None, observer=None) -> dict:
    """observer(t, state) is called after each step. It must not mutate state."""
    g.validate()
    steps = g.numerics.max_steps if steps is None else steps
    # MS-C-870, second independent audit. `steps` had no contract: `-1` was ACCEPTED,
    # ran nothing, and was written into the identity record as `steps: -1`, so a
    # provenance line described a run that never happened and the initial and final
    # hashes agreed because nothing moved. `True` was accepted too -- `range(True)` is
    # one step -- and recorded as `steps: true`, which is not a count. Only `1.5` failed,
    # and late, inside `range`. An explicit int, not a bool, not negative.
    # `bool` is checked first because `isinstance(True, int)` is True in Python.
    if isinstance(steps, bool):
        raise TypeError(
            f"steps must be an int, not a bool: {steps!r}. `range(True)` is one step, "
            "and the identity record then reads `steps: true`, which is not a count "
            "(MS-C-870).")
    if not isinstance(steps, int):
        raise TypeError(
            f"steps must be an int: {steps!r} ({type(steps).__name__}). This used to "
            "fail late and obscurely, inside `range` (MS-C-870).")
    if steps < 0:
        raise ValueError(
            f"steps must be >= 0: {steps!r}. A negative count ran nothing and was "
            "written into the record as if it had (MS-C-870).")
    s = s0.copy() if s0 is not None else make_initial_state(g, namespace, index)
    init_hash = s.hash()[:16]
    # MS-C-703: the docstring said the observer "must not mutate state" and then handed
    # it the live State. An observer doing `s.x[0,0] += 0.125` at t=0 produced an
    # identical genome_hash / namespace / index / code_hash / steps /
    # initial_state_hash and a DIFFERENT final_state_hash -- the identity quad no
    # longer determined the trajectory. It now receives a copy, so the contract is
    # enforced rather than requested.
    if observer is not None:
        observer(0, s.copy())
    for t in range(steps):
        s = step(g, s)
        if observer is not None:
            observer(t + 1, s.copy())
    return {"genome_hash": g.short_hash(),
            "universe_id": g.universe_id,
            "namespace": namespace,
            "index": index,
            "code_hash": code_hash(),
            "steps": steps,
            "initial_state_hash": init_hash,
            "final_state_hash": s.hash()[:16],
            "state": s}
