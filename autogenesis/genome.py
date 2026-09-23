"""Universe Genome: the complete, machine-readable specification of one universe.

INVARIANT: nothing in this engine reads a physics constant from anywhere else.
If a number influences a trajectory it lives in a Genome field, or it is a bug.

The genome is canonicalised to sorted-key JSON and SHA256-hashed.  That hash,
together with (seed, code_hash), fully determines a trajectory bit-for-bit.
"""
from __future__ import annotations
import json, hashlib, math
from dataclasses import dataclass, field, asdict

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Space:
    dimensions: int = 3          # U0 is 3D. See docs/DECISIONS.md D-001.


@dataclass(frozen=True)
class Physics:
    r0: float = 1.0              # bond equilibrium distance (length unit)
    mass: float = 1.0            # (unit choice)
    spring_k: float = 2.0        # k_s, bonded pairs only
    gamma: float = 1.0           # per-particle damping
    repulsion_cutoff: float = 0.75   # r_rep, NONBONDED pairs only
    repulsion_strength: float = 4.0  # k_rep


@dataclass(frozen=True)
class Topology:
    candidate_radius: float = 1.35   # R_candidate: formation + support radius
    break_radius: float = 1.60       # R_break: retention max distance
    valence: int = 6
    allocator: str = "B_support_orbit"   # A_batch_all | B_support_orbit | C_overload_prune
    # --- node rule (MF-L) ---------------------------------------------------
    # The frozen protocol carries SEPARATE birth and survival bands. The
    # original engine reads only `birth_...` and applies it to both -- a
    # silent capability gap: changing the survival band there has no effect
    # (audited 2026-08-21, see docs/DECISIONS.md D-010).
    #
    # Here the distinction is real:
    #   node currently PASSIVE -> birth band decides
    #   node currently ACTIVE  -> survival band decides
    # At U0 the two bands are equal, so this reduces exactly to the
    # state-independent rule and reproduces U0 bit-for-bit.
    node_rule: str = "MF-L"
    birth_lo: float = 1.0 / 6.0        # inclusive
    birth_hi: float = 0.5              # inclusive
    survival_lo: float = 1.0 / 6.0     # inclusive
    survival_hi: float = 0.5           # inclusive
    # --- bond rule (A*|A*|a1|fs1 + geometric hysteresis) --------------------
    bond_rule: str = "A*|A*|a1|fs1"
    retention_requires_active_endpoint: bool = True   # A* retention
    formation_requires_active_endpoint: bool = True   # A* formation
    formation_support_min: int = 1                    # fs1
    retention_support_min: int = 0

    def effective_bond_rule(self) -> str:
        """The bond rule these fields actually describe.

        `MS-C-851`, external audit Bulgu 12. `MS-C-723` closed `bond_rule` to a single
        label so that `bond_rule = "nonsense"` could no longer be accepted, and wrote
        the right principle beside it -- *an inert label that can describe an
        unimplemented behaviour is not metadata, it is a false record*. The closure
        stopped an UNKNOWN label. It did nothing about the KNOWN one sitting on a genome
        that is not that rule. Measured: four genomes with four different bond physics
        -- U0, `formation_support_min` 1 -> 0, `retention_requires_active_endpoint`
        True -> False, `formation_requires_active_endpoint` True -> False -- all four
        carry `A*|A*|a1|fs1` and all four pass `validate()`.

        The label decodes, from `PHASE_B_FROZEN_PROTOCOL.json`'s `bond_rule` block:
        `A*` retention endpoint types `["AA","AI"]`, `A*` formation endpoint types,
        `a1` = `alpha` (protocol-only and never read -- `C-012`), `fs1` =
        `formation_common_active_support_min`. `retention_support_min` has no component,
        so a genome varying it was mis-described even in principle; it is appended as
        `rs{n}` when non-zero rather than left silent.

        U0 renders `A*|A*|a1|fs1` byte for byte, so nothing about U0 moves.

        Note the identity was never at risk: the four genome hashes above already
        differed (`0a545a09`, `74dbfad6`, `92ea5907`, `3faa2bf8`), so no measurement was
        ever attributed to the wrong physics. This is a record defect, and only that.
        """
        r = "A*" if self.retention_requires_active_endpoint else "**"
        f = "A*" if self.formation_requires_active_endpoint else "**"
        label = f"{r}|{f}|a1|fs{self.formation_support_min}"
        if self.retention_support_min:
            label += f"|rs{self.retention_support_min}"
        return label


@dataclass(frozen=True)
class InitialConditions:
    N: int = 32
    cube_side: float = 3.6           # family B_very_sparse_bonded
    min_separation: float = 0.70     # rejection-sampling floor
    active_fraction: float = 0.5     # exactly round(N*f) active
    # Two ways to seed the initial bond graph.
    #   "probability"  -- the protocol's original recipe: Bernoulli(p) per
    #                     candidate edge, capped by initial_degree_cap.
    #   "mean_degree"  -- target an exact mean initial degree <k0>. This is the
    #                     PHYSICALLY meaningful knob: bond_probability is
    #                     entangled with density and R_candidate, whereas <k0>
    #                     is the quantity the node rule actually responds to.
    #                     Requested <k0> may be unreachable at low density; the
    #                     ACHIEVED value is always reported, never assumed.
    bond_init_mode: str = "probability"
    bond_probability: float = 0.08
    target_mean_degree: float = 0.0
    initial_degree_cap: int = 2
    initial_velocity: str = "zero"
    family: str = "B_very_sparse_bonded"


@dataclass(frozen=True)
class Numerics:
    dt: float = 0.02
    integrator: str = "semi_implicit_euler"
    max_steps: int = 4096


@dataclass(frozen=True)
class Genome:
    universe_id: str = "U0"
    schema_version: int = SCHEMA_VERSION
    space: Space = field(default_factory=Space)
    physics: Physics = field(default_factory=Physics)
    topology: Topology = field(default_factory=Topology)
    initial_conditions: InitialConditions = field(default_factory=InitialConditions)
    numerics: Numerics = field(default_factory=Numerics)

    # ---- serialisation ----------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    def canonical_json(self) -> str:
        """Byte-stable representation. universe_id is EXCLUDED from the hash so
        that two identically-parameterised universes collide on hash (which is
        what we want: the hash identifies the physics, the id identifies the
        registry entry)."""
        d = self.to_dict()
        d.pop("universe_id")
        return json.dumps(d, sort_keys=True, separators=(",", ":"))

    def hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()

    def short_hash(self) -> str:
        return self.hash()[:16]

    @staticmethod
    def from_dict(d: dict) -> "Genome":
        # MS-C-702: unknown top-level keys were silently ignored and any
        # `schema_version` was accepted, so `replace(toplogy={...})` returned an
        # UNCHANGED genome and `from_dict({..., "toplogy": ...})` dropped the field
        # without a word. A typo in a genome patch became a silent no-op.
        known = {"universe_id", "schema_version", "space", "physics", "topology",
                 "initial_conditions", "numerics"}
        unknown = sorted(set(d) - known)
        if unknown:
            raise ValueError(
                f"unknown genome field(s) {unknown}; expected a subset of "
                f"{sorted(known)}. A typo here is silent otherwise.")
        got = d.get("schema_version", SCHEMA_VERSION)
        if got != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version {got!r}; this engine reads "
                f"{SCHEMA_VERSION!r}. Migrate the record rather than loading it.")
        return Genome(
            universe_id=d.get("universe_id", "U?"),
            schema_version=d.get("schema_version", SCHEMA_VERSION),
            space=Space(**d["space"]),
            physics=Physics(**d["physics"]),
            topology=Topology(**d["topology"]),
            initial_conditions=InitialConditions(**d["initial_conditions"]),
            numerics=Numerics(**d["numerics"]),
        )

    def replace(self, **paths) -> "Genome":
        """g.replace(**{'physics.gamma': 0.7, 'topology.valence': 5})"""
        d = self.to_dict()
        for path, val in paths.items():
            head, _, tail = path.partition(".")
            if not tail:
                d[head] = val
            else:
                d[head][tail] = val
        g = Genome.from_dict(d)
        # MS-C-851: changing the bond physics must change the label that names it. Done
        # here so the two tools that vary these fields keep working and simply stop
        # carrying a false record; an explicit `topology.bond_rule` is left alone, so a
        # caller can still write a wrong label deliberately and have `validate()` say so.
        if "topology.bond_rule" not in paths:
            eff = g.topology.effective_bond_rule()
            if eff != g.topology.bond_rule:
                d["topology"]["bond_rule"] = eff
                g = Genome.from_dict(d)
        return g

    def validate(self) -> None:
        """Reject a physically invalid universe before it is ever simulated.

        F-02 / OB-131 / MS-C-710: every check here was an `assert`, which
        `python -O` deletes -- so under optimisation this function accepted anything.
        The band-bounds check was worse: `assert empty or 0.0 <= lo <= hi <= 1.0`
        short-circuits on `empty`, so `lo > hi` skipped the range test entirely and
        `[2.0, 1.0]` was a valid `Genome`. Both are fixed; `require` raises.

        MS-C-723: those two repairs were recorded as closing `OB-131`'s validate arm,
        and they did not. `OB-131` asks for COMPLETE FIELD RANGES, and eight of
        `MS-C-576`'s own examples were still accepted afterwards --
        `bond_probability = 2.0`, `bond_probability = -0.1`, `active_fraction = 1.2`,
        `initial_degree_cap = -1`, `candidate_radius = -1.0`, and the three pure
        labels below. Every field of every genome dataclass now has a constraint
        here, and `tests/test_genome_validate.py` FAILS if a field is added without
        one -- the arm is closed by a gate, not by a claim.
        """
        # AssertionError, not ValueError: 20 call sites across tools/ and tests/
        # already catch AssertionError from this function to skip an invalid genome,
        # and `python -O` strips only the `assert` STATEMENT, never an explicit
        # `raise`. Same type, same handlers, no longer optimised away (MS-C-711).
        def require(cond, msg):
            if not cond:
                raise AssertionError(f"invalid Genome: {msg}")

        p, t, ic, nm = self.physics, self.topology, self.initial_conditions, self.numerics

        # MS-C-737: the range checks were all written as `>=` / `<=` comparisons, and
        # in Python those succeed against the WRONG TYPE and against infinity. So
        # `N = 2.5`, `dimensions = 2.0`, `max_steps = 1.5`, `valence = 2.5`, the two
        # support minima at `1.5` and EVERY physical quantity at `inf` all validated
        # -- 19 invalid genomes accepted, four of which then died later with a
        # `TypeError` or an `OverflowError` far from the cause. A bound is not a type,
        # and `inf <= inf` is True.
        def whole(name, v):
            # bool is a subclass of int and must not pass as a count
            require(isinstance(v, int) and not isinstance(v, bool),
                    f"{name} must be an integer, got {type(v).__name__} {v!r}")

        def finite(name, v):
            require(isinstance(v, (int, float)) and not isinstance(v, bool)
                    and math.isfinite(v),
                    f"{name} must be a finite real, got {type(v).__name__} {v!r}")

        # One call per field, deliberately not a loop over tuples: `require`, `whole`
        # and `finite` are the three GATE calls, and `tests/test_genome_validate.py`
        # establishes coverage by finding each field inside one of them. Hiding the
        # fields in a loop iterator would put them out of the gate's sight, which is
        # the defect that test exists to catch (MS-C-737).
        whole("space.dimensions", self.space.dimensions)
        whole("topology.valence", t.valence)
        whole("topology.formation_support_min", t.formation_support_min)
        whole("topology.retention_support_min", t.retention_support_min)
        whole("initial_conditions.N", ic.N)
        whole("initial_conditions.initial_degree_cap", ic.initial_degree_cap)
        whole("numerics.max_steps", nm.max_steps)

        finite("physics.r0", p.r0)
        finite("physics.mass", p.mass)
        finite("physics.spring_k", p.spring_k)
        finite("physics.gamma", p.gamma)
        finite("physics.repulsion_strength", p.repulsion_strength)
        finite("physics.repulsion_cutoff", p.repulsion_cutoff)
        finite("topology.candidate_radius", t.candidate_radius)
        finite("topology.break_radius", t.break_radius)
        finite("topology.birth_lo", t.birth_lo)
        finite("topology.birth_hi", t.birth_hi)
        finite("topology.survival_lo", t.survival_lo)
        finite("topology.survival_hi", t.survival_hi)
        finite("initial_conditions.cube_side", ic.cube_side)
        finite("initial_conditions.min_separation", ic.min_separation)
        finite("initial_conditions.active_fraction", ic.active_fraction)
        finite("initial_conditions.bond_probability", ic.bond_probability)
        finite("initial_conditions.target_mean_degree", ic.target_mean_degree)
        finite("numerics.dt", nm.dt)
        # `from_dict` rejects a bad schema_version, but a Genome built directly --
        # `dataclasses.replace(g, schema_version=999)` -- never passes through it.
        require(self.schema_version == SCHEMA_VERSION,
                f"unsupported schema_version {self.schema_version!r}; this engine "
                f"reads {SCHEMA_VERSION!r}")

        require(self.space.dimensions in (2, 3), "dimensions must be 2 or 3")
        require(p.r0 > 0 and p.mass > 0 and p.spring_k > 0,
                f"r0/mass/spring_k must be positive: {p.r0}/{p.mass}/{p.spring_k}")
        require(p.gamma >= 0 and p.repulsion_strength >= 0,
                f"gamma/repulsion_strength must be non-negative: "
                f"{p.gamma}/{p.repulsion_strength}")
        require(p.repulsion_cutoff > 0,
                f"repulsion cutoff must be positive: {p.repulsion_cutoff}")
        require(t.candidate_radius > 0.0,
                f"R_candidate must be positive: {t.candidate_radius}")
        require(t.candidate_radius <= t.break_radius,
                f"hysteresis inverted: R_candidate {t.candidate_radius} must be "
                f"<= R_break {t.break_radius}")
        require(t.valence >= 1 and 0 <= ic.initial_degree_cap <= t.valence,
                f"valence {t.valence} must be >= 1 and initial_degree_cap "
                f"{ic.initial_degree_cap} must lie in [0, valence]")
        require(t.formation_support_min >= 0 and t.retention_support_min >= 0,
                f"support minima must be non-negative: "
                f"{t.formation_support_min}/{t.retention_support_min}")
        # The A* gates are read as truth values by the kernel, so ANY non-empty
        # string is `True` there: `formation_requires_active_endpoint = "false"`
        # would silently mean the opposite of what the record says.
        require(isinstance(t.formation_requires_active_endpoint, bool) and
                isinstance(t.retention_requires_active_endpoint, bool),
                f"the A* gates must be bool, got "
                f"{type(t.formation_requires_active_endpoint).__name__}/"
                f"{type(t.retention_requires_active_endpoint).__name__}")
        # lo > hi is the EMPTY band: it selects no activation fraction, so no node can
        # ever activate. That is a legitimate member of the enumerated rule space
        # (autogenesis.rules) and serves as the floor control, so it must be
        # representable rather than rejected. But its BOUNDS are still checked --
        # MS-C-620: the old short-circuit let `[2.0, 1.0]` through as "empty".
        for band_name, lo, hi in (("birth", t.birth_lo, t.birth_hi),
                                  ("survival", t.survival_lo, t.survival_hi)):
            require(0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0,
                    f"{band_name} band bounds out of range: [{lo}, {hi}] "
                    f"(an EMPTY band is lo > hi, not a bound outside [0, 1])")
        require(t.allocator in ("A_batch_all", "B_support_orbit", "C_overload_prune"),
                f"unknown allocator {t.allocator!r}")
        require(ic.N >= 2 and ic.cube_side > 0 and ic.min_separation > 0,
                f"N/cube_side/min_separation invalid: {ic.N}/{ic.cube_side}/"
                f"{ic.min_separation}")
        require(ic.bond_init_mode in ("probability", "mean_degree"),
                f"unknown bond_init_mode {ic.bond_init_mode!r}")
        require(ic.target_mean_degree >= 0.0,
                f"target_mean_degree must be non-negative: {ic.target_mean_degree}")
        require(0.0 <= ic.bond_probability <= 1.0,
                f"bond_probability is a probability: {ic.bond_probability}")
        require(0.0 <= ic.active_fraction <= 1.0,
                f"active_fraction is a fraction of N: {ic.active_fraction} "
                f"(> 1 asks init for more active nodes than there are nodes)")
        # MS-C-723: `node_rule`, `bond_rule` and `initial_velocity` are the fields
        # `tools/audit_parameters.py` classifies as PURE_LABELS -- the engine reads
        # none of them. That is precisely why they must be closed sets rather than
        # free text: a genome saying `initial_velocity = "random"` was ACCEPTED and
        # then simulated with `u = 0`, and `node_rule = "nonsense"` was accepted and
        # simulated with MF-L. An inert label that can describe an unimplemented
        # behaviour is not metadata, it is a false record. Add a value here only
        # together with the code that implements it.
        require(t.node_rule in ("MF-L",),
                f"unknown node_rule {t.node_rule!r}; this engine implements 'MF-L'")
        # MS-C-851: a closed SET of one label stops an unknown value and permits a false
        # one. The label must equal the rule the genome's own bond fields describe; that
        # is strictly stronger, since every renderable label is by construction one this
        # engine implements. `replace()` re-derives it, so a caller varying bond physics
        # gets an honest record without having to know this exists.
        require(t.bond_rule == t.effective_bond_rule(),
                f"bond_rule {t.bond_rule!r} does not describe this genome's bond fields, "
                f"which are {t.effective_bond_rule()!r} -- a label that names physics "
                f"the genome does not have is a false record (MS-C-723's own rule)")
        require(ic.initial_velocity in ("zero",),
                f"unknown initial_velocity {ic.initial_velocity!r}; "
                f"autogenesis.init always seeds u = 0")
        require(nm.dt > 0 and nm.max_steps >= 1,
                f"dt/max_steps invalid: {nm.dt}/{nm.max_steps}")
        require(nm.integrator == "semi_implicit_euler",
                f"unknown integrator {nm.integrator!r}")


def U0() -> Genome:
    """Baseline universe: an EXACT restatement of PHASE_B_FROZEN_PROTOCOL.json
    (Project Autogenesis, frozen 2026-08-13), family B, valence 6,
    allocator B_support_orbit -- the configuration M006 was found in.

    U0 is a BENCHMARK, not an assumed optimum.  See docs/U0_SPEC.md.
    """
    g = Genome(universe_id="U0")
    g.validate()
    return g
