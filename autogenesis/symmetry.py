"""Symmetry-quotiented discrete signatures.

Round-3 audit, section 4.1: our canonical `(active, bonds)` signature contains
PERSISTENT PARTICLE INDICES. The frozen physics is permutation-equivariant, so
two states related only by relabelling indistinguishable particles are the same
physical state -- but they receive different signatures, which inflates the
attractor count `k`.

Since every result we have counted attractors with the labelled signature, this
is not a hypothetical: it may have inflated every `k` in the project.

WHAT THIS COMPUTES
------------------
A 1-WL (colour-refinement) invariant of the active-coloured motif graph:

  * initial colour of a node = its activity bit
  * refine: colour <- (own colour, sorted multiset of neighbour colours)
  * iterate to a fixed point
  * signature = sorted multiset of final colours, plus the sorted multiset of
    (colour, colour) pairs over edges

DIRECTION OF ERROR, STATED UP FRONT
-----------------------------------
1-WL is an INVARIANT, not a complete canonical form. It can MERGE two
non-isomorphic graphs (notoriously, regular graphs), but it can never SPLIT two
isomorphic ones.

For our purpose that is the safe direction: we are worried about over-counting
attractors, and this can only reduce `k`, never raise it. A drop in `k` under
quotienting is evidence of label inflation. A drop is NOT proof the merged
states are physically identical -- 1-WL may have merged genuinely different
graphs, and any candidate that survives must be rechecked with an exact
isomorphism test.

ROUND-4 AUDIT, CONDITION C3 -- WL IS NO LONGER ALLOWED TO DECIDE
---------------------------------------------------------------
MEASUREMENT_SPEC_V4 sec 1.1 said that if the labelled and quotiented counts ever
disagree, "the quotiented count is authoritative". That contradicts the
paragraph directly above, which this module has carried since it was written.
The audit reached the same conclusion independently.

MS-C-859: for a year this reversal lived ONLY here. The spec kept stating the rule the
engine had stopped implementing, and the file that knew better was a module docstring --
the one place a reader of the specification would never look. sec 1.1 is now struck and
amended in the spec itself, using the amendment mechanism that file already applies in
sec M2. The rule is:

    labelled signature   -> debugging and tracking
    WL invariant         -> a fast BUCKET INDEX, never a verdict
    canonical_form()     -> the decision: exact active-coloured graph
                            isomorphism, by individualisation-refinement

A labelled/quotiented disagreement escalates to `canonical_form`; it does not
resolve by fiat in either direction.

`canonical_form` is exact: two active-coloured graphs get the same string if and
only if they are isomorphic as active-coloured graphs. It is exponential in the
worst case, so it carries an explicit budget and raises `BudgetExceeded` rather
than silently degrading to WL -- a silent fallback would reintroduce exactly the
defect this replaces (D-004, fail loud).
"""
from __future__ import annotations
import hashlib
from collections import Counter


def wl_colours(members, active, edges, rounds=None):
    """Colour-refinement over the motif subgraph. `edges` are pairs of motif
    members; `active` maps member -> bool."""
    idx = {m: k for k, m in enumerate(members)}
    n = len(members)
    adj = [[] for _ in range(n)]
    for i, j in edges:
        if i in idx and j in idx:
            adj[idx[i]].append(idx[j])
            adj[idx[j]].append(idx[i])

    colour = [1 if active[m] else 0 for m in members]
    rounds = n if rounds is None else rounds
    for _ in range(rounds):
        sig = [(colour[v], tuple(sorted(colour[w] for w in adj[v])))
               for v in range(n)]
        order = {s: k for k, s in enumerate(sorted(set(sig)))}
        new = [order[s] for s in sig]
        if new == colour:
            break
        colour = new
    return colour, adj


def quotient_signature(members, active, edges) -> str:
    """Label-independent signature of one motif state."""
    colour, adj = wl_colours(members, active, edges)
    node_hist = sorted(Counter(colour).items())
    edge_hist = sorted(Counter(tuple(sorted((colour[v], colour[w])))
                               for v in range(len(colour))
                               for w in adj[v] if v < w).items())
    return "N" + repr(node_hist) + "|E" + repr(edge_hist)


def labelled_signature(members, active, edges) -> str:
    """The signature we have been using: persistent indices included."""
    mset = set(members)
    a = "".join("1" if active[m] else "0" for m in members)
    b = ";".join(f"{i}-{j}" for i, j in sorted(edges)
                 if i in mset and j in mset)
    return a + "|" + b


def canonical_cycle(sigs) -> str:
    """Phase-invariant canonical form of a cycle of signatures."""
    p = len(sigs)
    r = min(range(p), key=lambda z: tuple(sigs[z:] + sigs[:z]))
    return "||".join(sigs[r:] + sigs[:r])


def short(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


# --------------------------------------------------------------- exact GI --
class BudgetExceeded(RuntimeError):
    """Raised instead of degrading to an approximate answer. See D-004."""


def _adjacency(members, active, edges):
    idx = {m: k for k, m in enumerate(members)}
    n = len(members)
    adj = [set() for _ in range(n)]
    for i, j in edges:
        if i in idx and j in idx and i != j:
            adj[idx[i]].add(idx[j])
            adj[idx[j]].add(idx[i])
    col = [1 if active[m] else 0 for m in members]
    return n, adj, col


def _refine(n, adj, colour):
    """1-WL to a stable colouring. Colours are renumbered by sorted signature,
    so the result is isomorphism-invariant."""
    while True:
        sig = [(colour[v], tuple(sorted(colour[w] for w in adj[v])))
               for v in range(n)]
        order = {s: k for k, s in enumerate(sorted(set(sig)))}
        new = [order[s] for s in sig]
        if new == colour:
            return colour
        colour = new


def _certificate(n, adj, colour, act):
    """Only valid when `colour` is discrete: colour[v] is then v's rank."""
    rank = colour
    pos = [0] * n
    for v in range(n):
        pos[rank[v]] = v
    bits = "".join("1" if act[pos[r]] else "0" for r in range(n))
    ed = sorted((min(rank[v], rank[w]), max(rank[v], rank[w]))
                for v in range(n) for w in adj[v] if v < w)
    return bits + "|" + ";".join(str(a) + "-" + str(b) for a, b in ed)


def canonical_form(members, active, edges, budget=200000) -> str:
    """Exact canonical form of the active-coloured motif graph.

    Equal strings <=> isomorphic as active-coloured graphs. Individualisation
    refinement: refine, and if the colouring is not yet discrete, individualise
    every vertex of the first non-singleton cell in turn and keep the smallest
    certificate. Target-cell choice is by colour value, which is
    isomorphism-invariant after refinement, so the recursion is too.
    """
    n, adj, col0 = _adjacency(members, active, edges)
    if n == 0:
        return "|"
    act = [bool(active[m]) for m in members]
    spent = [0]

    def search(colour):
        spent[0] += 1
        if spent[0] > budget:
            raise BudgetExceeded(
                "canonical_form exceeded " + str(budget) + " refinements on "
                + str(n) + " nodes; this is a real symmetry blow-up, not a "
                "reason to fall back on WL")
        colour = _refine(n, adj, colour)
        cells = {}
        for v in range(n):
            cells.setdefault(colour[v], []).append(v)
        target = None
        for c in sorted(cells):
            if len(cells[c]) > 1:
                target = cells[c]
                break
        if target is None:
            return _certificate(n, adj, colour, act)
        best = None
        for v in target:
            marked = [2 * colour[w] + (1 if w == v else 0) for w in range(n)]
            cert = search(marked)
            if best is None or cert < best:
                best = cert
        return best

    return search(list(col0))


def exact_signature(members, active, edges) -> str:
    """The C3 decision function. Same contract as `quotient_signature`, but
    exact: it never merges non-isomorphic graphs."""
    return canonical_form(members, active, edges)


def equivalence_classes(states):
    """Group `(members, active, edges)` triples under the C3 rule: WL buckets
    first, exact isomorphism decides inside each bucket.

    Returns a list of class ids, one per input, and the number of splits the
    exact test made that WL would have missed -- report it, never hide it.
    """
    buckets = {}
    for k, (m, a, e) in enumerate(states):
        buckets.setdefault(quotient_signature(m, a, e), []).append(k)
    ids = [None] * len(states)
    nxt, wl_missed = 0, 0
    for _, group in sorted(buckets.items()):
        seen = {}
        for k in group:
            m, a, e = states[k]
            cf = canonical_form(m, a, e)
            if cf not in seen:
                seen[cf] = nxt
                nxt += 1
            ids[k] = seen[cf]
        wl_missed += len(seen) - 1
    return ids, wl_missed
