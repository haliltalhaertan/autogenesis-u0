"""Seed namespaces.

Discovery and confirmation MUST NOT share seeds.  Rather than trusting anyone
to remember that, seeds are derived from a NAMED namespace:

    seed_for("pilot_v1", 17)      -> discovery
    seed_for("holdout_v1", 17)    -> confirmation, provably disjoint stream

The namespace string is hashed into the entropy, so the same (namespace, index)
gives the same stream on any machine, forever.  Reusing a discovery namespace
during confirmation is then a visible, greppable act -- not an accident.
"""
from __future__ import annotations
import hashlib
import numpy as np

# streams are spawned in this fixed order; never reorder or insert in the middle
STREAMS = ("coordinates", "activity", "bonds", "perturbation", "reserve")


def _entropy(namespace: str, index: int) -> int:
    h = hashlib.sha256(f"{namespace}|{int(index)}".encode()).digest()
    return int.from_bytes(h[:16], "big")


def seed_for(namespace: str, index: int) -> int:
    """Public, loggable integer seed."""
    return _entropy(namespace, index)


def streams(namespace: str, index: int) -> dict[str, np.random.Generator]:
    """One independent Generator per named purpose."""
    root = np.random.SeedSequence(_entropy(namespace, index))
    children = root.spawn(len(STREAMS))
    return {name: np.random.default_rng(ss) for name, ss in zip(STREAMS, children)}
