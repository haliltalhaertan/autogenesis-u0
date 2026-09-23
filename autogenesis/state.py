"""World state and its canonical hash."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import numpy as np


@dataclass
class State:
    x: np.ndarray          # (N, D) float64 positions
    u: np.ndarray          # (N, D) float64 velocities
    active: np.ndarray     # (N,)   bool
    bonds: np.ndarray      # (N, N) bool, symmetric, zero diagonal

    def copy(self) -> "State":
        return State(self.x.copy(), self.u.copy(), self.active.copy(), self.bonds.copy())

    @property
    def N(self) -> int:
        return self.x.shape[0]

    @property
    def D(self) -> int:
        return self.x.shape[1]

    def edge_list(self) -> list[tuple[int, int]]:
        ii, jj = np.nonzero(np.triu(self.bonds, 1))
        return sorted(zip(ii.tolist(), jj.tolist()))

    def degrees(self) -> np.ndarray:
        return self.bonds.sum(1).astype(np.int32)

    def check(self, valence: int) -> None:
        """Fail LOUD. Never repair silently (docs/DECISIONS.md D-004)."""
        # MS-C-700: shapes and dtypes were NOT validated, and `kernel._step` runs under
        # `boundscheck=False`. A state with x/u=(2,3), active=(1,), bonds=(2,2) passed
        # this check, and `step()` then read `active[1]` out of bounds and returned a
        # (2,) result with no error -- undefined behaviour that another numba or
        # platform build may resolve differently, or crash on.
        if self.x.ndim != 2:
            raise AssertionError(f"x must be (N, D), got shape {self.x.shape}")
        n, d = self.x.shape
        for name, arr, want in (("u", self.u, (n, d)),
                                ("active", self.active, (n,)),
                                ("bonds", self.bonds, (n, n))):
            if arr.shape != want:
                raise AssertionError(
                    f"{name} shape {arr.shape} != {want} implied by x {self.x.shape}")
        if self.x.dtype != np.float64 or self.u.dtype != np.float64:
            raise AssertionError(
                f"x/u must be float64, got {self.x.dtype}/{self.u.dtype}")
        if self.active.dtype != np.bool_ or self.bonds.dtype != np.bool_:
            raise AssertionError(
                f"active/bonds must be bool, got {self.active.dtype}/{self.bonds.dtype}")
        if not np.isfinite(self.x).all() or not np.isfinite(self.u).all():
            raise FloatingPointError("NaN/Inf in state")
        if not np.array_equal(self.bonds, self.bonds.T):
            raise AssertionError("bond asymmetry")
        if self.bonds.diagonal().any():
            raise AssertionError("self bond")
        d = self.degrees()
        if d.size and int(d.max()) > valence:
            raise AssertionError(f"strict valence violated {int(d.max())}>{valence}")

    def hash_v1(self) -> str:
        """The UNFRAMED hash, kept ONLY to read records written before the migration.

        MS-C-701: it concatenates raw bytes with no version, no N, no D, no dtype and
        no length framing, so two states of different size collide whenever their byte
        streams happen to line up. Reproduced: N=49 D=2 and N=33 D=3, both all-zero,
        are 1617 bytes either way and hash to
        7f4da80ffddd6aa6f53fc5e1280294406187d5f5f3506d4664f4f31cba4f5f1e.

        Not a SHA-256 collision -- a deterministic serialisation collision. The
        contract "the state hash identifies the physical state" was false for it.
        Never use this for new records.
        """
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(self.active.astype(np.uint8)).tobytes())
        h.update(np.array(self.edge_list(), dtype=np.int32).tobytes())
        h.update(np.ascontiguousarray(self.x, dtype=np.float64).tobytes())
        h.update(np.ascontiguousarray(self.u, dtype=np.float64).tobytes())
        return h.hexdigest()

    def hash(self) -> str:
        """Framed canonical hash (v2). Every field is length- and type-delimited, so
        no two distinct states can produce the same byte stream (MS-C-701)."""
        edges = np.array(self.edge_list(), dtype=np.int32).reshape(-1, 2)
        h = hashlib.sha256()
        h.update(b"autogenesis.State/v2\x00")
        h.update(np.array([self.x.shape[0], self.x.shape[1], edges.shape[0]],
                          dtype=np.int64).tobytes())
        for tag, arr in ((b"active/u8", self.active.astype(np.uint8)),
                         (b"edges/i32", edges),
                         (b"x/f64", np.ascontiguousarray(self.x, dtype=np.float64)),
                         (b"u/f64", np.ascontiguousarray(self.u, dtype=np.float64))):
            buf = np.ascontiguousarray(arr).tobytes()
            h.update(tag + b"\x00")
            h.update(np.array([len(buf)], dtype=np.int64).tobytes())
            h.update(buf)
        return h.hexdigest()

    def discrete_signature(self) -> str:
        """Topology+activity only. NOTE: two states sharing this signature may
        differ geometrically. Never read 'same signature' as 'same physical
        state' -- that error is logged as an audit target in the U0 project."""
        return ("".join("1" if a else "0" for a in self.active) + "|" +
                ";".join(f"{i}-{j}" for i, j in self.edge_list()))
