from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Generic, Iterator, List, TypeVar

import jpamb
from jpamb import jvm


AV = TypeVar("AV")  # abstract value type, e.g. SignSet


# --- Basic helper structures -------------------------------------------------


@dataclass
class Stack(Generic[AV]):
    """
    Tiny stack wrapper with an explicit 'items' list so we can
    deepcopy / join easily.
    """
    items: List[AV]

    @classmethod
    def empty(cls) -> "Stack[AV]":
        return cls([])

    def push(self, value: AV) -> None:
        self.items.append(value)

    def pop(self) -> AV:
        return self.items.pop()

    def peek(self) -> AV:
        return self.items[-1]

    def __len__(self) -> int:
        return len(self.items)

    def __bool__(self) -> bool:
        return bool(self.items)

    def copy(self) -> "Stack[AV]":
        return Stack(list(self.items))

    def __iter__(self) -> Iterator[AV]:
        return iter(self.items)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic only
        if not self.items:
            return "ϵ"
        return "[" + ", ".join(repr(v) for v in self.items) + "]"


@dataclass(frozen=True)
class PC:
    """Program counter: (method, instruction offset)."""
    method: jvm.AbsMethodID
    offset: int

    def next(self, delta: int = 1) -> "PC":
        return PC(self.method, self.offset + delta)

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.method}:{self.offset}"


# --- Bytecode access ---------------------------------------------------------


@dataclass
class Bytecode:
    """
    Lazy bytecode cache for a jpamb Suite.

    suite.method_opcodes(methodid) yields a list of jvm.Opcode objects.
    """
    suite: jpamb.Suite
    _cache: Dict[jvm.AbsMethodID, List[jvm.Opcode]] | None = None

    def __post_init__(self) -> None:
        if self._cache is None:
            self._cache = {}

    def __getitem__(self, pc: PC) -> jvm.Opcode:
        assert self._cache is not None
        try:
            ops = self._cache[pc.method]
        except KeyError:
            ops = list(self.suite.method_opcodes(pc.method))
            self._cache[pc.method] = ops
        return ops[pc.offset]


# --- Abstract frame/state ----------------------------------------------------


@dataclass
class PerVarFrame(Generic[AV]):
    """
    Abstract frame for one method.

    locals: mapping from local index -> abstract value
    stack:  abstract values on operand stack
    pc:     current program counter
    """
    locals: Dict[int, AV]
    stack: Stack[AV]
    pc: PC

    # -- Lattice structure over frames ---------------------------------------

    def __le__(self, other: "PerVarFrame[AV]") -> bool:
        """
        Pointwise ≤ over locals and stack.
        Missing locals are treated as bottom (<= any value).
        """
        if self.pc != other.pc:
            return False

        # locals: every value here must be ≤ corresponding value in other
        for idx, v in self.locals.items():
            ov = other.locals.get(idx)
            if ov is None:
                # other knows nothing about this local; treat as bottom ⇒ self !≤ other
                return False
            if not (v <= ov):  # type: ignore[operator]
                return False

        # stack: must be same height, compare element-wise
        if len(self.stack) != len(other.stack):
            return False

        for v1, v2 in zip(self.stack, other.stack):
            if not (v1 <= v2):  # type: ignore[operator]
                return False

        return True

    def join(self, other: "PerVarFrame[AV]") -> "PerVarFrame[AV]":
        """
        Pointwise join (⊔) of two frames at the same pc.
        """
        assert self.pc == other.pc

        # locals: union of keys, join overlapping
        new_locals: Dict[int, AV] = {}

        all_keys = set(self.locals) | set(other.locals)
        for idx in all_keys:
            v1 = self.locals.get(idx)
            v2 = other.locals.get(idx)
            if v1 is None:
                assert v2 is not None
                new_locals[idx] = v2
            elif v2 is None:
                new_locals[idx] = v1
            else:
                new_locals[idx] = v1 | v2  # type: ignore[operator]

        # stack: must have same height; join element-wise
        assert len(self.stack) == len(other.stack)
        new_stack_items: List[AV] = []
        for v1, v2 in zip(self.stack, other.stack):
            new_stack_items.append(v1 | v2)  # type: ignore[operator]

        return PerVarFrame(
            locals=new_locals,
            stack=Stack(new_stack_items),
            pc=self.pc,
        )

    # convenience alias so we can use frame1 | frame2
    def __or__(self, other: "PerVarFrame[AV]") -> "PerVarFrame[AV]":
        return self.join(other)

    # -- structural helpers ---------------------------------------------------

    def copy(self) -> "PerVarFrame[AV]":
        return PerVarFrame(
            locals=dict(self.locals),
            stack=self.stack.copy(),
            pc=self.pc,
        )

    def with_pc(self, pc: PC) -> "PerVarFrame[AV]":
        f = self.copy()
        f.pc = pc
        return f
