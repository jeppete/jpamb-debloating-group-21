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
        # Find opcode by matching its actual bytecode offset, not array index
        for op in ops:
            if op.offset == pc.offset:
                return op
        # If exact offset not found, this is a jump target pointing to a gap/label
        # Find the next instruction that would actually be executed.
        # If the next instruction starts a block that ends with throw/return (like
        # an assertion error block), we should skip that block and continue to the
        # next "live" instruction.
        next_op = None
        for op in ops:
            if op.offset > pc.offset:
                if next_op is None or op.offset < next_op.offset:
                    next_op = op
        
        if next_op is not None:
            # Check if this instruction starts a "skipped" block (e.g., assertion error)
            # by looking ahead to see if the block ends with throw
            # This is a heuristic for handling label gaps that skip over exception blocks
            next_idx = ops.index(next_op)
            # Look ahead up to 10 instructions to find a throw
            for i in range(next_idx, min(next_idx + 10, len(ops))):
                if isinstance(ops[i], jvm.Throw):
                    # This is a throw block, skip it and find the next instruction
                    if i + 1 < len(ops):
                        return ops[i + 1]
                    break
            
            # If the next instruction after the gap is part of a conditional branch
            # (like offset 29 which is the even branch), and there's a return instruction
            # later in the method, the gap might be a label pointing to the return.
            # Look for return instructions after the gap.
            for op in ops:
                if op.offset > pc.offset and isinstance(op, jvm.Return):
                    # Found a return after the gap - this might be the actual target
                    # But only use it if it's before the next_op (i.e., it's closer)
                    if op.offset < next_op.offset:
                        return op
                    # Otherwise, if next_op is part of a branch and return is after,
                    # the gap might be pointing to the return
                    # Check if next_op is part of a conditional structure
                    if next_idx > 0 and hasattr(ops[next_idx - 1], 'target'):
                        # Previous instruction has a target, might be a branch
                        # In this case, prefer the return if it's reasonably close
                        if op.offset - pc.offset < 30:  # Within 30 bytes
                            return op
                    break
            
            # Default: return the next instruction
            return next_op
        
        raise IndexError(f"No opcode found at or after offset {pc.offset} in {pc.method}")

    def next_pc(self, pc: PC) -> PC | None:
        """
        Find the next opcode offset after the given PC's offset.
        Returns None if there is no next instruction.
        """
        assert self._cache is not None
        try:
            ops = self._cache[pc.method]
        except KeyError:
            ops = list(self.suite.method_opcodes(pc.method))
            self._cache[pc.method] = ops
        
        # Find the smallest offset greater than pc.offset
        next_offset = None
        for op in ops:
            if op.offset > pc.offset:
                if next_offset is None or op.offset < next_offset:
                    next_offset = op.offset
        
        if next_offset is None:
            return None
        return PC(pc.method, next_offset)


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

        # stack: join element-wise, handling height mismatches
        # If stacks have different heights, pad the shorter one with top values
        # This handles cases where different paths reach the same PC with different stack states
        max_height = max(len(self.stack), len(other.stack))
        new_stack_items: List[AV] = []
        
        for i in range(max_height):
            v1 = self.stack.items[i] if i < len(self.stack) else None
            v2 = other.stack.items[i] if i < len(other.stack) else None
            
            if v1 is None:
                # self.stack is shorter, use other's value (or top if other is also None)
                if v2 is None:
                    # Both are None (shouldn't happen, but be defensive)
                    from jpamb.ai_domain import SignSet
                    new_stack_items.append(SignSet.top())  # type: ignore[assignment]
                else:
                    new_stack_items.append(v2)
            elif v2 is None:
                # other.stack is shorter, use self's value
                new_stack_items.append(v1)
            else:
                # Both have values, join them
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
