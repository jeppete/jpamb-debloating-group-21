# solutions/abstract_interpreter.py
from __future__ import annotations

from typing import Dict, Iterable, Tuple

import jpamb
from jpamb import jvm

from solutions.abstract_domain import SignSet, SignArithmetic
from solutions.abstract_state import Bytecode, PC, PerVarFrame, Stack


def _name(obj: object) -> str:
    n = getattr(obj, "name", None)
    return n.lower() if isinstance(n, str) else str(obj).lower()


def _eval_zero_compare(cond: object | None, v: SignSet) -> set[bool]:
    """
    Evaluate a zero comparison against a SignSet.

    Returns the set of booleans that are *possible* at runtime.
    """
    if cond is None:
        return {True, False}

    cname = _name(cond)
    out: set[bool] = set()

    # If v is ⊥ we treat it as “unknown” here, both branches possible
    for s in v.signs or {"+", "0", "-"}:
        if "eq" in cname:
            out.add(s == "0")
        elif "ne" in cname:
            out.add(s != "0")
        elif "lt" in cname:
            out.add(s == "-")
        elif "le" in cname:
            out.add(s in {"-", "0"})
        elif "gt" in cname:
            out.add(s == "+")
        elif "ge" in cname:
            out.add(s in {"+", "0"})
        else:
            # Unknown condition object, conservative
            return {True, False}

    return out or {True, False}



def astep(
    bc: Bytecode,
    frame: PerVarFrame[SignSet],
) -> Iterable[PerVarFrame[SignSet] | str]:
    """
    One abstract step from a single abstract frame.

    Yields successor frames and/or final outcome strings:

        "ok",
        "divide by zero",
        "null pointer",
        "out of bounds",
        "assertion error"

    This function is side‑effect free on the input frame, so it can be
    safely reused by the unbounded (fixpoint) analysis.
    """
    pc = frame.pc
    try:
        op = bc[pc]
    except IndexError:
        # Stepping into nowhere terminates this path silently
        return

    tname = type(op).__name__

    # ---------- PUSH ----------
    if tname == "Push":
        n = frame.copy()
        v = getattr(op, "value", None)
        inner = getattr(v, "value", None)  # jpamb may wrap constants

        if isinstance(v, int):
            n.stack.push(SignSet.const(v))
        elif isinstance(inner, int):
            n.stack.push(SignSet.const(int(inner)))
        else:
            # Non‑integral constants are abstracted as ⊤ in the sign domain
            n.stack.push(SignSet.top())

        nxt = bc.next_pc(pc)
        if nxt is not None:
            yield n.with_pc(nxt)
        return

    # ---------- LOAD ----------
    if tname == "Load":
        n = frame.copy()
        idx = int(getattr(op, "index", 0))
        n.stack.push(n.locals.get(idx, SignSet.top()))

        nxt = bc.next_pc(pc)
        if nxt is not None:
            yield n.with_pc(nxt)
        return

    # ---------- STORE ----------
    if tname == "Store":
        n = frame.copy()
        idx = int(getattr(op, "index", 0))
        val = n.stack.pop() if n.stack else SignSet.top()
        n.locals[idx] = val

        nxt = bc.next_pc(pc)
        if nxt is not None:
            yield n.with_pc(nxt)
        return

    # ---------- INCR (iinc) ----------
    if tname == "Incr":
        n = frame.copy()
        idx = int(getattr(op, "index", getattr(op, "var", 0)))
        delta = int(getattr(op, "value", getattr(op, "amount", getattr(op, "delta", 0))))
        base = n.locals.get(idx, SignSet.top())
        n.locals[idx] = SignArithmetic.add(base, SignSet.const(delta))

        nxt = bc.next_pc(pc)
        if nxt is not None:
            yield n.with_pc(nxt)
        return

    # ---------- CAST ----------
    if tname == "Cast":
        n = frame.copy()
        val = n.stack.pop() if n.stack else SignSet.top()
        to_ty = getattr(op, "type", getattr(op, "to", None))

        # Int cast: keep the same sign info
        if isinstance(to_ty, jvm.Int):
            out = val
        # Boolean cast: concrete values are 0 or 1, so signs ⊆ {0,+}
        elif isinstance(to_ty, jvm.Boolean):
            zero = "0" in val.signs
            truthy = any(s != "0" for s in val.signs) or val.is_top()
            signs: set[str] = set()
            if zero:
                signs.add("0")
            if truthy:
                signs.add("+")
            out = SignSet(frozenset(signs)) if signs else SignSet.top()
        else:
            # Other casts are irrelevant to the sign domain
            out = SignSet.top()

        n.stack.push(out)
        nxt = bc.next_pc(pc)
        if nxt is not None:
            yield n.with_pc(nxt)
        return

    # ---------- NEGATE ----------
    if tname == "Negate":
        n = frame.copy()
        v = n.stack.pop() if n.stack else SignSet.top()
        n.stack.push(SignArithmetic.neg(v))

        nxt = bc.next_pc(pc)
        if nxt is not None:
            yield n.with_pc(nxt)
        return

    # ---------- BINARY ----------
    if tname == "Binary":
        n = frame.copy()

        # Ensure stack height consistent with the JVM verifier
        while len(n.stack) < 2:
            n.stack.push(SignSet.top())

        right = n.stack.pop()
        left = n.stack.pop()
        kind = _name(getattr(op, "operant", None))

        # Helper flags for division / remainder.
        right_definitely_zero = right.signs == {"0"}

        if "add" in kind:
            n.stack.push(SignArithmetic.add(left, right))

        elif "sub" in kind:
            n.stack.push(SignArithmetic.sub(left, right))

        elif "mul" in kind:
            n.stack.push(SignArithmetic.mul(left, right))

        elif "div" in kind:
            res, dz = SignArithmetic.div(left, right)
            if dz:
                # Always report potential division by zero
                yield "divide by zero"
            # Only keep the normal successor if there is a non‑exceptional case
            if not right_definitely_zero and res:
                n.stack.push(res)
                nxt = bc.next_pc(pc)
                if nxt is not None:
                    yield n.with_pc(nxt)
            return

        elif "rem" in kind:
            # For remainder we treat “b == 0” like division
            if right_definitely_zero:
                yield "divide by zero"
                return

            res, dz = SignArithmetic.rem(left, right)
            if dz:
                yield "divide by zero"
            if res:
                n.stack.push(res)
            else:
                n.stack.push(SignSet.top())

        else:
            # Unknown operator, conservative
            n.stack.push(SignSet.top())

        nxt = bc.next_pc(pc)
        if nxt is not None:
            yield n.with_pc(nxt)
        return

    # ---------- IFZ (compare with 0) ----------
    if tname == "Ifz":
        # We must pop the condition in all successors
        n = frame.copy()
        v = n.stack.pop() if n.stack else SignSet.top()
        cond = getattr(op, "condition", None)
        poss = _eval_zero_compare(cond, v)

        target = getattr(op, "target", None)
        if True in poss and target is not None:
            t = n.with_pc(PC(pc.method, int(target)))
            yield t

        if False in poss:
            nxt = bc.next_pc(pc)
            if nxt is not None:
                yield n.with_pc(nxt)
        return

    # ---------- IF (icmp) ----------
    if tname == "If":
        # In the sign domain we cannot say much about comparisons of two
        # arbitrary integers, so both branches are conservatively possible
        n = frame.copy()
        while len(n.stack) < 2:
            n.stack.push(SignSet.top())
        n.stack.pop()
        n.stack.pop()

        target = getattr(op, "target", None)
        if target is not None:
            t = n.with_pc(PC(pc.method, int(target)))
            yield t

        nxt = bc.next_pc(pc)
        if nxt is not None:
            yield n.with_pc(nxt)
        return

    # ---------- GOTO ----------
    if tname == "Goto":
        n = frame.copy()
        target = getattr(op, "target", None)
        if target is not None:
            yield n.with_pc(PC(pc.method, int(target)))
        else:
            nxt = bc.next_pc(pc)
            if nxt is not None:
                yield n.with_pc(nxt)
        return

    # ---------- GET (fields) ----------
    if tname == "Get":
        n = frame.copy()
        static = bool(getattr(op, "static", False))
        field = getattr(op, "field", None)

        # Instance fields: pop receiver (may be null).
        if not static:
            if not n.stack:
                n.stack.push(SignSet.top())
            # Receiver might be null, we cannot track nullness in this
            # domain so we always consider the NPE possible
            yield "null pointer"
            _recv = n.stack.pop()

        # Push an abstract value for the field.
        ftype = getattr(getattr(field, "extension", field), "type", None)
        if isinstance(ftype, jvm.Int):
            n.stack.push(SignSet.top())
        elif isinstance(ftype, jvm.Boolean):
            # Booleans are 0/1 → {0,+}
            n.stack.push(SignSet(frozenset({"+", "0"})))
        else:
            n.stack.push(SignSet.top())

        nxt = bc.next_pc(pc)
        if nxt is not None:
            yield n.with_pc(nxt)
        return

    # ---------- ARRAYS (coarse) ----------
    if tname == "NewArray":
        n = frame.copy()
        if n.stack:
            _len = n.stack.pop()   # length (sign not relevant)
        n.stack.push(SignSet.top())  # array reference
        nxt = bc.next_pc(pc)
        if nxt is not None:
            yield n.with_pc(nxt)
        return

    if tname == "ArrayLoad":
        n = frame.copy()
        while len(n.stack) < 2:
            n.stack.push(SignSet.top())
        _idx = n.stack.pop()
        _arr = n.stack.pop()
        # Array may be null or index may be out of bounds
        yield "null pointer"
        yield "out of bounds"
        n.stack.push(SignSet.top())
        nxt = bc.next_pc(pc)
        if nxt is not None:
            yield n.with_pc(nxt)
        return

    if tname == "ArrayStore":
        n = frame.copy()
        while len(n.stack) < 3:
            n.stack.push(SignSet.top())
        _val = n.stack.pop()
        _idx = n.stack.pop()
        _arr = n.stack.pop()
        yield "null pointer"
        yield "out of bounds"
        nxt = bc.next_pc(pc)
        if nxt is not None:
            yield n.with_pc(nxt)
        return

    if tname == "ArrayLength":
        n = frame.copy()
        if n.stack:
            _arr = n.stack.pop()
        yield "null pointer"
        # Array length is ≥ 0, but we just approximate with ⊤ here
        n.stack.push(SignSet.top())
        nxt = bc.next_pc(pc)
        if nxt is not None:
            yield n.with_pc(nxt)
        return

    # ---------- INVOKEs (coarse, context‑insensitive) ----------
    if tname.startswith("Invoke"):
        n = frame.copy()

        # Extract arg types/return type robustly across jpamb versions
        def _desc(m):
            d = getattr(m, "desc", None)
            if d is not None:
                return list(getattr(d, "args", [])), getattr(d, "returns", None)
            ext = getattr(m, "extension", None)
            if ext is not None:
                return list(getattr(ext, "params", [])), getattr(ext, "return_type", None)
            return [], None

        method_obj = getattr(op, "method", None)
        arg_types, returns = _desc(method_obj)
        nargs = len(arg_types)

        # Non‑static invocations have a receiver on the stack
        access = getattr(op, "access", None)
        access_name = _name(access) if access is not None else ""
        is_static = "static" in access_name or tname == "InvokeStatic"
        if not is_static:
            nargs += 1  # receiver + args

        # Pop arguments (and receiver, if any).  If the stack is too
        # small we pad with ⊤ to keep heights consistent
        while len(n.stack) < nargs:
            n.stack.push(SignSet.top())
        for _ in range(nargs):
            n.stack.pop()

        # Receiver could be null.
        if not is_static:
            yield "null pointer"

        # Push abstract return value if any.
        if returns is not None and isinstance(returns, jvm.Int):
            n.stack.push(SignSet.top())

        nxt = bc.next_pc(pc)
        if nxt is not None:
            yield n.with_pc(nxt)
        return

    # ---------- NEW ----------
    if tname == "New":
        n = frame.copy()
        cname = getattr(op, "classname", None)

        # If the new object is AssertionError, some patterns immediately
        # throw it.  We let the Throw instruction handle the actual error
        # reporting, so here we only keep the stack well‑formed
        if cname is not None and getattr(cname, "slashed", lambda: "")() == "java/lang/AssertionError":
            # No push; Throw will terminate the path
            pass
        else:
            n.stack.push(SignSet.top())

        nxt = bc.next_pc(pc)
        if nxt is not None:
            yield n.with_pc(nxt)
        return

    # ---------- DUP ----------
    if tname == "Dup":
        n = frame.copy()
        if n.stack:
            n.stack.push(n.stack.peek())
        else:
            n.stack.push(SignSet.top())
        nxt = bc.next_pc(pc)
        if nxt is not None:
            yield n.with_pc(nxt)
        return

    # ---------- THROW ----------
    if tname == "Throw":
        # We conservatively treat all throws as assertion failures for the
        # purposes of this project
        yield "assertion error"
        return

    # ---------- RETURN ----------
    if tname.endswith("Return") or tname == "Return":
        # Any kind of return is treated as normal termination
        yield "ok"
        return

    # ---------- default: unknown opcode → fallthrough ----------
    n = frame.copy()
    nxt = bc.next_pc(pc)
    if nxt is not None:
        yield n.with_pc(nxt)


# ---------------------------------------------------------------------------
# Many‑step and bounded run
# ---------------------------------------------------------------------------


def manystep(
    bc: Bytecode,
    state: Dict[PC, PerVarFrame[SignSet]],
) -> Tuple[Dict[PC, PerVarFrame[SignSet]], set[str]]:
    """
    For every abstract frame in the current state we take a single abstract
    step, join frames that land at the same program counter, and collect
    any final outcomes encountered along the way
    """
    new_state: Dict[PC, PerVarFrame[SignSet]] = {}
    finals: set[str] = set()

    for pc, frame in state.items():
        for out in astep(bc, frame):
            if isinstance(out, str):
                finals.add(out)
            else:
                k = out.pc
                if k in new_state:
                    new_state[k] = new_state[k] | out
                else:
                    new_state[k] = out

    return new_state, finals


def bounded_abstract_run(
    suite: jpamb.Suite,
    method: jvm.AbsMethodID,
    init_locals: Dict[int, SignSet] | None = None,
    max_steps: int = 100,
) -> set[str]:
    """
    The unbounded static analysis will reuse the same `astep` / `manystep`
    primitives but iterate them to a fixpoint instead of truncating at
    max_steps
    """
    bc = Bytecode(suite)
    start = PC(method, 0)
    init = PerVarFrame(locals=dict(init_locals or {}), stack=Stack.empty(), pc=start)
    state: Dict[PC, PerVarFrame[SignSet]] = {start: init}

    finals: set[str] = set()
    for _ in range(max_steps):
        if not state:
            break
        state, step_finals = manystep(bc, state)
        finals |= step_finals

    return finals


class AbstractInterpreter:
    """
    Thin OO wrapper to mirror the concrete interpreter interface
    """

    def __init__(self, suite: jpamb.Suite, max_steps: int = 100) -> None:
        self.suite = suite
        self.max_steps = max_steps

    def analyze(self, method: jvm.AbsMethodID, init_locals: Dict[int, SignSet] | None = None) -> set[str]:
        return bounded_abstract_run(self.suite, method, init_locals=init_locals, max_steps=self.max_steps)
