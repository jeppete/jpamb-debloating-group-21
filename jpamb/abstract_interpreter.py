from __future__ import annotations

from typing import Dict, Iterable, Tuple

import jpamb
from jpamb import jvm

from .ai_domain import SignSet, SignArithmetic
from .ai_state import Bytecode, PC, PerVarFrame, Stack


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _binop_name(opr: object) -> str:
    """
    Best-effort name for a BinaryOpr value.

    We avoid depending on the exact enum type from jpamb.jvm;
    instead we normalise whatever we get to a lowercase string
    and pattern-match on substrings like "add", "sub", "mul", "div".
    """
    name = getattr(opr, "name", None)
    if isinstance(name, str):
        return name.lower()
    return str(opr).lower()


def _cond_name(cond: object) -> str:
    """
    Similar helper for jvm.Condition values (Eq, Ne, Lt, Le, Gt, Ge, ...).
    """
    name = getattr(cond, "name", None)
    if isinstance(name, str):
        return name.lower()
    return str(cond).lower()


def _eval_zero_compare(cond: object, v: SignSet) -> set[bool]:
    """
    Given a condition (like Eq, Ne, Lt, Le, Gt, Ge) and an abstract
    value v over signs, compute the possible truth values of
        (v <op> 0)
    where <op> is the condition used by Ifz.

    This is still a may-analysis: if both True and False are
    possible, we return {True, False}.
    """
    cname = _cond_name(cond)
    possible: set[bool] = set()

    for s in v.signs:
        # For each concrete sign, see what the condition would do.
        if "eq" in cname:
            possible.add(s == "0")
        elif "ne" in cname:
            possible.add(s != "0")
        elif "lt" in cname:
            possible.add(s == "-")
        elif "le" in cname:
            possible.add(s in {"-", "0"})
        elif "gt" in cname:
            possible.add(s == "+")
        elif "ge" in cname:
            possible.add(s in {"+", "0"})
        else:
            # Unknown condition – be maximally conservative
            possible.update({True, False})

    # If for some weird reason we saw no signs, be conservative.
    if not possible:
        possible.update({True, False})
    return possible


# ---------------------------------------------------------------------------
# One abstract step
# ---------------------------------------------------------------------------


def astep(
    bc: Bytecode,
    frame: PerVarFrame[SignSet],
) -> Iterable[PerVarFrame[SignSet] | str]:
    """
    One abstract step from a single abstract frame.

    May yield:
      * zero or more successor frames, and/or
      * final strings like "ok" or "err:divide_by_zero".
    """
    pc = frame.pc
    op = bc[pc]

    match op:

        # ------------------------------------------------------------------
        # Constants / loads / stores
        # ------------------------------------------------------------------

        # Push integer constant
        case jvm.Push(type=jvm.Int(), value=v):
            new = frame.copy()
            new.stack.push(SignSet.const(int(v.value)))
            new.pc = pc.next()
            yield new

        # If your jpamb has Push without a 'type' field, this generic
        # fallback still treats integer-like constants as ints.
        case jvm.Push(value=v):
            new = frame.copy()
            # Very coarse: assume non-int constants are "unknown"
            if isinstance(v.type, jvm.Int):
                new.stack.push(SignSet.const(int(v.value)))
            else:
                # placeholder for refs / other primitives
                new.stack.push(SignSet.top())
            new.pc = pc.next()
            yield new

        # Load from local
        case jvm.Load(type=jvm.Int(), index=i):
            new = frame.copy()
            av = new.locals.get(i, SignSet.top())
            new.stack.push(av)
            new.pc = pc.next()
            yield new

        # Store to local
        case jvm.Store(type=jvm.Int(), index=i):
            new = frame.copy()
            av = new.stack.pop()
            new.locals[i] = av
            new.pc = pc.next()
            yield new

        # ------------------------------------------------------------------
        # Integer arithmetic
        # ------------------------------------------------------------------

        case jvm.Binary(type=jvm.Int(), operant=opr):
            new = frame.copy()
            right = new.stack.pop()
            left = new.stack.pop()

            op_name = _binop_name(opr)

            if "add" in op_name:
                res = SignArithmetic.add(left, right)
                new.stack.push(res)
                new.pc = pc.next()
                yield new

            elif "sub" in op_name:
                res = SignArithmetic.sub(left, right)
                new.stack.push(res)
                new.pc = pc.next()
                yield new

            elif "mul" in op_name:
                res = SignArithmetic.mul(left, right)
                new.stack.push(res)
                new.pc = pc.next()
                yield new

            elif "div" in op_name:
                res_signs, may_div_zero = SignArithmetic.div(left, right)
                if may_div_zero:
                    # One abstract path where we divide by zero
                    yield "err:divide_by_zero"
                if res_signs:
                    # Path where divisor != 0
                    n2 = new
                    n2.stack.push(res_signs)
                    n2.pc = pc.next()
                    yield n2

            else:
                # Any other int binop: approximate result as ⊤
                new.stack.push(SignSet.top())
                new.pc = pc.next()
                yield new

        # ------------------------------------------------------------------
        # Comparisons / branches
        # ------------------------------------------------------------------

        # Ifz with condition (==, !=, <, <=, >, >= 0)
        case jvm.Ifz(condition=cond, target=target_off):
            new = frame.copy()
            av = new.stack.pop()

            possible = _eval_zero_compare(cond, av)
            can_true = True in possible
            can_false = False in possible

            if can_true:
                t_frame = new.copy()
                t_frame.pc = PC(pc.method, target_off)
                yield t_frame

            if can_false:
                f_frame = new.copy()
                f_frame.pc = pc.next()
                yield f_frame

        # Very simple unconditional jump
        case jvm.Goto(target=target_off):
            new = frame.copy()
            new.pc = PC(pc.method, target_off)
            yield new

        # ------------------------------------------------------------------
        # Arrays (very coarse modelling)
        # ------------------------------------------------------------------
        #
        # NOTE: The exact opcode class names for arrays may differ in
        # jpamb.jvm. Check jpamb/jvm/opcode.py and adjust the 'case'
        # lines accordingly (NewArray, ArrayLoad, ArrayStore, ArrayLength).

        # new int[length]
        case jvm.NewArray(component=jvm.Int()):
            new = frame.copy()
            _len = new.stack.pop()         # length, SignSet
            # We don't track array contents precisely; push an opaque ref.
            new.stack.push(SignSet.top())  # placeholder for "some ref"
            new.pc = pc.next()
            yield new

        # x = arr[i]
        case jvm.ArrayLoad(component=jvm.Int()):
            new = frame.copy()
            _index = new.stack.pop()       # SignSet
            _arr = new.stack.pop()         # opaque ref
            # We have no idea about element sign -> ⊤
            new.stack.push(SignSet.top())
            new.pc = pc.next()
            yield new

        # arr[i] = x
        case jvm.ArrayStore(component=jvm.Int()):
            new = frame.copy()
            _val = new.stack.pop()
            _index = new.stack.pop()
            _arr = new.stack.pop()
            # No observable int result, just fall through.
            new.pc = pc.next()
            yield new

        # len = arr.length
        case jvm.ArrayLength():
            new = frame.copy()
            _arr = new.stack.pop()
            new.stack.push(SignSet.top())  # length is some non-negative int
            new.pc = pc.next()
            yield new

        # ------------------------------------------------------------------
        # Method calls (context-insensitive)
        # ------------------------------------------------------------------
        #
        # We *do not* inline or analyse the callee here. Instead, we:
        #   - pop arguments (and 'this' for instance methods),
        #   - push ⊤ for an int return value, or nothing for void / refs.

        case jvm.Invoke(method=method, access=access):
            new = frame.copy()

            # Figure out how many stack elements to pop.
            # `method.desc.args` is a list of types.
            desc = getattr(method, "desc", None)
            arg_types = list(getattr(desc, "args", []))
            returns = getattr(desc, "returns", None)

            nargs = len(arg_types)

            # For non-static calls, there's an extra receiver on the stack.
            access_name = getattr(access, "name", "").lower()
            if "static" not in access_name:
                nargs += 1

            for _ in range(nargs):
                new.stack.pop()

            # Push abstract return value if needed.
            if returns is not None:
                if isinstance(returns, jvm.Int):
                    new.stack.push(SignSet.top())
                else:
                    # Reference or void: we treat refs as opaque and
                    # don't track them for sign analysis.
                    pass

            new.pc = pc.next()
            yield new

        # ------------------------------------------------------------------
        # Returns
        # ------------------------------------------------------------------

        case jvm.Return(type=jvm.Int()):
            # Top frame returns "ok".
            # (If you later add full call-stack analysis, adapt this.)
            yield "ok"

        # ------------------------------------------------------------------
        # Default: unknown opcode
        # ------------------------------------------------------------------

        case _:
            # For any opcode we don't understand yet, we conservatively
            # leave locals/stack alone and just step to the next pc.
            # This is unsound in general, but keeps the analysis running.
            new = frame.copy()
            new.pc = pc.next()
            yield new


# ---------------------------------------------------------------------------
# Global 'many-step' and bounded run
# ---------------------------------------------------------------------------


def manystep(
    bc: Bytecode,
    state: Dict[PC, PerVarFrame[SignSet]],
) -> Tuple[Dict[PC, PerVarFrame[SignSet]], set[str]]:
    """
    One global abstract step over all currently reachable PCs.

    Returns (new_state, final_results).
    """
    new_state: Dict[PC, PerVarFrame[SignSet]] = {}
    finals: set[str] = set()

    for pc, frame in state.items():
        for out in astep(bc, frame):
            if isinstance(out, str):
                finals.add(out)
            else:
                pc2 = out.pc
                if pc2 in new_state:
                    new_state[pc2] = new_state[pc2] | out
                else:
                    new_state[pc2] = out

    return new_state, finals


def bounded_abstract_run(
    suite: jpamb.Suite,
    method: jvm.AbsMethodID,
    init_locals: Dict[int, SignSet] | None = None,
    max_steps: int = 100,
) -> set[str]:
    """
    Run bounded abstract interpretation on a single method.

    - `init_locals` maps local index -> initial abstract value.
      If omitted, we default everything to SignSet.top().
    - `max_steps` bounds the number of global iterations.

    Returns a set of possible final outcomes, e.g.:
        {"ok", "err:divide_by_zero"}
    """
    bc = Bytecode(suite)

    # build initial frame
    pc0 = PC(method, 0)
    locals0: Dict[int, SignSet] = dict(init_locals or {})
    stack0 = Stack.empty()
    frame0 = PerVarFrame(locals=locals0, stack=stack0, pc=pc0)

    state: Dict[PC, PerVarFrame[SignSet]] = {pc0: frame0}
    finals: set[str] = set()

    for _ in range(max_steps):
        if not state:
            break
        state, new_finals = manystep(bc, state)
        finals |= new_finals

    return finals

class AbstractInterpreter:
    """
    Simple wrapper around bounded_abstract_run so you can do:

        from solutions.abstract_interpreter import AbstractInterpreter
        ai = AbstractInterpreter(suite)
        results = ai.analyze(methodid)

    """

    def __init__(self, suite: jpamb.Suite, max_steps: int = 100):
        self.suite = suite
        self.max_steps = max_steps

    def analyze(
        self,
        method: jvm.AbsMethodID,
        init_locals: dict[int, SignSet] | None = None,
    ) -> set[str]:
        """
        Run the abstract interpreter on a single method.

        Returns the set of possible final outcomes, e.g.:
            {"ok", "err:divide_by_zero"}
        """
        return bounded_abstract_run(
            self.suite,
            method,
            init_locals=init_locals,
            max_steps=self.max_steps,
        )
