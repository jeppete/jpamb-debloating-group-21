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
      * final strings like "ok" or "divide by zero".
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
            next_pc = bc.next_pc(pc)
            if next_pc is None:
                return  # End of method
            new.pc = next_pc
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
            next_pc = bc.next_pc(pc)
            if next_pc is None:
                return  # End of method
            new.pc = next_pc
            yield new

        # Load from local
        case jvm.Load(type=jvm.Int(), index=i):
            new = frame.copy()
            av = new.locals.get(i, SignSet.top())
            new.stack.push(av)
            next_pc = bc.next_pc(pc)
            if next_pc is None:
                return  # End of method
            new.pc = next_pc
            yield new

        # Store to local
        case jvm.Store(type=jvm.Int(), index=i):
            new = frame.copy()
            # Defensive: ensure stack has value
            if len(new.stack) == 0:
                new.stack.push(SignSet.top())
            av = new.stack.pop()
            new.locals[i] = av
            next_pc = bc.next_pc(pc)
            if next_pc is None:
                return  # End of method
            new.pc = next_pc
            yield new

        # ------------------------------------------------------------------
        # Field access
        # ------------------------------------------------------------------

        # Get static field (getstatic)
        case jvm.Get(static=True, field=field):
            new = frame.copy()
            # For static field access, particularly $assertionsDisabled
            if field.extension.name == "$assertionsDisabled":
                # Java assertions are typically disabled, so return false
                # For ifz, false = 0, so we push {"0"}
                new.stack.push(SignSet.const(0))
            else:
                # Default to top for other static fields (unknown value)
                if isinstance(field.extension.type, jvm.Boolean):
                    # Boolean could be true or false, approximate as top
                    new.stack.push(SignSet.top())
                elif isinstance(field.extension.type, jvm.Int):
                    new.stack.push(SignSet.top())
                else:
                    # Other types: push top (unknown)
                    new.stack.push(SignSet.top())
            next_pc = bc.next_pc(pc)
            if next_pc is None:
                return  # End of method
            new.pc = next_pc
            yield new

        # Get instance field (getfield) - pops object ref, pushes field value
        case jvm.Get(static=False, field=field):
            new = frame.copy()
            # Defensive: ensure stack has value
            if len(new.stack) == 0:
                new.stack.push(SignSet.top())
            # Check for null pointer
            yield "null pointer"
            _obj_ref = new.stack.pop()  # Pop object reference
            # Push abstract value for field (we don't track field values precisely)
            if isinstance(field.extension.type, jvm.Int):
                new.stack.push(SignSet.top())
            elif isinstance(field.extension.type, jvm.Boolean):
                new.stack.push(SignSet.top())
            else:
                # Reference or other types: push top
                new.stack.push(SignSet.top())
            next_pc = bc.next_pc(pc)
            if next_pc is None:
                return  # End of method
            new.pc = next_pc
            yield new

        # ------------------------------------------------------------------
        # Integer arithmetic
        # ------------------------------------------------------------------

        case jvm.Binary(type=jvm.Int(), operant=opr):
            new = frame.copy()
            # Defensive: ensure stack has enough values
            if len(new.stack) < 2:
                # Push top values to make stack valid
                while len(new.stack) < 2:
                    new.stack.push(SignSet.top())
            right = new.stack.pop()
            left = new.stack.pop()

            op_name = _binop_name(opr)

            if "add" in op_name:
                res = SignArithmetic.add(left, right)
                new.stack.push(res)
                next_pc = bc.next_pc(pc)
                if next_pc is None:
                    return  # End of method
                new.pc = next_pc
                yield new

            elif "sub" in op_name:
                res = SignArithmetic.sub(left, right)
                new.stack.push(res)
                next_pc = bc.next_pc(pc)
                if next_pc is None:
                    return  # End of method
                new.pc = next_pc
                yield new

            elif "mul" in op_name:
                res = SignArithmetic.mul(left, right)
                new.stack.push(res)
                next_pc = bc.next_pc(pc)
                if next_pc is None:
                    return  # End of method
                new.pc = next_pc
                yield new

            elif "div" in op_name:
                res_signs, may_div_zero = SignArithmetic.div(left, right)
                if may_div_zero:
                    # One abstract path where we divide by zero
                    yield "divide by zero"
                if res_signs:
                    # Path where divisor != 0
                    n2 = new
                    n2.stack.push(res_signs)
                    next_pc = bc.next_pc(pc)
                    if next_pc is None:
                        return  # End of method
                    n2.pc = next_pc
                    yield n2

            else:
                # Any other int binop: approximate result as ⊤
                new.stack.push(SignSet.top())
                next_pc = bc.next_pc(pc)
                if next_pc is None:
                    return  # End of method
                new.pc = next_pc
                yield new

        # ------------------------------------------------------------------
        # Comparisons / branches
        # ------------------------------------------------------------------

        # Ifz with condition (==, !=, <, <=, >, >= 0)
        case jvm.Ifz(condition=cond, target=target_off):
            new = frame.copy()
            # Handle case where stack might be empty (from malformed bytecode/jump)
            if len(new.stack) == 0:
                # Push top value conservatively
                new.stack.push(SignSet.top())
            av = new.stack.pop()

            possible = _eval_zero_compare(cond, av)
            can_true = True in possible
            can_false = False in possible

            if can_true:
                t_frame = new.copy()
                # Resolve the jump target (in case it points to a gap/label)
                target_pc = PC(pc.method, target_off)
                resolved_op = bc[target_pc]
                resolved_pc = PC(pc.method, resolved_op.offset)
                t_frame.pc = resolved_pc
                # If jumping to an If instruction, ensure stack has required values
                if isinstance(resolved_op, jvm.If) and len(t_frame.stack) < 2:
                    while len(t_frame.stack) < 2:
                        t_frame.stack.push(SignSet.top())
                yield t_frame

            if can_false:
                f_frame = new.copy()
                next_pc = bc.next_pc(pc)
                if next_pc is None:
                    return  # End of method
                f_frame.pc = next_pc
                yield f_frame

        # If with condition comparing two values (if_icmp)
        case jvm.If(condition=cond, target=target_off):
            new = frame.copy()
            # Pop two values for comparison
            # Handle case where stack might be empty (from malformed bytecode/jump)
            if len(new.stack) < 2:
                # If stack doesn't have enough values, push top values conservatively
                # This handles cases where we jump to this instruction with wrong stack state
                while len(new.stack) < 2:
                    new.stack.push(SignSet.top())
            right = new.stack.pop()  # Second value (top of stack)
            left = new.stack.pop()   # First value

            # For if_icmp, we compare two abstract values
            # Since we're using sign domain, we can't precisely compare,
            # so we conservatively assume both branches are possible
            # unless we can prove otherwise
            can_true = True   # Conservative: assume condition might be true
            can_false = True  # Conservative: assume condition might be false

            # TODO: Could add more precise comparison logic here
            # For now, be maximally conservative

            if can_true:
                t_frame = new.copy()
                # Resolve the jump target (in case it points to a gap/label)
                target_pc = PC(pc.method, target_off)
                resolved_op = bc[target_pc]
                t_frame.pc = PC(pc.method, resolved_op.offset)
                yield t_frame

            if can_false:
                f_frame = new.copy()
                next_pc = bc.next_pc(pc)
                if next_pc is None:
                    return  # End of method
                f_frame.pc = next_pc
                yield f_frame

        # Very simple unconditional jump
        case jvm.Goto(target=target_off):
            new = frame.copy()
            # Resolve the jump target (in case it points to a gap/label)
            target_pc = PC(pc.method, target_off)
            resolved_op = bc[target_pc]
            resolved_pc = PC(pc.method, resolved_op.offset)
            new.pc = resolved_pc
            # If jumping to an If instruction, ensure stack has required values
            if isinstance(resolved_op, jvm.If) and len(new.stack) < 2:
                while len(new.stack) < 2:
                    new.stack.push(SignSet.top())
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
            # Defensive: ensure stack has value
            if len(new.stack) == 0:
                new.stack.push(SignSet.top())
            _len = new.stack.pop()         # length, SignSet
            # We don't track array contents precisely; push an opaque ref.
            new.stack.push(SignSet.top())  # placeholder for "some ref"
            next_pc = bc.next_pc(pc)
            if next_pc is None:
                return  # End of method
            new.pc = next_pc
            yield new

        # x = arr[i]
        case jvm.ArrayLoad(component=jvm.Int()):
            new = frame.copy()
            # Defensive: ensure stack has enough values
            if len(new.stack) < 2:
                while len(new.stack) < 2:
                    new.stack.push(SignSet.top())
            _index = new.stack.pop()       # SignSet
            _arr = new.stack.pop()         # opaque ref
            
            # Check for null pointer: array reference could be null
            # Since we don't track nullness precisely, conservatively assume it might be null
            yield "null pointer"
            
            # Also check for out of bounds: index could be negative or too large
            # Since we don't track array lengths precisely, conservatively assume OOB is possible
            yield "out of bounds"
            
            # If array is not null and index is in bounds, push element value
            # We have no idea about element sign -> ⊤
            new.stack.push(SignSet.top())
            next_pc = bc.next_pc(pc)
            if next_pc is None:
                return  # End of method
            new.pc = next_pc
            yield new

        # arr[i] = x
        case jvm.ArrayStore(component=jvm.Int()):
            new = frame.copy()
            # Defensive: ensure stack has enough values
            if len(new.stack) < 3:
                while len(new.stack) < 3:
                    new.stack.push(SignSet.top())
            _val = new.stack.pop()
            _index = new.stack.pop()
            _arr = new.stack.pop()
            
            # Check for null pointer and out of bounds (same as ArrayLoad)
            yield "null pointer"
            yield "out of bounds"
            
            # If array is not null and index is in bounds, store succeeds
            next_pc = bc.next_pc(pc)
            if next_pc is None:
                return  # End of method
            new.pc = next_pc
            yield new

        # len = arr.length
        case jvm.ArrayLength():
            new = frame.copy()
            # Defensive: ensure stack has value
            if len(new.stack) == 0:
                new.stack.push(SignSet.top())
            _arr = new.stack.pop()
            
            # Check for null pointer
            yield "null pointer"
            
            # If array is not null, push length
            new.stack.push(SignSet.top())  # length is some non-negative int
            next_pc = bc.next_pc(pc)
            if next_pc is None:
                return  # End of method
            new.pc = next_pc
            yield new

        # ------------------------------------------------------------------
        # Method calls (context-insensitive)
        # ------------------------------------------------------------------
        #
        # We *do not* inline or analyse the callee here. Instead, we:
        #   - pop arguments (and 'this' for instance methods),
        #   - push ⊤ for an int return value, or nothing for void / refs.

        case jvm.InvokeVirtual(method=method):
            new = frame.copy()
            nargs = len(method.extension.params) + 1  # +1 for receiver
            # Defensive: ensure stack has enough values
            if len(new.stack) < nargs:
                while len(new.stack) < nargs:
                    new.stack.push(SignSet.top())
            # Check for null pointer on receiver (first argument)
            yield "null pointer"
            # Pop arguments
            for _ in range(nargs):
                new.stack.pop()
            if method.extension.return_type is not None and isinstance(method.extension.return_type, jvm.Int):
                new.stack.push(SignSet.top())
            next_pc = bc.next_pc(pc)
            if next_pc is None:
                return  # End of method
            new.pc = next_pc
            yield new

        case jvm.InvokeStatic(method=method):
            new = frame.copy()
            nargs = len(method.extension.params)  # No receiver for static
            # Defensive: ensure stack has enough values
            if len(new.stack) < nargs:
                while len(new.stack) < nargs:
                    new.stack.push(SignSet.top())
            # Pop arguments
            for _ in range(nargs):
                new.stack.pop()
            if method.extension.return_type is not None and isinstance(method.extension.return_type, jvm.Int):
                new.stack.push(SignSet.top())
            next_pc = bc.next_pc(pc)
            if next_pc is None:
                return  # End of method
            new.pc = next_pc
            yield new

        case jvm.InvokeSpecial(method=method):
            new = frame.copy()
            nargs = len(method.extension.params) + 1  # +1 for receiver
            # Defensive: ensure stack has enough values
            if len(new.stack) < nargs:
                while len(new.stack) < nargs:
                    new.stack.push(SignSet.top())
            # Check for null pointer on receiver (first argument)
            yield "null pointer"
            # Pop arguments
            for _ in range(nargs):
                new.stack.pop()
            if method.extension.return_type is not None and isinstance(method.extension.return_type, jvm.Int):
                new.stack.push(SignSet.top())
            next_pc = bc.next_pc(pc)
            if next_pc is None:
                return  # End of method
            new.pc = next_pc
            yield new

        case jvm.InvokeInterface(method=method):
            new = frame.copy()
            nargs = len(method.extension.params) + 1  # +1 for receiver
            # Defensive: ensure stack has enough values
            if len(new.stack) < nargs:
                while len(new.stack) < nargs:
                    new.stack.push(SignSet.top())
            # Check for null pointer on receiver (first argument)
            yield "null pointer"
            # Pop arguments
            for _ in range(nargs):
                new.stack.pop()
            if method.extension.return_type is not None and isinstance(method.extension.return_type, jvm.Int):
                new.stack.push(SignSet.top())
            next_pc = bc.next_pc(pc)
            if next_pc is None:
                return  # End of method
            new.pc = next_pc
            yield new

        # ------------------------------------------------------------------
        # Object creation and manipulation
        # ------------------------------------------------------------------

        # Create new object (new)
        case jvm.New(classname=classname):
            new = frame.copy()
            # Handle creation of new objects, especially exceptions
            if classname.slashed() == "java/lang/AssertionError":
                yield "assertion error"
            else:
                # For other classes, push an opaque reference (top)
                new.stack.push(SignSet.top())
                next_pc = bc.next_pc(pc)
                if next_pc is None:
                    return  # End of method
                new.pc = next_pc
                yield new

        # Duplicate stack top (dup)
        case jvm.Dup():
            new = frame.copy()
            if new.stack:
                top = new.stack.peek()
                new.stack.push(top)
            else:
                # Empty stack - push top as fallback
                new.stack.push(SignSet.top())
            next_pc = bc.next_pc(pc)
            if next_pc is None:
                return  # End of method
            new.pc = next_pc
            yield new

        # Throw exception (throw)
        case jvm.Throw():
            # Throwing an exception terminates execution
            # Check if it's AssertionError by looking at what's on stack
            # For now, just yield assertion error if we're throwing
            yield "assertion error"

        # ------------------------------------------------------------------
        # Returns
        # ------------------------------------------------------------------

        case jvm.Return(type=jvm.Int()):
            # Top frame returns "ok".
            # (If you later add full call-stack analysis, adapt this.)
            yield "ok"

        case jvm.Return(type=None):  # Void return
            yield "ok"

        # ------------------------------------------------------------------
        # Default: unknown opcode
        # ------------------------------------------------------------------

        case _:
            # For any opcode we don't understand yet, we conservatively
            # leave locals/stack alone and just step to the next pc.
            # This is unsound in general, but keeps the analysis running.
            new = frame.copy()
            next_pc = bc.next_pc(pc)
            if next_pc is None:
                return  # End of method
            new.pc = next_pc
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