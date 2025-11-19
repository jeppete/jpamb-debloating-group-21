from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, Set, Tuple

import jpamb
from jpamb import jvm

from jpamb.ai_domain import SignSet
from jpamb.ai_state import Bytecode, PC, PerVarFrame, Stack
from jpamb.abstract_interpreter import astep, _eval_zero_compare


# ---------------------------------------------------------------------------
# Result object
# ---------------------------------------------------------------------------


@dataclass
class UnboundedResult:
    """
    Result of unbounded static analysis for a whole suite (per method).

    - states[method][pc] = joined abstract frame at that pc
    - finals[method]     = set of final outcomes, e.g. {"ok", "err:divide_by_zero"}

    Helper methods let you ask about reachability and potential bloat.
    """
    states: Dict[jvm.AbsMethodID, Dict[PC, PerVarFrame[SignSet]]]
    finals: Dict[jvm.AbsMethodID, Set[str]]
    bc: Bytecode

    # ---- basic queries ----------------------------------------------------

    def method_state(self, method: jvm.AbsMethodID) -> Dict[PC, PerVarFrame[SignSet]]:
        return self.states.get(method, {})

    def method_finals(self, method: jvm.AbsMethodID) -> Set[str]:
        return self.finals.get(method, set())

    def reachable_pcs(self, method: jvm.AbsMethodID) -> Set[PC]:
        """
        All PCs that are reachable at fixpoint.
        """
        return set(self.method_state(method).keys())

    # ---- dead instructions -----------------------------------------------

    def _all_method_pcs(self, method: jvm.AbsMethodID) -> Set[PC]:
        """
        Helper: enumerate all PCs belonging to `method`.

        Gets all instruction offsets by loading the opcodes for the method.
        Uses actual bytecode offsets, not array indices.
        """
        pcs: Set[PC] = set()
        
        # Get all opcodes for this method and use their actual offsets
        for opcode in self.bc.suite.method_opcodes(method):
            pcs.add(PC(method, opcode.offset))

        return pcs

    def dead_instructions(self, method: jvm.AbsMethodID) -> Set[PC]:
        """
        PCs that are never reachable in the fixpoint (provably dead code).
        """
        all_pcs = self._all_method_pcs(method)
        reachable = self.reachable_pcs(method)
        return all_pcs - reachable

    # ---- dead branch edges -----------------------------------------------

    def dead_branch_edges(self, method: jvm.AbsMethodID) -> Set[Tuple[PC, PC]]:
        """
        Returns edges (from_pc, to_pc) that are infeasible in the fixpoint
        due to the Ifz condition always being true or always being false.

        NOTE: this only looks at local control-flow off Ifz; it does not
        build a full CFG. It is enough to mark obviously dead successors.
        """
        state = self.method_state(method)
        dead_edges: Set[Tuple[PC, PC]] = set()

        for pc, frame in state.items():
            op = self.bc[pc]

            match op:
                case jvm.Ifz(condition=cond, target=target_off):
                    # We *don't* want to mutate the frame, so we peek.
                    # Adjust `peek()` if your Stack API differs.
                    if not frame.stack:
                        # Very defensive; if analysis messed up stack, assume nothing.
                        continue

                    av: SignSet = frame.stack.peek()

                    possible = _eval_zero_compare(cond, av)
                    can_true = True in possible
                    can_false = False in possible

                    true_pc = PC(pc.method, target_off)
                    false_pc = self.bc.next_pc(pc)
                    if false_pc is None:
                        continue  # End of method, skip this edge

                    if not can_true:
                        dead_edges.add((pc, true_pc))
                    if not can_false:
                        dead_edges.add((pc, false_pc))

        return dead_edges


# ---------------------------------------------------------------------------
# Core unbounded analysis (per method)
# ---------------------------------------------------------------------------


def unbounded_abstract_run_method(
    bc: Bytecode,
    method: jvm.AbsMethodID,
    init_locals: Dict[int, SignSet] | None = None,
    max_iterations: int = 10000,
) -> Tuple[Dict[PC, PerVarFrame[SignSet]], Set[str]]:
    """
    Unbounded abstract interpretation (sign domain) for a single method,
    using a worklist and the `astep` abstract semantics.

    Returns:
        (state, finals) where
          - state[pc] = joined abstract frame at that pc
          - finals    = set of final outcomes, like {"ok", "err:divide_by_zero"}
                        or {"*"} if method doesn't terminate
    """
    # Initial abstract frame at pc = 0
    pc0 = PC(method, 0)
    locals0: Dict[int, SignSet] = dict(init_locals or {})
    stack0 = Stack.empty()
    frame0 = PerVarFrame(locals=locals0, stack=stack0, pc=pc0)

    state: Dict[PC, PerVarFrame[SignSet]] = {pc0: frame0}
    finals: Set[str] = set()

    # Classic worklist algorithm over PCs
    worklist: deque[PC] = deque([pc0])
    iterations = 0

    while worklist:
        iterations += 1
        if iterations > max_iterations:
            # Method doesn't terminate - report "*"
            return state, {"*"}
        
        pc = worklist.popleft()
        frame = state[pc]

        # Apply one abstract step from this instruction
        for out in astep(bc, frame):
            if isinstance(out, str):
                # Final outcomes like "ok", "divide by zero"
                finals.add(out)
                continue

            pc2 = out.pc
            old = state.get(pc2)

            if old is None:
                # First time we reach this instruction
                state[pc2] = out
                worklist.append(pc2)
            else:
                # Join new info into existing abstract frame
                joined = old | out   # relies on PerVarFrame.__or__ as join
                if joined != old:    # state at pc2 actually gained information
                    state[pc2] = joined
                    worklist.append(pc2)

    # If we completed but found no final outcomes, check if there are any
    # reachable return instructions. If not, the method doesn't terminate.
    if not finals:
        # Check if any reachable PC is a return instruction
        has_return = False
        for pc in state.keys():
            try:
                op = bc[pc]
                if isinstance(op, jvm.Return):
                    has_return = True
                    break
            except:
                pass
        
        # If no return is reachable, method doesn't terminate
        if not has_return:
            finals.add("*")
        else:
            # Method has return but no outcomes - this shouldn't happen,
            # but be conservative and assume it terminates normally
            finals.add("ok")
    
    return state, finals


# ---------------------------------------------------------------------------
# Suite-level unbounded analyzer
# ---------------------------------------------------------------------------


class UnboundedStaticAnalyzer:
    """
    Unbounded static analyzer built on top of the sign-based abstract
    interpreter (`astep`).

    Usage:

        suite = jpamb.suite(...)
        uba = UnboundedStaticAnalyzer(suite)

        # Analyse a single method:
        result = uba.analyze_method(methodid)
        dead = result.dead_instructions(methodid)

    Or:

        # Analyse a whole suite:
        result = uba.analyze_all()
        for m in result.states:
            print(m, result.dead_instructions(m))
    """

    def __init__(self, suite: jpamb.Suite):
        self.suite = suite
        self.bc = Bytecode(suite)

    # ---- entrypoints ------------------------------------------------------

    def analyze_method(
        self,
        method: jvm.AbsMethodID,
        init_locals: Dict[int, SignSet] | None = None,
    ) -> UnboundedResult:
        """
        Run unbounded analysis for a single method and return a Result
        object that still knows about the whole suite / bytecode.
        """
        state, finals = unbounded_abstract_run_method(self.bc, method, init_locals)

        return UnboundedResult(
            states={method: state},
            finals={method: finals},
            bc=self.bc,
        )

    def analyze_all(
        self,
        init_locals_for: Dict[jvm.AbsMethodID, Dict[int, SignSet]] | None = None,
    ) -> UnboundedResult:
        """
        Run unbounded analysis for all methods in the suite.

        `init_locals_for`, if given, maps method -> initial locals dict;
        otherwise we assume empty locals / all Top.
        """
        all_states: Dict[jvm.AbsMethodID, Dict[PC, PerVarFrame[SignSet]]] = {}
        all_finals: Dict[jvm.AbsMethodID, Set[str]] = {}

        init_locals_for = init_locals_for or {}

        # You may need to adjust this to however jpamb exposes all methods.
        for method in self._iter_all_methods():
            init_locals = init_locals_for.get(method)
            state, finals = unbounded_abstract_run_method(self.bc, method, init_locals)
            all_states[method] = state
            all_finals[method] = finals

        return UnboundedResult(states=all_states, finals=all_finals, bc=self.bc)

    # ---- helper to get all methods in the suite --------------------------

    def _iter_all_methods(self) -> Iterable[jvm.AbsMethodID]:
        """
        Enumerate all methods in the suite.

        Uses case_methods() to get all methods that have test cases.
        """
        for method, _ in self.suite.case_methods():
            yield method


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: unbound-static.py <method_id>", file=sys.stderr)
        print("Example: unbound-static.py 'jpamb.cases.Simple.divideByZero:()I'", file=sys.stderr)
        sys.exit(1)
    
    method_str = sys.argv[1]
    
    try:
        methodid = jvm.AbsMethodID.decode(method_str)
    except Exception as e:
        print(f"Error parsing method ID: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Run analysis
    suite = jpamb.Suite()
    analyzer = UnboundedStaticAnalyzer(suite)
    result = analyzer.analyze_method(methodid)
    
    # Print results
    print(f"Method: {methodid}")
    print("=" * 60)
    
    # Reachable PCs
    reachable = result.reachable_pcs(methodid)
    print(f"\nReachable PCs: {len(reachable)}")
    
    # Final outcomes
    finals = result.method_finals(methodid)
    if finals:
        print(f"\nFinal outcomes: {', '.join(sorted(finals))}")
    
    # Dead instructions
    dead = result.dead_instructions(methodid)
    print(f"\nDead instructions: {len(dead)}")
    if dead:
        print("Dead PCs:")
        for pc in sorted(dead, key=lambda p: p.offset):
            print(f"  - {pc}")
    
    # Dead branch edges
    dead_edges = result.dead_branch_edges(methodid)
    print(f"\nDead branch edges: {len(dead_edges)}")
    if dead_edges:
        print("Dead edges:")
        for from_pc, to_pc in sorted(dead_edges, key=lambda e: (e[0].offset, e[1].offset)):
            print(f"  - {from_pc} -> {to_pc}")
