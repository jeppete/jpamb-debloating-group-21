"""
solutions/cfg_builder.py

Control Flow Graph (CFG) construction from JVM bytecode.

This module builds CFGs from jvm2json bytecode output, handling:
- Basic block identification (leaders)
- Branch targets (if/ifz, goto, tableswitch)
- Exception handler edges
- Method invocations
- Return and throw instructions

DTU 02242 Program Analysis - Group 21
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jpamb.jvm import opcode as opc
from solutions.ir import CFGNode, BasicBlock, ExceptionHandler, NodeType


def classify_opcode(opcode: Any) -> NodeType:
    """
    Classify an opcode into a NodeType category.
    
    Args:
        opcode: Parsed opcode object from jpamb.jvm.opcode
        
    Returns:
        NodeType classification
    """
    tname = type(opcode).__name__
    
    # Map opcode class names to NodeTypes
    type_map = {
        "Push": NodeType.PUSH,
        "Load": NodeType.LOAD,
        "Store": NodeType.ASSIGN,
        "Binary": NodeType.BINARY,
        "Negate": NodeType.UNARY,
        "Cast": NodeType.UNARY,
        "Incr": NodeType.ASSIGN,
        "Dup": NodeType.DUP,
        "If": NodeType.BRANCH,
        "Ifz": NodeType.BRANCH,
        "Goto": NodeType.JUMP,
        "TableSwitch": NodeType.SWITCH,
        "Return": NodeType.RETURN,
        "Throw": NodeType.THROW,
        "InvokeVirtual": NodeType.INVOKE,
        "InvokeStatic": NodeType.INVOKE,
        "InvokeSpecial": NodeType.INVOKE,
        "InvokeInterface": NodeType.INVOKE,
        "New": NodeType.NEW,
        "NewArray": NodeType.NEW,
        "ArrayLoad": NodeType.ARRAY_ACCESS,
        "ArrayStore": NodeType.ARRAY_ACCESS,
        "ArrayLength": NodeType.ARRAY_ACCESS,
        "Get": NodeType.FIELD_ACCESS,
    }
    
    return type_map.get(tname, NodeType.OTHER)


class CFGBuilder:
    """
    Builds Control Flow Graph from JVM bytecode.
    
    The builder performs three passes:
    1. Parse opcodes and identify leaders (basic block starts)
    2. Build CFG nodes with successors
    3. Compute predecessors and basic blocks
    
    Example:
        builder = CFGBuilder(bytecode_list, exception_handlers)
        cfg = builder.build()
        blocks = builder.get_basic_blocks()
    """
    
    def __init__(
        self,
        bytecode: list[dict],
        exception_handlers: list[ExceptionHandler] = None
    ):
        """
        Initialize CFG builder.
        
        Args:
            bytecode: List of bytecode instruction dicts from jvm2json
            exception_handlers: List of exception handlers for the method
        """
        self.bytecode = bytecode
        self.exception_handlers = exception_handlers or []
        
        # Intermediate data structures
        self._opcodes: dict[int, Any] = {}  # pc -> parsed opcode
        self._pc_list: list[int] = []  # ordered list of PCs
        self._leaders: set[int] = set()  # basic block leaders
        self._cfg: dict[int, CFGNode] = {}  # final CFG
        self._basic_blocks: list[BasicBlock] = []
        
    def build(self) -> dict[int, CFGNode]:
        """
        Build the complete CFG.
        
        Returns:
            Dictionary mapping PC to CFGNode
        """
        if not self.bytecode:
            return {}
        
        # Pass 1: Parse opcodes and find leaders
        self._parse_opcodes()
        self._find_leaders()
        
        # Pass 2: Build CFG nodes with successors
        self._build_nodes()
        
        # Pass 3: Compute predecessors and assign basic blocks
        self._compute_predecessors()
        self._build_basic_blocks()
        
        return self._cfg
    
    def get_basic_blocks(self) -> list[BasicBlock]:
        """Get the computed basic blocks (after build())."""
        return self._basic_blocks
    
    def _parse_opcodes(self) -> None:
        """Parse all bytecode instructions into opcode objects."""
        for instr in self.bytecode:
            pc = instr["offset"]
            try:
                opcode = opc.Opcode.from_json(instr)
            except (NotImplementedError, KeyError) as e:
                # Create a placeholder for unsupported opcodes
                opcode = _PlaceholderOpcode(pc, instr.get("opr", "unknown"), instr)
            
            self._opcodes[pc] = opcode
            self._pc_list.append(pc)
        
        # Sort PC list
        self._pc_list.sort()
    
    def _find_leaders(self) -> None:
        """
        Identify basic block leaders.
        
        Leaders are:
        1. First instruction of method
        2. Target of any branch/jump
        3. Instruction following a branch/jump
        4. Exception handler entry points
        """
        if not self._pc_list:
            return
        
        # First instruction is always a leader
        self._leaders.add(self._pc_list[0])
        
        # Find targets and instructions after branches
        for i, pc in enumerate(self._pc_list):
            opcode = self._opcodes[pc]
            
            # Get branch/jump targets
            targets = self._get_branch_targets(opcode)
            
            if targets:
                # All targets are leaders
                self._leaders.update(targets)
                
                # Instruction after branch/jump is also a leader
                if i + 1 < len(self._pc_list):
                    next_pc = self._pc_list[i + 1]
                    # Only if this isn't an unconditional jump
                    if not isinstance(opcode, opc.Goto):
                        self._leaders.add(next_pc)
            
            # Instructions after gotos are leaders
            if isinstance(opcode, opc.Goto) and i + 1 < len(self._pc_list):
                self._leaders.add(self._pc_list[i + 1])
            
            # Instructions after returns/throws are leaders
            if isinstance(opcode, (opc.Return, opc.Throw)):
                if i + 1 < len(self._pc_list):
                    self._leaders.add(self._pc_list[i + 1])
        
        # Exception handler entry points are leaders
        for handler in self.exception_handlers:
            if handler.handler_pc in self._opcodes:
                self._leaders.add(handler.handler_pc)
    
    def _get_branch_targets(self, opcode: Any) -> list[int]:
        """Get all possible jump targets for an opcode."""
        targets = []
        
        if isinstance(opcode, (opc.If, opc.Ifz)):
            targets.append(opcode.target)
        elif isinstance(opcode, opc.Goto):
            targets.append(opcode.target)
        elif isinstance(opcode, opc.TableSwitch):
            targets.append(opcode.default)
            targets.extend(opcode.targets)
        
        return targets
    
    def _build_nodes(self) -> None:
        """Build CFG nodes with successor edges."""
        for i, pc in enumerate(self._pc_list):
            opcode = self._opcodes[pc]
            node_type = classify_opcode(opcode)
            
            # Get instruction string
            instr_str = str(opcode) if hasattr(opcode, '__str__') else repr(opcode)
            
            # Find exception handlers covering this instruction
            handlers = [
                h for h in self.exception_handlers
                if h.start_pc <= pc < h.end_pc
            ]
            
            # Create node
            node = CFGNode(
                pc=pc,
                opcode=opcode,
                instr_str=instr_str,
                node_type=node_type,
                is_leader=(pc in self._leaders),
                exception_handlers=handlers,
            )
            
            # Compute successors
            node.successors = self._compute_successors(i, opcode)
            
            # Add exception handler targets as successors
            for handler in handlers:
                if handler.handler_pc not in node.successors:
                    node.successors.append(handler.handler_pc)
            
            self._cfg[pc] = node
    
    def _compute_successors(self, index: int, opcode: Any) -> list[int]:
        """
        Compute successor PCs for an instruction.
        
        Handles:
        - Sequential flow (fallthrough)
        - Conditional branches (if/ifz)
        - Unconditional jumps (goto)
        - Switch statements
        - Returns and throws (no successors)
        """
        pc = self._pc_list[index]
        successors = []
        
        # Return and throw have no successors
        if isinstance(opcode, (opc.Return, opc.Throw)):
            return successors
        
        # Get next instruction PC (for fallthrough)
        next_pc = None
        if index + 1 < len(self._pc_list):
            next_pc = self._pc_list[index + 1]
        
        # Handle different instruction types
        if isinstance(opcode, (opc.If, opc.Ifz)):
            # Conditional branch: fallthrough + target
            if next_pc is not None:
                successors.append(next_pc)
            successors.append(opcode.target)
        
        elif isinstance(opcode, opc.Goto):
            # Unconditional jump: only target
            successors.append(opcode.target)
        
        elif isinstance(opcode, opc.TableSwitch):
            # Switch: default + all case targets
            successors.append(opcode.default)
            successors.extend(opcode.targets)
            # Remove duplicates while preserving order
            seen = set()
            unique = []
            for s in successors:
                if s not in seen:
                    seen.add(s)
                    unique.append(s)
            successors = unique
        
        else:
            # Sequential instruction: fallthrough
            if next_pc is not None:
                successors.append(next_pc)
        
        return successors
    
    def _compute_predecessors(self) -> None:
        """Compute predecessor edges from successors."""
        for pc, node in self._cfg.items():
            for succ_pc in node.successors:
                if succ_pc in self._cfg:
                    succ_node = self._cfg[succ_pc]
                    if pc not in succ_node.predecessors:
                        succ_node.predecessors.append(pc)
    
    def _build_basic_blocks(self) -> None:
        """
        Build basic blocks from leaders.
        
        A basic block contains:
        - A leader instruction
        - All following non-leader instructions until a terminator
        """
        # Sort leaders
        sorted_leaders = sorted(self._leaders)
        
        block_id = 0
        for i, leader_pc in enumerate(sorted_leaders):
            # Find end of this block
            if i + 1 < len(sorted_leaders):
                next_leader = sorted_leaders[i + 1]
            else:
                next_leader = float('inf')
            
            # Collect nodes in this block
            block_nodes = []
            for pc in self._pc_list:
                if pc >= leader_pc and pc < next_leader:
                    if pc in self._cfg:
                        node = self._cfg[pc]
                        node.basic_block_id = block_id
                        block_nodes.append(node)
            
            if block_nodes:
                block = BasicBlock(
                    block_id=block_id,
                    start_pc=block_nodes[0].pc,
                    end_pc=block_nodes[-1].pc,
                    nodes=block_nodes,
                )
                
                # Compute successor blocks
                last_node = block_nodes[-1]
                for succ_pc in last_node.successors:
                    if succ_pc in self._cfg:
                        succ_block_id = self._cfg[succ_pc].basic_block_id
                        if succ_block_id is not None:
                            if succ_block_id not in block.successor_blocks:
                                block.successor_blocks.append(succ_block_id)
                
                self._basic_blocks.append(block)
                block_id += 1
        
        # Second pass: compute predecessor blocks
        for block in self._basic_blocks:
            for succ_id in block.successor_blocks:
                for other in self._basic_blocks:
                    if other.block_id == succ_id:
                        if block.block_id not in other.predecessor_blocks:
                            other.predecessor_blocks.append(block.block_id)


@dataclass
class _PlaceholderOpcode:
    """Placeholder for unsupported opcodes."""
    offset: int
    opr: str
    raw: dict
    
    def __str__(self) -> str:
        return f"{self.opr} (unsupported)"
    
    def mnemonic(self) -> str:
        return self.opr


def build_cfg_from_json(method_json: dict) -> tuple[dict[int, CFGNode], list[BasicBlock]]:
    """
    Convenience function to build CFG from method JSON.
    
    Args:
        method_json: Method dict from jvm2json output
        
    Returns:
        Tuple of (cfg dict, basic_blocks list)
    """
    code = method_json.get("code", {})
    bytecode = code.get("bytecode", [])
    exceptions = code.get("exceptions", [])
    
    handlers = [ExceptionHandler.from_json(e) for e in exceptions]
    
    builder = CFGBuilder(bytecode, handlers)
    cfg = builder.build()
    blocks = builder.get_basic_blocks()
    
    return cfg, blocks
