#!/usr/bin/env python3
"""
Debloater - Orchestrator for finding dead code.

Coordinates two pipelines:
1. Source pipeline: Syntactic analysis on source code
2. Bytecode pipeline: Multiple analyses on bytecode (sequential optimization)

Then maps bytecode results to source level and combines both.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional
from pathlib import Path

import jpamb
from jpamb import jvm

# Import analysis modules
from bytecode_analysis import BytecodeAnalyzer, AnalysisResult as BytecodeResult

log = logging.getLogger(__name__)


@dataclass
class SourceFinding:
    """A single finding from source-level analysis."""
    line: int
    kind: str  # "unused_import", "dead_branch", "unused_field", etc.
    message: str
    details: dict = field(default_factory=dict)


@dataclass
class SourceAnalysisResult:
    """Results from source-level syntactic analysis."""
    findings: List[SourceFinding] = field(default_factory=list)
    dead_lines: Set[int] = field(default_factory=set)
    
    def add_finding(self, line: int, kind: str, message: str, **details):
        """Add a finding from source analysis."""
        self.findings.append(SourceFinding(line, kind, message, details))
        self.dead_lines.add(line)


@dataclass
class MappedBytecodeResult:
    """Bytecode results mapped to source code level."""
    dead_lines: Set[int] = field(default_factory=set)
    dead_methods: Set[str] = field(default_factory=set)
    details: List[dict] = field(default_factory=list)


@dataclass
class CombinedResult:
    """Combined results from both pipelines."""
    suggestions: List[dict] = field(default_factory=list)
    by_line: Dict[int, List[dict]] = field(default_factory=dict)
    total_dead_lines: int = 0


class Debloater:
    """
    Main debloater orchestrator.
    
    Runs two parallel pipelines:
    1. Source pipeline: Fast syntactic analysis on Java source
    2. Bytecode pipeline: Multi-phase analysis on bytecode (optimized sequentially)
    
    Then combines results and presents to developer.
    """
    
    def __init__(self, suite: jpamb.Suite):
        self.suite = suite
        self.results = {}
    
    def analyze_class(self, classname: jvm.ClassName, source_file: Optional[Path] = None):
        """
        Run complete debloating analysis on a class.
        
        Args:
            classname: The class to analyze
            source_file: Optional path to source file for source analysis
        """
        log.info("="*70)
        log.info("DEBLOATER ANALYSIS: %s", classname)
        log.info("="*70)
        
        # ===================================================================
        # PIPELINE 1: Source Analysis (if source file provided)
        # ===================================================================
        if source_file and source_file.exists():
            log.info("\n[Pipeline 1] Source Analysis")
            source_result = self.run_source_pipeline(source_file)
            self.results['source'] = source_result
        else:
            log.info("\n[Pipeline 1] Source Analysis: Skipped (no source file)")
            self.results['source'] = SourceAnalysisResult()
        
        # ===================================================================
        # PIPELINE 2: Bytecode Analysis (multi-phase, optimized)
        # ===================================================================
        log.info("\n[Pipeline 2] Bytecode Analysis")
        bytecode_result = self.run_bytecode_pipeline(classname)
        self.results['bytecode'] = bytecode_result
        
        # ===================================================================
        # MAPPING: Bytecode offsets → Source lines
        # ===================================================================
        log.info("\n[Mapping] Bytecode results to source level")
        mapped_bytecode = self.map_bytecode_to_source(classname, bytecode_result)
        self.results['bytecode_mapped'] = mapped_bytecode
        
        # ===================================================================
        # COMBINE: Merge and deduplicate results
        # ===================================================================
        log.info("\n[Combine] Merging results from both pipelines")
        combined = self.combine_results(
            self.results['source'],
            mapped_bytecode
        )
        self.results['combined'] = combined
        
        # ===================================================================
        # REPORT: Present findings
        # ===================================================================
        self.report()
    
    def run_source_pipeline(self, source_file: Path) -> SourceAnalysisResult:
        """
        Run source-level syntactic analysis.
        
        TODO: Integrate your source syntaxer here
        
        Args:
            source_file: Path to .java file
            
        Returns:
            SourceAnalysisResult with findings
        """
        result = SourceAnalysisResult()
        
        # TODO: Call your source syntaxer
        # Example integration:
        # from syntaxer import BloatFinder
        # parser = create_java_parser()
        # tree = parser.parse(source_file.read_bytes())
        # finder = BloatFinder(tree, source_file.read_bytes())
        # 
        # for issue in finder.issues:
        #     result.add_finding(
        #         line=issue.line,
        #         kind=issue.kind,
        #         message=issue.message
        #     )
        
        log.info("  Source findings: %d", len(result.findings))
        return result
    
    def run_bytecode_pipeline(self, classname: jvm.ClassName) -> BytecodeResult:
        """
        Run multi-phase bytecode analysis with sequential optimization.
        
        Phase 1: Bytecode syntactic (CFG + call graph)
        Phase 2: [Future] Abstract interpretation (optimized by Phase 1)
        Phase 3: [Future] Data flow (optimized by Phase 1+2)
        
        Args:
            classname: Class to analyze
            
        Returns:
            BytecodeResult with all findings
        """
        # Phase 1: Bytecode Syntactic Analysis
        log.info("  Phase 1: CFG + Call Graph Analysis")
        bytecode_analyzer = BytecodeAnalyzer(self.suite)
        
        try:
            result = bytecode_analyzer.analyze_class(classname)
            log.info("    → Found %d unreachable methods", len(result.unreachable_methods))
            log.info("    → Found %d methods with dead instructions", 
                    len(result.dead_instructions))
        except Exception as e:
            log.error("  Bytecode analysis failed: %s", e)
            raise
        
        # Phase 2: Abstract Interpretation (TODO)
        # log.info("  Phase 2: Abstract Interpretation")
        # pruned_cfgs = self.prune_dead_code(result.cfgs, result.dead_instructions)
        # abstract_result = AbstractInterpreter().analyze(classname, pruned_cfgs)
        # Merge abstract_result into result
        
        # Phase 3: Data Flow Analysis (TODO)
        # log.info("  Phase 3: Data Flow Analysis")
        # ...
        
        return result
    
    def map_bytecode_to_source(self, classname: jvm.ClassName, 
                               bytecode_result: BytecodeResult) -> MappedBytecodeResult:
        """
        Map bytecode findings to source code locations.
        
        Uses line number tables in bytecode to map offsets → lines.
        
        Args:
            classname: Class being analyzed
            bytecode_result: Results from bytecode analysis
            
        Returns:
            MappedBytecodeResult with source-level locations
        """
        mapped = MappedBytecodeResult()
        
        try:
            cls = self.suite.findclass(classname)
            methods = cls.get("methods", [])
            
            # Map dead instructions to source lines
            for method_dict in methods:
                method_name = method_dict.get("name", "<unknown>")
                full_name = f"{classname}.{method_name}"
                
                if full_name not in bytecode_result.dead_instructions:
                    continue
                
                dead_offsets = bytecode_result.dead_instructions[full_name]
                code = method_dict.get("code", {})
                line_table = code.get("lines", [])
                
                for offset in dead_offsets:
                    line_num = self.offset_to_line(offset, line_table)
                    if line_num:
                        mapped.dead_lines.add(line_num)
                        mapped.details.append({
                            'line': line_num,
                            'method': full_name,
                            'offset': offset,
                            'type': 'dead_instruction',
                            'source': 'bytecode'
                        })
            
            # Map unreachable methods to source lines
            for method_dict in methods:
                method_name = method_dict.get("name", "<unknown>")
                full_name = f"{classname}.{method_name}"
                
                if full_name in bytecode_result.unreachable_methods:
                    # Get method's starting line
                    code = method_dict.get("code", {})
                    line_table = code.get("lines", [])
                    if line_table:
                        method_line = line_table[0].get("line")
                        mapped.dead_methods.add(full_name)
                        mapped.dead_lines.add(method_line)
                        mapped.details.append({
                            'line': method_line,
                            'method': full_name,
                            'type': 'unreachable_method',
                            'source': 'bytecode'
                        })
            
            log.info("  Mapped to %d source lines", len(mapped.dead_lines))
        
        except Exception as e:
            log.warning("  Mapping failed: %s", e)
        
        return mapped
    
    def offset_to_line(self, offset: int, line_table: List[dict]) -> Optional[int]:
        """
        Convert bytecode offset to source line number.
        
        Args:
            offset: Bytecode offset
            line_table: Line number table from bytecode
            
        Returns:
            Source line number, or None if not found
        """
        best_line = None
        for entry in line_table:
            entry_offset = entry.get("offset", -1)
            if entry_offset <= offset:
                best_line = entry.get("line")
            else:
                break
        return best_line
    
    def combine_results(self, source_result: SourceAnalysisResult,
                       bytecode_mapped: MappedBytecodeResult) -> CombinedResult:
        """
        Combine results from both pipelines, deduplicating by line number.
        
        Args:
            source_result: Findings from source analysis
            bytecode_mapped: Bytecode findings mapped to source
            
        Returns:
            CombinedResult with merged, deduplicated suggestions
        """
        combined = CombinedResult()
        seen_lines = set()
        
        # Add source findings
        for finding in source_result.findings:
            key = (finding.line, finding.kind)
            if key not in seen_lines:
                seen_lines.add(key)
                suggestion = {
                    'line': finding.line,
                    'type': finding.kind,
                    'message': finding.message,
                    'source': 'source_analysis',
                    'details': finding.details
                }
                combined.suggestions.append(suggestion)
                
                if finding.line not in combined.by_line:
                    combined.by_line[finding.line] = []
                combined.by_line[finding.line].append(suggestion)
        
        # Add bytecode findings (deduplicate by line)
        for detail in bytecode_mapped.details:
            line = detail['line']
            key = (line, 'bytecode_dead')
            
            if key not in seen_lines:
                seen_lines.add(key)
                suggestion = {
                    'line': line,
                    'type': detail['type'],
                    'message': f"{detail['method']}: offset {detail['offset']}",
                    'source': 'bytecode_analysis',
                    'details': detail
                }
                combined.suggestions.append(suggestion)
                
                if line not in combined.by_line:
                    combined.by_line[line] = []
                combined.by_line[line].append(suggestion)
        
        combined.total_dead_lines = len(combined.by_line)
        log.info("  Combined: %d unique suggestions", len(combined.suggestions))
        
        return combined
    
    def report(self):
        """Generate comprehensive report of all findings."""
        source_result = self.results.get('source')
        bytecode_result = self.results.get('bytecode')
        combined = self.results.get('combined')
        
        print("\n" + "="*70)
        print("DEBLOATING ANALYSIS REPORT")
        print("="*70)
        
        # Source pipeline results
        if source_result and source_result.findings:
            print(f"\n📝 Source Analysis: {len(source_result.findings)} findings")
            for finding in source_result.findings[:5]:  # Show first 5
                print(f"  Line {finding.line}: {finding.message}")
            if len(source_result.findings) > 5:
                print(f"  ... and {len(source_result.findings) - 5} more")
        else:
            print(f"\n📝 Source Analysis: No findings")
        
        # Bytecode pipeline results
        if bytecode_result:
            print(f"\n🔍 Bytecode Analysis:")
            print(f"  Unreachable methods: {len(bytecode_result.unreachable_methods)}")
            total_dead_inst = sum(len(offsets) for offsets in bytecode_result.dead_instructions.values())
            print(f"  Dead instructions: {total_dead_inst}")
        
        # Combined results
        if combined:
            print(f"\n✨ Combined Results:")
            print(f"  Total suggestions: {len(combined.suggestions)}")
            print(f"  Lines with dead code: {combined.total_dead_lines}")
            
            print(f"\n📋 Suggestions by Line:")
            for line in sorted(combined.by_line.keys())[:10]:  # Show first 10
                suggestions = combined.by_line[line]
                for sugg in suggestions:
                    source_icon = "📝" if sugg['source'] == 'source_analysis' else "🔍"
                    print(f"  {source_icon} Line {line}: {sugg['message']}")
            
            if combined.total_dead_lines > 10:
                print(f"  ... and {combined.total_dead_lines - 10} more lines")
        
        print("\n" + "="*70)


def main():
    """Main entry point."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    if len(sys.argv) < 2:
        print("Usage: debloater.py <classname> [source_file]")
        print("\nExample:")
        print("  python debloater.py jpamb.cases.Simple")
        print("  python debloater.py jpamb.cases.Simple src/main/java/jpamb/cases/Simple.java")
        print("\nRuns both source and bytecode analysis pipelines.")
        sys.exit(1)
    
    classname_str = sys.argv[1]
    source_file = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    
    # Parse classname
    parts = classname_str.split(".")
    classname = jvm.ClassName("/".join(parts))
    
    suite = jpamb.Suite()
    debloater = Debloater(suite)
    
    try:
        debloater.analyze_class(classname, source_file)
    except Exception as e:
        log.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
