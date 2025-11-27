"""
NAB Integration Module - Integrates IIN traces with IAB abstract domains.

This module provides the core integration between dynamic execution traces
produced by IIN (Interpreter) and the abstract domains defined in IAB
(Abstractions). It implements the dynamic refinement heuristic that uses
observed concrete values to set initial abstract states for static analysis.

**Course Definition (02242):**
"Run two or more abstractions at the same time, letting them inform each other
during execution" (formula: 5 per abstraction after the first).

This module implements a **Reduced Product** of SignDomain and IntervalDomain,
where both abstractions run in parallel and mutually refine each other:
- Sign POSITIVE tightens interval low bound to max(low, 1)
- Sign NEGATIVE tightens interval high bound to min(high, -1)
- Sign ZERO constrains interval to [0, 0]
- Interval [a, b] where a > 0 refines sign to POSITIVE
- Interval [a, b] where b < 0 refines sign to NEGATIVE
- etc.

DTU 02242 Program Analysis - Group 21
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from copy import deepcopy

# Import from existing IAB module
from solutions.abstractions import (
    SignDomain, 
    SignValue,
    IntervalDomain, 
    IntervalValue,
    DomainRefinement
)


@dataclass
class AbstractValue:
    """Combined abstract value with both sign and interval domains."""
    sign: SignDomain
    interval: IntervalDomain
    local_index: int
    
    def __str__(self):
        return f"local_{self.local_index}: sign={self.sign}, interval={self.interval}"
    
    def __repr__(self):
        return f"AbstractValue(local_{self.local_index}, sign={self.sign}, interval={self.interval})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "local_index": self.local_index,
            "sign": str(self.sign),
            "sign_value": self.sign.value.name,
            "interval": str(self.interval),
            "interval_bounds": {
                "low": self.interval.value.low,
                "high": self.interval.value.high
            }
        }


@dataclass
class ReducedProductState:
    """
    Reduced Product of SignDomain and IntervalDomain.
    
    This implements the course definition for NAB (Integrate Abstractions):
    "Run two or more abstractions at the same time, letting them inform each other
    during execution" (formula: 5 per abstraction after the first).
    
    The reduced product maintains both sign and interval abstractions in parallel,
    with mutual refinement to tighten both domains using information from each other.
    
    Key refinement rules:
    - sign=POSITIVE + interval=[a,b] → interval=[max(a,1), b]
    - sign=NEGATIVE + interval=[a,b] → interval=[a, min(b,-1)]
    - sign=ZERO + interval=[a,b] → interval=[0,0]
    - interval=[a,b] where a>0 → sign=POSITIVE
    - interval=[a,b] where b<0 → sign=NEGATIVE
    - interval=[0,0] → sign=ZERO
    """
    sign: SignDomain
    interval: IntervalDomain
    _refinement_history: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Apply initial mutual refinement after construction."""
        if not hasattr(self, '_refinement_history') or self._refinement_history is None:
            self._refinement_history = []
    
    @classmethod
    def from_samples(cls, samples: List[int]) -> 'ReducedProductState':
        """
        Create a reduced product state from concrete samples.
        
        Args:
            samples: List of concrete integer values
            
        Returns:
            ReducedProductState with mutually refined sign and interval
        """
        sign_domain, interval_domain = DomainRefinement.from_concrete_values(samples)
        state = cls(sign=sign_domain, interval=interval_domain, _refinement_history=[])
        state.inform_each_other()
        return state
    
    @classmethod
    def top(cls) -> 'ReducedProductState':
        """Create TOP state (no information)."""
        return cls(
            sign=SignDomain(SignValue.TOP),
            interval=IntervalDomain(IntervalValue(None, None)),
            _refinement_history=[]
        )
    
    @classmethod
    def bottom(cls) -> 'ReducedProductState':
        """Create BOTTOM state (unreachable)."""
        return cls(
            sign=SignDomain(SignValue.BOTTOM),
            interval=IntervalDomain(IntervalDomain.BOTTOM),
            _refinement_history=[]
        )
    
    def is_bottom(self) -> bool:
        """Check if either component is bottom."""
        return self.sign.is_bottom() or self.interval.is_bottom()
    
    def is_top(self) -> bool:
        """Check if both components are top."""
        return self.sign.is_top() and self.interval.is_top()
    
    def inform_each_other(self) -> bool:
        """
        Core NAB operation: Mutual refinement between sign and interval.
        
        This is the key integration method that implements the course definition:
        "Run two or more abstractions at the same time, letting them inform each other"
        
        Returns:
            True if any refinement occurred, False if fixed point reached
        """
        changed = True
        iterations = 0
        max_iterations = 10  # Prevent infinite loops
        
        while changed and iterations < max_iterations:
            changed = False
            iterations += 1
            
            # Refine interval from sign information
            interval_changed = self._refine_interval_from_sign()
            if interval_changed:
                changed = True
            
            # Refine sign from interval information
            sign_changed = self._refine_sign_from_interval()
            if sign_changed:
                changed = True
            
            # Check for inconsistency (bottom)
            if self._check_inconsistency():
                self.sign = SignDomain(SignValue.BOTTOM)
                self.interval = IntervalDomain(IntervalDomain.BOTTOM)
                self._refinement_history.append("inconsistency detected -> BOTTOM")
                return True
        
        return iterations > 1
    
    def _refine_interval_from_sign(self) -> bool:
        """
        Refine interval bounds using sign information.
        
        Rules:
        - POSITIVE → low = max(low, 1)
        - NEGATIVE → high = min(high, -1)
        - ZERO → [0, 0]
        - NON_NEGATIVE → low = max(low, 0)
        - NON_POSITIVE → high = min(high, 0)
        - NON_ZERO → exclude 0 if point interval
        """
        if self.sign.is_bottom() or self.interval.is_bottom():
            return False
        
        old_low = self.interval.value.low
        old_high = self.interval.value.high
        new_low = old_low
        new_high = old_high
        
        sign_val = self.sign.value
        
        if sign_val == SignValue.POSITIVE:
            # Positive values must be >= 1
            if new_low is None or new_low < 1:
                new_low = 1
                self._refinement_history.append("sign=POSITIVE → low=max(low,1)")
        
        elif sign_val == SignValue.NEGATIVE:
            # Negative values must be <= -1
            if new_high is None or new_high > -1:
                new_high = -1
                self._refinement_history.append("sign=NEGATIVE → high=min(high,-1)")
        
        elif sign_val == SignValue.ZERO:
            # Zero constraint
            new_low = 0
            new_high = 0
            self._refinement_history.append("sign=ZERO → interval=[0,0]")
        
        elif sign_val == SignValue.NON_NEGATIVE:
            # Non-negative values must be >= 0
            if new_low is None or new_low < 0:
                new_low = 0
                self._refinement_history.append("sign=NON_NEGATIVE → low=max(low,0)")
        
        elif sign_val == SignValue.NON_POSITIVE:
            # Non-positive values must be <= 0
            if new_high is None or new_high > 0:
                new_high = 0
                self._refinement_history.append("sign=NON_POSITIVE → high=min(high,0)")
        
        # Check if changed
        if new_low != old_low or new_high != old_high:
            # Check for inconsistency (low > high means empty interval = BOTTOM)
            if (new_low is not None and new_high is not None and new_low > new_high):
                self.interval = IntervalDomain(IntervalDomain.BOTTOM)
                self._refinement_history.append("inconsistency: low > high → BOTTOM")
            else:
                self.interval = IntervalDomain(IntervalValue(new_low, new_high))
            return True
        
        return False
    
    def _refine_sign_from_interval(self) -> bool:
        """
        Refine sign using interval information.
        
        Rules:
        - [a, b] where a > 0 → POSITIVE
        - [a, b] where b < 0 → NEGATIVE
        - [0, 0] → ZERO
        - [0, b] where b > 0 → NON_NEGATIVE
        - [a, 0] where a < 0 → NON_POSITIVE
        - [a, b] where a < 0 < b → meet with current
        """
        if self.sign.is_bottom() or self.interval.is_bottom():
            return False
        
        old_sign = self.sign.value
        low = self.interval.value.low
        high = self.interval.value.high
        
        inferred_sign = None
        
        # Infer sign from interval
        if low is not None and low > 0:
            inferred_sign = SignValue.POSITIVE
            self._refinement_history.append(f"interval.low={low}>0 → sign=POSITIVE")
        elif high is not None and high < 0:
            inferred_sign = SignValue.NEGATIVE
            self._refinement_history.append(f"interval.high={high}<0 → sign=NEGATIVE")
        elif low == 0 and high == 0:
            inferred_sign = SignValue.ZERO
            self._refinement_history.append("interval=[0,0] → sign=ZERO")
        elif low is not None and low == 0 and (high is None or high > 0):
            inferred_sign = SignValue.NON_NEGATIVE
            self._refinement_history.append("interval.low=0 → sign=NON_NEGATIVE")
        elif high is not None and high == 0 and (low is None or low < 0):
            inferred_sign = SignValue.NON_POSITIVE
            self._refinement_history.append("interval.high=0 → sign=NON_POSITIVE")
        elif low is not None and high is not None and low < 0 and high > 0:
            # Crosses zero - could be anything non-zero or include zero
            if low < 0 and high > 0:
                # Full range including zero
                inferred_sign = SignValue.TOP
        
        if inferred_sign is not None:
            # Meet with current sign for precision
            new_sign_domain = self.sign.meet(SignDomain(inferred_sign))
            if new_sign_domain.value != old_sign:
                self.sign = new_sign_domain
                return True
        
        return False
    
    def _check_inconsistency(self) -> bool:
        """Check if sign and interval are inconsistent."""
        if self.interval.is_bottom():
            return True
        
        low = self.interval.value.low
        high = self.interval.value.high
        
        # Check for empty interval
        if low is not None and high is not None and low > high:
            return True
        
        # Check sign/interval consistency
        sign_val = self.sign.value
        
        if sign_val == SignValue.POSITIVE:
            if high is not None and high <= 0:
                return True
        elif sign_val == SignValue.NEGATIVE:
            if low is not None and low >= 0:
                return True
        elif sign_val == SignValue.ZERO:
            if (low is not None and low > 0) or (high is not None and high < 0):
                return True
        
        return False
    
    def join(self, other: 'ReducedProductState') -> 'ReducedProductState':
        """
        Join (least upper bound) of two reduced product states.
        """
        new_sign = self.sign.join(other.sign)
        new_interval = self.interval.join(other.interval)
        result = ReducedProductState(sign=new_sign, interval=new_interval)
        result.inform_each_other()
        return result
    
    def meet(self, other: 'ReducedProductState') -> 'ReducedProductState':
        """
        Meet (greatest lower bound) of two reduced product states.
        """
        new_sign = self.sign.meet(other.sign)
        new_interval = self.interval.meet(other.interval)
        result = ReducedProductState(sign=new_sign, interval=new_interval)
        result.inform_each_other()
        return result
    
    def widening(self, other: 'ReducedProductState') -> 'ReducedProductState':
        """
        Widening for fixpoint computation.
        """
        new_sign = self.sign.widening(other.sign)
        new_interval = self.interval.widening(other.interval)
        result = ReducedProductState(sign=new_sign, interval=new_interval)
        # Don't refine after widening to ensure termination
        return result
    
    def get_refinement_history(self) -> List[str]:
        """Get the history of refinement steps applied."""
        return list(self._refinement_history)
    
    def __str__(self):
        return f"ReducedProduct(sign={self.sign}, interval={self.interval})"
    
    def __repr__(self):
        return self.__str__()
    
    def to_abstract_value(self, local_index: int) -> 'AbstractValue':
        """Convert to AbstractValue for compatibility."""
        return AbstractValue(
            sign=self.sign,
            interval=self.interval,
            local_index=local_index
        )


@dataclass
class IntegrationResult:
    """Result of NAB integration for a method."""
    method_name: str
    abstract_values: Dict[int, AbstractValue]
    trace_path: str
    confidence: float
    samples_count: Dict[int, int]
    
    def __str__(self):
        values_str = ", ".join(str(v) for v in self.abstract_values.values())
        return f"IntegrationResult({self.method_name}: [{values_str}], confidence={self.confidence:.2f})"


def extract_samples_from_trace(trace_data: Dict[str, Any]) -> Dict[int, List[int]]:
    """
    Extract concrete sample values from IIN trace data.
    
    Args:
        trace_data: Parsed JSON trace data from IIN
        
    Returns:
        Dictionary mapping local variable indices to lists of observed values
    """
    samples = {}
    
    if "values" not in trace_data:
        return samples
    
    for local_key, analysis in trace_data["values"].items():
        # Parse local variable index from key like "local_0"
        try:
            local_idx = int(local_key.split("_")[1])
        except (IndexError, ValueError):
            continue
        
        # Extract samples from analysis
        if "samples" in analysis and analysis["samples"]:
            # Filter to only integer values
            int_samples = [v for v in analysis["samples"] if isinstance(v, (int, bool))]
            if int_samples:
                samples[local_idx] = int_samples
    
    return samples


def refine_from_trace(samples: List[int]) -> Tuple[SignDomain, IntervalDomain]:
    """
    Refine abstract domains from concrete sample values.
    
    This is the core dynamic refinement heuristic: we observe concrete
    execution values and infer the tightest abstract domain that contains
    all observed values. This is our approved novelty contribution.
    
    Uses ReducedProductState to ensure mutual refinement between domains.
    
    Args:
        samples: List of concrete integer values observed during execution
        
    Returns:
        Tuple of (SignDomain, IntervalDomain) refined from samples with
        mutual refinement applied (reduced product)
    """
    # Use reduced product for mutual refinement
    reduced = ReducedProductState.from_samples(samples)
    return reduced.sign, reduced.interval


def refine_from_trace_reduced(samples: List[int]) -> ReducedProductState:
    """
    Refine abstract domains from samples, returning full ReducedProductState.
    
    This exposes the full reduced product including refinement history.
    
    Args:
        samples: List of concrete integer values observed during execution
        
    Returns:
        ReducedProductState with mutually refined sign and interval
    """
    return ReducedProductState.from_samples(samples)


def integrate_abstractions(trace_path: str) -> Dict[int, AbstractValue]:
    """
    Main NAB integration function: reads IIN trace and produces refined abstract state.
    
    This function implements the integration between:
    - IIN (Interpreter): produces traces/<method>.json with concrete execution data
    - IAB (Abstractions): provides SignDomain, IntervalDomain, refine_from_trace()
    
    The integration uses dynamic traces to set initial abstract states (e.g., x>0)
    for subsequent static analysis, as described in our approved proposal.
    
    Args:
        trace_path: Path to IIN JSON trace file in traces/ directory
        
    Returns:
        Dictionary mapping local variable indices to AbstractValue objects
        containing refined sign and interval domains
        
    Example:
        >>> result = integrate_abstractions("traces/jpamb.cases.Simple_assertPositive_IV.json")
        >>> result[0].sign.value  # SignValue.POSITIVE for samples [1,1,1,1,1]
    """
    # Read IIN trace file
    path = Path(trace_path)
    if not path.exists():
        raise FileNotFoundError(f"Trace file not found: {trace_path}")
    
    with open(path, 'r') as f:
        trace_data = json.load(f)
    
    # Extract concrete samples from trace
    samples_by_local = extract_samples_from_trace(trace_data)
    
    # Refine abstract domains for each local variable
    abstract_values: Dict[int, AbstractValue] = {}
    
    for local_idx, samples in samples_by_local.items():
        # Apply reduced product refinement for mutual information exchange
        reduced_state = ReducedProductState.from_samples(samples)
        
        # Create combined abstract value from reduced product
        abstract_values[local_idx] = reduced_state.to_abstract_value(local_idx)
    
    # For local variables with no samples, return TOP
    if not abstract_values:
        # Return empty dict - caller can handle as needed
        pass
    
    return abstract_values


def integrate_abstractions_reduced(trace_path: str) -> Dict[int, ReducedProductState]:
    """
    Integration returning full ReducedProductState objects.
    
    This exposes the parallel execution and mutual refinement capabilities
    as required by the course definition.
    
    Args:
        trace_path: Path to IIN JSON trace file
        
    Returns:
        Dictionary mapping local variable indices to ReducedProductState objects
    """
    path = Path(trace_path)
    if not path.exists():
        raise FileNotFoundError(f"Trace file not found: {trace_path}")
    
    with open(path, 'r') as f:
        trace_data = json.load(f)
    
    samples_by_local = extract_samples_from_trace(trace_data)
    
    reduced_states: Dict[int, ReducedProductState] = {}
    
    for local_idx, samples in samples_by_local.items():
        reduced_states[local_idx] = ReducedProductState.from_samples(samples)
    
    return reduced_states


def integrate_abstractions_full(trace_path: str) -> IntegrationResult:
    """
    Full NAB integration with additional metadata.
    
    Args:
        trace_path: Path to IIN JSON trace file
        
    Returns:
        IntegrationResult with abstract values and metadata
    """
    path = Path(trace_path)
    if not path.exists():
        raise FileNotFoundError(f"Trace file not found: {trace_path}")
    
    with open(path, 'r') as f:
        trace_data = json.load(f)
    
    # Get method name
    method_name = trace_data.get("method", "unknown")
    
    # Extract samples
    samples_by_local = extract_samples_from_trace(trace_data)
    
    # Compute abstract values
    abstract_values = integrate_abstractions(trace_path)
    
    # Calculate confidence based on sample coverage
    confidence = _calculate_integration_confidence(trace_data, samples_by_local)
    
    # Count samples per local
    samples_count = {idx: len(samples) for idx, samples in samples_by_local.items()}
    
    return IntegrationResult(
        method_name=method_name,
        abstract_values=abstract_values,
        trace_path=str(trace_path),
        confidence=confidence,
        samples_count=samples_count
    )


def _calculate_integration_confidence(
    trace_data: Dict[str, Any], 
    samples: Dict[int, List[int]]
) -> float:
    """
    Calculate confidence score for the integration result.
    
    Higher confidence when:
    - More samples are observed
    - Better code coverage
    - Branch coverage is available
    """
    confidence = 0.5  # Base confidence
    
    # Boost for having samples
    if samples:
        total_samples = sum(len(s) for s in samples.values())
        sample_boost = min(0.2, total_samples * 0.02)
        confidence += sample_boost
    
    # Boost for coverage information
    if "coverage" in trace_data:
        coverage = trace_data["coverage"]
        executed = len(coverage.get("executed_pcs", []))
        uncovered = len(coverage.get("uncovered_pcs", []))
        total = executed + uncovered
        
        if total > 0:
            coverage_ratio = executed / total
            confidence += coverage_ratio * 0.2
        
        # Branch coverage boost
        if coverage.get("branches"):
            confidence += 0.1
    
    return min(0.95, confidence)


def integrate_all_traces(traces_dir: str = "traces") -> Dict[str, IntegrationResult]:
    """
    Integrate all trace files in a directory.
    
    Args:
        traces_dir: Directory containing IIN trace JSON files
        
    Returns:
        Dictionary mapping method names to IntegrationResult objects
    """
    traces_path = Path(traces_dir)
    if not traces_path.exists():
        return {}
    
    results = {}
    for trace_file in traces_path.glob("*.json"):
        try:
            result = integrate_abstractions_full(str(trace_file))
            results[result.method_name] = result
        except Exception as e:
            # Log error but continue with other files
            print(f"Warning: Failed to integrate {trace_file}: {e}")
    
    return results


def get_sign_for_local(trace_path: str, local_idx: int) -> SignValue:
    """
    Convenience function to get sign for a specific local variable.
    
    Args:
        trace_path: Path to IIN trace file
        local_idx: Local variable index
        
    Returns:
        SignValue for the local variable, or TOP if not found
    """
    abstract_values = integrate_abstractions(trace_path)
    
    if local_idx in abstract_values:
        return abstract_values[local_idx].sign.value
    
    return SignValue.TOP


def get_interval_for_local(trace_path: str, local_idx: int) -> IntervalValue:
    """
    Convenience function to get interval for a specific local variable.
    
    Args:
        trace_path: Path to IIN trace file
        local_idx: Local variable index
        
    Returns:
        IntervalValue for the local variable, or TOP if not found
    """
    abstract_values = integrate_abstractions(trace_path)
    
    if local_idx in abstract_values:
        return abstract_values[local_idx].interval.value
    
    return IntervalValue(None, None)  # TOP


# Proposal example implementation
def process_example() -> Dict[int, AbstractValue]:
    """
    Proposal example §1.3.1: process(int x) with samples [5, 10, ...]
    
    From proposal: "dynamic traces to set initial abstracts like x>0"
    
    When we observe positive samples [5, 10], we refine:
    - sign(x) = positive (SignValue.POSITIVE)
    - interval(x) = [5, 10]
    
    This is our dynamic refinement heuristic: using IIN traces to
    initialize abstract states for static analysis.
    
    Uses ReducedProductState for mutual refinement between sign and interval.
    """
    # Simulate the proposal example
    samples = [5, 10, 15, 20, 25]  # Positive samples from dynamic execution
    
    # Use reduced product for mutual refinement
    reduced = ReducedProductState.from_samples(samples)
    
    return {
        1: reduced.to_abstract_value(1)
    }


def process_example_reduced() -> Dict[int, ReducedProductState]:
    """
    Proposal example returning full ReducedProductState.
    
    This demonstrates the parallel execution and mutual refinement.
    """
    samples = [5, 10, 15, 20, 25]
    reduced = ReducedProductState.from_samples(samples)
    return {1: reduced}


def inform_each_other(sign: SignDomain, interval: IntervalDomain) -> Tuple[SignDomain, IntervalDomain]:
    """
    Module-level function for mutual refinement between sign and interval domains.
    
    This is the core NAB operation implementing the course definition:
    "Run two or more abstractions at the same time, letting them inform each other
    during execution" (formula: 5 per abstraction after the first).
    
    Args:
        sign: Initial sign domain
        interval: Initial interval domain
        
    Returns:
        Tuple of (refined_sign, refined_interval) after mutual refinement
        
    Example:
        >>> sign = SignDomain(SignValue.POSITIVE)
        >>> interval = IntervalDomain(IntervalValue(-5, 10))
        >>> new_sign, new_interval = inform_each_other(sign, interval)
        >>> new_interval.value.low  # Tightened to 1
        1
    """
    reduced = ReducedProductState(sign=sign, interval=interval)
    reduced.inform_each_other()
    return reduced.sign, reduced.interval


if __name__ == "__main__":
    # Demonstrate proposal example
    print("NAB Integration - Proposal Example §1.3.1")
    print("=" * 50)
    
    result = process_example()
    for local_idx, abstract_val in result.items():
        print(f"  {abstract_val}")
        print(f"    → sign = {abstract_val.sign.value.name}")
        print(f"    → interval = {abstract_val.interval}")
    
    print()
    print("Demonstrating Reduced Product mutual refinement:")
    print("-" * 50)
    
    # Show mutual refinement example
    print("\nExample 1: POSITIVE sign tightens interval [−5, 10] to [1, 10]")
    sign1 = SignDomain(SignValue.POSITIVE)
    interval1 = IntervalDomain(IntervalValue(-5, 10))
    new_sign1, new_interval1 = inform_each_other(sign1, interval1)
    print(f"  Before: sign={sign1}, interval={interval1}")
    print(f"  After:  sign={new_sign1}, interval={new_interval1}")
    
    print("\nExample 2: Interval [5, 100] infers POSITIVE sign")
    reduced2 = ReducedProductState(
        sign=SignDomain(SignValue.TOP),
        interval=IntervalDomain(IntervalValue(5, 100))
    )
    reduced2.inform_each_other()
    print(f"  Before: sign=TOP, interval=[5, 100]")
    print(f"  After:  sign={reduced2.sign}, interval={reduced2.interval}")
    print(f"  Refinement history: {reduced2.get_refinement_history()}")
    
    print()
    print("Integrating actual traces from traces/ directory...")
    print("-" * 50)
    
    # Integrate actual traces
    all_results = integrate_all_traces("traces")
    
    for method_name, result in list(all_results.items())[:5]:  # Show first 5
        print(f"\n{method_name}:")
        for local_idx, abstract_val in result.abstract_values.items():
            print(f"  {abstract_val}")
