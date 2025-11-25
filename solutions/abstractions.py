"""
Novel abstract domains for JPAMB static analysis with dynamic refinement.

This module implements sign and interval abstract domains for static analysis,
with heuristics to refine initial abstract states using dynamic traces from IIN.
Includes operations (join, meet, widening) and refinement from IIN JSON traces.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Set
from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod


class SignValue(Enum):
    """Sign domain abstract values."""
    BOTTOM = "⊥"      # Unreachable/impossible
    POSITIVE = "+"    # Strictly positive 
    NEGATIVE = "-"    # Strictly negative
    ZERO = "0"        # Exactly zero
    NON_ZERO = "≠0"   # Non-zero (positive or negative)
    NON_NEGATIVE = "≥0"  # Zero or positive
    NON_POSITIVE = "≤0"  # Zero or negative  
    TOP = "⊤"         # Unknown/any value

    def __str__(self):
        return self.value


@dataclass(frozen=True)
class IntervalValue:
    """Interval domain abstract values."""
    low: Optional[int]   # None represents -∞
    high: Optional[int]  # None represents +∞
    
    def __post_init__(self):
        # Validate interval bounds - but allow creation of BOTTOM during class definition
        if (self.low is not None and self.high is not None and 
            self.low > self.high and not (self.low == 1 and self.high == 0)):
            raise ValueError(f"Invalid interval: [{self.low}, {self.high}]")
    
    def __str__(self):
        low_str = "-∞" if self.low is None else str(self.low)
        high_str = "+∞" if self.high is None else str(self.high)
        return f"[{low_str}, {high_str}]"
    
    def is_bottom(self) -> bool:
        """Check if this is the bottom element (empty interval)."""
        return (self.low is not None and self.high is not None and 
                self.low > self.high)
    
    def is_top(self) -> bool:
        """Check if this is the top element (unbounded interval)."""
        return self.low is None and self.high is None
    
    def contains(self, value: int) -> bool:
        """Check if value is contained in this interval."""
        if self.is_bottom():
            return False
        low_ok = self.low is None or value >= self.low
        high_ok = self.high is None or value <= self.high
        return low_ok and high_ok


class AbstractDomain(ABC):
    """Abstract base class for abstract domains."""
    
    @abstractmethod
    def join(self, other: 'AbstractDomain') -> 'AbstractDomain':
        """Least upper bound (union) operation."""
        pass
    
    @abstractmethod
    def meet(self, other: 'AbstractDomain') -> 'AbstractDomain':
        """Greatest lower bound (intersection) operation.""" 
        pass
    
    @abstractmethod
    def widening(self, other: 'AbstractDomain') -> 'AbstractDomain':
        """Widening operation for loops."""
        pass
    
    @abstractmethod
    def is_bottom(self) -> bool:
        """Check if this is the bottom element."""
        pass
    
    @abstractmethod
    def is_top(self) -> bool:
        """Check if this is the top element."""
        pass


class SignDomain(AbstractDomain):
    """Sign domain for integer values."""
    
    def __init__(self, value: SignValue = SignValue.TOP):
        self.value = value
    
    def __str__(self):
        return str(self.value)
    
    def __repr__(self):
        return f"SignDomain({self.value})"
    
    def __eq__(self, other):
        return isinstance(other, SignDomain) and self.value == other.value
    
    def __hash__(self):
        return hash(self.value)
    
    def is_bottom(self) -> bool:
        return self.value == SignValue.BOTTOM
    
    def is_top(self) -> bool:
        return self.value == SignValue.TOP
    
    def join(self, other: 'SignDomain') -> 'SignDomain':
        """Least upper bound of two sign domains."""
        if not isinstance(other, SignDomain):
            raise TypeError(f"Cannot join SignDomain with {type(other)}")
        
        if self.is_bottom():
            return other
        if other.is_bottom():
            return self
        
        # Join table based on sign lattice structure
        join_table = {
            (SignValue.POSITIVE, SignValue.POSITIVE): SignValue.POSITIVE,
            (SignValue.NEGATIVE, SignValue.NEGATIVE): SignValue.NEGATIVE,
            (SignValue.ZERO, SignValue.ZERO): SignValue.ZERO,
            (SignValue.POSITIVE, SignValue.NEGATIVE): SignValue.NON_ZERO,
            (SignValue.NEGATIVE, SignValue.POSITIVE): SignValue.NON_ZERO,
            (SignValue.POSITIVE, SignValue.ZERO): SignValue.NON_NEGATIVE,
            (SignValue.ZERO, SignValue.POSITIVE): SignValue.NON_NEGATIVE,
            (SignValue.NEGATIVE, SignValue.ZERO): SignValue.NON_POSITIVE,
            (SignValue.ZERO, SignValue.NEGATIVE): SignValue.NON_POSITIVE,
            (SignValue.NON_ZERO, SignValue.NON_ZERO): SignValue.NON_ZERO,
            (SignValue.NON_NEGATIVE, SignValue.NON_NEGATIVE): SignValue.NON_NEGATIVE,
            (SignValue.NON_POSITIVE, SignValue.NON_POSITIVE): SignValue.NON_POSITIVE,
        }
        
        # Check direct combinations
        key = (self.value, other.value)
        if key in join_table:
            return SignDomain(join_table[key])
        
        # Handle combinations involving compound values
        if (self.value in [SignValue.NON_ZERO, SignValue.NON_NEGATIVE, SignValue.NON_POSITIVE] or
            other.value in [SignValue.NON_ZERO, SignValue.NON_NEGATIVE, SignValue.NON_POSITIVE]):
            
            # Convert to sets for easier handling
            self_values = self._to_concrete_values()
            other_values = other._to_concrete_values()
            union = self_values | other_values
            return self._from_concrete_values(union)
        
        # Default to TOP for any combination not handled
        return SignDomain(SignValue.TOP)
    
    def meet(self, other: 'SignDomain') -> 'SignDomain':
        """Greatest lower bound of two sign domains."""
        if not isinstance(other, SignDomain):
            raise TypeError(f"Cannot meet SignDomain with {type(other)}")
        
        if self.is_bottom() or other.is_bottom():
            return SignDomain(SignValue.BOTTOM)
        
        if self.is_top():
            return other
        if other.is_top():
            return self
        
        # Convert to concrete value sets and intersect
        self_values = self._to_concrete_values()
        other_values = other._to_concrete_values()
        intersection = self_values & other_values
        
        if not intersection:
            return SignDomain(SignValue.BOTTOM)
        
        return self._from_concrete_values(intersection)
    
    def widening(self, other: 'SignDomain') -> 'SignDomain':
        """Widening operation for fixpoint computation."""
        if not isinstance(other, SignDomain):
            raise TypeError(f"Cannot widen SignDomain with {type(other)}")
        
        # For sign domain, widening is the same as join
        # since the domain height is finite (no infinite chains)
        return self.join(other)
    
    def _to_concrete_values(self) -> Set[str]:
        """Convert abstract sign value to set of concrete possibilities."""
        mapping = {
            SignValue.POSITIVE: {"positive"},
            SignValue.NEGATIVE: {"negative"},
            SignValue.ZERO: {"zero"},
            SignValue.NON_ZERO: {"positive", "negative"},
            SignValue.NON_NEGATIVE: {"positive", "zero"},
            SignValue.NON_POSITIVE: {"negative", "zero"},
            SignValue.TOP: {"positive", "negative", "zero"},
            SignValue.BOTTOM: set()
        }
        return mapping[self.value]
    
    def _from_concrete_values(self, values: Set[str]) -> 'SignDomain':
        """Convert set of concrete possibilities to abstract sign value."""
        if not values:
            return SignDomain(SignValue.BOTTOM)
        elif values == {"positive"}:
            return SignDomain(SignValue.POSITIVE)
        elif values == {"negative"}:
            return SignDomain(SignValue.NEGATIVE)
        elif values == {"zero"}:
            return SignDomain(SignValue.ZERO)
        elif values == {"positive", "negative"}:
            return SignDomain(SignValue.NON_ZERO)
        elif values == {"positive", "zero"}:
            return SignDomain(SignValue.NON_NEGATIVE)
        elif values == {"negative", "zero"}:
            return SignDomain(SignValue.NON_POSITIVE)
        else:  # {"positive", "negative", "zero"}
            return SignDomain(SignValue.TOP)
    
    def add(self, other: 'SignDomain') -> 'SignDomain':
        """Addition operation for sign domain."""
        if self.is_bottom() or other.is_bottom():
            return SignDomain(SignValue.BOTTOM)
        
        # Addition rules for sign domain
        add_rules = {
            (SignValue.POSITIVE, SignValue.POSITIVE): SignValue.POSITIVE,
            (SignValue.NEGATIVE, SignValue.NEGATIVE): SignValue.NEGATIVE,
            (SignValue.ZERO, SignValue.ZERO): SignValue.ZERO,
            (SignValue.POSITIVE, SignValue.ZERO): SignValue.POSITIVE,
            (SignValue.ZERO, SignValue.POSITIVE): SignValue.POSITIVE,
            (SignValue.NEGATIVE, SignValue.ZERO): SignValue.NEGATIVE,
            (SignValue.ZERO, SignValue.NEGATIVE): SignValue.NEGATIVE,
        }
        
        key = (self.value, other.value)
        if key in add_rules:
            return SignDomain(add_rules[key])
        
        # For complex cases, be conservative and return TOP
        return SignDomain(SignValue.TOP)
    
    def mul(self, other: 'SignDomain') -> 'SignDomain':
        """Multiplication operation for sign domain."""
        if self.is_bottom() or other.is_bottom():
            return SignDomain(SignValue.BOTTOM)
        
        # Multiplication rules for sign domain
        mul_rules = {
            (SignValue.POSITIVE, SignValue.POSITIVE): SignValue.POSITIVE,
            (SignValue.NEGATIVE, SignValue.NEGATIVE): SignValue.POSITIVE,
            (SignValue.POSITIVE, SignValue.NEGATIVE): SignValue.NEGATIVE,
            (SignValue.NEGATIVE, SignValue.POSITIVE): SignValue.NEGATIVE,
            (SignValue.ZERO, SignValue.ZERO): SignValue.ZERO,
            (SignValue.POSITIVE, SignValue.ZERO): SignValue.ZERO,
            (SignValue.ZERO, SignValue.POSITIVE): SignValue.ZERO,
            (SignValue.NEGATIVE, SignValue.ZERO): SignValue.ZERO,
            (SignValue.ZERO, SignValue.NEGATIVE): SignValue.ZERO,
        }
        
        key = (self.value, other.value)
        if key in mul_rules:
            return SignDomain(mul_rules[key])
        
        # For complex cases, be conservative and return TOP
        return SignDomain(SignValue.TOP)


class IntervalDomain(AbstractDomain):
    """Interval domain for integer values."""
    
    # Special constants
    BOTTOM = IntervalValue(1, 0)  # Invalid interval representing bottom
    TOP = IntervalValue(None, None)  # Unbounded interval representing top
    
    def __init__(self, value: IntervalValue = None):
        if value is None:
            value = self.TOP
        self.value = value
    
    def __str__(self):
        return str(self.value)
    
    def __repr__(self):
        return f"IntervalDomain({self.value})"
    
    def __eq__(self, other):
        return (isinstance(other, IntervalDomain) and 
                self.value.low == other.value.low and 
                self.value.high == other.value.high)
    
    def __hash__(self):
        return hash((self.value.low, self.value.high))
    
    def is_bottom(self) -> bool:
        return self.value.is_bottom()
    
    def is_top(self) -> bool:
        return self.value.is_top()
    
    def join(self, other: 'IntervalDomain') -> 'IntervalDomain':
        """Union of two intervals (least upper bound)."""
        if not isinstance(other, IntervalDomain):
            raise TypeError(f"Cannot join IntervalDomain with {type(other)}")
        
        if self.is_bottom():
            return other
        if other.is_bottom():
            return self
        
        # Union: take minimum of lows and maximum of highs
        new_low = None
        if self.value.low is None or other.value.low is None:
            new_low = None
        else:
            new_low = min(self.value.low, other.value.low)
        
        new_high = None
        if self.value.high is None or other.value.high is None:
            new_high = None
        else:
            new_high = max(self.value.high, other.value.high)
        
        return IntervalDomain(IntervalValue(new_low, new_high))
    
    def meet(self, other: 'IntervalDomain') -> 'IntervalDomain':
        """Intersection of two intervals (greatest lower bound)."""
        if not isinstance(other, IntervalDomain):
            raise TypeError(f"Cannot meet IntervalDomain with {type(other)}")
        
        if self.is_bottom() or other.is_bottom():
            return IntervalDomain(self.BOTTOM)
        
        # Intersection: take maximum of lows and minimum of highs
        new_low = None
        if self.value.low is None:
            new_low = other.value.low
        elif other.value.low is None:
            new_low = self.value.low
        else:
            new_low = max(self.value.low, other.value.low)
        
        new_high = None
        if self.value.high is None:
            new_high = other.value.high
        elif other.value.high is None:
            new_high = self.value.high
        else:
            new_high = min(self.value.high, other.value.high)
        
        # Check if intersection is empty
        if (new_low is not None and new_high is not None and new_low > new_high):
            return IntervalDomain(self.BOTTOM)
        
        return IntervalDomain(IntervalValue(new_low, new_high))
    
    def widening(self, other: 'IntervalDomain') -> 'IntervalDomain':
        """Widening operation for loops to ensure termination."""
        if not isinstance(other, IntervalDomain):
            raise TypeError(f"Cannot widen IntervalDomain with {type(other)}")
        
        if self.is_bottom():
            return other
        if other.is_bottom():
            return self
        
        # Widening: if bound decreases/increases, make it infinite
        new_low = self.value.low
        new_high = self.value.high
        
        # If other's low bound is smaller, widen to -∞
        if (other.value.low is not None and self.value.low is not None and
            other.value.low < self.value.low):
            new_low = None
        
        # If other's high bound is larger, widen to +∞
        if (other.value.high is not None and self.value.high is not None and
            other.value.high > self.value.high):
            new_high = None
        
        return IntervalDomain(IntervalValue(new_low, new_high))
    
    def add(self, other: 'IntervalDomain') -> 'IntervalDomain':
        """Addition operation for intervals."""
        if self.is_bottom() or other.is_bottom():
            return IntervalDomain(self.BOTTOM)
        
        # Addition: [a,b] + [c,d] = [a+c, b+d]
        new_low = None
        if self.value.low is not None and other.value.low is not None:
            new_low = self.value.low + other.value.low
        
        new_high = None
        if self.value.high is not None and other.value.high is not None:
            new_high = self.value.high + other.value.high
        
        return IntervalDomain(IntervalValue(new_low, new_high))
    
    def mul(self, other: 'IntervalDomain') -> 'IntervalDomain':
        """Multiplication operation for intervals."""
        if self.is_bottom() or other.is_bottom():
            return IntervalDomain(self.BOTTOM)
        
        # Handle unbounded cases
        if self.is_top() or other.is_top():
            return IntervalDomain(self.TOP)
        
        # Bounded case: [a,b] * [c,d] = [min(ac,ad,bc,bd), max(ac,ad,bc,bd)]
        if (self.value.low is None or self.value.high is None or
            other.value.low is None or other.value.high is None):
            return IntervalDomain(self.TOP)
        
        products = [
            self.value.low * other.value.low,
            self.value.low * other.value.high,
            self.value.high * other.value.low,
            self.value.high * other.value.high
        ]
        
        return IntervalDomain(IntervalValue(min(products), max(products)))


class DomainRefinement:
    """Refines abstract domains using dynamic traces from IIN."""
    
    @staticmethod
    def from_concrete_values(values: List[int]) -> tuple[SignDomain, IntervalDomain]:
        """Create refined domains from concrete execution values."""
        if not values:
            return SignDomain(SignValue.BOTTOM), IntervalDomain(IntervalDomain.BOTTOM)
        
        # Refine sign domain
        sign_domain = DomainRefinement._refine_sign_domain(values)
        
        # Refine interval domain
        interval_domain = DomainRefinement._refine_interval_domain(values)
        
        return sign_domain, interval_domain
    
    @staticmethod
    def _refine_sign_domain(values: List[int]) -> SignDomain:
        """Refine sign domain from concrete values."""
        if not values:
            return SignDomain(SignValue.BOTTOM)
        
        has_positive = any(v > 0 for v in values)
        has_negative = any(v < 0 for v in values)
        has_zero = any(v == 0 for v in values)
        
        if has_positive and has_negative and has_zero:
            return SignDomain(SignValue.TOP)
        elif has_positive and has_negative:
            return SignDomain(SignValue.NON_ZERO)
        elif has_positive and has_zero:
            return SignDomain(SignValue.NON_NEGATIVE)
        elif has_negative and has_zero:
            return SignDomain(SignValue.NON_POSITIVE)
        elif has_positive:
            return SignDomain(SignValue.POSITIVE)
        elif has_negative:
            return SignDomain(SignValue.NEGATIVE)
        else:  # has_zero only
            return SignDomain(SignValue.ZERO)
    
    @staticmethod
    def _refine_interval_domain(values: List[int]) -> IntervalDomain:
        """Refine interval domain from concrete values."""
        if not values:
            return IntervalDomain(IntervalDomain.BOTTOM)
        
        min_val = min(values)
        max_val = max(values)
        
        # Use exact bounds from observed values
        return IntervalDomain(IntervalValue(min_val, max_val))
    
    @staticmethod
    def from_iin_trace(trace_path: Union[str, Path]) -> Dict[str, tuple[SignDomain, IntervalDomain]]:
        """Refine domains from IIN JSON trace file."""
        with open(trace_path, 'r') as f:
            trace_data = json.load(f)
        
        refined_domains = {}
        
        # Extract value analysis from trace
        if "values" not in trace_data:
            return refined_domains
        
        for local_name, analysis in trace_data["values"].items():
            # Extract concrete samples if available
            samples = analysis.get("samples", [])
            
            if samples:
                sign_domain, interval_domain = DomainRefinement.from_concrete_values(samples)
            else:
                # Fallback: use analysis properties to create domains
                sign_domain = DomainRefinement._domain_from_analysis_properties(analysis)
                interval_domain = DomainRefinement._interval_from_analysis_properties(analysis)
            
            refined_domains[local_name] = (sign_domain, interval_domain)
        
        return refined_domains
    
    @staticmethod
    def _domain_from_analysis_properties(analysis: Dict[str, Any]) -> SignDomain:
        """Create sign domain from trace analysis properties."""
        if analysis.get("always_positive", False):
            return SignDomain(SignValue.POSITIVE)
        elif analysis.get("sign") == "negative":
            return SignDomain(SignValue.NEGATIVE)
        elif analysis.get("sign") == "zero":
            return SignDomain(SignValue.ZERO)
        elif analysis.get("never_negative", False):
            return SignDomain(SignValue.NON_NEGATIVE)
        elif analysis.get("sign") == "mixed":
            return SignDomain(SignValue.TOP)
        else:
            return SignDomain(SignValue.TOP)
    
    @staticmethod
    def _interval_from_analysis_properties(analysis: Dict[str, Any]) -> IntervalDomain:
        """Create interval domain from trace analysis properties."""
        interval = analysis.get("interval")
        if not interval:
            return IntervalDomain(IntervalDomain.TOP)
        
        low = interval[0] if len(interval) > 0 else None
        high = interval[1] if len(interval) > 1 and interval[1] is not None else low
        
        return IntervalDomain(IntervalValue(low, high))


# Example usage and test method matching proposal example
def process_example() -> Dict[str, tuple[SignDomain, IntervalDomain]]:
    """
    Example from proposal §1.3.1: process(int x) method with samples [5,10]
    should refine local_1 to positive sign domain.
    """
    # Simulate trace data for process(int x) with samples [5, 10]
    example_samples = [5, 10]
    
    # Refine domains from samples
    sign_domain, interval_domain = DomainRefinement.from_concrete_values(example_samples)
    
    return {
        "local_1": (sign_domain, interval_domain)
    }


if __name__ == "__main__":
    # Test the example from the proposal
    result = process_example()
    print("Proposal example (§1.3.1) refinement result:")
    for local_name, (sign, interval) in result.items():
        print(f"  {local_name}: sign={sign}, interval={interval}")