"""
Comprehensive pytest tests for novel abstract domains.

Tests cover sign and interval domains, their operations (join, meet, widening),
and refinement from IIN JSON traces. Includes edge cases and proposal examples.
"""

import pytest
import json
import tempfile
from pathlib import Path
from solutions.abstractions import (
    SignDomain, SignValue, IntervalDomain, IntervalValue, 
    DomainRefinement, process_example
)


class TestSignDomain:
    """Test cases for SignDomain class."""
    
    def test_sign_domain_creation(self):
        """Test basic sign domain creation and properties."""
        # Test default constructor (TOP)
        top = SignDomain()
        assert top.is_top()
        assert not top.is_bottom()
        
        # Test specific values
        positive = SignDomain(SignValue.POSITIVE)
        assert not positive.is_top()
        assert not positive.is_bottom()
        assert str(positive) == "+"
        
        bottom = SignDomain(SignValue.BOTTOM)
        assert bottom.is_bottom()
        assert not bottom.is_top()
    
    def test_sign_domain_join_basic(self):
        """Test basic join operations for sign domain."""
        pos = SignDomain(SignValue.POSITIVE)
        neg = SignDomain(SignValue.NEGATIVE)
        zero = SignDomain(SignValue.ZERO)
        bottom = SignDomain(SignValue.BOTTOM)
        
        # Basic joins
        assert pos.join(pos) == pos
        assert pos.join(neg) == SignDomain(SignValue.NON_ZERO)
        assert pos.join(zero) == SignDomain(SignValue.NON_NEGATIVE)
        assert neg.join(zero) == SignDomain(SignValue.NON_POSITIVE)
        
        # Bottom is identity
        assert pos.join(bottom) == pos
        assert bottom.join(neg) == neg
    
    def test_sign_domain_join_mixed_values(self):
        """Test join operations with mixed values (covers mixed values → top)."""
        pos = SignDomain(SignValue.POSITIVE)
        neg = SignDomain(SignValue.NEGATIVE)
        zero = SignDomain(SignValue.ZERO)
        
        # Multiple joins leading to TOP
        result = pos.join(neg).join(zero)
        assert result == SignDomain(SignValue.TOP)
        
        # Non-zero joined with zero gives TOP
        non_zero = SignDomain(SignValue.NON_ZERO)
        result = non_zero.join(zero)
        assert result == SignDomain(SignValue.TOP)
    
    def test_sign_domain_meet(self):
        """Test meet operations for sign domain."""
        pos = SignDomain(SignValue.POSITIVE)
        neg = SignDomain(SignValue.NEGATIVE)
        top = SignDomain(SignValue.TOP)
        bottom = SignDomain(SignValue.BOTTOM)
        non_neg = SignDomain(SignValue.NON_NEGATIVE)
        
        # Basic meets
        assert pos.meet(pos) == pos
        assert pos.meet(neg) == bottom  # No intersection
        assert pos.meet(top) == pos
        assert top.meet(pos) == pos
        
        # Meet with compound values
        assert pos.meet(non_neg) == pos
        assert non_neg.meet(pos) == pos
        
        # Bottom absorbs everything
        assert pos.meet(bottom) == bottom
        assert bottom.meet(pos) == bottom
    
    def test_sign_domain_widening_loops(self):
        """Test widening operation for loops."""
        pos = SignDomain(SignValue.POSITIVE)
        neg = SignDomain(SignValue.NEGATIVE)
        
        # For sign domain, widening equals join (finite height)
        result = pos.widening(neg)
        expected = pos.join(neg)
        assert result == expected
    
    def test_sign_domain_arithmetic_add(self):
        """Test addition operation for sign domain."""
        pos = SignDomain(SignValue.POSITIVE)
        neg = SignDomain(SignValue.NEGATIVE)
        zero = SignDomain(SignValue.ZERO)
        bottom = SignDomain(SignValue.BOTTOM)
        
        # Basic addition rules
        assert pos.add(pos) == pos  # + + + = +
        assert neg.add(neg) == neg  # - + - = -
        assert zero.add(zero) == zero  # 0 + 0 = 0
        assert pos.add(zero) == pos  # + + 0 = +
        assert neg.add(zero) == neg  # - + 0 = -
        
        # Bottom absorbs
        assert pos.add(bottom) == bottom
        assert bottom.add(pos) == bottom
    
    def test_sign_domain_arithmetic_mul(self):
        """Test multiplication operation for sign domain."""
        pos = SignDomain(SignValue.POSITIVE)
        neg = SignDomain(SignValue.NEGATIVE)
        zero = SignDomain(SignValue.ZERO)
        
        # Basic multiplication rules
        assert pos.mul(pos) == pos  # + * + = +
        assert neg.mul(neg) == pos  # - * - = +
        assert pos.mul(neg) == neg  # + * - = -
        assert neg.mul(pos) == neg  # - * + = -
        assert zero.mul(pos) == zero  # 0 * + = 0
        assert zero.mul(neg) == zero  # 0 * - = 0


class TestIntervalDomain:
    """Test cases for IntervalDomain class."""
    
    def test_interval_domain_creation(self):
        """Test basic interval domain creation."""
        # Default constructor (TOP)
        top = IntervalDomain()
        assert top.is_top()
        assert not top.is_bottom()
        
        # Specific interval
        interval = IntervalDomain(IntervalValue(1, 10))
        assert not interval.is_top()
        assert not interval.is_bottom()
        assert str(interval) == "[1, 10]"
        
        # Bottom interval
        bottom = IntervalDomain(IntervalValue(1, 0))  # Invalid interval
        assert bottom.is_bottom()
        
        # Unbounded intervals
        unbounded_low = IntervalDomain(IntervalValue(None, 10))
        assert str(unbounded_low) == "[-∞, 10]"
        
        unbounded_high = IntervalDomain(IntervalValue(1, None))
        assert str(unbounded_high) == "[1, +∞]"
    
    def test_interval_domain_contains(self):
        """Test interval containment check."""
        interval = IntervalValue(1, 10)
        
        assert interval.contains(5)
        assert interval.contains(1)  # Boundary
        assert interval.contains(10)  # Boundary
        assert not interval.contains(0)
        assert not interval.contains(11)
        
        # Unbounded intervals
        unbounded = IntervalValue(None, None)
        assert unbounded.contains(0)
        assert unbounded.contains(-1000)
        assert unbounded.contains(1000)
    
    def test_interval_domain_join_union(self):
        """Test union (join) operations for intervals."""
        i1 = IntervalDomain(IntervalValue(1, 5))
        i2 = IntervalDomain(IntervalValue(3, 8))
        i3 = IntervalDomain(IntervalValue(10, 15))
        bottom = IntervalDomain(IntervalValue(1, 0))  # Bottom
        
        # Overlapping intervals
        result = i1.join(i2)
        expected = IntervalDomain(IntervalValue(1, 8))
        assert result == expected
        
        # Non-overlapping intervals (still union)
        result = i1.join(i3)
        expected = IntervalDomain(IntervalValue(1, 15))
        assert result == expected
        
        # Bottom is identity
        assert i1.join(bottom) == i1
        assert bottom.join(i1) == i1
    
    def test_interval_domain_meet_intersection(self):
        """Test intersection (meet) operations for intervals."""
        i1 = IntervalDomain(IntervalValue(1, 8))
        i2 = IntervalDomain(IntervalValue(5, 12))
        i3 = IntervalDomain(IntervalValue(10, 15))  # No overlap with i1
        top = IntervalDomain()  # Unbounded
        bottom = IntervalDomain(IntervalValue(1, 0))  # Bottom
        
        # Overlapping intervals
        result = i1.meet(i2)
        expected = IntervalDomain(IntervalValue(5, 8))
        assert result == expected
        
        # Non-overlapping intervals
        result = i1.meet(i3)
        assert result.is_bottom()
        
        # Meet with TOP
        assert i1.meet(top) == i1
        assert top.meet(i1) == i1
        
        # Bottom absorbs
        assert i1.meet(bottom).is_bottom()
        assert bottom.meet(i1).is_bottom()
    
    def test_interval_domain_widening_loops(self):
        """Test widening operation for loops with increasing bounds."""
        # Simulate loop iteration where bounds increase
        i1 = IntervalDomain(IntervalValue(0, 5))
        i2 = IntervalDomain(IntervalValue(-1, 10))  # Bounds expanded
        
        # Widening should make expanding bounds infinite
        result = i1.widening(i2)
        
        # Lower bound decreased (-1 < 0), so should become -∞
        # Upper bound increased (10 > 5), so should become +∞
        expected = IntervalDomain(IntervalValue(None, None))  # TOP
        assert result == expected
        
        # Test partial widening
        i3 = IntervalDomain(IntervalValue(0, 10))  # Only upper bound increased
        result = i1.widening(i3)
        expected = IntervalDomain(IntervalValue(0, None))  # Widen upper bound only
        assert result == expected
    
    def test_interval_domain_arithmetic_add(self):
        """Test addition for interval domain."""
        i1 = IntervalDomain(IntervalValue(1, 3))
        i2 = IntervalDomain(IntervalValue(2, 4))
        
        result = i1.add(i2)
        expected = IntervalDomain(IntervalValue(3, 7))  # [1,3] + [2,4] = [3,7]
        assert result == expected
        
        # Addition with unbounded intervals
        unbounded = IntervalDomain(IntervalValue(None, None))
        result = i1.add(unbounded)
        assert result.is_top()
    
    def test_interval_domain_arithmetic_mul(self):
        """Test multiplication for interval domain."""
        i1 = IntervalDomain(IntervalValue(2, 3))
        i2 = IntervalDomain(IntervalValue(4, 5))
        
        result = i1.mul(i2)
        # [2,3] * [4,5] = [min(8,10,12,15), max(8,10,12,15)] = [8,15]
        expected = IntervalDomain(IntervalValue(8, 15))
        assert result == expected
        
        # Multiplication with negative values
        i3 = IntervalDomain(IntervalValue(-3, -1))
        i4 = IntervalDomain(IntervalValue(1, 2))
        
        result = i3.mul(i4)
        # [-3,-1] * [1,2] = [min(-6,-3,-2,-1), max(-6,-3,-2,-1)] = [-6,-1]
        expected = IntervalDomain(IntervalValue(-6, -1))
        assert result == expected


class TestDomainRefinement:
    """Test cases for domain refinement from concrete values and traces."""
    
    def test_refinement_from_concrete_values(self):
        """Test domain refinement from concrete execution values."""
        # All positive values
        values = [1, 5, 10]
        sign, interval = DomainRefinement.from_concrete_values(values)
        
        assert sign == SignDomain(SignValue.POSITIVE)
        assert interval == IntervalDomain(IntervalValue(1, 10))
        
        # Mixed positive and negative
        values = [1, -2, 5]
        sign, interval = DomainRefinement.from_concrete_values(values)
        
        assert sign == SignDomain(SignValue.NON_ZERO)  # Has both positive and negative
        assert interval == IntervalDomain(IntervalValue(-2, 5))
        
        # Contains zero
        values = [0, 1, 2]
        sign, interval = DomainRefinement.from_concrete_values(values)
        
        assert sign == SignDomain(SignValue.NON_NEGATIVE)  # Has positive and zero
        assert interval == IntervalDomain(IntervalValue(0, 2))
    
    def test_refinement_empty_values(self):
        """Test refinement with empty value list."""
        sign, interval = DomainRefinement.from_concrete_values([])
        
        assert sign.is_bottom()
        assert interval.is_bottom()
    
    def test_refinement_proposal_example(self):
        """Test the specific example from proposal §1.3.1."""
        # process(int x) with samples [5, 10] should refine local_1 to positive
        result = process_example()
        
        assert "local_1" in result
        sign, interval = result["local_1"]
        
        # Should be positive sign domain
        assert sign == SignDomain(SignValue.POSITIVE)
        
        # Should be interval [5, 10]
        assert interval == IntervalDomain(IntervalValue(5, 10))
    
    def test_refinement_from_iin_trace(self):
        """Test refinement from IIN JSON trace file."""
        # Create a mock trace file
        trace_data = {
            "method": "jpamb.cases.Simple.process:(I)V",
            "coverage": {
                "executed_pcs": [0, 1, 2, 3],
                "uncovered_pcs": [],
                "branches": {}
            },
            "values": {
                "local_0": {
                    "samples": [5, 10, 8],
                    "always_positive": True,
                    "never_negative": True,
                    "never_zero": True,
                    "sign": "positive",
                    "interval": [5, 10]
                },
                "local_1": {
                    "samples": [-2, -1],
                    "always_positive": False,
                    "never_negative": False,
                    "never_zero": True,
                    "sign": "negative",
                    "interval": [-2, -1]
                }
            }
        }
        
        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(trace_data, f)
            temp_path = f.name
        
        try:
            # Test refinement from trace
            refined = DomainRefinement.from_iin_trace(temp_path)
            
            assert "local_0" in refined
            assert "local_1" in refined
            
            # Check local_0 (positive values)
            sign0, interval0 = refined["local_0"]
            assert sign0 == SignDomain(SignValue.POSITIVE)
            assert interval0 == IntervalDomain(IntervalValue(5, 10))
            
            # Check local_1 (negative values) 
            sign1, interval1 = refined["local_1"]
            assert sign1 == SignDomain(SignValue.NEGATIVE)
            assert interval1 == IntervalDomain(IntervalValue(-2, -1))
            
        finally:
            # Cleanup
            Path(temp_path).unlink()
    
    def test_refinement_from_trace_properties_only(self):
        """Test refinement when only properties available (no samples)."""
        trace_data = {
            "method": "jpamb.cases.Simple.test:()V",
            "values": {
                "local_0": {
                    "always_positive": True,
                    "never_negative": True,
                    "sign": "positive",
                    "interval": [1, None]  # Open interval
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(trace_data, f)
            temp_path = f.name
        
        try:
            refined = DomainRefinement.from_iin_trace(temp_path)
            
            sign, interval = refined["local_0"]
            assert sign == SignDomain(SignValue.POSITIVE)
            assert interval == IntervalDomain(IntervalValue(1, 1))  # Fallback to single value
            
        finally:
            Path(temp_path).unlink()


class TestDomainIntegration:
    """Integration tests for complete domain operations."""
    
    def test_complex_join_operations(self):
        """Test complex sequences of join operations."""
        # Start with specific domains and join progressively
        pos = SignDomain(SignValue.POSITIVE)
        neg = SignDomain(SignValue.NEGATIVE)
        zero = SignDomain(SignValue.ZERO)
        
        # Progressive joining should reach TOP
        result = pos
        result = result.join(neg)  # NON_ZERO
        result = result.join(zero)  # TOP
        
        assert result == SignDomain(SignValue.TOP)
    
    def test_domain_lattice_properties(self):
        """Test that domains satisfy lattice properties."""
        pos = SignDomain(SignValue.POSITIVE)
        neg = SignDomain(SignValue.NEGATIVE)
        top = SignDomain(SignValue.TOP)
        bottom = SignDomain(SignValue.BOTTOM)
        
        # Commutativity: a ⊔ b = b ⊔ a
        assert pos.join(neg) == neg.join(pos)
        
        # Associativity: (a ⊔ b) ⊔ c = a ⊔ (b ⊔ c) 
        zero = SignDomain(SignValue.ZERO)
        left = pos.join(neg).join(zero)
        right = pos.join(neg.join(zero))
        assert left == right
        
        # Idempotence: a ⊔ a = a
        assert pos.join(pos) == pos
        
        # Absorption: a ⊔ ⊤ = ⊤
        assert pos.join(top) == top
        
        # Identity: a ⊔ ⊥ = a
        assert pos.join(bottom) == pos
    
    def test_real_world_loop_widening(self):
        """Test widening in a realistic loop scenario."""
        # Simulate loop counter: initially [0,0], then [0,1], [0,2], etc.
        counter = IntervalDomain(IntervalValue(0, 0))
        
        # First iteration: counter becomes [0,1]
        iter1 = IntervalDomain(IntervalValue(0, 1))
        counter = counter.widening(iter1)
        # Upper bound increased (1 > 0), so should widen to [0,+∞] immediately
        expected = IntervalDomain(IntervalValue(0, None))
        assert counter == expected
        
        # Further iterations should stabilize at [0,+∞]
        iter2 = IntervalDomain(IntervalValue(0, 2))
        result = counter.widening(iter2)
        assert result == counter  # Should remain [0,+∞]
        
        # Even with much larger bounds
        iter3 = IntervalDomain(IntervalValue(0, 100))
        result = counter.widening(iter3)
        assert result == counter  # Should remain [0,+∞]
    
    def test_refinement_integration_workflow(self):
        """Test complete workflow from traces to refined domains."""
        # Simulate a method that processes positive integers in a loop
        samples = [1, 5, 10, 15, 20]  # Increasing positive values
        
        # Refine initial domains
        sign, interval = DomainRefinement.from_concrete_values(samples)
        
        # Should get positive sign and tight interval
        assert sign == SignDomain(SignValue.POSITIVE)
        assert interval == IntervalDomain(IntervalValue(1, 20))
        
        # Simulate widening during loop analysis
        # If we see [1,25] in next iteration, widen upper bound
        next_iter = IntervalDomain(IntervalValue(1, 25))
        widened = interval.widening(next_iter)
        
        # Should widen upper bound to infinity
        assert widened == IntervalDomain(IntervalValue(1, None))
    
    def test_error_cases(self):
        """Test error handling and edge cases.""" 
        sign = SignDomain(SignValue.POSITIVE)
        interval = IntervalDomain(IntervalValue(1, 5))
        
        # Type errors for operations
        with pytest.raises(TypeError):
            sign.join("not_a_domain")
        
        with pytest.raises(TypeError):
            interval.meet(42)
        
        # Invalid interval creation (not the special BOTTOM case)
        with pytest.raises(ValueError):
            IntervalValue(10, 5)  # low > high (but not the special 1,0 case)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])