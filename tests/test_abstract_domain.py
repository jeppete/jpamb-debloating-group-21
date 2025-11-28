"""
Test suite for abstract domains (SignSet, IntervalDomain).

Tests cover sign and interval domains, their operations (join, meet, widening),
and arithmetic operations. These are the core building blocks for abstract interpretation.

DTU 02242 Program Analysis - Group 21
"""

import pytest
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from solutions.abstract_domain import (
    SignSet,
    SignArithmetic,
    IntervalDomain,
    IntervalValue,
    IntervalArithmetic,
)


class TestSignSetCreation:
    """Test cases for SignSet creation and basic properties."""
    
    def test_signset_bottom(self):
        """Test SignSet.bottom() creates empty set."""
        bottom = SignSet.bottom()
        assert bottom.signs == frozenset()
        assert bottom.is_bottom()
        assert not bottom.is_top()
        assert not bottom  # bool(bottom) should be False
    
    def test_signset_top(self):
        """Test SignSet.top() creates full set."""
        top = SignSet.top()
        assert top.signs == frozenset({"+", "0", "-"})
        assert top.is_top()
        assert not top.is_bottom()
        assert top  # bool(top) should be True
    
    def test_signset_const_positive(self):
        """Test SignSet.const for positive values."""
        s = SignSet.const(5)
        assert s.signs == frozenset({"+"})
        assert 5 in s
        assert -5 not in s
        assert 0 not in s
    
    def test_signset_const_negative(self):
        """Test SignSet.const for negative values."""
        s = SignSet.const(-5)
        assert s.signs == frozenset({"-"})
        assert -5 in s
        assert 5 not in s
        assert 0 not in s
    
    def test_signset_const_zero(self):
        """Test SignSet.const for zero."""
        s = SignSet.const(0)
        assert s.signs == frozenset({"0"})
        assert 0 in s
        assert 5 not in s
        assert -5 not in s
    
    def test_signset_abstract_positive_values(self):
        """Test SignSet.abstract for positive values."""
        s = SignSet.abstract([1, 5, 100])
        assert s.signs == frozenset({"+"})
    
    def test_signset_abstract_mixed_values(self):
        """Test SignSet.abstract for mixed values."""
        s = SignSet.abstract([-5, 0, 5])
        assert s.signs == frozenset({"+", "0", "-"})
        assert s.is_top()
    
    def test_signset_abstract_empty(self):
        """Test SignSet.abstract for empty iterable."""
        s = SignSet.abstract([])
        assert s.is_bottom()
    
    def test_signset_repr(self):
        """Test SignSet string representation."""
        assert repr(SignSet.bottom()) == "⊥"
        assert repr(SignSet.top()) == "⊤"
        assert repr(SignSet.const(5)) == "{+}"


class TestSignSetLattice:
    """Test SignSet lattice operations."""
    
    def test_signset_order_bottom_le_all(self):
        """Bottom is below all other elements."""
        bottom = SignSet.bottom()
        top = SignSet.top()
        pos = SignSet.const(5)
        
        assert bottom <= top
        assert bottom <= pos
        assert bottom <= bottom
    
    def test_signset_order_all_le_top(self):
        """All elements are below top."""
        bottom = SignSet.bottom()
        top = SignSet.top()
        pos = SignSet.const(5)
        
        assert bottom <= top
        assert pos <= top
        assert top <= top
    
    def test_signset_join_basic(self):
        """Test SignSet join (union) operation."""
        pos = SignSet.const(5)
        neg = SignSet.const(-5)
        zero = SignSet.const(0)
        
        # Basic joins
        assert (pos | pos).signs == frozenset({"+"})
        assert (pos | neg).signs == frozenset({"+", "-"})
        assert (pos | zero).signs == frozenset({"+", "0"})
        assert (neg | zero).signs == frozenset({"-", "0"})
    
    def test_signset_join_with_bottom(self):
        """Bottom is identity for join."""
        pos = SignSet.const(5)
        bottom = SignSet.bottom()
        
        assert pos | bottom == pos
        assert bottom | pos == pos
    
    def test_signset_join_to_top(self):
        """Join of all signs gives top."""
        pos = SignSet.const(5)
        neg = SignSet.const(-5)
        zero = SignSet.const(0)
        
        result = pos | neg | zero
        assert result.is_top()
    
    def test_signset_meet_basic(self):
        """Test SignSet meet (intersection) operation."""
        pos = SignSet.const(5)
        neg = SignSet.const(-5)
        non_neg = SignSet(frozenset({"+", "0"}))
        
        # Intersection with same = same
        assert (pos & pos).signs == frozenset({"+"})
        
        # Intersection of disjoint = bottom
        assert (pos & neg).is_bottom()
        
        # Intersection with superset = self
        assert (pos & non_neg).signs == frozenset({"+"})
    
    def test_signset_meet_with_top(self):
        """Top is identity for meet."""
        pos = SignSet.const(5)
        top = SignSet.top()
        
        assert pos & top == pos
        assert top & pos == pos
    
    def test_signset_meet_with_bottom(self):
        """Bottom absorbs in meet."""
        pos = SignSet.const(5)
        bottom = SignSet.bottom()
        
        assert (pos & bottom).is_bottom()
        assert (bottom & pos).is_bottom()


class TestSignArithmetic:
    """Test SignArithmetic operations."""
    
    def test_add_positive_positive(self):
        """+ + + = +."""
        pos = SignSet.const(5)
        result = SignArithmetic.add(pos, pos)
        assert result.signs == frozenset({"+"})
    
    def test_add_negative_negative(self):
        """- + - = -."""
        neg = SignSet.const(-5)
        result = SignArithmetic.add(neg, neg)
        assert result.signs == frozenset({"-"})
    
    def test_add_positive_negative(self):
        """+ + - = {+, 0, -}."""
        pos = SignSet.const(5)
        neg = SignSet.const(-5)
        result = SignArithmetic.add(pos, neg)
        assert result.is_top()
    
    def test_add_zero_anything(self):
        """0 + x = x."""
        zero = SignSet.const(0)
        pos = SignSet.const(5)
        neg = SignSet.const(-5)
        
        assert SignArithmetic.add(zero, pos).signs == frozenset({"+"})
        assert SignArithmetic.add(zero, neg).signs == frozenset({"-"})
        assert SignArithmetic.add(zero, zero).signs == frozenset({"0"})
    
    def test_add_with_bottom(self):
        """Adding with bottom gives bottom."""
        pos = SignSet.const(5)
        bottom = SignSet.bottom()
        
        assert SignArithmetic.add(pos, bottom).is_bottom()
        assert SignArithmetic.add(bottom, pos).is_bottom()
    
    def test_sub_positive_positive(self):
        """+ - + can be anything."""
        pos = SignSet.const(5)
        result = SignArithmetic.sub(pos, pos)
        assert result.is_top()  # 5-3=2 (+), 3-3=0 (0), 3-5=-2 (-)
    
    def test_sub_positive_negative(self):
        """+ - (-) = +."""
        pos = SignSet.const(5)
        neg = SignSet.const(-5)
        result = SignArithmetic.sub(pos, neg)
        assert result.signs == frozenset({"+"})  # 5 - (-5) = 10
    
    def test_mul_positive_positive(self):
        """+ * + = +."""
        pos = SignSet.const(5)
        result = SignArithmetic.mul(pos, pos)
        assert result.signs == frozenset({"+"})
    
    def test_mul_negative_negative(self):
        """- * - = +."""
        neg = SignSet.const(-5)
        result = SignArithmetic.mul(neg, neg)
        assert result.signs == frozenset({"+"})
    
    def test_mul_positive_negative(self):
        """+ * - = -."""
        pos = SignSet.const(5)
        neg = SignSet.const(-5)
        result = SignArithmetic.mul(pos, neg)
        assert result.signs == frozenset({"-"})
    
    def test_mul_zero_anything(self):
        """0 * anything = 0."""
        zero = SignSet.const(0)
        pos = SignSet.const(5)
        neg = SignSet.const(-5)
        
        assert SignArithmetic.mul(zero, pos).signs == frozenset({"0"})
        assert SignArithmetic.mul(zero, neg).signs == frozenset({"0"})
        assert SignArithmetic.mul(pos, zero).signs == frozenset({"0"})
    
    def test_div_positive_positive(self):
        """+ / + = + or 0."""
        pos = SignSet.const(5)
        result, may_div_zero = SignArithmetic.div(pos, pos)
        assert "+" in result.signs or "0" in result.signs
        assert not may_div_zero
    
    def test_div_by_zero_detected(self):
        """Division by zero is detected."""
        pos = SignSet.const(5)
        zero = SignSet.const(0)
        result, may_div_zero = SignArithmetic.div(pos, zero)
        assert may_div_zero or result.is_bottom()
    
    def test_div_by_maybe_zero(self):
        """Division by possibly-zero value flags may_div_zero."""
        pos = SignSet.const(5)
        non_neg = SignSet(frozenset({"+", "0"}))
        result, may_div_zero = SignArithmetic.div(pos, non_neg)
        assert may_div_zero
    
    def test_neg_positive(self):
        """neg(+) = -."""
        pos = SignSet.const(5)
        result = SignArithmetic.neg(pos)
        assert result.signs == frozenset({"-"})
    
    def test_neg_negative(self):
        """neg(-) = +."""
        neg = SignSet.const(-5)
        result = SignArithmetic.neg(neg)
        assert result.signs == frozenset({"+"})
    
    def test_neg_zero(self):
        """neg(0) = 0."""
        zero = SignSet.const(0)
        result = SignArithmetic.neg(zero)
        assert result.signs == frozenset({"0"})
    
    def test_rem_positive_positive(self):
        """+ % + = {+, 0}."""
        pos = SignSet.const(5)
        result, may_div_zero = SignArithmetic.rem(pos, pos)
        assert "+" in result.signs or "0" in result.signs
        assert not may_div_zero


class TestIntervalValueCreation:
    """Test IntervalValue dataclass."""
    
    def test_interval_value_bounded(self):
        """Test bounded interval."""
        iv = IntervalValue(1, 10)
        assert iv.low == 1
        assert iv.high == 10
        assert not iv.is_bottom()
        assert not iv.is_top()
        assert str(iv) == "[1, 10]"
    
    def test_interval_value_unbounded_low(self):
        """Test interval with unbounded low."""
        iv = IntervalValue(None, 10)
        assert iv.low is None
        assert iv.high == 10
        assert not iv.is_top()
        assert str(iv) == "[-∞, 10]"
    
    def test_interval_value_unbounded_high(self):
        """Test interval with unbounded high."""
        iv = IntervalValue(5, None)
        assert iv.low == 5
        assert iv.high is None
        assert not iv.is_top()
        assert str(iv) == "[5, +∞]"
    
    def test_interval_value_top(self):
        """Test top interval (fully unbounded)."""
        iv = IntervalValue(None, None)
        assert iv.is_top()
        assert str(iv) == "[-∞, +∞]"
    
    def test_interval_value_point(self):
        """Test point interval [n, n]."""
        iv = IntervalValue(5, 5)
        assert iv.low == 5
        assert iv.high == 5
        assert iv.contains(5)
        assert not iv.contains(4)
        assert not iv.contains(6)
    
    def test_interval_value_contains(self):
        """Test contains method."""
        iv = IntervalValue(1, 10)
        assert iv.contains(1)
        assert iv.contains(5)
        assert iv.contains(10)
        assert not iv.contains(0)
        assert not iv.contains(11)


class TestIntervalDomainCreation:
    """Test IntervalDomain creation and methods."""
    
    def test_interval_domain_bottom(self):
        """Test IntervalDomain.bottom()."""
        bottom = IntervalDomain.bottom()
        assert bottom.is_bottom()
        assert not bottom.is_top()
        assert not bottom  # bool should be False
    
    def test_interval_domain_top(self):
        """Test IntervalDomain.top()."""
        top = IntervalDomain.top()
        assert top.is_top()
        assert not top.is_bottom()
        assert top  # bool should be True
    
    def test_interval_domain_const(self):
        """Test IntervalDomain.const() for point interval."""
        c = IntervalDomain.const(5)
        assert c.value.low == 5
        assert c.value.high == 5
        assert 5 in c
        assert 4 not in c
    
    def test_interval_domain_range(self):
        """Test IntervalDomain.range() for bounded interval."""
        r = IntervalDomain.range(1, 10)
        assert r.value.low == 1
        assert r.value.high == 10
    
    def test_interval_domain_abstract(self):
        """Test IntervalDomain.abstract() from values."""
        d = IntervalDomain.abstract([3, 7, 5, 1, 9])
        assert d.value.low == 1
        assert d.value.high == 9
    
    def test_interval_domain_abstract_empty(self):
        """Test IntervalDomain.abstract() with empty list."""
        d = IntervalDomain.abstract([])
        assert d.is_bottom()


class TestIntervalDomainLattice:
    """Test IntervalDomain lattice operations."""
    
    def test_interval_order_subset(self):
        """Test subset ordering."""
        small = IntervalDomain.range(3, 7)
        large = IntervalDomain.range(1, 10)
        
        assert small <= large
        assert not large <= small
    
    def test_interval_join_overlapping(self):
        """Test join of overlapping intervals."""
        a = IntervalDomain.range(1, 5)
        b = IntervalDomain.range(3, 10)
        
        result = a | b
        assert result.value.low == 1
        assert result.value.high == 10
    
    def test_interval_join_disjoint(self):
        """Test join of disjoint intervals."""
        a = IntervalDomain.range(1, 3)
        b = IntervalDomain.range(7, 10)
        
        result = a | b
        assert result.value.low == 1
        assert result.value.high == 10
    
    def test_interval_join_with_bottom(self):
        """Bottom is identity for join."""
        a = IntervalDomain.range(1, 5)
        bottom = IntervalDomain.bottom()
        
        assert a | bottom == a
        assert bottom | a == a
    
    def test_interval_meet_overlapping(self):
        """Test meet of overlapping intervals."""
        a = IntervalDomain.range(1, 7)
        b = IntervalDomain.range(5, 10)
        
        result = a & b
        assert result.value.low == 5
        assert result.value.high == 7
    
    def test_interval_meet_disjoint(self):
        """Test meet of disjoint intervals gives bottom."""
        a = IntervalDomain.range(1, 3)
        b = IntervalDomain.range(7, 10)
        
        result = a & b
        assert result.is_bottom()
    
    def test_interval_meet_with_top(self):
        """Top is identity for meet."""
        a = IntervalDomain.range(1, 5)
        top = IntervalDomain.top()
        
        assert a & top == a
        assert top & a == a
    
    def test_interval_widening_expanding(self):
        """Test widening expands bounds to infinity."""
        a = IntervalDomain.range(0, 10)
        b = IntervalDomain.range(0, 20)  # high is growing
        
        result = a.widening(b)
        assert result.value.low == 0
        assert result.value.high is None  # widened to +∞
    
    def test_interval_widening_shrinking_low(self):
        """Test widening expands low to -∞ when shrinking."""
        a = IntervalDomain.range(5, 10)
        b = IntervalDomain.range(2, 10)  # low is shrinking
        
        result = a.widening(b)
        assert result.value.low is None  # widened to -∞
        assert result.value.high == 10


class TestIntervalArithmetic:
    """Test IntervalArithmetic operations."""
    
    def test_add_bounded(self):
        """Test addition of bounded intervals."""
        a = IntervalDomain.range(1, 5)
        b = IntervalDomain.range(10, 20)
        
        result = IntervalArithmetic.add(a, b)
        assert result.value.low == 11
        assert result.value.high == 25
    
    def test_add_with_bottom(self):
        """Adding with bottom gives bottom."""
        a = IntervalDomain.range(1, 5)
        bottom = IntervalDomain.bottom()
        
        assert IntervalArithmetic.add(a, bottom).is_bottom()
    
    def test_sub_bounded(self):
        """Test subtraction of bounded intervals."""
        a = IntervalDomain.range(10, 20)
        b = IntervalDomain.range(1, 5)
        
        result = IntervalArithmetic.sub(a, b)
        assert result.value.low == 5   # 10 - 5 = 5
        assert result.value.high == 19  # 20 - 1 = 19
    
    def test_mul_bounded_positive(self):
        """Test multiplication of positive intervals."""
        a = IntervalDomain.range(2, 4)
        b = IntervalDomain.range(3, 5)
        
        result = IntervalArithmetic.mul(a, b)
        assert result.value.low == 6   # 2 * 3
        assert result.value.high == 20  # 4 * 5
    
    def test_mul_mixed_signs(self):
        """Test multiplication with mixed signs."""
        a = IntervalDomain.range(-2, 3)
        b = IntervalDomain.range(-4, 5)
        
        result = IntervalArithmetic.mul(a, b)
        # Products: (-2)*(-4)=8, (-2)*5=-10, 3*(-4)=-12, 3*5=15
        assert result.value.low == -12
        assert result.value.high == 15
    
    def test_neg_interval(self):
        """Test interval negation."""
        a = IntervalDomain.range(3, 10)
        result = IntervalArithmetic.neg(a)
        assert result.value.low == -10
        assert result.value.high == -3


class TestDomainContainment:
    """Test containment checks for abstract domains."""
    
    def test_signset_contains_concrete(self):
        """Test SignSet.__contains__ for concrete values."""
        non_neg = SignSet(frozenset({"+", "0"}))
        
        assert 5 in non_neg
        assert 0 in non_neg
        assert -5 not in non_neg
    
    def test_interval_contains_concrete(self):
        """Test IntervalDomain.__contains__ for concrete values."""
        interval = IntervalDomain.range(1, 10)
        
        assert 1 in interval
        assert 5 in interval
        assert 10 in interval
        assert 0 not in interval
        assert 11 not in interval


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
