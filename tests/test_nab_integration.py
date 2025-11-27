"""
Test suite for NAB Integration module.

Tests the integration between IIN (dynamic traces) and IAB (abstract domains).
Verifies that dynamic refinement heuristics correctly infer abstract states
from concrete execution samples.

**Course Definition (02242):**
"Run two or more abstractions at the same time, letting them inform each other
during execution" (formula: 5 per abstraction after the first).

This test suite verifies the Reduced Product implementation where SignDomain
and IntervalDomain run in parallel with mutual refinement.

DTU 02242 Program Analysis - Group 21
"""

import pytest
import json
import tempfile
import os
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import NAB integration module
from solutions.nab_integration import (
    integrate_abstractions,
    integrate_abstractions_full,
    integrate_abstractions_reduced,
    integrate_all_traces,
    extract_samples_from_trace,
    refine_from_trace,
    refine_from_trace_reduced,
    get_sign_for_local,
    get_interval_for_local,
    process_example,
    process_example_reduced,
    inform_each_other,
    AbstractValue,
    IntegrationResult,
    ReducedProductState,
)

# Import IAB abstract domains
from solutions.abstractions import (
    SignDomain,
    SignValue,
    IntervalDomain,
    IntervalValue,
)


class TestProposalExample:
    """Tests for the proposal example from §1.3.1."""
    
    def test_proposal_example_positive_samples(self):
        """
        Proposal example: process(int x) with samples [5, 10, ...]
        should refine local_1 to sign = positive.
        """
        result = process_example()
        
        assert 1 in result
        assert result[1].sign.value == SignValue.POSITIVE
        assert result[1].local_index == 1
    
    def test_proposal_example_interval(self):
        """
        Proposal example: samples [5, 10, 15, 20, 25] should give
        interval [5, 25].
        """
        result = process_example()
        
        assert 1 in result
        assert result[1].interval.value.low == 5
        assert result[1].interval.value.high == 25


class TestSignDomainRefinement:
    """Tests for sign domain refinement from samples."""
    
    def test_positive_samples_give_positive_sign(self):
        """All positive samples should refine to POSITIVE sign."""
        samples = [5, 10, 15, 100, 999]
        sign_domain, _ = refine_from_trace(samples)
        
        assert sign_domain.value == SignValue.POSITIVE
    
    def test_negative_samples_give_negative_sign(self):
        """All negative samples should refine to NEGATIVE sign."""
        samples = [-5, -10, -15, -100, -999]
        sign_domain, _ = refine_from_trace(samples)
        
        assert sign_domain.value == SignValue.NEGATIVE
    
    def test_zero_samples_give_zero_sign(self):
        """All zero samples should refine to ZERO sign."""
        samples = [0, 0, 0, 0]
        sign_domain, _ = refine_from_trace(samples)
        
        assert sign_domain.value == SignValue.ZERO
    
    def test_mixed_positive_negative_give_non_zero(self):
        """Mixed positive and negative (no zero) should give NON_ZERO."""
        samples = [5, -10, 15, -20]
        sign_domain, _ = refine_from_trace(samples)
        
        assert sign_domain.value == SignValue.NON_ZERO
    
    def test_positive_and_zero_give_non_negative(self):
        """Positive and zero samples should give NON_NEGATIVE."""
        samples = [0, 5, 10, 0, 15]
        sign_domain, _ = refine_from_trace(samples)
        
        assert sign_domain.value == SignValue.NON_NEGATIVE
    
    def test_negative_and_zero_give_non_positive(self):
        """Negative and zero samples should give NON_POSITIVE."""
        samples = [0, -5, -10, 0, -15]
        sign_domain, _ = refine_from_trace(samples)
        
        assert sign_domain.value == SignValue.NON_POSITIVE
    
    def test_all_signs_give_top(self):
        """Positive, negative, and zero should give TOP."""
        samples = [-5, 0, 5, -10, 10]
        sign_domain, _ = refine_from_trace(samples)
        
        assert sign_domain.value == SignValue.TOP
    
    def test_empty_samples_give_bottom(self):
        """Empty samples should give BOTTOM."""
        samples = []
        sign_domain, _ = refine_from_trace(samples)
        
        assert sign_domain.value == SignValue.BOTTOM


class TestIntervalDomainRefinement:
    """Tests for interval domain refinement from samples."""
    
    def test_interval_bounds_from_samples(self):
        """Interval should have min and max of samples."""
        samples = [5, 10, 15, 20, 25]
        _, interval_domain = refine_from_trace(samples)
        
        assert interval_domain.value.low == 5
        assert interval_domain.value.high == 25
    
    def test_single_sample_interval(self):
        """Single sample should give point interval."""
        samples = [42]
        _, interval_domain = refine_from_trace(samples)
        
        assert interval_domain.value.low == 42
        assert interval_domain.value.high == 42
    
    def test_negative_interval(self):
        """Negative samples should give negative interval."""
        samples = [-100, -50, -25, -10]
        _, interval_domain = refine_from_trace(samples)
        
        assert interval_domain.value.low == -100
        assert interval_domain.value.high == -10
    
    def test_mixed_interval(self):
        """Mixed samples should span negative to positive."""
        samples = [-50, -25, 0, 25, 50]
        _, interval_domain = refine_from_trace(samples)
        
        assert interval_domain.value.low == -50
        assert interval_domain.value.high == 50
    
    def test_empty_samples_give_bottom_interval(self):
        """Empty samples should give bottom interval."""
        samples = []
        _, interval_domain = refine_from_trace(samples)
        
        assert interval_domain.is_bottom()


class TestTraceExtraction:
    """Tests for extracting samples from IIN trace data."""
    
    def test_extract_samples_from_trace_data(self):
        """Extract samples from typical IIN trace format."""
        trace_data = {
            "method": "test.Method.foo:(I)V",
            "values": {
                "local_0": {
                    "samples": [1, 2, 3, 4, 5],
                    "sign": "positive"
                }
            }
        }
        
        samples = extract_samples_from_trace(trace_data)
        
        assert 0 in samples
        assert samples[0] == [1, 2, 3, 4, 5]
    
    def test_extract_multiple_locals(self):
        """Extract samples for multiple local variables."""
        trace_data = {
            "values": {
                "local_0": {"samples": [1, 2, 3]},
                "local_1": {"samples": [-1, -2, -3]},
                "local_2": {"samples": [0, 0, 0]}
            }
        }
        
        samples = extract_samples_from_trace(trace_data)
        
        assert len(samples) == 3
        assert samples[0] == [1, 2, 3]
        assert samples[1] == [-1, -2, -3]
        assert samples[2] == [0, 0, 0]
    
    def test_extract_empty_values(self):
        """Handle trace with no values section."""
        trace_data = {
            "method": "test.Method.foo:(I)V",
            "coverage": {"executed_pcs": [0, 1, 2]}
        }
        
        samples = extract_samples_from_trace(trace_data)
        
        assert samples == {}


class TestIntegration:
    """Integration tests using actual trace file format."""
    
    @pytest.fixture
    def temp_trace_file(self):
        """Create a temporary trace file for testing."""
        trace_data = {
            "method": "jpamb.cases.Test.process:(I)V",
            "coverage": {
                "executed_pcs": [0, 1, 2, 3],
                "uncovered_pcs": [4, 5],
                "branches": {"1": [True]}
            },
            "values": {
                "local_0": {
                    "samples": [5, 10, 15, 20, 25],
                    "always_positive": True,
                    "never_negative": True,
                    "never_zero": True,
                    "sign": "positive",
                    "interval": [5, 25]
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(trace_data, f)
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        os.unlink(temp_path)
    
    def test_integrate_abstractions_from_file(self, temp_trace_file):
        """Test main integration function with temp file."""
        result = integrate_abstractions(temp_trace_file)
        
        assert 0 in result
        assert result[0].sign.value == SignValue.POSITIVE
        assert result[0].interval.value.low == 5
        assert result[0].interval.value.high == 25
    
    def test_integrate_abstractions_full(self, temp_trace_file):
        """Test full integration with metadata."""
        result = integrate_abstractions_full(temp_trace_file)
        
        assert isinstance(result, IntegrationResult)
        assert result.method_name == "jpamb.cases.Test.process:(I)V"
        assert 0 in result.abstract_values
        assert result.confidence > 0.5
    
    def test_abstract_value_to_dict(self, temp_trace_file):
        """Test AbstractValue serialization."""
        result = integrate_abstractions(temp_trace_file)
        
        assert 0 in result
        value_dict = result[0].to_dict()
        
        assert "local_index" in value_dict
        assert "sign" in value_dict
        assert "interval" in value_dict
        assert value_dict["local_index"] == 0


class TestRealTraces:
    """Tests against actual traces in traces/ directory."""
    
    @pytest.fixture
    def traces_dir(self):
        """Get the traces directory path."""
        base_dir = Path(__file__).parent.parent
        traces_path = base_dir / "traces"
        if traces_path.exists():
            return traces_path
        return None
    
    def test_integrate_real_trace_positive(self, traces_dir):
        """Test integration with real assertPositive trace."""
        if traces_dir is None:
            pytest.skip("traces directory not found")
        
        trace_file = traces_dir / "jpamb.cases.Simple_assertPositive_IV.json"
        if not trace_file.exists():
            pytest.skip("assertPositive trace not found")
        
        result = integrate_abstractions(str(trace_file))
        
        # local_0 should have positive samples [1,1,1,1,1]
        assert 0 in result
        assert result[0].sign.value == SignValue.POSITIVE
    
    def test_integrate_all_traces(self, traces_dir):
        """Test integration of all traces in directory."""
        if traces_dir is None:
            pytest.skip("traces directory not found")
        
        results = integrate_all_traces(str(traces_dir))
        
        # Should have results for multiple methods
        assert len(results) > 0
        
        # Each result should have valid structure
        for method_name, result in results.items():
            assert isinstance(result, IntegrationResult)
            assert result.method_name == method_name


class TestEdgeCases:
    """Edge case tests."""
    
    def test_file_not_found(self):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            integrate_abstractions("nonexistent_file.json")
    
    def test_get_sign_for_local_convenience(self):
        """Test convenience function for getting sign."""
        # Create temp file
        trace_data = {
            "values": {
                "local_0": {"samples": [5, 10, 15]}
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(trace_data, f)
            temp_path = f.name
        
        try:
            sign = get_sign_for_local(temp_path, 0)
            assert sign == SignValue.POSITIVE
            
            # Non-existent local should return TOP
            sign_missing = get_sign_for_local(temp_path, 99)
            assert sign_missing == SignValue.TOP
        finally:
            os.unlink(temp_path)
    
    def test_get_interval_for_local_convenience(self):
        """Test convenience function for getting interval."""
        trace_data = {
            "values": {
                "local_0": {"samples": [5, 10, 15]}
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(trace_data, f)
            temp_path = f.name
        
        try:
            interval = get_interval_for_local(temp_path, 0)
            assert interval.low == 5
            assert interval.high == 15
            
            # Non-existent local should return TOP
            interval_missing = get_interval_for_local(temp_path, 99)
            assert interval_missing.low is None
            assert interval_missing.high is None
        finally:
            os.unlink(temp_path)


class TestBooleanSamples:
    """Tests for handling boolean samples."""
    
    def test_boolean_true_samples(self):
        """Boolean True values should be treated as 1."""
        trace_data = {
            "values": {
                "local_0": {"samples": [True, True, True]}
            }
        }
        
        samples = extract_samples_from_trace(trace_data)
        
        assert 0 in samples
        # True is treated as integer 1
        assert all(isinstance(s, (int, bool)) for s in samples[0])
    
    def test_mixed_boolean_integer(self):
        """Mixed boolean and integer samples."""
        trace_data = {
            "values": {
                "local_0": {"samples": [True, 1, False, 0]}
            }
        }
        
        samples = extract_samples_from_trace(trace_data)
        
        assert 0 in samples
        assert len(samples[0]) == 4


class TestReducedProductState:
    """
    Tests for ReducedProductState - the core NAB implementation.
    
    Course Definition: "Run two or more abstractions at the same time,
    letting them inform each other during execution"
    """
    
    def test_reduced_product_from_samples_positive(self):
        """Positive samples should create reduced product with mutual refinement."""
        samples = [5, 10, 15, 20, 25]
        reduced = ReducedProductState.from_samples(samples)
        
        assert reduced.sign.value == SignValue.POSITIVE
        assert reduced.interval.value.low == 5
        assert reduced.interval.value.high == 25
    
    def test_reduced_product_from_samples_negative(self):
        """Negative samples should create reduced product with mutual refinement."""
        samples = [-25, -20, -15, -10, -5]
        reduced = ReducedProductState.from_samples(samples)
        
        assert reduced.sign.value == SignValue.NEGATIVE
        assert reduced.interval.value.low == -25
        assert reduced.interval.value.high == -5
    
    def test_reduced_product_from_samples_zero(self):
        """Zero samples should refine to ZERO sign and [0,0] interval."""
        samples = [0, 0, 0]
        reduced = ReducedProductState.from_samples(samples)
        
        assert reduced.sign.value == SignValue.ZERO
        assert reduced.interval.value.low == 0
        assert reduced.interval.value.high == 0
    
    def test_reduced_product_top(self):
        """TOP state should have no constraints."""
        reduced = ReducedProductState.top()
        
        assert reduced.sign.value == SignValue.TOP
        assert reduced.interval.value.low is None
        assert reduced.interval.value.high is None
        assert reduced.is_top()
    
    def test_reduced_product_bottom(self):
        """BOTTOM state should be unreachable."""
        reduced = ReducedProductState.bottom()
        
        assert reduced.sign.value == SignValue.BOTTOM
        assert reduced.is_bottom()


class TestParallelExecution:
    """
    Tests for parallel execution of sign and interval domains.
    
    These tests verify the course requirement:
    "Run two or more abstractions at the same time, letting them inform
    each other during execution" (formula: 5 per abstraction after the first)
    """
    
    def test_parallel_positive_sign_tightens_interval_low(self):
        """
        Parallel execution: POSITIVE sign should tighten interval low bound to 1.
        
        Course definition: abstractions inform each other during execution.
        """
        sign = SignDomain(SignValue.POSITIVE)
        interval = IntervalDomain(IntervalValue(-5, 10))
        
        new_sign, new_interval = inform_each_other(sign, interval)
        
        # Sign POSITIVE should tighten low bound to max(-5, 1) = 1
        assert new_interval.value.low == 1
        assert new_interval.value.high == 10
        assert new_sign.value == SignValue.POSITIVE
    
    def test_parallel_negative_sign_tightens_interval_high(self):
        """
        Parallel execution: NEGATIVE sign should tighten interval high bound to -1.
        """
        sign = SignDomain(SignValue.NEGATIVE)
        interval = IntervalDomain(IntervalValue(-10, 5))
        
        new_sign, new_interval = inform_each_other(sign, interval)
        
        # Sign NEGATIVE should tighten high bound to min(5, -1) = -1
        assert new_interval.value.low == -10
        assert new_interval.value.high == -1
        assert new_sign.value == SignValue.NEGATIVE
    
    def test_parallel_zero_sign_constrains_interval(self):
        """
        Parallel execution: ZERO sign should constrain interval to [0, 0].
        """
        sign = SignDomain(SignValue.ZERO)
        interval = IntervalDomain(IntervalValue(-10, 10))
        
        new_sign, new_interval = inform_each_other(sign, interval)
        
        # Sign ZERO should constrain interval to [0, 0]
        assert new_interval.value.low == 0
        assert new_interval.value.high == 0
        assert new_sign.value == SignValue.ZERO
    
    def test_parallel_interval_positive_infers_sign(self):
        """
        Parallel execution: Interval [a, b] where a > 0 should refine sign to POSITIVE.
        """
        sign = SignDomain(SignValue.TOP)
        interval = IntervalDomain(IntervalValue(5, 100))
        
        new_sign, new_interval = inform_each_other(sign, interval)
        
        # Interval [5, 100] should infer POSITIVE sign
        assert new_sign.value == SignValue.POSITIVE
    
    def test_parallel_interval_negative_infers_sign(self):
        """
        Parallel execution: Interval [a, b] where b < 0 should refine sign to NEGATIVE.
        """
        sign = SignDomain(SignValue.TOP)
        interval = IntervalDomain(IntervalValue(-100, -5))
        
        new_sign, new_interval = inform_each_other(sign, interval)
        
        # Interval [-100, -5] should infer NEGATIVE sign
        assert new_sign.value == SignValue.NEGATIVE
    
    def test_parallel_interval_zero_infers_sign(self):
        """
        Parallel execution: Interval [0, 0] should refine sign to ZERO.
        """
        sign = SignDomain(SignValue.TOP)
        interval = IntervalDomain(IntervalValue(0, 0))
        
        new_sign, new_interval = inform_each_other(sign, interval)
        
        # Interval [0, 0] should infer ZERO sign
        assert new_sign.value == SignValue.ZERO
    
    def test_parallel_non_negative_sign_tightens_interval(self):
        """
        Parallel execution: NON_NEGATIVE sign should tighten interval low to 0.
        """
        sign = SignDomain(SignValue.NON_NEGATIVE)
        interval = IntervalDomain(IntervalValue(-10, 10))
        
        new_sign, new_interval = inform_each_other(sign, interval)
        
        # NON_NEGATIVE should tighten low bound to max(-10, 0) = 0
        assert new_interval.value.low == 0
        assert new_interval.value.high == 10
    
    def test_parallel_non_positive_sign_tightens_interval(self):
        """
        Parallel execution: NON_POSITIVE sign should tighten interval high to 0.
        """
        sign = SignDomain(SignValue.NON_POSITIVE)
        interval = IntervalDomain(IntervalValue(-10, 10))
        
        new_sign, new_interval = inform_each_other(sign, interval)
        
        # NON_POSITIVE should tighten high bound to min(10, 0) = 0
        assert new_interval.value.low == -10
        assert new_interval.value.high == 0
    
    def test_parallel_refinement_bidirectional(self):
        """
        Test bidirectional refinement: both domains inform each other.
        
        This is the key test for the course requirement.
        """
        # Start with sign POSITIVE and interval including negatives
        reduced = ReducedProductState(
            sign=SignDomain(SignValue.POSITIVE),
            interval=IntervalDomain(IntervalValue(-5, 100))
        )
        
        # Apply mutual refinement
        changed = reduced.inform_each_other()
        
        # Sign POSITIVE should have tightened interval to [1, 100]
        assert reduced.interval.value.low == 1
        assert reduced.interval.value.high == 100
        assert changed  # Refinement should have occurred
    
    def test_parallel_refinement_history_tracked(self):
        """
        Test that refinement history is tracked for debugging.
        """
        reduced = ReducedProductState(
            sign=SignDomain(SignValue.POSITIVE),
            interval=IntervalDomain(IntervalValue(-5, 10))
        )
        
        reduced.inform_each_other()
        
        history = reduced.get_refinement_history()
        assert len(history) > 0
        assert any("POSITIVE" in h for h in history)


class TestMutualRefinementEdgeCases:
    """
    Edge case tests for mutual refinement.
    """
    
    def test_inconsistent_sign_interval_becomes_bottom(self):
        """
        Inconsistent sign and interval should result in BOTTOM.
        """
        # POSITIVE sign with all-negative interval is inconsistent
        reduced = ReducedProductState(
            sign=SignDomain(SignValue.POSITIVE),
            interval=IntervalDomain(IntervalValue(-100, -1))
        )
        
        reduced.inform_each_other()
        
        # Should be bottom due to inconsistency
        assert reduced.is_bottom()
    
    def test_already_consistent_no_change(self):
        """
        Already consistent state should not change.
        """
        reduced = ReducedProductState(
            sign=SignDomain(SignValue.POSITIVE),
            interval=IntervalDomain(IntervalValue(1, 10))
        )
        
        # Already consistent - low >= 1 for POSITIVE
        changed = reduced.inform_each_other()
        
        # May have minor refinement but result should be same
        assert reduced.sign.value == SignValue.POSITIVE
        assert reduced.interval.value.low == 1
        assert reduced.interval.value.high == 10
    
    def test_empty_samples_gives_bottom(self):
        """
        Empty samples should give BOTTOM reduced product.
        """
        reduced = ReducedProductState.from_samples([])
        
        assert reduced.is_bottom()
    
    def test_single_sample_point_interval(self):
        """
        Single sample should give point interval with appropriate sign.
        """
        reduced = ReducedProductState.from_samples([42])
        
        assert reduced.sign.value == SignValue.POSITIVE
        assert reduced.interval.value.low == 42
        assert reduced.interval.value.high == 42


class TestReducedProductOperations:
    """
    Tests for join, meet, and widening operations on reduced product.
    """
    
    def test_reduced_product_join(self):
        """
        Join of two reduced products should be least upper bound.
        """
        rp1 = ReducedProductState.from_samples([5, 10])
        rp2 = ReducedProductState.from_samples([15, 20])
        
        joined = rp1.join(rp2)
        
        # Join should give interval [5, 20] and POSITIVE sign
        assert joined.sign.value == SignValue.POSITIVE
        assert joined.interval.value.low == 5
        assert joined.interval.value.high == 20
    
    def test_reduced_product_meet(self):
        """
        Meet of two reduced products should be greatest lower bound.
        """
        rp1 = ReducedProductState(
            sign=SignDomain(SignValue.POSITIVE),
            interval=IntervalDomain(IntervalValue(1, 20))
        )
        rp2 = ReducedProductState(
            sign=SignDomain(SignValue.POSITIVE),
            interval=IntervalDomain(IntervalValue(10, 30))
        )
        
        met = rp1.meet(rp2)
        
        # Meet should give interval [10, 20]
        assert met.sign.value == SignValue.POSITIVE
        assert met.interval.value.low == 10
        assert met.interval.value.high == 20
    
    def test_reduced_product_widening(self):
        """
        Widening should ensure termination in fixpoint computation.
        """
        rp1 = ReducedProductState(
            sign=SignDomain(SignValue.POSITIVE),
            interval=IntervalDomain(IntervalValue(1, 10))
        )
        rp2 = ReducedProductState(
            sign=SignDomain(SignValue.POSITIVE),
            interval=IntervalDomain(IntervalValue(1, 20))
        )
        
        widened = rp1.widening(rp2)
        
        # Widening should expand upper bound to infinity
        assert widened.interval.value.low == 1
        assert widened.interval.value.high is None  # +∞


class TestIntegrationWithReducedProduct:
    """
    Tests for integration functions using reduced product.
    """
    
    @pytest.fixture
    def temp_trace_file(self):
        """Create a temporary trace file for testing."""
        trace_data = {
            "method": "jpamb.cases.Test.process:(I)V",
            "coverage": {
                "executed_pcs": [0, 1, 2, 3],
                "uncovered_pcs": [4, 5],
                "branches": {"1": [True]}
            },
            "values": {
                "local_0": {
                    "samples": [5, 10, 15, 20, 25],
                    "always_positive": True,
                    "never_negative": True,
                    "never_zero": True,
                    "sign": "positive",
                    "interval": [5, 25]
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(trace_data, f)
            temp_path = f.name
        
        yield temp_path
        
        os.unlink(temp_path)
    
    def test_integrate_abstractions_reduced(self, temp_trace_file):
        """Test integration returning ReducedProductState."""
        result = integrate_abstractions_reduced(temp_trace_file)
        
        assert 0 in result
        assert isinstance(result[0], ReducedProductState)
        assert result[0].sign.value == SignValue.POSITIVE
        assert result[0].interval.value.low == 5
        assert result[0].interval.value.high == 25
    
    def test_refine_from_trace_reduced(self):
        """Test refine_from_trace_reduced returns ReducedProductState."""
        samples = [10, 20, 30]
        reduced = refine_from_trace_reduced(samples)
        
        assert isinstance(reduced, ReducedProductState)
        assert reduced.sign.value == SignValue.POSITIVE
        assert reduced.interval.value.low == 10
        assert reduced.interval.value.high == 30
    
    def test_process_example_reduced(self):
        """Test process_example_reduced returns ReducedProductState."""
        result = process_example_reduced()
        
        assert 1 in result
        assert isinstance(result[1], ReducedProductState)
        assert result[1].sign.value == SignValue.POSITIVE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
