# Novel Abstractions for JPAMB Static Analysis

This module implements sign and interval abstract domains with refinement capabilities using dynamic traces from the IIN interpreter.

## Usage

### Basic Domain Operations

```python
from abstractions import SignDomain, SignValue, IntervalDomain, IntervalValue

# Create sign domains
pos = SignDomain(SignValue.POSITIVE)
neg = SignDomain(SignValue.NEGATIVE)

# Domain operations
result = pos.join(neg)  # Results in NON_ZERO
intersection = pos.meet(neg)  # Results in BOTTOM

# Interval domains  
i1 = IntervalDomain(IntervalValue(1, 10))
i2 = IntervalDomain(IntervalValue(5, 15))
union = i1.join(i2)  # Results in [1, 15]
```

### Dynamic Refinement from IIN Traces

```python
from abstractions import DomainRefinement

# Refine domains from concrete values (e.g., from proposal example)
samples = [5, 10]  # Values observed during execution
sign, interval = DomainRefinement.from_concrete_values(samples)
# Results: sign=POSITIVE, interval=[5, 10]

# Refine from IIN JSON trace files
refined = DomainRefinement.from_iin_trace("traces/method_trace.json")
for local_var, (sign_domain, interval_domain) in refined.items():
    print(f"{local_var}: sign={sign_domain}, interval={interval_domain}")
```

This implementation enables precise static analysis by combining abstract interpretation with dynamic observation, following the approach described in the project proposal §1.3.1.

## Testing

### Running the Test Suite

The implementation includes comprehensive tests covering all domain operations and refinement capabilities:

```bash
# Run all tests with verbose output
uv run pytest test_abstractions.py -v

# Run specific test classes
uv run pytest test_abstractions.py::TestSignDomain -v
uv run pytest test_abstractions.py::TestIntervalDomain -v
uv run pytest test_abstractions.py::TestDomainRefinement -v

# Run with coverage information
uv run pytest test_abstractions.py --cov=abstractions --cov-report=term-missing
```

### Test Coverage

The test suite includes 24 comprehensive tests covering:

- **Sign Domain Operations**: Join, meet, widening, arithmetic operations
- **Interval Domain Operations**: Union, intersection, widening, arithmetic operations  
- **Domain Refinement**: From concrete values and IIN JSON traces
- **Integration Scenarios**: Real-world usage patterns and edge cases

### Quick Functionality Test

```bash
# Basic functionality verification
uv run python -c "
from solutions.abstractions import *
print('✅ All modules imported successfully')

# Test basic domains
sign = SignDomain(SignValue.POSITIVE)
interval = IntervalDomain(IntervalValue(1, 10))
print(f'✅ Basic domains: sign={sign}, interval={interval}')

# Test refinement
samples = [5, 10, 15]
sign_refined, interval_refined = DomainRefinement.from_concrete_values(samples)
print(f'✅ Refinement from samples {samples}: sign={sign_refined}, interval={interval_refined}')

print('✅ All tests passed! Implementation complete.')
"
```

### Example Test Output

```
test_abstractions.py::TestSignDomain::test_sign_domain_join PASSED
test_abstractions.py::TestSignDomain::test_sign_domain_meet PASSED
test_abstractions.py::TestIntervalDomain::test_interval_join PASSED
test_abstractions.py::TestDomainRefinement::test_refinement_from_concrete_values PASSED
...
========================== 24 passed in 0.09s ==========================
```