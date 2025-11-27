# NAB – Integration of Abstractions

## Overview

NAB (Novel Abstract integration) integrates dynamic execution traces from IIN with abstract domains from IAB to produce refined initial abstract states for static analysis.

This module implements the core innovation of our DTU 02242 project: **dynamic refinement heuristics** that use observed concrete execution values to set initial abstract states, enabling more precise static analysis.

## Course Definition (02242)

> **"Run two or more abstractions at the same time, letting them inform each other during execution"**
> 
> (formula: 5 per abstraction after the first)

This module implements a **Reduced Product** of SignDomain and IntervalDomain, satisfying the course definition by:
1. Running both sign and interval abstractions **in parallel**
2. **Mutual refinement**: sign information tightens interval bounds, and vice versa
3. **Dynamic refinement**: using IIN traces to initialize abstract states

### Scoring
- **First abstraction (SignDomain)**: base
- **Second abstraction (IntervalDomain)**: +5 points
- **Parallel execution with mutual refinement**: implements "letting them inform each other"

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     NAB Integration Layer                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   IIN (Interpreter)              IAB (Abstractions)             │
│   ─────────────────              ──────────────────             │
│   • Executes bytecode            • SignDomain                   │
│   • Produces traces/             • IntervalDomain               │
│   • Records samples              • DomainRefinement             │
│                                                                  │
│              ↓                           ↓                       │
│        traces/*.json           refine_from_trace()              │
│              ↓                           ↓                       │
│   ┌──────────────────────────────────────────────────┐          │
│   │          ReducedProductState                      │          │
│   │  ┌─────────────┐    ←───→    ┌───────────────┐   │          │
│   │  │  SignDomain │  inform_    │ IntervalDomain│   │          │
│   │  │  (parallel) │  each_other │   (parallel)  │   │          │
│   │  └─────────────┘             └───────────────┘   │          │
│   │                                                   │          │
│   │  Mutual Refinement Rules:                        │          │
│   │  • POSITIVE → low = max(low, 1)                  │          │
│   │  • NEGATIVE → high = min(high, -1)               │          │
│   │  • [a,b] where a>0 → sign = POSITIVE             │          │
│   │  • [0,0] → sign = ZERO                           │          │
│   └──────────────────────────────────────────────────┘          │
│                         ↓                                        │
│              Dict[int, ReducedProductState]                      │
│              (mutually refined sign + interval)                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Proposal Reference

From our approved proposal (20 October 2025), page 2, §1.3:

> "dynamic traces to set initial abstracts like x>0"

This dynamic refinement heuristic, combined with classic sign/interval domains running in parallel with mutual refinement, is our approved novelty contribution in 02242.

### Proposal §1.3.1 - Dynamic Refinement Heuristic

The proposal describes using observed execution traces to refine initial abstract states:
- When we observe samples `[5, 10, 15, 20, 25]`, we refine:
  - `sign(x) = POSITIVE` (all samples > 0)
  - `interval(x) = [5, 25]` (min/max of samples)
- The **Reduced Product** then applies mutual refinement to ensure consistency

## Key Components

### 1. ReducedProductState (Core NAB Implementation)

The `ReducedProductState` class implements the course definition by running sign and interval in parallel with mutual refinement:

```python
from solutions.nab_integration import ReducedProductState, inform_each_other
from solutions.abstractions import SignDomain, SignValue, IntervalDomain, IntervalValue

# Create reduced product from samples
reduced = ReducedProductState.from_samples([5, 10, 15, 20, 25])
print(reduced.sign)      # SignDomain(POSITIVE)
print(reduced.interval)  # IntervalDomain([5, 25])

# Mutual refinement example: POSITIVE tightens interval
sign = SignDomain(SignValue.POSITIVE)
interval = IntervalDomain(IntervalValue(-5, 10))
new_sign, new_interval = inform_each_other(sign, interval)
print(new_interval)  # IntervalDomain([1, 10]) - low tightened to 1!

# Check refinement history
reduced = ReducedProductState(
    sign=SignDomain(SignValue.POSITIVE),
    interval=IntervalDomain(IntervalValue(-5, 100))
)
reduced.inform_each_other()
print(reduced.get_refinement_history())
# ['sign=POSITIVE → low=max(low,1)']
```

### 2. IIN Integration (Interpreter)

IIN (`solutions/interpreter.py`) produces JSON trace files in `traces/` directory:

```json
{
  "method": "jpamb.cases.Simple.assertPositive:(I)V",
  "coverage": {
    "executed_pcs": [0, 1, 2, 3, 8],
    "uncovered_pcs": [4, 5, 6, 7],
    "branches": {"1": [false], "3": [true]}
  },
  "values": {
    "local_0": {
      "samples": [1, 1, 1, 1, 1],
      "always_positive": true,
      "never_negative": true,
      "never_zero": true,
      "sign": "positive",
      "interval": [1, null]
    }
  }
}
```

### 3. IAB Integration (Abstractions)

IAB (`solutions/abstractions.py`) provides:

- **SignDomain**: POSITIVE, NEGATIVE, ZERO, NON_ZERO, NON_NEGATIVE, NON_POSITIVE, TOP, BOTTOM
- **IntervalDomain**: [low, high] bounds
- **DomainRefinement.from_concrete_values()**: Refines domains from samples

### 4. NAB Module Functions

NAB (`solutions/nab_integration.py`) provides:

- **`ReducedProductState`**: Core class for parallel execution with mutual refinement
- **`inform_each_other(sign, interval)`**: Apply mutual refinement between domains
- **`integrate_abstractions(trace_path)`**: Main integration function
- **`integrate_abstractions_reduced(trace_path)`**: Returns ReducedProductState objects
- **`refine_from_trace_reduced(samples)`**: Create ReducedProductState from samples
- **`AbstractValue`**: Combined sign + interval for each local
- **`IntegrationResult`**: Full result with metadata

## Usage

### Basic Usage

```python
from solutions.nab_integration import integrate_abstractions

# Integrate trace to get refined abstract state
result = integrate_abstractions("traces/jpamb.cases.Simple_assertPositive_IV.json")

# Access refined domains for local_0
print(result[0].sign)      # SignDomain(SignValue.POSITIVE)
print(result[0].interval)  # IntervalDomain([1, 1])
```

### Reduced Product Usage (Parallel Execution)

```python
from solutions.nab_integration import (
    ReducedProductState, 
    inform_each_other,
    integrate_abstractions_reduced
)
from solutions.abstractions import SignDomain, SignValue, IntervalDomain, IntervalValue

# Create reduced product directly
reduced = ReducedProductState.from_samples([5, 10, 15])
print(reduced)  # ReducedProduct(sign=+, interval=[5, 15])

# Apply mutual refinement manually
sign = SignDomain(SignValue.POSITIVE)
interval = IntervalDomain(IntervalValue(-10, 50))
refined_sign, refined_interval = inform_each_other(sign, interval)
print(f"Before: interval=[-10, 50]")
print(f"After:  interval={refined_interval}")  # [1, 50]

# Integrate with reduced product
states = integrate_abstractions_reduced("traces/jpamb.cases.Simple_assertPositive_IV.json")
for idx, state in states.items():
    print(f"local_{idx}: {state}")
    print(f"  Refinement history: {state.get_refinement_history()}")
```

### Proposal Example

From proposal §1.3.1: `process(int x)` with samples `[5, 10, 15, 20, 25]`:

```python
from solutions.nab_integration import process_example, process_example_reduced

result = process_example()

# local_1 should be refined to:
# - sign = POSITIVE (all samples > 0)
# - interval = [5, 25]
print(result[1].sign.value)  # SignValue.POSITIVE
print(result[1].interval)    # [5, 25]

# Get full reduced product with history
reduced_result = process_example_reduced()
print(reduced_result[1].get_refinement_history())
```

### Batch Processing

```python
from solutions.nab_integration import integrate_all_traces

# Process all traces in directory
all_results = integrate_all_traces("traces")

for method_name, result in all_results.items():
    print(f"{method_name}:")
    for idx, abstract_val in result.abstract_values.items():
        print(f"  local_{idx}: sign={abstract_val.sign}, interval={abstract_val.interval}")
```

## Mutual Refinement Rules

### Sign → Interval Refinement

| Sign Value | Interval Refinement |
|------------|---------------------|
| POSITIVE | low = max(low, 1) |
| NEGATIVE | high = min(high, -1) |
| ZERO | interval = [0, 0] |
| NON_NEGATIVE | low = max(low, 0) |
| NON_POSITIVE | high = min(high, 0) |

### Interval → Sign Refinement

| Interval Condition | Sign Refinement |
|-------------------|-----------------|
| low > 0 | POSITIVE |
| high < 0 | NEGATIVE |
| [0, 0] | ZERO |
| low = 0, high > 0 | NON_NEGATIVE |
| low < 0, high = 0 | NON_POSITIVE |

## Sign Domain Refinement Rules

| Observed Samples | Refined Sign |
|------------------|--------------|
| All positive | POSITIVE |
| All negative | NEGATIVE |
| All zero | ZERO |
| Positive + negative | NON_ZERO |
| Positive + zero | NON_NEGATIVE |
| Negative + zero | NON_POSITIVE |
| All three | TOP |
| Empty | BOTTOM |

## Interval Domain Refinement Rules

- **Interval bounds**: `[min(samples), max(samples)]`
- **Single sample**: Point interval `[x, x]`
- **Empty samples**: BOTTOM

## Testing

Run the full test suite:

```bash
pytest tests/test_nab_integration.py -v
```

Run only parallel execution tests:

```bash
pytest tests/test_nab_integration.py -v -k "Parallel"
```

Run reduced product tests:

```bash
pytest tests/test_nab_integration.py -v -k "ReducedProduct"
```

Test coverage includes:
- **Proposal example verification**
- **Sign domain refinement** (all 8 cases)
- **Interval domain refinement**
- **Parallel execution tests** (10+ tests for mutual refinement)
- **ReducedProductState operations** (join, meet, widening)
- **Mutual refinement edge cases** (inconsistency detection)
- **Trace file parsing**
- **Real trace file integration**

## File Structure

```
solutions/
├── interpreter.py      # IIN - produces traces/*.json
├── abstractions.py     # IAB - SignDomain, IntervalDomain
└── nab_integration.py  # NAB - ReducedProductState, inform_each_other()

tests/
└── test_nab_integration.py  # NAB tests (includes parallel execution tests)

traces/
└── *.json             # IIN output files
```

## Dependencies

NAB uses only existing project code:
- `solutions.abstractions` (IAB)
- Standard library: `json`, `pathlib`, `typing`, `dataclasses`, `copy`

No new external dependencies required.

## Summary

This module implements the NAB (Integrate Abstractions) requirement for 02242:

| Requirement | Implementation |
|-------------|----------------|
| "Run two or more abstractions" | SignDomain + IntervalDomain |
| "at the same time" | ReducedProductState runs both in parallel |
| "letting them inform each other" | `inform_each_other()` mutual refinement |
| "during execution" | Applied during trace integration |
| "5 per abstraction after the first" | 2 abstractions = 5 points |

## Authors

DTU 02242 Program Analysis - Group 21

## License

See project LICENSE file.
