# JPAMB Interpreter User Guide

## Overview

The JPAMB Interpreter is a comprehensive Java bytecode analysis tool that provides both dynamic execution tracing and static analysis integration. It allows you to execute Java methods, collect detailed coverage and value information, and refine traces into abstract states for static analysis tools.

## Quick Start

### Prerequisites
- Python 3.8+ with uv package manager
- JPAMB project setup in your workspace

### Installation
The interpreter is included with the JPAMB project. Ensure you have the project dependencies installed:

```bash
uv sync
```

## Command Reference

### 1. Generate Dynamic Analysis Traces

Execute all test cases and generate comprehensive trace files:

```bash
uv run jpamb trace --trace-dir traces
```

**Options:**
- `--trace-dir PATH`: Directory to write trace files (default: "traces")

**Output:** JSON trace files containing:
- PC (Program Counter) coverage information
- Branch coverage data
- Concrete value analysis for local variables
- Statistical summaries (sign analysis, intervals, properties)

**Example Output:**
```
[INFO] Generating traces in traces
[INFO] Tracing jpamb.cases.Simple.justAdd:(II)I
[SUCCESS] Generated traces for 67 methods in traces
```

### 2. Refine Traces to Abstract States

Convert dynamic traces into initial abstract states for static analysis:

```bash
uv run jpamb refine --trace-dir traces --output initial_states.json
```

**Options:**
- `--trace-dir PATH`: Directory containing trace files to refine (default: "traces")
- `--output PATH`: Output file for initial abstract states (default: "initial_states.json")

**Output:** JSON file with abstract state information including:
- Initial abstract states per method
- Confidence metrics
- Abstract domain mappings
- Coverage point analysis

**Example Output:**
```
[INFO] Refining traces from traces
[SUCCESS] Refined 30 methods
[INFO] Average confidence: 0.95
[INFO] High confidence (>0.8): 30/30
[SUCCESS] Generated initial state file: initial_states.json
```

### 3. Execute Single Method

Run a specific method with detailed tracing:

```bash
# Using jpamb interpret command (recommended)
uv run jpamb interpret --filter "Simple.divideByN" --with-python solutions/interpreter.py

# Direct execution for debugging
uv run python solutions/interpreter.py 'jpamb.cases.Simple.divideByN:(I)I' '(5)'
```

**Parameters for direct execution:**
- First argument: Method signature in format `package.Class.method:(params)returnType`
- Second argument: Input values in format `(value1,value2,...)`

**Example Output:**
```
[DEBUG] STEP push:I 1
[DEBUG] STEP load:I 0  
[DEBUG] STEP binary:I div
[DEBUG] STEP return:I
ok
```

### 4. View Available Commands

See all available JPAMB commands:

```bash
uv run jpamb --help
```

## Complete Workflow Example

### Step 1: Generate Traces
```bash
# Create traces for all test methods
uv run jpamb trace --trace-dir traces
```

### Step 2: Refine to Abstract States
```bash
# Convert traces to abstract states for static analysis
uv run jpamb refine --trace-dir traces --output my_states.json
```

### Step 3: Analyze Results
```bash
# View generated files
ls traces/          # Individual trace JSON files
cat my_states.json     # Abstract states summary
```

## Trace File Format

### Dynamic Trace JSON Structure
```json
{
  "method": "jpamb.cases.Simple.justAdd:(II)I",
  "coverage": {
    "executed_pcs": [0, 1, 2, 3],
    "uncovered_pcs": [],
    "branches": {
      "1": [false, true]
    }
  },
  "values": {
    "local_0": {
      "samples": [1, 1, 1, 1],
      "always_positive": true,
      "never_negative": true,
      "never_zero": true,
      "sign": "positive",
      "interval": [1, null]
    }
  }
}
```

### Abstract States JSON Structure
```json
{
  "format_version": "1.0",
  "description": "Initial abstract states refined from dynamic analysis traces",
  "methods": {
    "jpamb.cases.Simple.justAdd:(II)I": {
      "initial_states": [
        "jpamb.cases.Simple.justAdd:(II)I:0[0:+,1:+]"
      ],
      "confidence": 0.95,
      "coverage_points": [0, 1, 2, 3],
      "abstract_domains": {
        "0": ["+"],
        "1": ["+"],
        "2": ["+"],
        "3": ["+"]
      }
    }
  }
}
```

## Abstract Domain Reference

The interpreter uses the following abstract domains:

| Symbol | Domain | Description |
|--------|--------|-------------|
| `⊤` | TOP | Unknown/any value |
| `⊥` | BOTTOM | Impossible/no value |
| `0` | ZERO | Exactly zero |
| `+` | POSITIVE | Positive integers |
| `-` | NEGATIVE | Negative integers |
| `≠0` | NON_ZERO | Non-zero integers |
| `≥0` | NON_NEGATIVE | Zero or positive |
| `≤0` | NON_POSITIVE | Zero or negative |

## Testing

### Run Interpreter Tests
```bash
# Test the interpreter implementation
uv run jpamb test --filter "Simple" --with-python solutions/interpreter.py
```

### Run Specific Test Categories
```bash
# Test dynamic analysis components
uv run jpamb test --filter "Simple" --with-python solutions/interpreter.py

# Test refinement components (unit tests)
uv run python -m pytest test/test_refinement.py -v

# Run all Python unit tests
uv run python -m pytest test/ -v
```

### Validate Complete Workflow
```bash
# Test interpreter with jpamb CLI
uv run jpamb interpret --filter "Simple" --with-python solutions/interpreter.py

# Test trace generation and refinement
uv run jpamb trace --trace-dir test_traces
uv run jpamb refine --trace-dir test_traces --output test_states.json
```

## Troubleshooting

### Common Issues

**Issue:** `ModuleNotFoundError: No module named 'jpamb'`
```bash
# Ensure you're in the project directory and dependencies are installed
uv sync
```

**Issue:** `No trace files found`
```bash
# Make sure trace directory exists and contains .json files
ls traces/
uv run jpamb trace --trace-dir traces  # Generate traces first
```

**Issue:** `command not found: jpamb`
```bash
# Ensure you're using uv to run jpamb commands
uv run jpamb --help
```

### Debug Mode

Enable verbose logging for detailed debugging:

```bash
uv run jpamb -v trace --trace-dir traces
uv run jpamb -v refine --trace-dir traces --output states.json
```

## Advanced Usage

### Batch Processing Multiple Directories
```bash
# Process multiple trace directories
for dir in traces_*; do
  uv run jpamb refine --trace-dir "$dir" --output "${dir}_states.json"
done
```

### Integration with Static Analysis Tools

The generated `initial_states.json` file is designed to integrate with static analysis frameworks:

1. **Abstract Interpreters**: Use initial states as entry points
2. **Verification Tools**: Import as preconditions
3. **Bug Detectors**: Load as constraint systems
4. **Test Generators**: Utilize for guided input generation

## Performance Notes

- **Trace Generation**: Approximately 1-2 seconds per method
- **Refinement**: Near-instantaneous for typical trace sets
- **Memory Usage**: ~50MB for 100+ trace files
- **Disk Usage**: ~1KB per trace file

## Important: Generated Files

**Do not commit generated files to version control!**

The following files are automatically generated and should not be committed:
- `traces/` - Directory containing individual trace JSON files
- `initial_states.json` - Abstract states output file
- `*_states.json` - Any custom-named state files
- `*.trace.json` - Individual trace files

These files are already included in `.gitignore` and will be ignored by Git automatically.

## Support

For issues or questions:
1. Check the test files in `test/` for usage examples
2. Review the complete implementation in `solutions/interpreter.py`
3. Consult the main documentation in `README.md`

## Version Information

- **Format Version**: 1.0
- **Supported Java Bytecode**: JVM specification compatible
- **Python Requirements**: 3.8+
- **Dependencies**: See `pyproject.toml`