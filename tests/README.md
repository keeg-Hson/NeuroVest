# NeuroVest Tests

This directory contains unit tests for critical system components.

## Running Tests

### Install Test Dependencies

```bash
pip install pytest pytest-cov
```

### Run All Tests

```bash
# From project root
pytest tests/ -v
```

### Run Specific Test Files

```bash
pytest tests/test_risk_management.py -v
pytest tests/test_transaction_costs.py -v
```

### Run with Coverage Report

```bash
pytest tests/ --cov=core --cov-report=html
```

Then open `htmlcov/index.html` in your browser to see coverage report.

## Test Files

### `test_risk_management.py`

Tests for the enforced risk management system that actually works (unlike the original).

**Key Tests**:
- Position size limits are enforced
- Stop losses actually trigger
- Gap risk is modeled realistically
- Daily loss limits halt trading
- Portfolio heat limits prevent over-concentration
- Max drawdown forces liquidation
- Leverage limits are enforced
- Maximum position count is enforced

**Example**:
```bash
pytest tests/test_risk_management.py::TestEnforcedRiskManager::test_stop_loss_enforcement -v
```

### `test_transaction_costs.py`

Tests for transaction cost modeling.

**Key Tests**:
- Stock trading costs calculated correctly
- Crypto trading costs are higher
- Volatility increases costs
- Liquidity affects costs
- Market orders have slippage
- Limit orders don't have slippage
- Round trip costs include entry + exit
- Leverage adds financing costs
- Conservative model is more expensive

**Example**:
```bash
pytest tests/test_transaction_costs.py::TestTransactionCostModel::test_volatility_increases_costs -v
```

## Test Coverage Goals

**Current Status**: Initial tests created

**Goals**:
- [ ] Risk Management: 80%+ coverage
- [ ] Transaction Costs: 80%+ coverage
- [ ] Validation Framework: 70%+ coverage
- [ ] Data Loading: 60%+ coverage
- [ ] Backtesting Engine: 70%+ coverage

## Writing New Tests

### Test Structure

```python
import pytest
from core.your_module import YourClass

class TestYourClass:
    """Tests for YourClass"""

    def test_specific_behavior(self):
        """Test that specific behavior works correctly"""
        # Arrange
        obj = YourClass()

        # Act
        result = obj.do_something()

        # Assert
        assert result == expected_value
```

### Best Practices

1. **One test, one behavior**: Each test should test exactly one thing
2. **Descriptive names**: Test names should describe what they test
3. **Arrange-Act-Assert**: Structure tests clearly
4. **Use fixtures**: For common setup code
5. **Test edge cases**: Not just happy path
6. **Test failures**: Ensure errors are handled correctly

### Example: Testing Exception Handling

```python
def test_invalid_input_raises_error(self):
    """Test that invalid input raises appropriate error"""
    with pytest.raises(ValueError, match="Invalid input"):
        process_data(invalid_input)
```

### Example: Testing Numerical Approximations

```python
def test_calculation_is_approximately_correct(self):
    """Test numerical calculation within tolerance"""
    result = calculate_sharpe_ratio(returns)
    assert abs(result - 1.5) < 0.01  # Within 0.01 of expected
```

## Why Testing Matters

The original NeuroVest system had **zero tests**. This meant:

❌ No way to verify code works correctly
❌ Changes could break everything silently
❌ No confidence in system reliability
❌ Bugs could cause catastrophic trading losses

With proper testing:

✅ Verify code behavior is correct
✅ Detect breaking changes immediately
✅ Build confidence in system reliability
✅ Catch bugs before they cause losses
✅ Enable safe refactoring and improvements

## Continuous Integration

**Future Goal**: Run tests automatically on every commit

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run tests
        run: pytest tests/ --cov=core
```

## Test-Driven Development

**Recommended Workflow**:

1. Write test for new feature (test fails - red)
2. Implement feature (test passes - green)
3. Refactor code while tests pass (refactor)
4. Repeat

This ensures all code is tested and prevents regressions.

## Running Tests Before Commits

**Git Hook** (`.git/hooks/pre-commit`):

```bash
#!/bin/bash
echo "Running tests before commit..."
pytest tests/ -v
if [ $? -ne 0 ]; then
    echo "Tests failed! Commit aborted."
    exit 1
fi
```

## Notes

- Tests are isolated (don't depend on external services or files)
- Tests are fast (run in seconds, not minutes)
- Tests are deterministic (same input = same output)
- Tests are independent (can run in any order)

## Further Reading

- [pytest documentation](https://docs.pytest.org/)
- [Python Testing Best Practices](https://docs.python-guide.org/writing/tests/)
- [Test-Driven Development](https://www.obeythetestinggoat.com/)
