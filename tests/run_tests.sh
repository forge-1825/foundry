#!/bin/bash
# Run tests for Foundry pipeline

echo "======================================"
echo "Foundry Pipeline Test Suite"
echo "======================================"
echo ""

# Change to the foundry directory
cd "$(dirname "$0")/.." || exit 1

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo "Error: pytest is not installed. Please install it with:"
    echo "  pip install pytest pytest-cov pytest-mock"
    exit 1
fi

# Default test type
TEST_TYPE="${1:-smoke}"

case "$TEST_TYPE" in
    "smoke")
        echo "Running smoke tests (basic functionality)..."
        echo "========================================="
        pytest tests/ -v -m smoke
        ;;
    
    "integration")
        echo "Running integration tests (requires services)..."
        echo "=============================================="
        pytest tests/ -v -m integration
        ;;
    
    "all")
        echo "Running all tests..."
        echo "==================="
        pytest tests/ -v
        ;;
    
    "coverage")
        echo "Running tests with coverage report..."
        echo "===================================="
        pytest tests/ -v --cov=scripts --cov=backend --cov-report=term-missing --cov-report=html
        echo ""
        echo "Coverage report saved to htmlcov/index.html"
        ;;
    
    "quick")
        echo "Running quick smoke tests..."
        echo "==========================="
        pytest tests/test_imports.py -v
        ;;
    
    "models")
        echo "Running model connectivity tests..."
        echo "================================="
        pytest tests/test_model_connectivity.py -v
        ;;
    
    *)
        echo "Usage: $0 [smoke|integration|all|coverage|quick|models]"
        echo ""
        echo "Test types:"
        echo "  smoke       - Basic functionality tests (default)"
        echo "  integration - Tests requiring external services"
        echo "  all         - Run all tests"
        echo "  coverage    - Run with coverage report"
        echo "  quick       - Only import tests"
        echo "  models      - Only model connectivity tests"
        exit 1
        ;;
esac

# Get exit code
EXIT_CODE=$?

echo ""
echo "======================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Tests completed successfully!"
else
    echo "Tests failed with exit code: $EXIT_CODE"
fi
echo "======================================"

exit $EXIT_CODE