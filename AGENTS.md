# AGENTS.md

This file contains guidelines for agentic coding assistants working on the better-kan repository.

## Project Overview

**better-kan** is a pure-PyTorch implementation of Kolmogorov-Arnold Networks (KANs) with plotting, pruning, and regularization capabilities. The library focuses on performance optimization and extensibility, particularly for equivariant neural networks and graph-structured data.

## Build/Test Commands

### Testing
```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest tests/test_layer.py

# Run tests with verbose output
python -m pytest -v

# Run tests with coverage (if coverage is installed)
python -m pytest --cov=better_kan
```

### Package Management
```bash
# Install dependencies (requires PDM)
pdm install

# Build package
pdm build

# Install in development mode
pdm install -e
```

### Running Examples
```bash
# Examples are in the examples/ directory
python examples/basic_usage.py
```

## Code Style Guidelines

### Import Organization
```python
# Standard library imports first
import copy
import numpy as np

# Third-party imports second
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Local imports last
from .layers import KANLayer
from .functions import Grid
from ..utils import assign_parameters
```

### Naming Conventions
- **Classes**: `PascalCase` (e.g., `KANLayer`, `BasisFunction`, `ActivationFunction`)
- **Functions/Variables**: `snake_case` (e.g., `build_KAN`, `reset_parameters`, `in_features`)
- **Constants**: `UPPER_CASE` (e.g., `GRID_SIZE`, `DEFAULT_LAMBDA`)
- **Private methods**: Prefix with underscore (e.g., `_compute_grid`)

### Type Hints
- Use type hints for function signatures and public APIs
- PyTorch tensors should be annotated as `torch.Tensor`
- Return types should be specified for public methods

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    pass

def build_layer(in_features: int, out_features: int, functions_type: type) -> KANLayer:
    pass
```

### Code Formatting
- **Indentation**: 4 spaces
- **Line length**: ~100-120 characters
- **Spacing**: Around operators and after commas
- **Blank lines**: Between logical sections

### Docstrings
Use triple quotes with descriptive text:

```python
def build_KAN(layers_types, layers_hidden, **kwargs):
    """
    Helper function to build stack of KAN layers
    
    Args:
        layers_types: List of layer types or single type for all layers
        layers_hidden: List of hidden dimensions including input and output
        **kwargs: Additional arguments passed to layer constructors
    
    Returns:
        KAN: Constructed KAN model
    """
```

## Architecture Patterns

### Layer Building Pattern
Use the `build_layer` helper function for creating layers:

```python
from .layers import build_layer

layer = build_layer(
    in_features, 
    out_features, 
    functions_type, 
    grid_size=grid_size,
    fast_version=False
)
```

### Function Registration
New activation functions should inherit from `BasisFunction` or `ActivationFunction`:

```python
class NewFunction(BasisFunction):
    def __init__(self, in_features, out_features, grid, **kwargs):
        super().__init__(in_features, out_features)
        self.n_basis_function = N  # Define number of basis functions
```

### Module Structure
- `better_kan/`: Main package code
- `better_kan/functions/`: Activation function implementations
- `better_kan/equivariance/`: Equivariance-related code
- `tests/`: Test suite using pytest
- `examples/`: Usage examples

## Error Handling

### Parameter Validation
```python
if not isinstance(layers_types, list):
    layers_types = [layers_types] * (len(layers_hidden) - 1)

if pooling_op.lower() not in ["sum", "prod", "logsumexp", "min", "max"]:
    raise ValueError(f"Unsupported pooling operation: {pooling_op}")
```

### Graceful Degradation
```python
@torch.no_grad()
def reset_parameters(self, init_type="uniform", **init_kwargs):
    for layer in self.layers:
        if hasattr(layer, "reset_parameters"):
            try:
                layer.reset_parameters(init_type, **init_kwargs)
            except TypeError:
                layer.reset_parameters()  # Fallback to default
```

## Testing Guidelines

### Test Structure
```python
import pytest
import torch
from better_kan import KANLayer

@pytest.fixture
def sample_layer():
    """Provides a sample layer for testing."""
    return KANLayer(in_features=2, out_features=3, functions=functions_set)

def test_layer_initialization(sample_layer):
    """Tests layer initialization."""
    assert sample_layer.in_features == 2
    assert sample_layer.out_features == 3
```

### Test Coverage
- Test all public methods
- Test edge cases and error conditions
- Use fixtures for common test data
- Test both slow and fast versions when applicable

## Performance Considerations

### Fast vs Slow Versions
- Use `fast_version=True` for performance-critical code
- Maintain both versions for compatibility
- Fast versions may sacrifice some features for speed

### Memory Management
- Use `@torch.no_grad()` for inference-only methods
- Register buffers instead of parameters for non-trainable tensors
- Be mindful of GPU memory usage in large models

## Development Workflow

1. **Feature Development**: Create new functions in `better_kan/functions/`
2. **Testing**: Add corresponding tests in `tests/`
3. **Documentation**: Update docstrings and add examples if needed
4. **Benchmarking**: Use `benchmark/` directory for performance tests

## Common Patterns

### Grid Management
```python
from .functions import Grid

grid = Grid(in_features, grid_size)
function = Splines(in_features, out_features, grid)
```

### Model Building
```python
from better_kan import build_KAN

model = build_KAN(
    layers_types=[Splines, ActivationFunction],
    layers_hidden=[input_dim, hidden_dim, output_dim],
    add_batch_norm=True
)
```

### Training Loop
```python
from better_kan.utils import train

train(model, dataset, opt="LBFGS", steps=100, lamb=1e-2)
```

## Dependencies

- **PyTorch** (>=2.2.0): Core tensor operations and neural network modules
- **pytest** (>=8.2.0): Testing framework
- **tqdm** (>=4.66.2): Progress bars
- **networkx**: Graph operations for graph neural network extensions
- **numpy**: Numerical operations, particularly for polynomial functions

## Notes for AI Assistants

- This is a research-oriented library focused on performance and extensibility
- Maintain compatibility with PyTorch conventions
- Prioritize numerical stability and performance
- When adding new features, consider both slow (full-featured) and fast versions
- Follow the existing code patterns for consistency