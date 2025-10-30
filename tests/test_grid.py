import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Save your provided code in a file named `grid_module.py`
# Note: The Grid class __init__ was modified to include missing attributes for testing
# self.auto_grid_allow_outside_points = 0.1
# self.auto_grid_allow_empty_bins = 5
from better_kan.functions import Grid, ParametrizedGrid


# Fixture to provide common parameters for grid creation
@pytest.fixture(params=[0, 3])  # Allow for testting both grid at zero order and grid of higher order
def grid_params(request):
    """Provides a standard set of parameters for creating grids."""
    return {"in_features": 3, "size": 10, "order": request.param, "grid_range": (-1.0, 1.0)}


# --- Tests for the base Grid Class ---


@pytest.mark.parametrize("grid_class", [Grid, ParametrizedGrid])
def test_grid_initialization(grid_class, grid_params):
    """
    Tests the constructor and the initial state of the Grid class.
    """
    grid = grid_class(**grid_params)

    # Verify that constructor parameters are stored correctly
    assert grid.grid_size == grid_params["size"]
    assert grid.order == grid_params["order"]
    assert torch.all(grid.grid_range[:, 0] == grid_params["grid_range"][0])
    assert torch.all(grid.grid_range[:, 1] == grid_params["grid_range"][1])

    # The total number of grid points should be the size plus padding for the order
    expected_rows = grid_params["size"] + 2 * grid_params["order"]
    expected_cols = grid_params["in_features"]
    assert grid.grid.shape == (expected_rows, expected_cols)

    # Check if the initial grid is uniformly spaced
    h = (grid_params["grid_range"][1] - grid_params["grid_range"][0]) / (grid_params["size"] - 1)
    # The first point of the 'inner' grid should be at the start of the range
    assert torch.allclose(grid.grid[grid_params["order"], :], torch.tensor([grid_params["grid_range"][0]] * expected_cols))
    # The first extrapolated point should be `order * h` before the start of the range
    print(grid.grid[0, :], torch.tensor([grid_params["grid_range"][0]] * expected_cols) - grid_params["order"] * h)
    assert torch.allclose(grid.grid[0, :], torch.tensor([grid_params["grid_range"][0]] * expected_cols) - grid_params["order"] * h)


def test_grid_update():
    """
    Tests the `update` method, which adjusts the grid based on input data.
    """
    in_features = 2
    grid = Grid(in_features=in_features, size=10, order=1)
    original_grid_mean = grid.grid.mean()

    # Create input data with a different distribution from the initial grid
    input_data = torch.randn(200, in_features) * 3 + 10  # Centered at 10

    # Update the grid based on the new data
    grid.update(input_data, grid_size=15, margin=0.1)

    # Check that the grid size was successfully updated
    assert grid.grid_size == 15
    assert grid.grid.shape == (15 + 2 * 1, in_features)

    # The grid's mean should have shifted towards the input data's mean
    assert torch.abs(grid.grid.mean() - original_grid_mean) > 5.0
    assert torch.abs(grid.grid.mean() - input_data.mean()) < 2.0


def test_grid_update_no_data():
    """
    Tests the `update` method, which adjusts the grid based on input data.
    """
    in_features = 2
    grid = Grid(in_features=in_features, size=10, order=1)
    # Update the grid based on the new data
    grid.update(None, grid_size=15, margin=0.1)
    # Check that the grid size was successfully updated
    assert grid.grid_size == 15
    assert grid.grid.shape == (15 + 2 * grid.order, in_features)


def test_grid_trigger_update():
    """
    Tests the `trigger_grid_update` method that decides when to update the grid.
    """
    grid = Grid(in_features=1, size=10, order=0, grid_range=(-1, 1))

    # Manually set the trigger thresholds for the test
    grid.auto_grid_allow_outside_points = 0.1  # Trigger if >10% of points are outside
    grid.auto_grid_allow_empty_bins = 5  # Trigger if >5 bins are empty

    # Case 1: No trigger - all points are within the grid range
    x_inside = torch.linspace(-0.8, 0.8, 100).unsqueeze(1)
    assert not grid.trigger_grid_update(x_inside)

    # Case 2: Trigger - more than 10% of points are outside the grid range
    x_outside = torch.cat([torch.linspace(-0.5, 0.5, 85), torch.linspace(1.1, 1.5, 15)]).unsqueeze(1)
    assert grid.trigger_grid_update(x_outside)

    # Case 3: Trigger - too many empty bins because all data is in one bin
    x_concentrated = torch.zeros(100, 1)
    # This leaves 8 of the 9 bins empty, which is more than the threshold of 5
    assert grid.trigger_grid_update(x_concentrated)


# --- Tests for the ParametrizedGrid Class ---


def test_parametrized_grid_initialization(grid_params):
    """
    Tests the constructor of the ParametrizedGrid.
    """
    p_grid = ParametrizedGrid(**grid_params)

    # The grid should be defined by a learnable parameter 's'
    assert isinstance(p_grid.s, nn.Parameter)

    # The shape of 's' should be (size - 1, in_features)
    expected_shape = (grid_params["size"] - 1, grid_params["in_features"])
    assert p_grid.s.shape == expected_shape


def test_parametrized_grid_getter(grid_params):
    """
    Tests the `grid` property getter, which computes the grid from the 's' parameter.
    """
    p_grid = ParametrizedGrid(**grid_params)
    grid_val = p_grid.grid

    # Check if the grid has the correct shape after computation
    expected_rows = grid_params["size"] + 2 * grid_params["order"]
    expected_cols = grid_params["in_features"]
    assert grid_val.shape == (expected_rows, expected_cols)

    # The grid points should be monotonically increasing within the core (non-padded) section
    core_grid = grid_val[grid_params["order"] : -grid_params["order"], :]
    diffs = torch.diff(core_grid, dim=0)
    assert torch.all(diffs >= 0)

    # The end-points of the inner grid should match the specified grid_range
    # NOTE: This test might fail if the getter has the suspected concatenation bug.
    inner_grid_start_point = grid_val[grid_params["order"], :]
    inner_grid_end_point = grid_val[grid_params["order"] + grid_params["size"] - 1, :]

    assert torch.allclose(inner_grid_start_point, torch.full((grid_params["in_features"],), grid_params["grid_range"][0]))
    assert torch.allclose(inner_grid_end_point, torch.full((grid_params["in_features"],), grid_params["grid_range"][1]))


def test_parametrized_grid_setter(grid_params):
    """
    This test is skipped because the setter method has a syntax error
    in the provided code (`grid = grid[:, self.order : -self.order]`).
    """
    grid = ParametrizedGrid(**grid_params)
    h = torch.ones(1, (grid_params["in_features"])) * 0.25
    grid.grid = torch.arange(0, grid_params["size"] + 2 * grid_params["order"]).unsqueeze(1) * h

    expected_rows = grid_params["size"] + 2 * grid_params["order"]
    expected_cols = grid_params["in_features"]
    assert grid.grid.shape == (expected_rows, expected_cols)
