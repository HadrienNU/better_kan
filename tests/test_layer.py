import pytest
import torch
import torch.nn as nn

from better_kan import KANLayer
from better_kan.functions import Grid, Splines, ActivationFunction


@pytest.fixture
def functions_set():
    """Provides a list of functions."""
    grid = Grid(2, 5)
    return nn.ModuleList([Splines(2, 3, grid), ActivationFunction(2, 3, base_activation=nn.SiLU)])


@pytest.fixture
def kan_layer_slow(functions_set):
    """Provides a KANLayer instance in slow mode."""
    return KANLayer(in_features=2, out_features=3, functions=functions_set)


def test_kanlayer_initialization(functions_set):
    """Tests the initialization of the KANLayer."""
    layer = KANLayer(in_features=2, out_features=3, functions=functions_set, bias=torch.tensor(0.5))
    assert layer.in_features == 2
    assert layer.out_features == 3
    assert torch.all(layer.bias == 0.5)
    assert not layer.fast_mode

    # Test initialization with default bias
    layer_default_bias = KANLayer(in_features=2, out_features=3, functions=functions_set)
    assert torch.all(layer_default_bias.bias == 0)


def test_kanlayer_forward_slow_sum(kan_layer_slow):
    """Tests the forward pass in slow mode with sum pooling."""
    input_tensor = torch.randn(4, 2)  # Batch size of 4
    output = kan_layer_slow(input_tensor)
    assert output.shape == (4, 3)
    assert hasattr(kan_layer_slow, "l1_norm")
    assert hasattr(kan_layer_slow, "min_vals")
    assert hasattr(kan_layer_slow, "max_vals")


def test_kanlayer_forward_slow_prod(functions_set):
    """Tests the forward pass in slow mode with prod pooling."""
    layer = KANLayer(in_features=2, out_features=3, functions=functions_set, pooling_op="prod")
    input_tensor = torch.randn(4, 2)
    output = layer(input_tensor)
    assert output.shape == (4, 3)


def test_kanlayer_forward_fast(functions_set):
    """Tests the forward pass in fast mode."""
    layer = KANLayer(in_features=2, out_features=3, functions=functions_set, fast_version=True)
    input_tensor = torch.randn(4, 2)
    output = layer(input_tensor)
    assert output.shape == (4, 3)
    assert not hasattr(layer, "l1_norm")


def test_regularization_loss(kan_layer_slow):
    """Tests the regularization loss calculation."""
    # Before forward pass, loss should be 0
    assert kan_layer_slow.regularization_loss() == 0.0

    # After forward pass, loss should be calculated
    input_tensor = torch.randn(4, 2)
    kan_layer_slow(input_tensor)
    loss = kan_layer_slow.regularization_loss()
    assert isinstance(loss, torch.Tensor)
    assert loss.item() > 0


def test_set_from_another_layer(functions_set):
    """Tests setting a layer's parameters from another layer."""
    parent_layer = KANLayer(in_features=5, out_features=5, functions=functions_set, bias=torch.randn(5))
    child_layer = KANLayer(in_features=2, out_features=3, functions=functions_set)

    out_id = [0, 2, 4]
    child_layer.set_from_another_layer(parent_layer, out_id=out_id)

    assert torch.allclose(child_layer.bias, parent_layer.bias[out_id])


def test_set_speed_mode(kan_layer_slow):
    """Tests toggling the speed mode."""
    # Initially in slow mode
    assert not kan_layer_slow.fast_mode

    # Switch to fast mode
    kan_layer_slow.set_speed_mode(True)
    assert kan_layer_slow.fast_mode

    # Switch back to slow mode
    kan_layer_slow.set_speed_mode(False)
    assert not kan_layer_slow.fast_mode


def test_invalid_pooling_op(functions_set):
    """Tests that an invalid pooling operation raises a ValueError."""
    with pytest.raises(ValueError):
        KANLayer(in_features=2, out_features=3, functions=functions_set, pooling_op="invalid_op")


if __name__ == "__main__":
    pytest.main()
