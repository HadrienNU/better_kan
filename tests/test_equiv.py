import pytest
import torch
import torch.nn as nn
from torch.nn.utils import parametrize

from better_kan.equivariance import (
    EquivariantMatrix,
    EquivariantVector,
    equivariant_permutations_inputs,
)


@pytest.fixture
def test_setup():
    """Provides a common setup for the equivariance tests."""
    # Define a cyclic group of order 4 (rotations on a 4-element vector)
    generators = equivariant_permutations_inputs([1, 1, 0, 0])
    in_channels = 2
    out_channels = 3
    in_features = len(generators[0])
    out_features = len(generators[0])

    # Return a dictionary of the setup variables
    return {
        "generators": generators,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "in_features": in_features,
        "out_features": out_features,
    }


def test_equivariants_inputs():
    """
    Tests the helper function for creating equivariant permutation generators.
    """
    # Test with numeric labels
    generators_numeric = equivariant_permutations_inputs([1, 1, 1, 1])
    assert len(generators_numeric) == 2
    assert len(generators_numeric[0]) == 4

    # Test with string labels
    generators_string = equivariant_permutations_inputs(["a", "a", "b", "b", "a"])
    assert len(generators_string) == 3


@pytest.mark.parametrize(
    "model_class, constructor_args, expected_shape_func",
    [
        (
            EquivariantMatrix,
            lambda p: (p["generators"], p["generators"], p["in_channels"], p["out_channels"]),
            lambda p: (p["out_features"] * p["out_channels"], p["in_features"] * p["in_channels"]),
        ),
        (
            EquivariantVector,
            lambda p: (p["generators"], p["out_channels"]),
            lambda p: (p["out_features"] * p["out_channels"],),
        ),
    ],
    ids=["weight", "bias"],
)
def test_forward_shape(test_setup, model_class, constructor_args, expected_shape_func):
    """
    Tests the forward pass output shape for both EquivariantMatrix and EquivariantVector.
    """
    params = test_setup
    param_obj = model_class(*constructor_args(params))
    expected_shape = expected_shape_func(params)

    stored_tensor = torch.randn(param_obj.num_weights)
    full_tensor = param_obj(stored_tensor)

    assert full_tensor.shape == expected_shape


@pytest.mark.parametrize(
    "model_class, constructor_args",
    [
        (EquivariantMatrix, lambda p: (p["generators"], p["generators"], p["in_channels"], p["out_channels"])),
        (EquivariantVector, lambda p: (p["generators"], p["out_channels"])),
    ],
    ids=["weight", "bias"],
)
def test_right_inverse(test_setup, model_class, constructor_args):
    """
    Tests that right_inverse is the correct inverse of the forward pass for both classes.
    """
    params = test_setup
    param_obj = model_class(*constructor_args(params))

    stored_tensor = torch.randn(param_obj.num_weights)

    # Generate the full tensor and then recover the original stored tensor
    full_tensor = param_obj.forward(stored_tensor)
    recovered_tensor = param_obj.right_inverse(full_tensor)

    # The recovered tensor should be identical to the original one
    assert torch.allclose(stored_tensor, recovered_tensor, atol=1e-6)

    # Applying forward then right_inverse on the full tensor should be idempotent
    reconstructed_full_tensor = param_obj.forward(recovered_tensor)
    assert torch.allclose(reconstructed_full_tensor, full_tensor, atol=1e-6)


def test_parametrization_on_linear(test_setup):
    """Tests applying EquivariantMatrix directly to a standard nn.Linear layer."""
    params = test_setup
    in_feats, out_feats = params["in_features"], params["out_features"]
    in_ch, out_ch = params["in_channels"], params["out_channels"]

    linear_layer = nn.Linear(in_feats * in_ch, out_feats * out_ch, bias=False)
    torch.nn.init.normal_(linear_layer.weight)

    weight_parametrization = EquivariantMatrix(params["generators"], params["generators"], in_ch, out_ch)
    parametrize.register_parametrization(linear_layer, "weight", weight_parametrization)

    # 1. Test that the forward pass of the parametrization is applied correctly
    expected_weight = weight_parametrization.forward(linear_layer.parametrizations.weight.original)
    assert torch.equal(linear_layer.weight, expected_weight)

    # 2. Test that updating the original (raw) tensor propagates to the public tensor
    with torch.no_grad():
        linear_layer.parametrizations.weight.original.data.normal_()
    expected_weight_after_update = weight_parametrization.forward(linear_layer.parametrizations.weight.original)
    assert torch.equal(linear_layer.weight, expected_weight_after_update)

    # 3. Test that setting the public tensor correctly updates the original via the inverse
    with torch.no_grad():
        new_weight = torch.randn_like(linear_layer.weight)
        linear_layer.weight = new_weight
    expected_original_after_update = weight_parametrization.right_inverse(new_weight)
    assert torch.allclose(linear_layer.parametrizations.weight.original, expected_original_after_update)


# TODO: Add test for layer parametrization for equivariance
@pytest.mark.skip(reason="no way of currently testing this")
def test_parametrization_layer():
    generators = equivariant_permutations_inputs(["a", "a", "b", "b", "c", "a", "b"])
    in_channels = 2
    out_channels = 3
    in_features = len(generators[0])
    out_features = len(generators[0])
