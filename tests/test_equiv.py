import unittest
import torch
import torch.nn as nn
from torch.nn.utils import parametrize

from better_kan.equivariance import (
    EquivariantMatrix,
    EquivariantVector,
    equivariant_permutations_inputs,
)


class TestParametrizations(unittest.TestCase):
    def setUp(self):
        """Set up common variables for the tests."""
        # Define a cyclic group of order 4 (rotations on a 4-element vector)
        # This means features [0, 1, 2, 3] are permuted cyclically

        self.generators = equivariant_permutations_inputs([1, 1, 0, 0])

        # self.generators = [
        #     torch.tensor([1, 2, 3, 0]),
        #     torch.tensor([2, 3, 0, 1]),
        #     torch.tensor([3, 0, 1, 2]),
        # ]
        self.in_channels = 2
        self.out_channels = 3
        self.in_features = len(self.generators[0])
        self.out_features = len(self.generators[0])

    def test_equivariants_inputs(self):

        generators = equivariant_permutations_inputs([1, 1, 1, 1])

        self.assertEqual(len(generators), 2)
        self.assertEqual(len(generators[0]), 4)
        generators = equivariant_permutations_inputs(["a", "a", "b", "b", "a"])

        self.assertEqual(len(generators), 3)

    def test_weight_forward_shape(self):
        """Test the shape of the output from EquivariantMatrix forward pass."""
        weight_param = EquivariantMatrix(self.generators, self.generators, self.in_channels, self.out_channels)
        stored_weight = torch.randn(weight_param.num_weights)
        full_weight = weight_param(stored_weight)  # Argument is a dummy

        expected_shape = (
            self.out_features * self.out_channels,
            self.in_features * self.in_channels,
        )
        self.assertEqual(full_weight.shape, expected_shape)

    def test_bias_forward_shape(self):
        """Test the shape of the output from EquivariantVector forward pass."""
        bias_param = EquivariantVector(self.generators, self.out_channels)
        stored_bias = torch.randn(bias_param.num_weights)
        full_bias = bias_param(stored_bias)

        expected_shape = (self.out_features * self.out_channels,)
        self.assertEqual(full_bias.shape, expected_shape)

    def test_weight_right_inverse(self):
        """Test the right_inverse method for EquivariantMatrix."""
        weight_param = EquivariantMatrix(self.generators, self.generators, self.in_channels, self.out_channels)

        stored_weight = torch.randn(weight_param.num_weights)

        # Generate the full weight matrix
        full_weight_matrix = weight_param.forward(stored_weight)

        # Use right_inverse to recover the unique weights
        recovered_unique_weights = weight_param.right_inverse(full_weight_matrix)

        # The recovered weights should be identical to the original ones
        self.assertTrue(torch.allclose(stored_weight, recovered_unique_weights, atol=1e-6))

    def test_bias_right_inverse(self):
        """Test the right_inverse method for EquivariantVector."""
        bias_param = EquivariantVector(self.generators, self.out_channels)

        # Manually set unique biases
        stored_bias = torch.randn(bias_param.num_weights)

        # Generate the full bias vector
        full_bias_vector = bias_param.forward(stored_bias)

        # Recover the unique biases
        recovered_unique_biases = bias_param.right_inverse(full_bias_vector)

        # Check for correctness
        self.assertTrue(torch.allclose(stored_bias, recovered_unique_biases, atol=1e-6))

    def test_parametrization_on_linear(self):
        """Tests applying EquivariantWeight directly to a standard nn.Linear layer."""
        in_feats, out_feats = self.in_features, self.out_features
        in_ch, out_ch = self.in_channels, self.out_channels

        linear_layer = nn.Linear(in_feats * in_ch, out_feats * out_ch, bias=False)

        torch.nn.init.normal_(linear_layer.weight)

        weight_parametrization = EquivariantMatrix(self.generators, self.generators, in_ch, out_ch)
        parametrize.register_parametrization(linear_layer, "weight", weight_parametrization)

        expected_weight = weight_parametrization.forward(linear_layer.parametrizations.weight.original)
        actual_weight = linear_layer.weight
        self.assertTrue(torch.equal(actual_weight, expected_weight))

        with torch.no_grad():
            linear_layer.parametrizations.weight.original.data.normal_()

        expected_weight_after_update = weight_parametrization.forward(linear_layer.parametrizations.weight.original)
        actual_weight_after_update = linear_layer.weight
        self.assertTrue(torch.equal(actual_weight_after_update, expected_weight_after_update))

        with torch.no_grad():
            new_weight = torch.randn_like(linear_layer.weight)
            linear_layer.weight = new_weight

        expected_weight_after_update = weight_parametrization.right_inverse(new_weight)
        actual_weight_after_update = linear_layer.parametrizations.weight.original

        self.assertTrue(torch.equal(actual_weight_after_update, expected_weight_after_update))


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
