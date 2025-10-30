# test_kan_fcts.py

import pytest
import torch
import torch.nn as nn

from better_kan.functions import Grid
from better_kan.functions import ActivationFunction, ChebyshevPolynomial, HermitePolynomial, Splines, GridReLU, RBFFunction


fct_test_cases = [
    pytest.param(ActivationFunction, {"in_features": 3, "out_features": 2, "base_activation": nn.ReLU}, id="ActivationFunction"),
    pytest.param(ChebyshevPolynomial, {"in_features": 3, "out_features": 2, "poly_order": 4}, id="ChebyshevPolynomial"),
    pytest.param(Splines, {"in_features": 4, "out_features": 5, "grid": Grid(4, 15, order=3)}, id="Splines"),
    pytest.param(GridReLU, {"in_features": 5, "out_features": 2, "grid": Grid(5, 10, order=2)}, id="GridReLU"),
    pytest.param(RBFFunction, {"in_features": 2, "out_features": 5, "grid": Grid(2, 15, order=0), "rbf_kernel": "gaussian"}, id="RBF"),
]

init_testcase = [
    pytest.param("uniform", {}, id="uniform"),
    pytest.param("noise", {"scale": 0.1}, id="noise"),
]


@pytest.mark.parametrize("fct_class, fct_kwargs", fct_test_cases)
class TestAllFunctions:
    """
    A single test suite that runs against all specified BasisFunction children.
    The `@parametrize` decorator above the class will feed each test method
    with the `fct_class` and `fct_kwargs` for every case defined.
    """

    # --- Fixtures ---
    # These fixtures create instances and data based on the parameters
    # passed to the class by the decorator.

    @pytest.fixture
    def fct(self, fct_class, fct_kwargs):
        """Fixture to create an instance of the fct to be tested."""
        return fct_class(**fct_kwargs)

    @pytest.fixture
    def input_data_2d(self, fct):
        """Provides a standard 2D input tensor for the created fct."""
        batch_size = 10
        return torch.randn(batch_size, fct.in_features)

    @pytest.fixture
    def input_data_4d(self, fct):
        """Provides a 4D input tensor to test reshaping logic."""
        batch_size = 10
        return torch.randn(2, 5, batch_size, fct.in_features)

    # --- Generic Tests (will run for ALL parametrized classes) ---
    @pytest.mark.parametrize("type_init, init_kwargs", init_testcase)
    def test_initialization(self, fct, type_init, init_kwargs):
        """Generic Test: Ensure correct shapes after initialization."""
        assert isinstance(fct, torch.nn.Module)
        expected_weights_shape = (fct.out_features, fct.n_basis_function, fct.in_features)
        assert fct.weights.shape == expected_weights_shape
        fct.reset_parameters(type_init, **init_kwargs)
        assert fct.weights.shape == expected_weights_shape

    def test_activations(self, fct, input_data_2d):
        """Generic Test: Activations produces correct shape."""
        output = fct.activations_eval(input_data_2d)
        assert output.shape == (input_data_2d.shape[0], fct.out_features, input_data_2d.shape[-1])

    def test_forward(self, fct, input_data_2d):
        """Generic Test: Forward pass produces correct shape."""
        output = fct(input_data_2d)
        assert output.shape == (input_data_2d.shape[0], fct.out_features)

    def test_forward_high_dims(self, fct, input_data_4d):
        """Generic Test: Fast forward pass handles high-dimensional input."""
        output = fct(input_data_4d)
        expected_shape = input_data_4d.shape[:-1] + (fct.out_features,)
        assert output.shape == expected_shape

    def test_curve2coeff_shape(self, fct, input_data_2d):
        """Generic Test: curve2coeff method returns the correct shape."""
        y = torch.randn(input_data_2d.shape[0], fct.in_features, fct.out_features)
        coeffs = fct.curve2coeff(input_data_2d, y)
        expected_shape = (fct.out_features, fct.n_basis_function, fct.in_features)
        assert coeffs.shape == expected_shape

    def test_update_and_project(self, fct, input_data_2d):
        if hasattr(fct, "update_grid"):
            pytest.skip("Not implemented")
            pass
            # TODO  write the test and check that the residuals at some collocations points are low
        else:
            pytest.skip("Not applicable")
