import pytest
import torch
import torch
from better_kan import build_KAN
from better_kan.functions import Splines
from better_kan.equivariance import S, Trivial, V, parametrize_kan_equivariance, unparametrize_kan
from better_kan.equivariance.parametrizations import EquivariantVector, EquivariantGrid, EquivariantMatrix, EquivariantBasisWeight
from torch.nn.utils.parametrize import is_parametrized


# Fixture to provide a common setup for the tests
@pytest.fixture
def setup_groups_and_reps():
    """Provides predefined groups and representations for testing."""
    G = S(5)
    G2 = S(5)
    rep = V(G)
    rep2 = V(G2)
    return rep, rep2


@pytest.fixture(params=[[5, 5, 5], [5, 25, 1]])
def setup_groups_and_reps_model(request):
    """Provides predefined groups and representations for testing."""
    rep_in = V(S(request.param[0]))
    rep_mid = V(S(request.param[1]))
    rep_out = V(Trivial(request.param[2]))
    return rep_in, rep_mid, rep_out


def test_equivariant_vector(setup_groups_and_reps):
    """
    Tests the forward and right_inverse methods of the EquivariantMatrix class.
    """
    rep, _ = setup_groups_and_reps

    # Initialize the EquivariantMatrix
    vec = EquivariantVector(rep)
    n_size_basis = vec.basis.shape[-1]
    # 1. Test the shape of the basis
    assert vec.basis.shape == (rep.size(), n_size_basis)

    # 2. Test the right_inverse method
    a = torch.rand(rep.size())
    w = vec.right_inverse(a)
    assert w.shape == (n_size_basis,)

    # 3. Test the forward pass
    output = vec(w)
    assert output.shape == (rep.size(),)

    # 4. Check if right_inverse(forward()) is the identity
    w_bis = vec.right_inverse(output)
    torch.testing.assert_close(w_bis, w)


def test_equivariant_grid(setup_groups_and_reps):
    """
    Tests the forward and right_inverse methods of the EquivariantMatrix class.
    """
    rep, _ = setup_groups_and_reps

    # Initialize the EquivariantMatrix
    vec = EquivariantGrid(rep)
    n_size_basis = vec.basis.shape[-1]
    # 1. Test the shape of the basis
    assert vec.basis.shape == (rep.size(), n_size_basis)

    # 2. Test the right_inverse method
    a = torch.rand(15, rep.size())
    w = vec.right_inverse(a)
    assert w.shape == (a.shape[0], n_size_basis)

    # 3. Test the forward pass
    output = vec(w)
    assert output.shape == (a.shape[0], rep.size())

    # 4. Check if right_inverse(forward()) is the identity
    w_bis = vec.right_inverse(output)
    torch.testing.assert_close(w_bis, w)


def test_equivariant_matrix(setup_groups_and_reps):
    """
    Tests the forward and right_inverse methods of the EquivariantMatrix class.
    """
    rep, rep2 = setup_groups_and_reps

    # Initialize the EquivariantMatrix
    mat = EquivariantMatrix(rep, rep2)

    # 1. Test the shape of the basis
    assert mat.basis.shape == (rep.size() * rep2.size(), 2)

    # 2. Test the right_inverse method
    a = torch.rand(rep2.size(), rep.size())
    w = mat.right_inverse(a)
    assert w.shape == (2,)

    # 3. Test the forward pass
    output = mat(w)
    assert output.shape == (rep2.size(), rep.size())

    # 4. Check if right_inverse(forward()) is the identity
    w_bis = mat.right_inverse(output)
    torch.testing.assert_close(w_bis, w)


def test_equivariant_basis_weight(setup_groups_and_reps):
    """
    Tests the forward and right_inverse methods of the EquivariantBasisWeight class.
    """
    rep, rep2 = setup_groups_and_reps

    # Initialize the EquivariantBasisWeight
    mat = EquivariantBasisWeight(rep, rep2)

    # 1. Test the shape of the basis
    assert mat.basis.shape == (rep.size() * rep2.size(), 2)

    # 2. Test the right_inverse method
    a = torch.rand(rep2.size(), 3, 2, rep.size())
    w = mat.right_inverse(a)
    assert w.shape == (3, 2, 2)

    # 3. Test the forward pass
    output = mat(w)
    assert output.shape == (rep2.size(), 3, 2, rep.size())

    # 4. Check if right_inverse(forward()) is the identity
    w_bis = mat.right_inverse(output)
    torch.testing.assert_close(w_bis, w)


def test_parametrize_and_unparametrize_kan(setup_groups_and_reps_model):
    """
    Tests that parametrize_kan_equivariance and unparametrize_kan correctly
    add and remove parametrizations from a KAN-like model.
    """
    rep_in, rep_mid, rep_out = setup_groups_and_reps_model
    # 1. Setup the model and representations
    model = build_KAN(Splines, [rep_in.size(), rep_mid.size(), rep_out.size()], grid_size=5, fast_version=False)
    rep_list = [rep_in, rep_mid, rep_out]

    # 2. Apply parametrization
    parametrize_kan_equivariance(model, rep_list)

    # 3. Assert that all relevant parameters are now parametrized
    for layer in model.layers:
        assert is_parametrized(layer, "bias")
        assert isinstance(layer.parametrizations.bias[0], EquivariantVector)
        for fct in layer.functions:
            assert is_parametrized(fct, "weights")
            assert isinstance(fct.parametrizations.weights[0], EquivariantBasisWeight)
            assert hasattr(fct, "grid")
            assert is_parametrized(fct.grid, "grid")
            assert isinstance(fct.grid.parametrizations.grid[0], EquivariantGrid)

    # 4. Apply unparametrization
    unparametrize_kan(model)

    # 5. Assert that all parametrizations have been removed
    for layer in model.layers:
        assert not is_parametrized(layer, "bias")
        for fct in layer.functions:
            assert not is_parametrized(fct, "weights")
            assert hasattr(fct, "grid")
            assert not is_parametrized(fct.grid, "grid")
