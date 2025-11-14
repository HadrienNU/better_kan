import numpy as np
from numpy import testing as npt
import pytest
import logging

from better_kan.equivariance.representation import *
from better_kan.equivariance.groups import *

# A list of test groups from the EMLP library
test_groups = (
    [SO(n) for n in [2, 3, 4]]
    + [O(n) for n in [2, 3, 4]]
    + [SU(n) for n in [2, 3]]
    + [U(n) for n in [2, 3]]
    + [SL(n) for n in [2, 3]]
    + [GL(n) for n in [2, 3]]
    + [C(k) for k in [2, 3, 4, 8]]
    + [D(k) for k in [2, 3, 4, 8]]
    + [S(n) for n in [2, 4, 6]]
    + [Z(n) for n in [2, 4, 6]]
    + [SO11p(), SO13p(), SO13(), O13()]
    + [Sp(n) for n in [1, 2]]
    + [RubiksCube(), Cube()]
)

# Create IDs for the test_groups list to make pytest output readable
test_group_ids = [str(g) for g in test_groups]


@pytest.mark.parametrize("G", test_groups, ids=test_group_ids)
def test_sum(G):
    """Test invariance of vectors in the trivial subspace of a direct sum representation."""
    N = 5
    rep = T(0, 2) + 3 * (T(0, 0) + T(1, 0)) + T(0, 0) + T(1, 1) + 2 * T(1, 0)
    rep = rep(G)

    P = rep.equivariant_projector()
    v = np.random.rand(rep.size())
    v_inv = P @ v  # Project to the invariant subspace

    gs = G.samples(N)
    rho_gs = np.array([rep.rho(g) for g in gs])
    gv = np.einsum("nij,j->ni", rho_gs, v_inv)

    # # Assert that gv is close to v_inv for all g in the batch
    for i in range(N):
        npt.assert_allclose(gv[i], v_inv, rtol=1e-4, atol=1e-5, err_msg=f"Symmetric vector fails for G={G}")


small_test_groups = [group for group in test_groups if group.d < 5]
small_test_group_ids = [str(g) for g in small_test_groups]


@pytest.mark.parametrize("G", small_test_groups, ids=small_test_group_ids)
def test_prod(G):
    """Test invariance of vectors in the trivial subspace of a tensor product representation."""
    N = 5
    rep = T(0, 1) * T(0, 0) * T(1, 0) ** 2
    rep = rep(G)

    Q = rep.equivariant_basis()
    v_inv = Q @ np.random.rand(Q.shape[-1])

    gs = G.samples(N)
    rho_gs = np.array([rep.rho(g) for g in gs])
    gv = np.einsum("nij,j->ni", rho_gs, v_inv)

    # Assert that gv is close to v_inv for all g in the batch
    for i in range(N):
        npt.assert_allclose(gv[i], v_inv, rtol=1e-4, atol=1e-5, err_msg=f"Symmetric vector fails for G={G}")


@pytest.mark.parametrize("G", test_groups, ids=test_group_ids)
def test_high_rank_representations(G):
    """Test invariance for high-rank tensor representations."""
    N = 5
    r = 10
    for p in range(r + 1):
        for q in range(r - p + 1):
            if G.is_orthogonal and q > 0:
                continue

            rep = T(p, q)(G)
            # Skip representations that are too large to handle
            if rep.size() > 2000:
                continue

            P = rep.equivariant_projector()
            v = np.random.rand(rep.size())
            v_inv = P @ v

            gs = G.samples(N)
            g_rho = np.array([rep.rho(g) for g in gs])
            gv = np.einsum("nij,j->ni", g_rho, v_inv)
            for i in range(N):
                npt.assert_allclose(gv[i], v_inv, rtol=1e-4, atol=1e-5, err_msg=f"Symmetric vector fails with T{p,q} and G={G}")
            logging.info(f"Success with T{p,q} and G={G}")


equivariant_matrix_cases = [
    (SO(3), T(1) + 2 * T(0), T(1) + T(2) + 2 * T(0) + T(1)),
    (SO(3), 5 * T(0) + 5 * T(1), 3 * T(0) + T(2) + 2 * T(1)),
    (SO13p(), T(2) + 4 * T(1, 0) + T(0, 1), 10 * T(0) + 3 * T(1, 0)),
    (SU(3), T(2, 0) + T(1, 1), V + V.T + T(0)),
]
equivariant_matrix_ids = [f"{c[0]}-{c[1]}-{c[2]}" for c in equivariant_matrix_cases]


@pytest.mark.parametrize("G", test_groups, ids=test_group_ids)
def test_large_representations(G):
    """Test equivariance for larger, uniform representations."""
    if G.is_trivial:
        return
    N = 5
    ch = 32
    repin = repout = uniform_rep(ch, G)

    # Skip if representations are too large
    if repin.size() > 1000:
        return

    repW = repin >> repout

    P = repW.equivariant_projector()
    W = np.random.rand(repout.size(), repin.size())
    W_equiv = (P @ W.reshape(-1)).reshape(*W.shape)

    x = np.random.rand(N, repin.size())
    gs = G.samples(N)

    rho_in = np.array([repin.rho(g) for g in gs])
    rho_out = np.array([repout.rho(g) for g in gs])

    gx = np.einsum("nij,nj->ni", rho_in, x)
    W_gx = gx @ W_equiv.T
    g_Wx = np.einsum("nij,nj->ni", rho_out, x @ W_equiv.T)
    for i in range(N):
        npt.assert_allclose(W_gx[i], g_Wx[i], rtol=1e-4, atol=1e-5, err_msg=f"Large rep gWx=Wgx fails for G={G}")
    logging.info(f"Success with G={G}")
