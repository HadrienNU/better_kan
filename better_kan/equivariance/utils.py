import numpy as np
import torch
import matplotlib.pyplot as plt
import hashlib
import itertools
from tqdm import tqdm


def consistent_hash(value):
    return int(hashlib.sha256(str(value).encode()).hexdigest(), 16)


def draw_matrix_parametrizations(parametrizations, X, cmap="rainbow", slice_dims=(-2, -1)):

    weights = parametrizations.right_inverse(X)

    num_weights = weights.numel()
    weights_shape = weights.shape

    wvals = torch.arange(num_weights, dtype=torch.float).reshape(weights_shape)
    w = parametrizations.forward(wvals).numpy()
    if len(w.shape) == 1:
        w = w.reshape(-1, 1)

    def plot_single_matrix(ax, matrix):
        ax.imshow(matrix[::-1, :].T, cmap=cmap, alpha=0.8, vmin=w.min(), vmax=w.max(), origin="lower", aspect="equal")
        ax.set_xticks(np.arange(matrix.shape[0] - 1) + 0.5, [])
        ax.set_yticks(np.arange(matrix.shape[1] - 1) + 0.5, [])
        ax.set_xlim(-0.5, matrix.shape[0] - 0.5)
        ax.set_ylim(-0.5, matrix.shape[1] - 0.5)
        ax.grid(which="major", color="grey", linestyle="-")
        ax.xaxis.set_ticks_position("none")
        ax.yaxis.set_ticks_position("none")

    # If w is a high-dimensional tensor, treat it as a collection of matrices
    if len(w.shape) > 2:
        # Identify dimensions other than the slice
        slice_dims = tuple(d if d >= 0 else w.ndim + d for d in slice_dims)  # Support negative dimension
        all_dims = list(range(w.ndim))
        iter_dims = [d for d in all_dims if d not in slice_dims]

        # Generate all index combinations for the other dims
        indices = list(itertools.product(*(range(w.shape[d]) for d in iter_dims)))

        num_matrices = len(indices)
        ncols = int(np.ceil(np.sqrt(num_matrices)))
        nrows = int(np.ceil(num_matrices / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
        # Flatten axes array for easy iteration
        axes = np.atleast_1d(axes).ravel()

        for i, idx in enumerate(indices):
            sel = [slice(None)] * w.ndim
            for d, val in zip(iter_dims, idx):
                sel[d] = val
            sub_matrix = w[tuple(sel)]

            # for i in range(num_matrices):
            #     sub_matrix = w[i]
            ax = axes[i]
            plot_single_matrix(ax, sub_matrix)
            ax.set_title(f"Submatrix {idx}")

        # Hide any unused subplots
        for i in range(num_matrices, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

    else:  # Original behavior for 2D or 1D arrays
        fig, ax = plt.subplots(figsize=(8, 8))
        plot_single_matrix(ax, w)


def get_mask_parametrizations(parametrizations, X):
    """
    For a parametrizations, get the mask of unique values
    """
    weights = parametrizations.right_inverse(X)
    num_weights = weights.numel()
    weights_shape = weights.shape

    wvals = torch.arange(num_weights, dtype=torch.float).reshape(weights_shape)
    w = parametrizations.forward(wvals).numpy()

    w = np.round(w, 4)  # For tolerance between float
    _, mask = np.unique_inverse(w[:, 0, :])
    return mask


def krylov_constraint_solve(C, tol=1e-5):
    import logging

    """Computes the solution basis Q for the linear constraint CQ=0  and QᵀQ=I
    up to specified tolerance with C expressed as a LinearOperator."""
    r = 5
    if C.shape[0] * r * 2 > 2e9:
        raise Exception(f"Solns for contraints {C.shape} too large to fit in memory")
    found_rank = 5
    while found_rank == r:
        r *= 2  # Iterative doubling of rank until large enough to include the full solution space
        if C.shape[0] * r > 2e9:
            logging.error(f"Hit memory limits, switching to sample equivariant subspace of size {found_rank}")
            break
        Q = krylov_constraint_solve_upto_r(C, r, tol)
        found_rank = Q.shape[-1]
    return Q


class ConvergenceError(Exception):
    pass


def krylov_constraint_solve_upto_r(C, r, tol=1e-5, lr=1e-2):
    """
    Iterative routine to compute the solution basis to the constraint CQ=0 and Q.T @ Q = I
    up to the rank r, with given tolerance. Uses gradient descent (+ momentum) on the
    objective |CQ|^2.
    """
    # Initialize W
    n_in = C.shape[-1]
    W = np.random.randn(n_in, r) / np.sqrt(n_in)

    # Ensure complex support if C is complex
    if np.iscomplexobj(C):
        W = W.astype(C.dtype)

    # Momentum state initialization (Optax sgd default momentum is usually 0.9)
    momentum = 0.9
    velocity = np.zeros_like(W)

    def get_loss_and_grad(W):
        # Forward: CW
        CW = C @ W
        # Loss: 0.5 * ||CW||^2
        loss = 0.5 * np.sum(np.abs(CW) ** 2)
        # Gradient: C^H @ (C @ W)
        # Note: .adjoint().T handles Hermitian transpose for complex, normal transpose for real
        grad = C.adjoint().T @ CW
        return loss, grad

    lstart, _ = get_loss_and_grad(W)

    # Setup progress bar
    pbar = tqdm(total=100, desc=f"Krylov Solving for Equivariant Subspace r<={r}", bar_format="{l_bar}{bar}| {n:.3g}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]")
    prog_val = 0

    for i in range(20000):
        lossval, grad = get_loss_and_grad(W)

        # Manual SGD with Momentum update (equivalent to optax.sgd(lr, 0.9))
        # v_{t+1} = mu * v_t + g_t
        # w_{t+1} = w_t - lr * v_{t+1}
        velocity = momentum * velocity + grad
        W = W - lr * velocity

        # Update progress bar
        # Avoid log(0) issues if loss drops to exactly 0
        safe_loss = max(lossval, 1e-30)
        ratio = np.log(safe_loss / lstart) / np.log(tol**2 / lstart)
        progress = max(100 * ratio - prog_val, 0)
        progress = min(100 - prog_val, progress)

        if progress > 0:
            prog_val += progress
            pbar.update(progress)

        # Check convergence condition
        if np.sqrt(lossval) < tol:
            pbar.close()
            break

        # Check divergence
        if lossval > 2e3 and i > 100:
            pbar.close()
            logging.warning(f"Constraint solving diverged, trying lower learning rate {lr/3:.2e}")
            if lr < 1e-4:
                raise ConvergenceError(f"Failed to converge even with smaller learning rate {lr:.2e}")
            return krylov_constraint_solve_upto_r(C, r, tol, lr=lr / 3)
    else:
        # Loop finished without breaking
        pbar.close()
        raise ConvergenceError("Failed to converge.")

    # Orthogonalize solution at the end using SVD
    # full_matrices=False performs Economy SVD (O(n*r^2) approx), solving the concern in original code
    U, S, VT = np.linalg.svd(W, full_matrices=False)

    # Determine Rank
    rank = (S > 10 * tol).sum()
    Q = U[:, :rank]

    # Verify final error
    final_L, _ = get_loss_and_grad(Q)

    if final_L > tol:
        logging.warning(f"Normalized basis has too high error {final_L:.2e} for tol {tol:.2e}")

    scutoff = S[rank] if r > rank else 0

    # Validity assertion
    # Note: S[rank-1] might index out of bounds if rank is 0, handled logic safely below
    if rank > 0:
        gap_condition = scutoff < S[rank - 1] / 100
        assert gap_condition, f"Singular value gap too small: {S[rank-1]:.2e} above cutoff " f"{scutoff:.2e} below cutoff. Final L {final_L:.2e}"

    return Q


def equivariant_permutations_inputs(inputs):
    """
    Return the generators of permutation group for a given input
    inputs is an array that contain the type of the entry, i.e. equivariant entry share the same type

    Return the generators of the symmetry group
    """
    from .representation import V, SumRep
    from .groups import S, Trivial

    inputs = np.asarray(inputs).ravel()
    nb_inputs = len(inputs)
    types = np.unique(inputs)
    reps = []
    perm = np.arange(nb_inputs, dtype=int)
    m = 0
    for t in types:
        if np.sum(inputs == t) > 1:  # Only if more than one entry share the same type
            reps.append(V(S(np.sum(inputs == t))))
            # Ensuite il faut construire l'array de permutation
            inds = np.nonzero(inputs == t)[0]
            for i in inds:
                perm[m] = i
                m += 1
        else:
            reps.append(V(Trivial(1)))
            inds = np.nonzero(inputs == t)[0]
            perm[m] = inds[0]
            m += 1
    return SumRep(*reps, extra_perm=perm), perm
