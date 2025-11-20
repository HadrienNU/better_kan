# modified from https://github.com/mfinzi/equivariant-MLP/blob/master/emlp/reps/representation.py and https://github.com/mfinzi/equivariant-MLP/blob/master/emlp/reps/product_sum_reps.py

import numpy as np
import torch
from scipy.special import binom
from functools import reduce
import itertools
from collections import defaultdict
from .utils import consistent_hash
from .lazy_operations import LinearOperator, ConcatLazy, LazyPerm, LazyDirectSum, LazyKron, LazyKronsum, I
from .lazy_operations import lazify, lazy_direct_matmat, densify

product = lambda c: reduce(lambda a, b: a * b, c)


class Rep(object):
    r"""The base Representation class. Representation objects formalize the vector space V
    on which the group acts, the group representation matrix ρ(g), and the Lie Algebra
    representation dρ(A) in a single object. Representations act as types for vectors coming
    from V. These types can be manipulated and transformed with the built in operators
    ⊕,⊗,dual, as well as incorporating custom representations. Rep objects should
    be immutable.

    At minimum, new representations need to implement ``rho``, ``__str__``."""

    is_permutation = False

    def rho(self, M):
        """Group representation of the matrix M of shape (d,d)"""
        raise NotImplementedError

    def drho(self, A):
        """Lie Algebra representation of the matrix A of shape (d,d)"""
        raise NotImplementedError

    def __call__(self, G):
        """Instantiate (non concrete) representation with a given symmetry group"""
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        d1 = tuple([(k, v) for k, v in self.__dict__.items() if (k not in ["_size", "is_permutation", "is_orthogonal"])])
        d2 = tuple([(k, v) for k, v in other.__dict__.items() if (k not in ["_size", "is_permutation", "is_orthogonal"])])
        return d1 == d2

    def __hash__(self):
        d1 = tuple([(k, v) for k, v in self.__dict__.items() if (k not in ["_size", "is_permutation", "is_orthogonal"])])
        return consistent_hash((type(self), d1))

    def size(self):
        if hasattr(self, "_size"):
            return self._size
        elif self.concrete and hasattr(self, "G"):
            self._size = self.rho(self.G.sample()).shape[-1]
            return self._size
        else:
            raise NotImplementedError

    def canonicalize(self):
        return self, np.arange(self.size())

    def rho_dense(self, M):
        """A convenience function which returns rho(M) as a dense matrix."""
        return densify(self.rho(M))

    def drho_dense(self, A):
        """A convenience function which returns drho(A) as a dense matrix."""
        return densify(self.drho(A))

    def constraint_matrix(self):
        n = self.size()
        constraints = []
        constraints.extend([lazify(self.rho(h)) - I(n) for h in self.G.discrete_generators])
        constraints.extend([lazify(self.drho(A)) for A in self.G.lie_algebra])
        return ConcatLazy(constraints) if constraints else lazify(np.zeros((1, n)))

    solcache = {}

    def equivariant_basis(self):
        if self == Scalar:
            return lazify(np.ones((1, 1)))
        canon_rep, perm = self.canonicalize()
        invperm = np.argsort(perm)
        if canon_rep not in self.solcache:
            C_lazy = canon_rep.constraint_matrix()
            # if C_lazy.shape[0] * C_lazy.shape[1] > 3e7:  # Too large to use SVD
            #     result = krylov_constraint_solve(C_lazy)
            # else:
            C = C_lazy.to_dense()
            result = orthogonal_complement(C)
            self.solcache[canon_rep] = result
        return self.solcache[canon_rep][invperm]

    def equivariant_projector(self):
        Q = self.equivariant_basis()
        Q_lazy = lazify(Q)
        P = Q_lazy @ Q_lazy.H
        return P

    @property
    def concrete(self):
        return hasattr(self, "G") and self.G is not None

    def __add__(self, other):
        if isinstance(other, int):
            if other == 0:
                return self
            else:
                return self + other * Scalar
        elif both_concrete(self, other):
            return SumRep(self, other)
        else:
            return DeferredSumRep(self, other)

    def __radd__(self, other):
        if isinstance(other, int):
            if other == 0:
                return self
            else:
                return other * Scalar + self
        else:
            return NotImplemented

    def __mul__(self, other):
        return mul_reps(self, other)

    def __rmul__(self, other):
        return mul_reps(other, self)

    def __pow__(self, other):
        assert isinstance(other, int), f"Power only supported for integers, not {type(other)}"
        assert other >= 0, f"Negative powers {other} not supported"
        return reduce(lambda a, b: a * b, other * [self], Scalar)

    def __rshift__(self, other):
        return other * self.T

    def __lshift__(self, other):
        return self * other.T

    def __lt__(self, other):
        if other == Scalar:
            return False
        try:
            if self.G < other.G:
                return True
            if self.G > other.G:
                return False
        except (AttributeError, TypeError):
            pass
        if self.size() < other.size():
            return True
        if self.size() > other.size():
            return False
        return consistent_hash(self) < consistent_hash(other)

    def __mod__(self, other):
        raise NotImplementedError

    @property
    def T(self):
        if hasattr(self, "G") and (self.G is not None) and self.G.is_orthogonal:
            return self
        return Dual(self)


class ScalarRep(Rep):
    def __init__(self, G=None):
        self.G = G
        self.is_permutation = True

    def __call__(self, G):
        self.G = G
        return self

    def size(self):
        return 1

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "V⁰"

    @property
    def T(self):
        return self

    def rho(self, M):
        return np.eye(1)

    def drho(self, M):
        return 0 * np.eye(1)

    def __hash__(self):
        return 0

    def __eq__(self, other):
        return isinstance(other, ScalarRep)

    def __mul__(self, other):
        if isinstance(other, int):
            return super().__mul__(other)
        return other

    def __rmul__(self, other):
        if isinstance(other, int):
            return super().__rmul__(other)
        return other

    @property
    def concrete(self):
        return True


class Base(Rep):
    def __init__(self, G=None):
        self.G = G
        if G is not None:
            self.is_permutation = G.is_permutation

    def __call__(self, G):
        return self.__class__(G)

    def rho(self, M):
        if hasattr(self, "G") and isinstance(M, dict):
            M = M[self.G]
        return M

    def drho(self, A):
        if hasattr(self, "G") and isinstance(A, dict):
            A = A[self.G]
        return A

    def size(self):
        assert self.G is not None, f"must know G to find size for rep={self}"
        return self.G.d

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "V"

    def __hash__(self):
        return consistent_hash((type(self), self.G))

    def __eq__(self, other):
        return type(other) == type(self) and self.G == other.G

    def __lt__(self, other):
        if isinstance(other, Dual):
            return True
        return super().__lt__(other)


class Dual(Rep):
    def __init__(self, rep):
        self.rep = rep
        self.G = rep.G
        if hasattr(rep, "is_permutation"):
            self.is_permutation = rep.is_permutation

    def __call__(self, G):
        return self.rep(G).T

    def rho(self, M):
        rho = self.rep.rho(M)
        rhoinvT = np.linalg.inv(rho).T
        return rhoinvT

    def drho(self, A):
        return -self.rep.drho(A).T

    def __str__(self):
        return str(self.rep) + "*"

    def __repr__(self):
        return str(self)

    @property
    def T(self):
        return self.rep

    def __eq__(self, other):
        return type(other) == type(self) and self.rep == other.rep

    def __hash__(self):
        return consistent_hash((type(self), self.rep))

    def __lt__(self, other):
        if other == self.rep:
            return False
        return super().__lt__(other)

    def size(self):
        return self.rep.size()


V = Vector = Base()

Scalar = ScalarRep()


def T(p, q=0, G=None):
    return (V**p * V.T**q)(G)


def orthogonal_complement(proj):
    U, S, VH = np.linalg.svd(proj, full_matrices=True)
    rank = (S > 1e-5).sum()
    return VH[rank:].conj().T


class SumRep(Rep):
    def __init__(self, *reps, extra_perm=None):
        reps = [SumRepFromCollection({Scalar: rep}) if isinstance(rep, int) else rep for rep in reps]
        reps, perms = zip(*[rep.canonicalize() for rep in reps])
        rep_counters = [rep.reps if isinstance(rep, SumRep) else {rep: 1} for rep in reps]
        self.reps, perm = self.compute_canonical(rep_counters, perms)
        self.perm = extra_perm[perm] if extra_perm is not None else perm
        self.invperm = np.argsort(self.perm)
        self.canonical = (self.perm == np.arange(len(self.perm))).all()
        self.is_permutation = all(rep.is_permutation for rep in self.reps.keys())

    def size(self):
        return sum(rep.size() * count for rep, count in self.reps.items())

    def rho(self, M):
        rhos = [rep.rho(M) for rep in self.reps]
        multiplicities = self.reps.values()
        return LazyPerm(self.invperm) @ LazyDirectSum(rhos, multiplicities) @ LazyPerm(self.perm)

    def drho(self, A):
        drhos = [rep.drho(A) for rep in self.reps]
        multiplicities = self.reps.values()
        return LazyPerm(self.invperm) @ LazyDirectSum(drhos, multiplicities) @ LazyPerm(self.perm)

    def __eq__(self, other):
        return self.reps == other.reps and (self.perm == other.perm).all()

    def __hash__(self):
        assert self.canonical
        return consistent_hash(tuple(self.reps.items()))

    @property
    def T(self):
        return SumRep(*[rep.T for rep, c in self.reps.items() for _ in range(c)], extra_perm=self.perm)

    def __repr__(self):
        return "+".join(f"{count if count > 1 else ''}{repr(rep)}" for rep, count in self.reps.items())

    def __str__(self):
        tensors = "+".join(f"{count if count > 1 else ''}{rep}" for rep, count in self.reps.items())
        return tensors

    def canonicalize(self):
        return SumRepFromCollection(self.reps), self.perm

    def __call__(self, G):
        return SumRepFromCollection({rep(G): c for rep, c in self.reps.items()}, perm=self.perm)

    @property
    def concrete(self):
        return True

    def equivariant_basis(self):
        Qs = {rep: rep.equivariant_basis() for rep in self.reps}
        active_dims = sum([self.reps[rep] * Qs[rep].shape[-1] for rep in Qs.keys()])
        multiplicities = self.reps.values()

        def lazy_Q(array):
            return lazy_direct_matmat(array, Qs.values(), multiplicities)[self.invperm]

        return LinearOperator(shape=(self.size(), active_dims), matvec=lazy_Q, matmat=lazy_Q)

    def equivariant_projector(self, lazy=False):
        Ps = {rep: rep.equivariant_projector() for rep in self.reps}
        multiplicities = self.reps.values()

        def lazy_P(array):
            return lazy_direct_matmat(array[self.perm], Ps.values(), multiplicities)[self.invperm]  # [:,self.invperm]

        return LinearOperator(shape=(self.size(), self.size()), matvec=lazy_P, matmat=lazy_P)

    @staticmethod
    def compute_canonical(rep_cnters, rep_perms):
        unique_reps = sorted(reduce(lambda a, b: a | b, [cnter.keys() for cnter in rep_cnters]))
        merged_cnt = defaultdict(int)
        permlist = []
        ids = [0] * len(rep_cnters)
        shifted_perms = []
        n = 0
        for perm in rep_perms:
            shifted_perms.append(n + perm)
            n += len(perm)
        for rep in unique_reps:
            for i in range(len(ids)):
                c = rep_cnters[i].get(rep, 0)
                permlist.append(shifted_perms[i][ids[i] : ids[i] + c * rep.size()])
                ids[i] += +c * rep.size()
                merged_cnt[rep] += c
        return dict(merged_cnt), np.concatenate(permlist)

    def __iter__(self):
        return (rep for rep, c in self.reps.items() for _ in range(c))

    def __len__(self):
        return sum(multiplicity for multiplicity in self.reps.values())

    def as_dict(self, v):
        out_dict = {}
        i = 0
        for rep, c in self.reps.items():
            chunk = c * rep.size()
            out_dict[rep] = v[..., self.perm[i : i + chunk]].reshape(v.shape[:-1] + (c, rep.size()))
            i += chunk
        return out_dict


def both_concrete(rep1, rep2):
    return all(rep.concrete for rep in (rep1, rep2) if hasattr(rep, "concrete"))


def mul_reps(ra, rb):
    if isinstance(rb, int):
        if rb == 1:
            return ra
        if rb == 0:
            return 0
        if (not hasattr(ra, "concrete")) or ra.concrete:
            return SumRep(*(rb * [ra]))
        else:
            return DeferredSumRep(*(rb * [ra]))
    if isinstance(ra, int):
        if ra == 1:
            return rb
        if ra == 0:
            return 0
        if (not hasattr(rb, "concrete")) or rb.concrete:
            return SumRep(*(ra * [rb]))
        else:
            return DeferredSumRep(*(ra * [rb]))
    if (isinstance(ra, SumRep) and isinstance(rb, Rep)) or (isinstance(ra, Rep) and isinstance(rb, SumRep)) or (isinstance(ra, SumRep) and isinstance(rb, SumRep)):
        if not both_concrete(ra, rb):
            return DeferredProductRep(ra, rb)
        return distribute_product([ra, rb])
    if type(ra) is ScalarRep:
        return rb
    if type(rb) is ScalarRep:
        return ra
    if not both_concrete(ra, rb):
        return DeferredProductRep(ra, rb)
    if hasattr(ra, "G") and hasattr(rb, "G") and ra.G == rb.G:
        return ProductRep(ra, rb)
    return DirectProduct(ra, rb)


class SumRepFromCollection(SumRep):
    def __init__(self, counter, perm=None):
        self.reps = counter
        self.perm = np.arange(self.size()) if perm is None else perm
        self.reps, self.perm = self.compute_canonical([counter], [self.perm])
        self.invperm = np.argsort(self.perm)
        self.canonical = (self.perm == np.arange(len(self.perm))).all()
        self.is_permutation = all(rep.is_permutation for rep in self.reps.keys())


def distribute_product(reps, extra_perm=None):
    reps, perms = zip(*[repsum.canonicalize() for repsum in reps])
    reps = [rep if isinstance(rep, SumRep) else SumRepFromCollection({rep: 1}) for rep in reps]

    axis_sizes = [len(perm) for perm in perms]

    order = np.arange(product(axis_sizes)).reshape(tuple(len(perm) for perm in perms))
    for i, perm in enumerate(perms):
        order = np.swapaxes(np.swapaxes(order, 0, i)[perm, ...], 0, i)
    order = order.reshape(-1)
    repsizes_all = []
    for rep in reps:
        this_rep_sizes = []
        for r, c in rep.reps.items():
            this_rep_sizes.extend([c * r.size()])
        repsizes_all.append(tuple(this_rep_sizes))
    block_perm = rep_permutation(tuple(repsizes_all))

    ordered_reps = []
    each_perm = []
    i = 0
    for prod in itertools.product(*[rep.reps.items() for rep in reps]):
        rs, cs = zip(*prod)
        prod_rep, canonicalizing_perm = (product(cs) * reduce(lambda a, b: a * b, rs)).canonicalize()
        ordered_reps.append(prod_rep)
        shape = []
        for r, c in prod:
            shape.extend([c, r.size()])
        axis_perm = np.concatenate([2 * np.arange(len(prod)), 2 * np.arange(len(prod)) + 1])
        mul_perm = np.arange(len(canonicalizing_perm)).reshape(shape).transpose(axis_perm).reshape(-1)
        each_perm.append(mul_perm[canonicalizing_perm] + i)
        i += len(canonicalizing_perm)
    each_perm = np.concatenate(each_perm)
    total_perm = order[block_perm[each_perm]]
    if extra_perm is not None:
        total_perm = extra_perm[total_perm]
    return SumRep(*ordered_reps, extra_perm=total_perm)


def rep_permutation(repsizes_all):
    """Permutation from block ordering to flattened ordering"""
    size_cumsums = [np.cumsum([0] + [size for size in repsizes]) for repsizes in repsizes_all]
    permutation = np.zeros([cumsum[-1] for cumsum in size_cumsums]).astype(int)
    arange = np.arange(permutation.size)
    indices_iter = itertools.product(*[range(len(repsizes)) for repsizes in repsizes_all])
    i = 0
    for indices in indices_iter:
        slices = tuple([slice(cumsum[idx], cumsum[idx + 1]) for idx, cumsum in zip(indices, size_cumsums)])
        slice_lengths = [sl.stop - sl.start for sl in slices]
        chunk_size = np.prod(slice_lengths)
        permutation[slices] += arange[i : i + chunk_size].reshape(*slice_lengths)
        i += chunk_size
    return np.argsort(permutation.reshape(-1))


class ProductRep(Rep):
    def __init__(self, *reps, extra_perm=None, counter=None):
        if counter is not None:
            self.reps = counter
            self.reps, self.perm = self.compute_canonical([counter], [np.arange(self.size()) if extra_perm is None else extra_perm])
        else:
            reps, perms = zip(*[rep.canonicalize() for rep in reps])
            rep_counters = [rep.reps if type(rep) == ProductRep else {rep: 1} for rep in reps]
            self.reps, perm = self.compute_canonical(rep_counters, perms)
            self.perm = extra_perm[perm] if extra_perm is not None else perm

        self.invperm = np.argsort(self.perm)
        self.canonical = (self.perm == self.invperm).all()
        Gs = tuple(set(rep.G for rep in self.reps.keys()))
        assert len(Gs) == 1, f"Multiple different groups {Gs} in product rep {self}"
        self.G = Gs[0]
        self.is_permutation = all(rep.is_permutation for rep in self.reps.keys())

    def size(self):
        return product([rep.size() ** count for rep, count in self.reps.items()])

    def rho(self, Ms, lazy=False):
        if hasattr(self, "G") and isinstance(Ms, dict):
            Ms = Ms[self.G]
        canonical_lazy = LazyKron([rep.rho(Ms) for rep, c in self.reps.items() for _ in range(c)])
        return LazyPerm(self.invperm) @ canonical_lazy @ LazyPerm(self.perm)

    def drho(self, As):
        if hasattr(self, "G") and isinstance(As, dict):
            As = As[self.G]
        canonical_lazy = LazyKronsum([rep.drho(As) for rep, c in self.reps.items() for _ in range(c)])
        return LazyPerm(self.invperm) @ canonical_lazy @ LazyPerm(self.perm)

    def __hash__(self):
        assert self.canonical, f"Not canonical {repr(self)}? perm {self.perm}"
        return consistent_hash(tuple(self.reps.items()))

    def __eq__(self, other):
        return isinstance(other, ProductRep) and self.reps == other.reps and (self.perm == other.perm).all()

    @property
    def concrete(self):
        return True

    @property
    def T(self):
        return self.__class__(*[rep.T for rep, c in self.reps.items() for _ in range(c)], extra_perm=self.perm)

    def __str__(self):
        superscript = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
        return "⊗".join([str(rep) + (f"{c}".translate(superscript) if c > 1 else "") for rep, c in self.reps.items()])

    def canonicalize(self):
        return self.__class__(counter=self.reps), self.perm

    @staticmethod
    def compute_canonical(rep_cnters, rep_perms):
        order = np.arange(product(len(perm) for perm in rep_perms))
        unique_reps = sorted(reduce(lambda a, b: a | b, [cnter.keys() for cnter in rep_cnters]))
        merged_cnt = defaultdict(int)
        order = order.reshape(tuple(len(perm) for perm in rep_perms))
        for i, perm in enumerate(rep_perms):
            order = np.moveaxis(np.moveaxis(order, i, 0)[perm, ...], 0, i)
        axis_ids = []
        n = 0
        for cnter in rep_cnters:
            axis_idsi = {}
            for rep, c in cnter.items():
                axis_idsi[rep] = n + np.arange(c)
                n += c
            axis_ids.append(axis_idsi)
        axes_perm = []
        for rep in unique_reps:
            for i in range(len(rep_perms)):
                c = rep_cnters[i].get(rep, 0)
                if c != 0:
                    axes_perm.append(axis_ids[i][rep])
                    merged_cnt[rep] += c
        axes_perm = np.concatenate(axes_perm)
        order = order.reshape(tuple(rep.size() for cnter in rep_cnters for rep, c in cnter.items() for _ in range(c)))
        final_order = np.transpose(order, axes_perm)
        return dict(merged_cnt), final_order.reshape(-1)


class DirectProduct(ProductRep):
    def __init__(self, *reps, counter=None, extra_perm=None):
        if counter is not None:
            self.reps = counter
            self.reps, perm = self.compute_canonical([counter], [np.arange(self.size())])
            self.perm = extra_perm[perm] if extra_perm is not None else perm
        else:
            reps, perms = zip(*[rep.canonicalize() for rep in reps])
            rep_counters = [rep.reps if type(rep) == DirectProduct else {rep: 1} for rep in reps]
            reps, perm = self.compute_canonical(rep_counters, perms)
            group_dict = defaultdict(lambda: 1)
            for rep, c in reps.items():
                group_dict[rep.G] = group_dict[rep.G] * rep**c
            sub_products = {rep: 1 for G, rep in group_dict.items()}
            self.reps = counter = sub_products
            self.reps, perm2 = self.compute_canonical([counter], [np.arange(self.size())])
            self.perm = extra_perm[perm[perm2]] if extra_perm is not None else perm[perm2]
        self.invperm = np.argsort(self.perm)
        self.canonical = (self.perm == self.invperm).all()
        self.is_permutation = all(rep.is_permutation for rep in self.reps.keys())
        assert all(count == 1 for count in self.reps.values())

    def equivariant_basis(self):
        canon_Q = LazyKron([rep.equivariant_basis() for rep, c in self.reps.items()])
        return LazyPerm(self.invperm) @ canon_Q

    def equivariant_projector(self):
        canon_P = LazyKron([rep.equivariant_projector() for rep, c in self.reps.items()])
        return LazyPerm(self.invperm) @ canon_P @ LazyPerm(self.perm)

    def rho(self, Ms):
        canonical_lazy = LazyKron([rep.rho(Ms) for rep, c in self.reps.items() for _ in range(c)])
        return LazyPerm(self.invperm) @ canonical_lazy @ LazyPerm(self.perm)

    def drho(self, As):
        canonical_lazy = LazyKronsum([rep.drho(As) for rep, c in self.reps.items() for _ in range(c)])
        return LazyPerm(self.invperm) @ canonical_lazy @ LazyPerm(self.perm)

    def __str__(self):
        superscript = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
        return "⊗".join([str(rep) + f"_{rep.G}" for rep, c in self.reps.items()])


class DeferredSumRep(Rep):
    def __init__(self, *reps):
        self.to_sum = []
        for rep in reps:
            self.to_sum.extend(rep.to_sum if isinstance(rep, DeferredSumRep) else [rep])

    def __call__(self, G):
        if G is None:
            return self
        return SumRep(*[rep(G) for rep in self.to_sum])

    def __repr__(self):
        return "(" + "+".join(f"{rep}" for rep in self.to_sum) + ")"

    def __str__(self):
        return repr(self)

    @property
    def T(self):
        return DeferredSumRep(*[rep.T for rep in self.to_sum])

    @property
    def concrete(self):
        return False


class DeferredProductRep(Rep):
    def __init__(self, *reps):
        self.to_prod = []
        for rep in reps:
            assert not isinstance(rep, ProductRep)
            self.to_prod.extend(rep.to_prod if isinstance(rep, DeferredProductRep) else [rep])

    def __call__(self, G):
        if G is None:
            return self
        return reduce(lambda a, b: a * b, [rep(G) for rep in self.to_prod])

    def __repr__(self):
        return "⊗".join(f"{rep}" for rep in self.to_prod)

    def __str__(self):
        return repr(self)

    @property
    def T(self):
        return DeferredProductRep(*[rep.T for rep in self.to_prod])

    @property
    def concrete(self):
        return False


def gated(ch_rep: Rep) -> Rep:
    if isinstance(ch_rep, SumRep):
        return ch_rep + sum([Scalar(rep.G) for rep in ch_rep if rep != Scalar and not rep.is_permutation])
    else:
        return ch_rep + Scalar(ch_rep.G) if not ch_rep.is_permutation else ch_rep


def gate_indices(ch_rep: Rep) -> np.ndarray:
    channels = ch_rep.size()
    if hasattr(ch_rep, "perm"):
        perm = ch_rep.perm
    else:
        perm = np.arange(channels)
    indices = np.arange(channels)

    if not isinstance(ch_rep, SumRep):
        return indices if ch_rep.is_permutation else np.ones(ch_rep.size()) * ch_rep.size()

    num_nonscalars = 0
    i = 0
    for rep in ch_rep:
        if rep != Scalar and not rep.is_permutation:
            indices[perm[i : i + rep.size()]] = channels + num_nonscalars
            num_nonscalars += 1
        i += rep.size()
    return indices


def lambertW(ch, d):
    max_rank = 0
    while (max_rank + 1) * d**max_rank <= ch:
        max_rank += 1
    max_rank -= 1
    return max_rank


def binomial_allocation(N, rank, G):
    if N == 0:
        return 0
    n_binoms = N // (2**rank)
    n_leftover = N % (2**rank)
    even_split = sum([n_binoms * int(binom(rank, k)) * T(k, rank - k, G) for k in range(rank + 1)])
    ps = np.random.binomial(rank, 0.5, n_leftover)
    ragged = sum([T(int(p), rank - int(p), G) for p in ps])
    out = even_split + ragged
    return out


def uniform_rep(ch, group):
    """A heuristic method for allocating a given number of channels (ch)
    into tensor types. Attempts to distribute the channels evenly across
    the different tensor types. Useful for hands off layer construction.

    Args:
        ch (int): total number of channels
        group (Group): symmetry group

    Returns:
        SumRep: The direct sum representation with dim(V)=ch
    """
    d = group.d
    Ns = np.zeros((lambertW(ch, d) + 1), int)
    while ch > 0:
        max_rank = lambertW(ch, d)
        Ns[: max_rank + 1] += np.array([d ** (max_rank - r) for r in range(max_rank + 1)], dtype=int)
        ch -= (max_rank + 1) * d**max_rank
    sum_rep = sum([binomial_allocation(nr, r, group) for r, nr in enumerate(Ns)])
    sum_rep, perm = sum_rep.canonicalize()
    return sum_rep


def lazy_P(P_terms, multiplicities, perm, invperm, v):
    if len(v.shape) == 1:
        v = v.unsqueeze(1)
    v = v[perm]
    n = v.shape[0]
    k = v.shape[1] if len(v.shape) > 1 else 1
    i = 0
    y = []
    for P_term, multiplicity in zip(P_terms, multiplicities):
        i_end = i + multiplicity * P_term.shape[-1]
        elems = P_term @ v[i:i_end].T.reshape(k * multiplicity, P_term.shape[-1]).T
        y.append(elems.T.reshape(k, multiplicity * P_term.shape[0]).T)
        i = i_end
    y = torch.cat(y, axis=0)
    return y[invperm]


def lazy_Pinv(P_terms, multiplicities, perm, invperm, v):
    if len(v.shape) == 1:
        v = v.unsqueeze(1)
    v = v[perm]
    n = v.shape[0]
    k = v.shape[1] if len(v.shape) > 1 else 1
    i = 0
    y = []
    for P_term, multiplicity in zip(P_terms, multiplicities):
        i_end = i + multiplicity * P_term.shape[0]
        elems = torch.linalg.pinv(P_term) @ v[i:i_end].T.reshape(k * multiplicity, P_term.shape[0]).T
        y.append(elems.T.reshape(k, multiplicity * P_term.shape[-1]).T)
        i = i_end
    y = torch.cat(y, axis=0)
    return y[invperm]
