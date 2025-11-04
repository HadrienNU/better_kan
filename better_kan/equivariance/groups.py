# modified from https://github.com/mfinzi/equivariant-MLP/blob/master/emlp/groups.py

import numpy as np
from scipy.linalg import expm
from scipy.linalg import block_diag
from utils import consistent_hash


def Kron(Ms):
    result = np.eye(1)
    for M in Ms:
        result = np.kron(result, M)
    return result


def Kronsum(Ms):
    result = np.zeros(1)
    for M in Ms:
        result = np.kron(result, np.eye(M.shape[0])) + np.kron(np.eye(result.shape[0]), M)
    return result


def DirectSum(Ms, multiplicities=None):
    multiplicities = [1 for M in Ms] if multiplicities is None else multiplicities
    Ms_all = [M for M, c in zip(Ms, multiplicities) for _ in range(c)]
    return block_diag(*Ms_all)


class Named(type):
    def __str__(self):
        return self.__name__

    def __repr__(self):
        return self.__name__


def rel_err(A, B):
    return np.mean(np.abs(A - B)) / (np.mean(np.abs(A)) + np.mean(np.abs(B)) + 1e-6)


class Group(object, metaclass=Named):
    lie_algebra = NotImplemented
    discrete_generators = NotImplemented
    z_scale = None
    is_orthogonal = None
    is_permutation = None
    d = NotImplemented

    def __init__(self, *args, **kwargs):
        if self.d is NotImplemented:
            if self.lie_algebra is not NotImplemented and len(self.lie_algebra):
                self.d = self.lie_algebra[0].shape[-1]
            if self.discrete_generators is not NotImplemented and len(self.discrete_generators):
                self.d = self.discrete_generators[0].shape[-1]

        if self.lie_algebra is NotImplemented:
            self.lie_algebra = np.zeros((0, self.d, self.d))
        if self.discrete_generators is NotImplemented:
            self.discrete_generators = np.zeros((0, self.d, self.d))

        self.args = args

        if self.is_permutation:
            self.is_orthogonal = True
        if self.is_orthogonal is None:
            self.is_orthogonal = True
            if len(self.lie_algebra) != 0:
                A_dense = np.stack([Ai @ np.eye(self.d) for Ai in self.lie_algebra])
                self.is_orthogonal &= rel_err(-A_dense.transpose((0, 2, 1)), A_dense) < 1e-6
            if len(self.discrete_generators) != 0:
                h_dense = np.stack([hi @ np.eye(self.d) for hi in self.discrete_generators])
                self.is_orthogonal &= rel_err(h_dense.transpose((0, 2, 1)) @ h_dense, np.eye(self.d)) < 1e-6

        if self.is_orthogonal and (self.is_permutation is None):
            self.is_permutation = True
            self.is_permutation &= len(self.lie_algebra) == 0
            if len(self.discrete_generators) != 0:
                h_dense = np.stack([hi @ np.eye(self.d) for hi in self.discrete_generators])
                self.is_permutation &= ((h_dense == 1).astype(int).sum(-1) == 1).all()

    def exp(self, A):
        return expm(A)

    def num_constraints(self):
        return len(self.lie_algebra) + len(self.discrete_generators)

    def sample(self):
        return self.samples(1)[0]

    def samples(self, N):
        A_dense = np.stack([Ai @ np.eye(self.d) for Ai in self.lie_algebra]) if len(self.lie_algebra) else np.zeros((0, self.d, self.d))
        h_dense = np.stack([hi @ np.eye(self.d) for hi in self.discrete_generators]) if len(self.discrete_generators) else np.zeros((0, self.d, self.d))
        z = np.random.randn(N, A_dense.shape[0])
        if self.z_scale is not None:
            z *= self.z_scale
        k = np.random.randint(-5, 5, size=(N, h_dense.shape[0], 3))
        numpy_seed = np.random.randint(100)
        return noise2samples(z, k, A_dense, h_dense, numpy_seed)

    def check_valid_group_elems(self, g):
        return True

    def __str__(self):
        return repr(self)

    def __repr__(self):
        outstr = f"{self.__class__}"
        if self.args:
            outstr += "(" + "".join(repr(arg) for arg in self.args) + ")"
        return outstr

    def __eq__(self, G2):
        return repr(self) == repr(G2)

    def __hash__(self):
        return consistent_hash(repr(self))

    def __lt__(self, other):
        return consistent_hash(self) < consistent_hash(other)

    def __mul__(self, other):
        return DirectProduct(self, other)


def matrix_power_simple(M, n):
    if n < 0:
        return np.linalg.matrix_power(np.linalg.inv(M), -n)
    else:
        return np.linalg.matrix_power(M, n)


def noise2sample(z, ks, lie_algebra, discrete_generators, seed=0):
    g = np.eye(lie_algebra.shape[-1])
    if lie_algebra.shape[0]:
        A = (z[:, None, None] * lie_algebra).sum(0)
        g = g @ expm(A)
    np.random.seed(seed)
    M, K = ks.shape
    if M == 0:
        return g
    for k in range(K):
        for i in np.random.permutation(M):
            g = g @ matrix_power_simple(discrete_generators[i], ks[i, k])
    return g


def noise2samples(zs, ks, lie_algebra, discrete_generators, seed=0):
    samples = []
    for i in range(zs.shape[0]):
        sample = noise2sample(zs[i], ks[i], lie_algebra, discrete_generators, seed)
        samples.append(sample)
    return np.array(samples)


class Trivial(Group):
    def __init__(self, n):
        self.d = n
        super().__init__(n)


class SO(Group):
    def __init__(self, n):
        self.lie_algebra = np.zeros(((n * (n - 1)) // 2, n, n))
        k = 0
        for i in range(n):
            for j in range(i):
                self.lie_algebra[k, i, j] = 1
                self.lie_algebra[k, j, i] = -1
                k += 1
        super().__init__(n)


class O(SO):
    def __init__(self, n):
        self.discrete_generators = np.eye(n)[None]
        self.discrete_generators[0, 0, 0] = -1
        super().__init__(n)


class C(Group):
    def __init__(self, k):
        theta = 2 * np.pi / k
        self.discrete_generators = np.zeros((1, 2, 2))
        self.discrete_generators[0, :, :] = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        super().__init__(k)


class D(C):
    def __init__(self, k):
        super().__init__(k)
        self.discrete_generators = np.concatenate((self.discrete_generators, np.array([[[-1, 0], [0, 1]]])))


class Scaling(Group):
    def __init__(self, n):
        self.lie_algebra = np.eye(n)[None]
        super().__init__(n)


class Parity(Group):
    discrete_generators = -np.eye(4)[None]
    discrete_generators[0, 0, 0] = 1


class TimeReversal(Group):
    discrete_generators = np.eye(4)[None]
    discrete_generators[0, 0, 0] = -1


class SO13p(Group):
    lie_algebra = np.zeros((6, 4, 4))
    lie_algebra[3:, 1:, 1:] = SO(3).lie_algebra
    for i in range(3):
        lie_algebra[i, 1 + i, 0] = lie_algebra[i, 0, 1 + i] = 1.0

    z_scale = np.array([0.3, 0.3, 0.3, 1, 1, 1])


class SO13(SO13p):
    discrete_generators = -np.eye(4)[None]


class O13(SO13p):
    discrete_generators = np.eye(4)[None] + np.zeros((2, 1, 1))
    discrete_generators[0] *= -1
    discrete_generators[1, 0, 0] = -1


class Lorentz(O13):
    pass


class SO11p(Group):
    lie_algebra = np.array([[0.0, 1.0], [1.0, 0.0]])[None]


class O11(SO11p):
    discrete_generators = np.eye(2)[None] + np.zeros((2, 1, 1))
    discrete_generators[0] *= -1
    discrete_generators[1, 0, 0] = -1


class Sp(Group):
    def __init__(self, m):
        self.lie_algebra = np.zeros((m * (2 * m + 1), 2 * m, 2 * m))
        k = 0
        for i in range(m):
            for j in range(m):
                self.lie_algebra[k, i, j] = 1
                self.lie_algebra[k, m + j, m + i] = -1
                k += 1
        for i in range(m):
            for j in range(i + 1):
                self.lie_algebra[k, m + i, j] = 1
                self.lie_algebra[k, m + j, i] = 1
                k += 1
                self.lie_algebra[k, i, m + j] = 1
                self.lie_algebra[k, j, m + i] = 1
                k += 1
        super().__init__(m)


class Z(Group):
    def __init__(self, n):
        self.discrete_generators = np.roll(np.eye(n), 1, axis=0)[None]
        super().__init__(n)


class S(Group):
    def __init__(self, n):
        perms = np.arange(n)[None] + np.zeros((n - 1, 1)).astype(int)
        perms[:, 0] = np.arange(1, n)
        perms[np.arange(n - 1), np.arange(1, n)[None]] = 0
        self.discrete_generators = np.array([np.eye(n)[perm] for perm in perms])
        super().__init__(n)


class SL(Group):
    def __init__(self, n):
        self.lie_algebra = np.zeros((n * n - 1, n, n))
        k = 0
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                self.lie_algebra[k, i, j] = 1
                k += 1
        for l in range(n - 1):
            self.lie_algebra[k, l, l] = 1
            self.lie_algebra[k, -1, -1] = -1
            k += 1
        super().__init__(n)


class GL(Group):
    def __init__(self, n):
        self.lie_algebra = np.zeros((n * n, n, n))
        k = 0
        for i in range(n):
            for j in range(n):
                self.lie_algebra[k, i, j] = 1
                k += 1
        super().__init__(n)


class U(Group):
    def __init__(self, n):
        lie_algebra_real = np.zeros((n**2, n, n))
        lie_algebra_imag = np.zeros((n**2, n, n))
        k = 0
        for i in range(n):
            for j in range(i):
                lie_algebra_real[k, i, j] = 1
                lie_algebra_real[k, j, i] = -1
                k += 1
                lie_algebra_imag[k, i, j] = 1
                lie_algebra_imag[k, j, i] = 1
                k += 1
        for i in range(n):
            lie_algebra_imag[k, i, i] = 1
            k += 1
        self.lie_algebra = lie_algebra_real + lie_algebra_imag * 1j
        super().__init__(n)


class SU(Group):
    def __init__(self, n):
        if n == 1:
            return Trivial(1)
        lie_algebra_real = np.zeros((n**2 - 1, n, n))
        lie_algebra_imag = np.zeros((n**2 - 1, n, n))
        k = 0
        for i in range(n):
            for j in range(i):
                lie_algebra_real[k, i, j] = 1
                lie_algebra_real[k, j, i] = -1
                k += 1
                lie_algebra_imag[k, i, j] = 1
                lie_algebra_imag[k, j, i] = 1
                k += 1
        for i in range(n - 1):
            lie_algebra_imag[k, i, i] = 1
            for j in range(n):
                if i == j:
                    continue
                lie_algebra_imag[k, j, j] = -1 / (n - 1)
            k += 1
        self.lie_algebra = lie_algebra_real + lie_algebra_imag * 1j
        super().__init__(n)


class Cube(Group):
    def __init__(self):
        Fperm = np.array([4, 1, 0, 3, 5, 2])
        Lperm = np.array([3, 0, 2, 5, 4, 1])
        self.discrete_generators = np.array([np.eye(6)[perm] for perm in [Fperm, Lperm]])
        super().__init__()


def pad(permutation):
    assert len(permutation) == 48
    padded = np.zeros((6, 9)).astype(permutation.dtype)
    padded[:, :4] = permutation.reshape(6, 8)[:, :4]
    padded[:, 5:] = permutation.reshape(6, 8)[:, 4:]
    return padded


def unpad(padded_perm):
    return np.concatenate([padded_perm[:, :4], padded_perm[:, 5:]], -1).reshape(-1)


class RubiksCube(Group):
    def __init__(self):
        order = np.arange(48)
        order_padded = pad(order)
        order_padded[0, :] = np.rot90(order_padded[0].reshape(3, 3), 1).reshape(9)
        FRBL = np.array([1, 2, 3, 4])
        order_padded[FRBL, :3] = order_padded[np.roll(FRBL, 1), :3]
        Uperm = unpad(order_padded)
        RotFront = pad(np.arange(48))
        URDL = np.array([0, 2, 5, 4])
        RotFront[URDL, :] = RotFront[np.roll(URDL, 1), :]
        RotFront = unpad(RotFront)
        RotBack = np.argsort(RotFront)
        RotLeft = pad(np.arange(48))
        UFDB = np.array([0, 1, 5, 3])
        RotLeft[UFDB, :] = RotLeft[np.roll(UFDB, 1), :]
        RotLeft = unpad(RotLeft)
        RotRight = np.argsort(RotLeft)

        Fperm = RotRight[Uperm[RotLeft]]
        Rperm = RotBack[Uperm[RotFront]]
        Bperm = RotLeft[Uperm[RotRight]]
        Lperm = RotFront[Uperm[RotBack]]
        Dperm = RotRight[RotRight[Uperm[RotLeft[RotLeft]]]]
        self.discrete_generators = np.array([np.eye(48)[perm] for perm in [Uperm, Fperm, Rperm, Bperm, Lperm, Dperm]])
        super().__init__()


def Rot90(n, k):
    R = np.zeros((n * n, n * n))
    for i in range(n * n):
        V = np.zeros(n * n)
        V[i] = 1
        RV = np.rot90(V.reshape((n, n)), k).reshape(V.shape)
        R[:, i] = RV
    return R


class ZksZnxZn(Group):
    def __init__(self, k, n):
        Zn = Z(n)
        Zk = Z(k)
        nshift = Zn.discrete_generators[0]
        kshift = Zk.discrete_generators[0]
        In = np.eye(n)
        Ik = np.eye(k)
        assert k in [2, 4]
        self.discrete_generators = []
        self.discrete_generators.append(np.kron(np.kron(Ik, nshift), In))
        self.discrete_generators.append(np.kron(np.kron(Ik, In), nshift))
        self.discrete_generators.append(np.kron(kshift, Rot90(n, 4 // k)))
        self.discrete_generators = np.array(self.discrete_generators)
        super().__init__(k, n)


class Embed(Group):
    def __init__(self, G, d, slice):
        self.lie_algebra = np.zeros((G.lie_algebra.shape[0], d, d))
        self.discrete_generators = np.zeros((G.discrete_generators.shape[0], d, d))
        self.discrete_generators += np.eye(d)
        self.lie_algebra[:, slice, slice] = G.lie_algebra
        self.discrete_generators[:, slice, slice] = G.discrete_generators
        self.name = f"{G}_R{d}"
        super().__init__()

    def __repr__(self):
        return self.name


def SO2eR3():
    return Embed(SO(2), 3, slice(2))


def O2eR3():
    return Embed(O(2), 3, slice(2))


def DkeR3(k):
    return Embed(D(k), 3, slice(2))


class DirectProduct(Group):
    def __init__(self, G1, G2):
        I1, I2 = np.eye(G1.d), np.eye(G2.d)
        self.lie_algebra = np.array([Kronsum([A1, 0 * I2]) for A1 in G1.lie_algebra] + [Kronsum([0 * I1, A2]) for A2 in G2.lie_algebra])
        self.discrete_generators = np.array([np.kron(M1, I2) for M1 in G1.discrete_generators] + [np.kron(I1, M2) for M2 in G2.discrete_generators])
        self.names = (repr(G1), repr(G2))
        super().__init__()

    def __repr__(self):
        return f"{self.names[0]}x{self.names[1]}"


class WreathProduct(Group):
    def __init__(self, G1, G2):
        raise NotImplementedError


class SemiDirectProduct(Group):
    def __init__(self, G1, G2, phi):
        raise NotImplementedError
