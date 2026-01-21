from better_kan.equivariance import S, V, Trivial, equivariant_permutations_inputs
from better_kan.equivariance.representation import SumRep
from better_kan.equivariance.utils import draw_matrix_parametrizations
from better_kan.equivariance.parametrizations import EquivariantMatrix, EquivariantVector
import numpy as np
import matplotlib.pyplot as plt

rep_perm, perm = equivariant_permutations_inputs(["a", "a", "a", "a", "a", "a", "a"])
# print(perm)

print(rep_perm.equivariant_basis().shape)
in_vars = 4

G = S(in_vars)  # Permutation invariance of all input in this case
rep = V(S(5)) * V(S(5))
G2 = S(in_vars)  # Permutation invariance of all input in this case
rep2 = V(G2) + V(S(3))

rep_all = V(S(7))

print(rep.equivariant_basis().shape)
print(type(rep))
print(len(G.discrete_generators))

reptot = rep_perm * rep_perm.T
print(type(reptot))
# print(reptot.perm)
print(reptot.reps)
print(reptot.equivariant_basis().shape)

# draw_matrix_parametrizations(EquivariantVector(reptot), np.zeros((reptot.size())))
draw_matrix_parametrizations(EquivariantMatrix(rep, rep), np.zeros((rep.size(), rep.size())))

# draw_matrix_parametrizations(EquivariantMatrix(rep_perm, rep_perm), np.zeros((rep_all.size(), rep_all.size())))
plt.show()
