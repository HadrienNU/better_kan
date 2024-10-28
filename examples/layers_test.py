import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from better_kan import RBFKANLayer, SplinesKANLayer, ChebyshevKANLayer, ReLUKANLayer


x = torch.linspace(-1, 2, 150).unsqueeze(1)
x_plot = x.squeeze().detach().numpy()


sp_layer = ReLUKANLayer(1, 4, grid_size=5, scale_basis=1.0, scale_base=0.0, grid_alpha=1.0, grid_range=[0, 1])
sp_layer_bis = RBFKANLayer(1, 4, grid_size=50, scale_basis=1.0, scale_base=0.0, grid_alpha=0.5)
sp_layer_bis.set_from_another_layer(sp_layer)


# print(sp_layer.trigger_grid_update(x))
#
# res_sp = sp_layer(x).squeeze().detach().numpy()
# plt.plot(x_plot, res_sp)
# # plt.gca().set_prop_cycle(None)
# print(sp_layer.grid)
#
#
print(sp_layer_bis.grid)

basis_values = sp_layer.basis(x)
print(basis_values.shape)
plt.plot(x_plot, basis_values.squeeze().detach().numpy(), "-")
# unreduced_basis_output = torch.sum(basis_values.unsqueeze(1) * sp_layer.scaled_weights.unsqueeze(0), dim=2)  # (batch, out, in)
# unreduced_basis_output = unreduced_basis_output.transpose(1, 2)  # (batch, in, out)

# plt.plot(x_plot, unreduced_basis_output.squeeze().detach().numpy(), "x")


# # sp_layer.update_grid(x, margin=0.0)
#
# # A = sp_layer.basis(x).permute(2, 0, 1)  # (in_features, batch_size, n_basis_function)
# # B = unreduced_basis_output.transpose(0, 1)  # (in_features, batch_size, out_features)
# # res = torch.linalg.lstsq(A, B, driver="gelsd")  # (in_features, n_basis_function, out_features)
# # print(res)
# # result = res.solution.permute(2, 1, 0)  # (out_features, n_basis_function, in_features)
# # sp_layer.scaled_weights = result.contiguous()
#
# plt.gca().set_prop_cycle(None)
# res_sp = sp_layer_bis(x).squeeze().detach().numpy()
# plt.plot(x_plot, res_sp, "--")
#
#
plt.show()
