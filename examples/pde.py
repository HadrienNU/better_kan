import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import autograd
from tqdm import tqdm
from better_kan import KAN, build_splines_layers, plot

dim = 2
np_i = 21  # number of interior points (along each dimension)
np_b = 21  # number of boundary points (along each dimension)
ranges = [-1, 1]


model = KAN(build_splines_layers([2, 2, 1], grid_size=5, grid_alpha=1.0, scale_basis=0.25))


def batch_jacobian(func, x, create_graph=False):
    # x in shape (Batch, Length)
    def _func_sum(x):
        return func(x).sum(dim=0)

    return autograd.functional.jacobian(_func_sum, x, create_graph=create_graph).permute(1, 0, 2)


# define solution
sol_fun = lambda x: torch.sin(torch.pi * x[:, [0]]) * torch.sin(torch.pi * x[:, [1]])
source_fun = lambda x: -2 * torch.pi**2 * torch.sin(torch.pi * x[:, [0]]) * torch.sin(torch.pi * x[:, [1]])

# interior
sampling_mode = "random"  # 'radnom' or 'mesh'

x_mesh = torch.linspace(ranges[0], ranges[1], steps=np_i)
y_mesh = torch.linspace(ranges[0], ranges[1], steps=np_i)
X, Y = torch.meshgrid(x_mesh, y_mesh, indexing="ij")
if sampling_mode == "mesh":
    # mesh
    x_i = torch.stack([X.reshape(-1), Y.reshape(-1)]).permute(1, 0)
else:
    # random
    x_i = torch.rand((np_i**2, 2)) * 2 - 1

# boundary, 4 sides
helper = lambda X, Y: torch.stack([X.reshape(-1), Y.reshape(-1)]).permute(1, 0)
xb1 = helper(X[0], Y[0])
xb2 = helper(X[-1], Y[0])
xb3 = helper(X[:, 0], Y[:, 0])
xb4 = helper(X[:, 0], Y[:, -1])
x_b = torch.cat([xb1, xb2, xb3, xb4], dim=0)

steps = 60
alpha = 0.1
log = 1


def train():
    optimizer = torch.optim.LBFGS(model.parameters(), lr=1, history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32)

    pbar = tqdm(range(steps), desc="description")

    for n in pbar:

        def closure():
            global pde_loss, bc_loss
            optimizer.zero_grad()
            # interior loss
            sol = sol_fun(x_i)
            sol_D1_fun = lambda x: batch_jacobian(model, x, create_graph=True)[:, 0, :]
            sol_D1 = sol_D1_fun(x_i)
            sol_D2 = batch_jacobian(sol_D1_fun, x_i, create_graph=True)[:, :, :]
            lap = torch.sum(torch.diagonal(sol_D2, dim1=1, dim2=2), dim=1, keepdim=True)
            source = source_fun(x_i)
            pde_loss = torch.mean((lap - source) ** 2)

            # boundary loss
            bc_true = sol_fun(x_b)
            bc_pred = model(x_b)
            bc_loss = torch.mean((bc_pred - bc_true) ** 2)

            loss = alpha * pde_loss + bc_loss
            loss.backward()
            return loss

        update_grid = n % 5 == 0 and n < 50

        optimizer.step(closure)
        sol = sol_fun(x_i)
        loss = alpha * pde_loss + bc_loss
        if update_grid:
            model.update_grid()
        l2 = torch.mean((model(x_i) - sol) ** 2)

        if n % log == 0:
            pbar.set_description("pde loss: %.2e | bc loss: %.2e | l2: %.2e " % (pde_loss.cpu().detach().numpy(), bc_loss.cpu().detach().numpy(), l2.detach().numpy()))


train()


plot(model, beta=10)

X_plot = x_mesh.cpu().detach().numpy()
Y_plot = y_mesh.cpu().detach().numpy()

exact_sol = np.sin(np.pi * X.cpu().detach().numpy()) * np.sin(np.pi * Y.cpu().detach().numpy())

x_i = torch.stack([X.reshape(-1), Y.reshape(-1)]).permute(1, 0)
res_kan = model(x_i).cpu().detach().numpy()[:, 0].reshape(X_plot.shape[0], Y_plot.shape[0])

fig, axs = plt.subplots(1, 3)

axs[0].set_title("Exact Solution")
h = axs[0].contourf(X_plot, Y_plot, exact_sol)
plt.colorbar(h, ax=axs[0])

axs[1].set_title("KAN result")
h = axs[1].contourf(X_plot, Y_plot, res_kan)
plt.colorbar(h, ax=axs[1])

axs[2].set_title("Difference")
h = axs[2].contourf(X_plot, Y_plot, np.abs(res_kan - exact_sol))
plt.colorbar(h, ax=axs[2])
plt.show()
