"""
In this example, we investigate the loss increase after grid update

"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from better_kan import build_KAN, create_dataset, plot, train
from better_kan.functions import Splines
from better_kan.utils import transition_optimizer_state

fig, axs = plt.subplots(nrows=4, ncols=4)  # 7*2+2 or 2*5+5 activations functions to plot


def compute_act(model, x_test, layers_range=None):
    ranges_all = []
    act_vals_all = []
    names = []
    x_layers = []
    for la in range(2):
        # Take for ranges, either the extremal of the centers or the min/max of the data
        # Get range from test_data
        if layers_range is None:
            min_vals = torch.min(x_test, dim=0).values
            max_vals = torch.max(x_test, dim=0).values
            ranges = [torch.linspace(min_vals[d], max_vals[d], 150) for d in range(model.width[la])]
            x_in = torch.stack(ranges, dim=1)
        else:
            x_in = layers_range[la]
        x_layers.append(x_in)
        acts_vals = model.layers[la].activations_eval(x_in).cpu().detach().numpy()
        x_ranges = x_in.cpu().detach().numpy()
        for i in range(model.width[la]):
            for j in range(model.width[la + 1]):
                ranges_all.append(x_ranges[:, i])
                act_vals_all.append(acts_vals[:, j, i])
                names.append(f"Layer {la}  ({i},{j})")
        if layers_range is None:
            x_test = model.layers[la].forward(x_test)
    return ranges_all, act_vals_all, x_layers, names


def train_loc(
    kan,
    dataset,
    opt="LBFGS",
    steps=100,
    log=1,
    lamb=1.0e-2,
    lamb_l1=1.0,
    lamb_entropy=1.0,
    grid_upds={},
    use_data_for_update=True,
    loss_fn=None,
    lr=1.0,
    batch=-1,
    metrics=None,
    sglr_avoid=False,
):

    pbar = tqdm(range(steps), desc="description")

    if loss_fn is None:
        loss_fn = loss_fn_eval = lambda x, y: torch.mean((x - y) ** 2)
    else:
        loss_fn = loss_fn_eval = loss_fn

    if opt == "Adam":
        optimizer = torch.optim.Adam(kan.parameters(), lr=lr)
    elif opt == "LBFGS":
        optimizer = torch.optim.LBFGS(kan.parameters(), lr=lr, history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32)
    results = {}
    results["train_loss"] = []
    results["test_loss"] = []
    results["reg"] = []
    if metrics is not None:
        for i in range(len(metrics)):
            results[metrics[i].__name__] = []

    global train_loss, reg_

    def closure():
        global train_loss, reg_
        optimizer.zero_grad()
        pred = kan.forward(dataset["train_input"])
        train_loss = loss_fn(pred, dataset["train_label"])
        reg_ = kan.regularization_loss()
        objective = train_loss + lamb * reg_
        objective.backward()
        return objective

    for epoch in pbar:

        if epoch in grid_upds.keys():
            old_params = dict(kan.named_parameters())
            print(sum(p.numel() for p in model.parameters() if p.requires_grad), [(k, p.shape) for k, p in kan.named_parameters()])
            ranges, act_vals_before, x_layers, names = compute_act(kan, dataset["test_input"], layers_range=None)
            if use_data_for_update:
                kan.update_grid(dataset["train_input"], grid_size=grid_upds[epoch])
            else:
                kan.update_grid(None, grid_size=grid_upds[epoch])

            _, act_vals_after, x_layers, _ = compute_act(kan, None, layers_range=x_layers)

            for k in range(len(ranges)):
                axs[k // 4, k % 4].set_title(names[k])
                axs[k // 4, k % 4].plot(ranges[k], np.abs(act_vals_after[k] - act_vals_before[k]), "-")
                # axs[k // 4, k % 4].plot(ranges[k], act_vals_after[k], "+")
            # Reset optimisers
            # print([(k, p.shape) for k, p in kan.named_parameters()])
            if opt == "Adam":
                #     new_params = dict(kan.named_parameters())
                #     # new_optimizer_state = transition_optimizer_state(old_params, new_params, optimizer)
                #     # print("New",new_optimizer_state)
                optimizer = torch.optim.Adam(kan.parameters(), lr=optimizer.param_groups[0]["lr"])
            #     # optimizer.load_state_dict(new_optimizer_state)
            elif opt == "LBFGS":
                optimizer = torch.optim.LBFGS(kan.parameters(), lr=lr, history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32)

        if opt == "LBFGS":
            optimizer.step(closure)

        if opt == "Adam":
            pred = kan.forward(dataset["train_input"])
            train_loss = loss_fn(pred, dataset["train_label"])
            reg_ = kan.regularization_loss(lamb_l1, lamb_entropy)
            loss = train_loss + lamb * reg_
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        test_loss = loss_fn_eval(kan.forward(dataset["test_input"]), dataset["test_label"])

        if epoch % log == 0:
            pbar.set_description(
                "train loss: %.2e | test loss: %.2e | reg: %.2e " % (torch.sqrt(train_loss).cpu().detach().numpy(), torch.sqrt(test_loss).cpu().detach().numpy(), reg_.cpu().detach().numpy())
            )

        if metrics is not None:
            for i in range(len(metrics)):
                results[metrics[i].__name__].append(metrics[i]().item())

        results["train_loss"].append(torch.sqrt(train_loss).cpu().detach().numpy())
        results["test_loss"].append(torch.sqrt(test_loss).cpu().detach().numpy())
        results["reg"].append(reg_.cpu().detach().numpy())

    return results


torch.manual_seed(7)

# model = build_KAN(Splines, [7, 2, 1], grid_size=5, fast_version=True)
# f = lambda x: 1 / (1 + torch.exp(-(torch.sin(torch.pi * x[:, [0]]) + x[:, [1]] ** 2 + x[:, [2]] ** 2 + x[:, [3]] ** 2 + x[:, [4]] ** 2 - (x[:, [5]] - x[:, [6]]).abs() ** 0.5)))
# dataset = create_dataset(f, n_var=7)
# grid_upds = {250: 5, 500: 5, 750: 5, 2000: None}
# use_data_for_update = True

# Alternative dataset
f = lambda x: torch.exp(torch.sin(torch.pi * x[:, [0]]) + x[:, [1]] ** 2)
dataset = create_dataset(f, n_var=2)
model = build_KAN(Splines, [2, 5, 1], grid_size=5)
grid_upds = {250: 6, 500: 10, 750: 15, 2000: None}
use_data_for_update = True

print(dataset["train_input"].shape, dataset["train_label"].shape)

print(sum(p.numel() for p in model.parameters() if p.requires_grad), [(k, p.shape) for k, p in model.named_parameters()])

model.update_grid(dataset["train_input"])


# train_losses = []
# test_losses = []
# step = 0
# for n, g in grid_upds.items():
#     results = train(model, dataset, opt="Adam", steps=n - step, lamb=0.0, lr=1e-2)
#     if g is not None:
#         if use_data_for_update:
#             model.update_grid(dataset["train_input"], grid_size=g)
#         else:
#             model.update_grid(None, grid_size=g)
#     print(sum(p.numel() for p in model.parameters() if p.requires_grad), [(k, p.shape) for k, p in model.named_parameters()])
#     print([(k, p.shape) for k, p in model.named_buffers()])
#     step = n
#     train_losses = np.hstack((train_losses, results["train_loss"]))
#     test_losses = np.hstack((test_losses, results["test_loss"]))
# print(train_losses.shape)

results = train_loc(model, dataset, opt="Adam", steps=1000, grid_upds=grid_upds, use_data_for_update=use_data_for_update, lamb=0.00, lr=1e-2)
train_losses = results["train_loss"]
test_losses = results["test_loss"]

plt.figure()

plt.plot(train_losses)
plt.plot(test_losses)
plt.legend(["train", "test"])
plt.ylabel("RMSE")
plt.xlabel("step")
plt.yscale("log")

# for la in model.layers:
#     la.set_speed_mode(False)
# model(dataset["test_input"])
# plot(model, title="KAN_after training", tick=False, norm_alpha=True)

# new_model = model.prune(mode="auto")

# new_model(dataset["train_input"])
# plot(new_model, title="KAN after pruning", tick=False)


# fig, axs = plt.subplots(2, 2)

# fitted_value = model(dataset["test_input"])
# with torch.no_grad():
#     x_sorted = np.sort(dataset["test_label"])
#     axs[0, 0].set_title("After training")
#     axs[0, 0].plot(x_sorted, x_sorted, "-", color="red")
#     axs[0, 0].scatter(dataset["test_label"], model(dataset["test_input"]))
#     axs[0, 1].set_title("After training")
#     axs[0, 1].hexbin(dataset["test_label"], model(dataset["test_input"]), gridsize=25)

#     axs[1, 0].set_title("After pruning")
#     axs[1, 0].plot(x_sorted, x_sorted, "-", color="red")
#     axs[1, 0].scatter(dataset["test_label"], model(dataset["test_input"]))
#     axs[1, 1].set_title("After pruning")
#     axs[1, 1].hexbin(dataset["test_label"], model(dataset["test_input"]), gridsize=25)
# plt.xlabel("True Value")
# plt.ylabel("Estimated Value")


plt.show()
