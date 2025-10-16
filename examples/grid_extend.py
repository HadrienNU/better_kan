import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from better_kan import KAN, build_splines_layers, build_rbf_layers, create_dataset, plot
from better_kan.utils import transition_optimizer_state


def train(
    kan,
    dataset,
    opt="LBFGS",
    steps=100,
    log=1,
    lamb=1.0e-2,
    lamb_l1=1.0,
    lamb_entropy=1.0,
    grid_upds={},
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
            # print([(k,p.shape) for k,p in kan.named_parameters()])
            kan.update_grid(dataset["train_input"], grid_size=grid_upds[epoch])
            # Reset optimisers
            print([(k,p.shape) for k,p in kan.named_parameters()])
            if opt == "Adam":
                new_params = dict(kan.named_parameters())
                new_optimizer_state = transition_optimizer_state(old_params, new_params, optimizer)
                # print("New",new_optimizer_state)
                optimizer = torch.optim.Adam(kan.parameters(), lr=optimizer.param_groups[0]['lr'])
                optimizer.load_state_dict(new_optimizer_state)
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
            pbar.set_description("train loss: %.2e | test loss: %.2e | reg: %.2e " % (torch.sqrt(train_loss).cpu().detach().numpy(), torch.sqrt(test_loss).cpu().detach().numpy(), reg_.cpu().detach().numpy()))

        if metrics is not None:
            for i in range(len(metrics)):
                results[metrics[i].__name__].append(metrics[i]().item())

        results["train_loss"].append(torch.sqrt(train_loss).cpu().detach().numpy())
        results["test_loss"].append(torch.sqrt(test_loss).cpu().detach().numpy())
        results["reg"].append(reg_.cpu().detach().numpy())

    return results


torch.manual_seed(7)

model = KAN(build_splines_layers([4, 10, 1], grid_size=3, fast_version=True))


f = lambda x: torch.exp(0.5*torch.sin((torch.pi * x[:, [0]]**2) + (torch.pi * x[:, [1]]**2)) + 0.5*torch.sin((torch.pi * x[:, [2]]**2) + (torch.pi * x[:, [3]]**2)))
dataset = create_dataset(f, n_var=4, train_num=3200, test_num=800)


# train_dat=np.load("/home/vroylan241/Projets/Libraries/ExternalsLibraries/jaxKAN/docs/tutorials/train.dat.npy")
# test_dat=np.load("/home/vroylan241/Projets/Libraries/ExternalsLibraries/jaxKAN/docs/tutorials/test.dat.npy")
# print(train_dat.shape, test_dat.shape)
# dataset={}
# dataset["train_input"] = torch.from_numpy(train_dat[:,:-1])
# dataset["train_label"] = torch.from_numpy(train_dat[:,-1:])
# dataset["test_input"] = torch.from_numpy(test_dat[:,:-1])
# dataset["test_label"] = torch.from_numpy(test_dat[:,-1:])

# print(dataset["train_input"].shape, dataset["train_label"].shape)

# model.update_grid(dataset["train_input"])
# plot(model, title="KAN_initialisation", tick=False)
grid_upds = {200 : 6} #, 300 : 10, 450 : 24}

results = train(model, dataset, opt="Adam", steps=600, grid_upds=grid_upds, lamb=0.05, lr=0.06)

plt.figure()

plt.plot(results["train_loss"])
plt.plot(results["test_loss"])
plt.plot(results["reg"])
plt.legend(["train", "test", "reg"])
plt.ylabel("RMSE")
plt.xlabel("step")
plt.yscale("log")


# plot(model, title="KAN_after training", tick=False)

plt.show()
