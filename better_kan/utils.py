import torch
import numpy as np
from tqdm import tqdm


def train(
    kan,
    dataset,
    opt="LBFGS",
    steps=100,
    log=1,
    lamb=1.0e-2,
    lamb_l1=1.0,
    lamb_entropy=1.0,
    update_grid=True,
    grid_update_num=10,
    loss_fn=None,
    lr=1.0,
    stop_grid_update_step=50,
    batch=-1,
    metrics=None,
    sglr_avoid=False,
):
    """
    training

    Args:
    -----
        dataset : dic
            contains dataset['train_input'], dataset['train_label'], dataset['test_input'], dataset['test_label']
        opt : str
            "LBFGS" or "Adam"
        steps : int
            training steps
        log : int
            logging frequency
        lamb : float
            overall penalty strength
        lamb_l1 : float
            l1 penalty strength
        lamb_entropy : float
            entropy penalty strength
        lamb_coef : float
            coefficient magnitude penalty strength
        lamb_coefdiff : float
            difference of nearby coefficits (smoothness) penalty strength
        update_grid : bool
            If True, update grid regularly before stop_grid_update_step
        grid_update_num : int
            the number of grid updates before stop_grid_update_step
        stop_grid_update_step : int
            no grid updates after this training step
        batch : int
            batch size, if -1 then full.
        small_mag_threshold : float
            threshold to determine large or small numbers (may want to apply larger penalty to smaller numbers)
        small_reg_factor : float
            penalty strength applied to small factors relative to large factos

    Returns:
    --------
        results : dic
            results['train_loss'], 1D array of training losses (RMSE)
            results['test_loss'], 1D array of test losses (RMSE)
            results['reg'], 1D array of regularization

    Example
    -------
    >>> # for interactive examples, please see demos
    >>> from utils import create_dataset
    >>> model = KAN(width=[2,5,1], grid=5, k=3, noise_scale=0.1, seed=0)
    >>> f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
    >>> dataset = create_dataset(f, n_var=2)
    >>> model.train(dataset, opt='LBFGS', steps=50, lamb=0.01);
    >>> model.plot()
    """

    pbar = tqdm(range(steps), desc="description", ncols=100)

    if loss_fn is None:
        loss_fn = loss_fn_eval = lambda x, y: torch.mean((x - y) ** 2)
    else:
        loss_fn = loss_fn_eval = loss_fn

    grid_update_freq = int(stop_grid_update_step / grid_update_num)

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

    if batch == -1 or batch > dataset["train_input"].shape[0]:
        batch_size = dataset["train_input"].shape[0]
        batch_size_test = dataset["test_input"].shape[0]
    else:
        batch_size = batch
        batch_size_test = batch

    global train_loss, reg_

    def closure():
        global train_loss, reg_
        optimizer.zero_grad()
        pred = kan.forward(dataset["train_input"][train_id])
        if sglr_avoid is True:  # Remove any nan
            id_ = torch.where(torch.isnan(torch.sum(pred, dim=1)) == False)[0]
            train_loss = loss_fn(pred[id_], dataset["train_label"][train_id][id_])
        else:
            train_loss = loss_fn(pred, dataset["train_label"][train_id])
        reg_ = kan.regularization_loss()
        objective = train_loss + lamb * reg_
        objective.backward()
        return objective

    for _ in pbar:

        train_id = np.random.choice(dataset["train_input"].shape[0], batch_size, replace=False)
        test_id = np.random.choice(dataset["test_input"].shape[0], batch_size_test, replace=False)

        if _ % grid_update_freq == 0 and _ < stop_grid_update_step and update_grid:
            kan.forward(dataset["train_input"][train_id], update_grid=True)

        if opt == "LBFGS":
            optimizer.step(closure)

        if opt == "Adam":
            pred = kan.forward(dataset["train_input"][train_id])
            if sglr_avoid is True:
                id_ = torch.where(torch.isnan(torch.sum(pred, dim=1)) == False)[0]
                train_loss = loss_fn(pred[id_], dataset["train_label"][train_id][id_])
            else:
                train_loss = loss_fn(pred, dataset["train_label"][train_id])
            reg_ = kan.regularization_loss(lamb_l1, lamb_entropy)
            loss = train_loss + lamb * reg_
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        test_loss = loss_fn_eval(kan.forward(dataset["test_input"][test_id]), dataset["test_label"][test_id])

        if _ % log == 0:
            pbar.set_description("train loss: %.2e | test loss: %.2e | reg: %.2e " % (torch.sqrt(train_loss).cpu().detach().numpy(), torch.sqrt(test_loss).cpu().detach().numpy(), reg_.cpu().detach().numpy()))

        if metrics is not None:
            for i in range(len(metrics)):
                results[metrics[i].__name__].append(metrics[i]().item())

        results["train_loss"].append(torch.sqrt(train_loss).cpu().detach().numpy())
        results["test_loss"].append(torch.sqrt(test_loss).cpu().detach().numpy())
        results["reg"].append(reg_.cpu().detach().numpy())

    return results


def plot(kan, beta=3, mask=False, scale=1.0, tick=False, in_vars=None, out_vars=None, title=None, ax=None):
    """
    plot KAN. Before plot, kan(x) should be run on a typical input to collect statistics on activations functions.

    Args:
    -----
        beta : float
            positive number. control the transparency of each activation. transparency = tanh(beta*l1).
        mask : bool
            If True, plot with mask (need to run prune() first to obtain mask). If False (by default), plot all activation functions.
        mode : bool
            "supervised" or "unsupervised". If "supervised", l1 is measured by absolution value (not subtracting mean); if "unsupervised", l1 is measured by standard deviation (subtracting mean).
        scale : float
            control the size of the insert plot of the activation functions
        in_vars: None or list of str
            the name(s) of input variables
        out_vars: None or list of str
            the name(s) of output variables
        title: None or str
            title

    Returns:
    --------
        Figure

    Example
    -------
    >>> # see more interactive examples in demos
    >>> model = KAN(width=[2,3,1], grid=3, k=3, noise_scale=1.0)
    >>> x = torch.normal(0,1,size=(100,2))
    >>> model(x) # do a forward pass to obtain model.acts
    >>> model.plot()
    """

    import matplotlib.pyplot as plt
    import networkx as nx

    depth = len(kan.width) - 1

    # Add nodes to graph, choose position at the same time
    pos = {}
    G = nx.Graph()
    for n, l in enumerate(kan.width):
        for m in range(l):
            G.add_node((n, m))
            pos[(n, m)] = [(1 / (2 * l) + m / l) * (1 - 0.1 * (n % 2)), n]

    # Add network edges
    # TODO Check value of mask to not plotting some edges
    for la in range(depth):
        for i in range(kan.width[la]):
            for j in range(kan.width[la + 1]):
                G.add_edge((la, i), (la + 1, j))

    if ax is None:
        _, ax = plt.subplots()

    nx.draw_networkx_nodes(G, pos, ax=ax)
    # Plot in and out vars if available
    offset = 0  # 0.15  # Find offset as size of a node ??
    if in_vars is not None:
        name_attrs = {}
        for m in range(kan.width[0]):
            name_attrs[(0, m)] = in_vars[m]
        nx.draw_networkx_labels(G, {n: (x, y - offset) for n, (x, y) in pos.items()}, labels=name_attrs, font_color="red", ax=ax)

    if out_vars is not None:
        name_attrs = {}
        for m in range(kan.width[-1]):
            name_attrs[(len(kan.width) - 1, m)] = out_vars[m]
        nx.draw_networkx_labels(G, {n: (x, y + offset) for n, (x, y) in pos.items()}, labels=name_attrs, font_color="red", ax=ax)

    def score2alpha(score):
        return np.tanh(beta * score)

    # Add insert plot of each activation functions
    for la in range(depth):
        if hasattr(kan.layers[la], "l1_norm"):
            alpha = score2alpha(kan.layers[la].l1_norm.cpu().detach().numpy())
            # Take for ranges, either the extremal of the centers or the min/max of the data
            ranges = [torch.linspace(kan.layers[la].min_vals[d], kan.layers[la].max_vals[d], 150) for d in range(kan.width[la])]
            x_in = torch.stack(ranges, dim=1)
            acts_vals = kan.layers[la].activations_eval(x_in).cpu().detach().numpy()
            x_ranges = x_in.cpu().detach().numpy()
            for i in range(kan.width[la]):
                for j in range(kan.width[la + 1]):
                    u, v = (la, i), (la + 1, j)
                    nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], alpha=alpha[j, i], ax=ax)

                    # Compute central position of the edge
                    x = (pos[u][0] + pos[v][0]) / 2
                    y = (pos[u][1] + pos[v][1]) / 2

                    width = scale * 0.1
                    height = scale * 0.1

                    # Créer un axe en insert
                    inset_ax = ax.inset_axes([x - 0.5 * width, y - 0.5 * height, width, height], transform=ax.transData, box_aspect=1.0)
                    if tick is False:
                        inset_ax.set_xticks([])
                        inset_ax.set_yticks([])
                    else:
                        inset_ax.tick_params(axis="both", which="both", length=0)  # Rendre les ticks invisibles

                        for label in inset_ax.get_xticklabels() + inset_ax.get_yticklabels():
                            label.set_alpha(alpha[j, i])

                    inset_ax.plot(x_ranges[:, i], acts_vals[:, j, i], "-", color="red", alpha=alpha[j, i])
                    for spine in inset_ax.spines.values():
                        spine.set_alpha(alpha[j, i])
                    inset_ax.patch.set_alpha(alpha[j, i])
        else:
            for i in range(kan.width[la]):
                for j in range(kan.width[la + 1]):
                    u, v = (la, i), (la + 1, j)
                    nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], ax=ax)

    if title is not None:
        ax.set_title(title)


def create_dataset(f, n_var=2, ranges=[-1, 1], train_num=1000, test_num=1000, normalize_input=False, normalize_label=False, seed=0):
    """
    create dataset

    Args:
    -----
        f : function
            the symbolic formula used to create the synthetic dataset
        ranges : list or np.array; shape (2,) or (n_var, 2)
            the range of input variables. Default: [-1,1].
        train_num : int
            the number of training samples. Default: 1000.
        test_num : int
            the number of test samples. Default: 1000.
        normalize_input : bool
            If True, apply normalization to inputs. Default: False.
        normalize_label : bool
            If True, apply normalization to labels. Default: False.
        device : str
            device. Default: 'cpu'.
        seed : int
            random seed. Default: 0.

    Returns:
    --------
        dataset : dic
            Train/test inputs/labels are dataset['train_input'], dataset['train_label'],
                        dataset['test_input'], dataset['test_label']

    Example
    -------
    >>> f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
    >>> dataset = create_dataset(f, n_var=2, train_num=100)
    >>> dataset['train_input'].shape
    torch.Size([100, 2])
    """

    np.random.seed(seed)
    torch.manual_seed(seed)

    if len(np.array(ranges).shape) == 1:
        ranges = np.array(ranges * n_var).reshape(n_var, 2)
    else:
        ranges = np.array(ranges)

    train_input = torch.zeros(train_num, n_var)
    test_input = torch.zeros(test_num, n_var)
    for i in range(n_var):
        train_input[:, i] = (
            torch.rand(
                train_num,
            )
            * (ranges[i, 1] - ranges[i, 0])
            + ranges[i, 0]
        )
        test_input[:, i] = (
            torch.rand(
                test_num,
            )
            * (ranges[i, 1] - ranges[i, 0])
            + ranges[i, 0]
        )

    train_label = f(train_input)
    test_label = f(test_input)

    def normalize(data, mean, std):
        return (data - mean) / std

    if normalize_input is True:
        mean_input = torch.mean(train_input, dim=0, keepdim=True)
        std_input = torch.std(train_input, dim=0, keepdim=True)
        train_input = normalize(train_input, mean_input, std_input)
        test_input = normalize(test_input, mean_input, std_input)

    if normalize_label is True:
        mean_label = torch.mean(train_label, dim=0, keepdim=True)
        std_label = torch.std(train_label, dim=0, keepdim=True)
        train_label = normalize(train_label, mean_label, std_label)
        test_label = normalize(test_label, mean_label, std_label)

    dataset = {}
    dataset["train_input"] = train_input
    dataset["test_input"] = test_input

    dataset["train_label"] = train_label
    dataset["test_label"] = test_label

    return dataset
