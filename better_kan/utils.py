import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize
import numpy as np
from tqdm import tqdm
import copy


def interpolate_moments_pytorch(mu_old, nu_old, new_shape):
    """
    Performs a linear interpolation to assign values to the first and second-order moments of gradients of the c_i basis functions coefficients after grid extension.

    Args:
        mu_old (torch.Tensor):
            First-order moments before extension.
        nu_old (torch.Tensor):
            Second-order moments before extension.
        new_shape (tuple):
            The new desired shape

    Returns:
        mu_new, nu_new (tuple):
            First- and second-order moments after extension, shape new_shape.
    """
    old_shape = mu_old.shape
    # Return original moments if the dimension hasn't changed, avoiding unnecessary computation.
    if old_shape == new_shape:
        return mu_old, nu_old
    # Detect which dimension has changed
    changed_dims = [i for i, (o, n) in enumerate(zip(old_shape, new_shape)) if o != n]
    if len(changed_dims) != 1:
        raise ValueError(f"Expected exactly one dimension to change, got {changed_dims}")
    interp_dim = changed_dims[0]

    old_len = old_shape[interp_dim]
    new_len = new_shape[interp_dim]

    # Move the interpolation dimension to the last position
    permute_order = list(range(len(old_shape)))
    permute_order[interp_dim], permute_order[-1] = permute_order[-1], permute_order[interp_dim]

    mu_perm = mu_old.permute(*permute_order)
    nu_perm = nu_old.permute(*permute_order)

    # Flatten all leading dimensions for interpolation
    batch_size = int(np.prod(mu_perm.shape[:-1]))
    mu_2d = mu_perm.reshape(batch_size, old_len).unsqueeze(1)  # (batch_size, 1, old_len)
    nu_2d = nu_perm.reshape(batch_size, old_len).unsqueeze(1)

    # Perform 1D linear interpolation
    mu_new_3d = F.interpolate(mu_2d, size=new_len, mode="linear", align_corners=True)
    nu_new_3d = F.interpolate(nu_2d, size=new_len, mode="linear", align_corners=True)
    # Reshape back and invert permutation
    mu_new = mu_new_3d.squeeze(1).reshape(*mu_perm.shape[:-1], new_len).permute(*np.argsort(permute_order))
    nu_new = nu_new_3d.squeeze(1).reshape(*nu_perm.shape[:-1], new_len).permute(*np.argsort(permute_order))

    print("Interpolated shape:", mu_new.shape)
    return mu_new, nu_new


def transition_optimizer_state(old_params_dict, new_params_dict, old_optimizer):
    """
    Implements Algorithm 1 from the paper. It transitions the optimizer state
    from an old model configuration to a new one after grid adaptation.

    Args:
        old_params_dict (dict): A dictionary of {name: param} from the model before the update.
        new_params_dict (dict): A dictionary of {name: param} from the model after the update.
        old_optimizer (torch.optim.Optimizer): The optimizer instance before the update.

    Returns:
        dict: The new state_dict for the optimizer.
    """
    old_state_dict = old_optimizer.state_dict()

    # Create a mapping from old parameter IDs (used in state_dict) to their names
    old_idx_to_name = {i: name for i, name in enumerate(old_params_dict.keys())}
    new_name_to_idx = {name: i for i, name in enumerate(new_params_dict.keys())}

    # Copy the hyperparameter groups from the old state_dict
    new_param_groups = copy.deepcopy(old_state_dict["param_groups"])

    # Rebuild the 'state' dictionary for the new model's parameters
    new_state = {}

    # Iterate through the state of the old parameters, where `param_idx` is the key.
    for param_idx, state in old_state_dict["state"].items():
        # Find the name of the old parameter using its index.
        param_name = old_idx_to_name.get(param_idx)
        if not param_name:
            continue

        # Find the corresponding parameter and its index in the new model.
        new_p = new_params_dict.get(param_name)
        new_param_idx = new_name_to_idx.get(param_name)

        if new_p is None or new_param_idx is None:
            continue

        # Make a deep copy of the state to avoid modifying the original.
        new_param_state = copy.deepcopy(state)
        if "exp_avg" in new_param_state and "exp_avg_sq" in new_param_state and "weights" in param_name:  # Only interpolate for weights
            mu_old = state["exp_avg"]
            nu_old = state["exp_avg_sq"]
            print(param_idx, param_name, new_p.shape)
            print(old_params_dict.get(param_name).shape)
            # Interpolate to the shape of the new parameter

            mu_new, nu_new = interpolate_moments_pytorch(mu_old, nu_old, new_p.shape)

            # Update the state with the interpolated moments
            new_param_state["exp_avg"] = mu_new
            new_param_state["exp_avg_sq"] = nu_new

        # The key in the new state dictionary must be the ID of the new parameter
        new_state[new_param_idx] = new_param_state

    # Construct the final state_dict
    new_state_dict = {
        "state": new_state,
        "param_groups": new_param_groups,
    }

    return new_state_dict


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
    grid_update_freq=50,
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

    """

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
            kan.update_grid(dataset["train_input"][train_id])

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


def plot(kan, beta=3, norm_alpha=False, scale=1.0, tick=False, in_vars=None, out_vars=None, title=None, ax=None):
    """
    plot KAN. Before plot, kan(x) should be run on a typical input to collect statistics on activations functions.

    Args:
    -----
        beta : float
            positive number. control the transparency of each activation. transparency = tanh(beta*l1).
        norm_alpha: bool, default False
            If True, normalize transparency within layer such that higher alpha is set to 1
        scale : float
            control the size of the insert plot of the activation functions
        in_vars: None or list of str
            the name(s) of input variables
        out_vars: None or list of str
            the name(s) of output variables
        title: None or str
            title
        tick: bool, default False
            draw ticks on insert plot
        ax: Axes, default None
            If None, create a new figure

    Returns:
    --------
        Figure

    Example
    -------
    >>> # see more interactive examples in demos
    >>> plot(model)
    """

    import matplotlib.pyplot as plt
    import networkx as nx

    depth = len(kan.layers)

    # Add nodes to graph, choose position at the same time
    pos = {}
    G = nx.Graph()
    for n, l in enumerate(kan.width):
        for m in range(l):
            G.add_node((n, m))
            pos[(n, m)] = [(1 / (2 * l) + m / l) * (1 - 0.1 * (n % 2)), n]

    # Add network edges
    for la in range(depth):
        for i in range(kan.width[la]):
            for j in range(kan.width[la + 1]):
                G.add_edge((la, i), (la + 1, j))

    if ax is None:
        _, ax = plt.subplots()

    nx.draw_networkx_nodes(G, pos, ax=ax)
    # Plot in and out vars if available
    offset = 0  # 0.15  # Find offset as size of a node ??
    mask_in = kan.layers[0].mask.cpu().detach().numpy()
    if in_vars is not None:
        name_attrs = {(0, m): in_vars[m] for m in range(kan.width[0])}
        nx.draw_networkx_labels(G, {n: (x, y - offset) for n, (x, y) in pos.items()}, labels=name_attrs, font_color="red", ax=ax)
    elif mask_in.shape[0] != mask_in.shape[1]:  # If there is some permutation invariants inputs, lets labels them appropeially
        groups = np.argmax(mask_in, axis=0)  # Group to which belong each input
        name_attrs = {(0, m): groups[m] for m in range(kan.width[0])}
        nx.draw_networkx_labels(G, {n: (x, y - offset) for n, (x, y) in pos.items()}, labels=name_attrs, font_color="yellow", ax=ax)
    if out_vars is not None:
        name_attrs = {}
        for m in range(kan.width[-1]):
            name_attrs[(len(kan.width) - 1, m)] = out_vars[m]
        nx.draw_networkx_labels(G, {n: (x, y + offset) for n, (x, y) in pos.items()}, labels=name_attrs, font_color="red", ax=ax)

    def score2alpha(score):
        return np.tanh(beta * score)

    # Add insert plot of each activation functions
    inserts_axes = []
    act_lines = []

    for la in range(depth):
        inserts_axes.append([[None for _ in range(kan.width[la + 1])] for _ in range(kan.width[la])])
        act_lines.append([[None for _ in range(kan.width[la + 1])] for _ in range(kan.width[la])])
        if hasattr(kan.layers[la], "l1_norm"):
            alpha = score2alpha(kan.layers[la].l1_norm.cpu().detach().numpy())
            alpha = alpha / (alpha.max() if norm_alpha else 1.0)
            # Take for ranges, either the extremal of the centers or the min/max of the data
            ranges = [torch.linspace(kan.layers[la].min_vals[d], kan.layers[la].max_vals[d], 150) for d in range(kan.width[la])]
            x_in = torch.stack(ranges, dim=1)
            acts_vals = kan.layers[la].activations_eval(x_in).cpu().detach().numpy()
            x_ranges = x_in.cpu().detach().numpy()
            # Take mask into account
            mask_la = kan.layers[la].mask.cpu().detach().numpy()  # L'idée c'est d'avoir mask [i][j] = True/False pour savoir si on plot
            mask = np.zeros(mask_la.shape[1], dtype=bool)
            mask[np.argmax(mask_la, axis=1)] = True  # Only plot first graph for each group

            for i in range(kan.width[la]):
                for j in range(kan.width[la + 1]):
                    u, v = (la, i), (la + 1, j)
                    nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], alpha=alpha[j, i], ax=ax)
                    if mask[i]:
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

                        act_lines[la][i][j] = inset_ax.plot(x_ranges[:, i], acts_vals[:, j, i], "-", color="red", alpha=alpha[j, i])[0]
                        for spine in inset_ax.spines.values():
                            spine.set_alpha(alpha[j, i])
                        inset_ax.patch.set_alpha(alpha[j, i])
                        inserts_axes[la][i][j] = inset_ax
        else:
            for i in range(kan.width[la]):
                for j in range(kan.width[la + 1]):
                    u, v = (la, i), (la + 1, j)
                    nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], ax=ax)

    if title is not None:
        ax.set_title(title)
    return inserts_axes, act_lines


def update_plot(kan, inserts_axes, act_lines, beta=3, norm_alpha=False, tick=False):
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

    depth = len(kan.layers)

    def score2alpha(score):
        return np.tanh(beta * score)

    for la in range(depth):
        if hasattr(kan.layers[la], "l1_norm"):
            alpha = score2alpha(kan.layers[la].l1_norm.cpu().detach().numpy())
            alpha = alpha / (alpha.max() if norm_alpha else 1.0)
            # Take for ranges, either the extremal of the centers or the min/max of the data
            ranges = [torch.linspace(kan.layers[la].min_vals[d], kan.layers[la].max_vals[d], 150) for d in range(kan.width[la])]
            x_in = torch.stack(ranges, dim=1)
            acts_vals = kan.layers[la].activations_eval(x_in).cpu().detach().numpy()
            x_ranges = x_in.cpu().detach().numpy()
            for i in range(kan.width[la]):
                for j in range(kan.width[la + 1]):
                    inset_ax = inserts_axes[la][i][j]
                    if tick is False:
                        inset_ax.set_xticks([])
                        inset_ax.set_yticks([])
                    else:
                        inset_ax.tick_params(axis="both", which="both", length=0)  # Rendre les ticks invisibles

                        for label in inset_ax.get_xticklabels() + inset_ax.get_yticklabels():
                            label.set_alpha(alpha[j, i])
                    act_lines[la][i][j].set_xdata(x_ranges[:, i])
                    act_lines[la][i][j].set_ydata(acts_vals[:, j, i])
                    act_lines[la][i][j].set_alpha(alpha[j, i])

                    inset_ax.set_xlim(x_ranges[0, i], x_ranges[-1, i])
                    inset_ax.set_ylim(acts_vals[:, j, i].min(), acts_vals[:, j, i].max())
                    # act_lines[i][j] = inset_ax.plot(x_ranges[:, i], acts_vals[:, j, i], "-", color="red", alpha=alpha[j, i])
                    for spine in inset_ax.spines.values():
                        spine.set_alpha(alpha[j, i])
                    inset_ax.patch.set_alpha(alpha[j, i])


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


@torch.no_grad()
def assign_parameters(module, param, new_value):
    """
    Workaround to the difference of assignation for paramerized and non parametrized Parameters
    """
    if parametrize.is_parametrized(module, param):  # TODO, to check
        old_param = getattr(module, param)
        if old_param.shape == new_value.shape:
            setattr(module, param, new_value)
        else:
            plist = getattr(module.parametrizations, param)
            # We remove the parametrizations
            for P in reversed(plist):
                parametrize.remove_parametrizations(module, param, P, leave_parametrized=True)
            setattr(module, param, nn.Parameter(new_value))
            # and but them back
            for P in reversed(plist):
                parametrize.register_parametrization(module, param, P, unsafe=True)

    else:
        old_param = getattr(module, param)
        if old_param.shape == new_value.shape:
            old_param.copy_(new_value)
        else:
            setattr(module, param, nn.Parameter(new_value))


def svd_lstsq(AA, BB, tol=1e-12):
    """
    Workaround to SVD when on CUDA, to allow lstsq on rank-deficient matrices
    Waiting for PR https://github.com/pytorch/pytorch/pull/126652 to be merged
    """
    U, S, Vh = torch.linalg.svd(AA, full_matrices=False)
    Spinv = torch.zeros_like(S)

    Spinv[S > tol * max(AA.shape) * S[0]] = 1 / S[S > tol * max(AA.shape) * S[0]]
    UhBB = U.adjoint() @ BB
    if Spinv.ndim != UhBB.ndim:
        Spinv = Spinv.unsqueeze(-1)
    SpinvUhBB = Spinv * UhBB
    return Vh.adjoint() @ SpinvUhBB
