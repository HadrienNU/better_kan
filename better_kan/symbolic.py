"""
Symbolic KAN layer, adapted from pyKAN
"""

import numpy as np
import torch
import torch.nn as nn
import sympy
from sklearn.linear_model import LinearRegression


# sigmoid = sympy.Function('sigmoid')
# name: (torch implementation, sympy implementation)
SYMBOLIC_LIB = {
    "x": (lambda x: x, lambda x: x),
    "x^2": (lambda x: x**2, lambda x: x**2),
    "x^3": (lambda x: x**3, lambda x: x**3),
    "x^4": (lambda x: x**4, lambda x: x**4),
    "1/x": (lambda x: 1 / x, lambda x: 1 / x),
    "1/x^2": (lambda x: 1 / x**2, lambda x: 1 / x**2),
    "1/x^3": (lambda x: 1 / x**3, lambda x: 1 / x**3),
    "1/x^4": (lambda x: 1 / x**4, lambda x: 1 / x**4),
    "sqrt": (lambda x: torch.sqrt(x), lambda x: sympy.sqrt(x)),
    "1/sqrt(x)": (lambda x: 1 / torch.sqrt(x), lambda x: 1 / sympy.sqrt(x)),
    "exp": (lambda x: torch.exp(x), lambda x: sympy.exp(x)),
    "log": (lambda x: torch.log(x), lambda x: sympy.log(x)),
    "abs": (lambda x: torch.abs(x), lambda x: sympy.Abs(x)),
    "sin": (lambda x: torch.sin(x), lambda x: sympy.sin(x)),
    "tan": (lambda x: torch.tan(x), lambda x: sympy.tan(x)),
    "tanh": (lambda x: torch.tanh(x), lambda x: sympy.tanh(x)),
    "sigmoid": (lambda x: torch.sigmoid(x), sympy.Function("sigmoid")),
    #'relu': (lambda x: torch.relu(x), relu),
    "sgn": (lambda x: torch.sign(x), lambda x: sympy.sign(x)),
    "arcsin": (lambda x: torch.arcsin(x), lambda x: sympy.arcsin(x)),
    "arctan": (lambda x: torch.arctan(x), lambda x: sympy.atan(x)),
    "arctanh": (lambda x: torch.arctanh(x), lambda x: sympy.atanh(x)),
    "0": (lambda x: x * 0, lambda x: x * 0),
    "gaussian": (lambda x: torch.exp(-(x**2)), lambda x: sympy.exp(-(x**2))),
    "cosh": (lambda x: torch.cosh(x), lambda x: sympy.cosh(x)),
    #'logcosh': (lambda x: torch.log(torch.cosh(x)), lambda x: sympy.log(sympy.cosh(x))),
    #'cosh^2': (lambda x: torch.cosh(x)**2, lambda x: sympy.cosh(x)**2),
}


def add_symbolic(name, fun):
    """
    add a symbolic function to library

    Args:
    -----
        name : str
            name of the function
        fun : fun
            torch function or lambda function

    Returns:
    --------
        None

    Example
    -------
    >>> print(SYMBOLIC_LIB['Bessel'])
    KeyError: 'Bessel'
    >>> add_symbolic('Bessel', torch.special.bessel_j0)
    >>> print(SYMBOLIC_LIB['Bessel'])
    (<built-in function special_bessel_j0>, Bessel)
    """
    exec(f"globals()['{name}'] = sympy.Function('{name}')")
    SYMBOLIC_LIB[name] = (fun, globals()[name])


def fit_params(x, y, fun, a_range=(-10, 10), b_range=(-10, 10), grid_number=101, iteration=3, verbose=True):
    """
    fit a, b, c, d such that

    .. math::
        |y-(cf(ax+b)+d)|^2

    is minimized. Both x and y are 1D array. Sweep a and b, find the best fitted model.

    Args:
    -----
        x : 1D array
            x values
        y : 1D array
            y values
        fun : function
            symbolic function
        a_range : tuple
            sweeping range of a
        b_range : tuple
            sweeping range of b
        grid_num : int
            number of steps along a and b
        iteration : int
            number of zooming in
        verbose : bool
            print extra information if True

    Returns:
    --------
        a_best : float
            best fitted a
        b_best : float
            best fitted b
        c_best : float
            best fitted c
        d_best : float
            best fitted d
        r2_best : float
            best r2 (coefficient of determination)

    Example
    -------
    >>> num = 100
    >>> x = torch.linspace(-1,1,steps=num)
    >>> noises = torch.normal(0,1,(num,)) * 0.02
    >>> y = 5.0*torch.sin(3.0*x + 2.0) + 0.7 + noises
    >>> fit_params(x, y, torch.sin)
    r2 is 0.9999727010726929
    (tensor([2.9982, 1.9996, 5.0053, 0.7011]), tensor(1.0000))
    """
    # fit a, b, c, d such that y=c*fun(a*x+b)+d; both x and y are 1D array.
    # sweep a and b, choose the best fitted model
    for _ in range(iteration):
        a_ = torch.linspace(a_range[0], a_range[1], steps=grid_number)
        b_ = torch.linspace(b_range[0], b_range[1], steps=grid_number)
        a_grid, b_grid = torch.meshgrid(a_, b_, indexing="ij")
        post_fun = fun(a_grid[None, :, :] * x[:, None, None] + b_grid[None, :, :])
        x_mean = torch.mean(post_fun, dim=[0], keepdim=True)
        y_mean = torch.mean(y, dim=[0], keepdim=True)
        numerator = torch.sum((post_fun - x_mean) * (y - y_mean)[:, None, None], dim=0) ** 2
        denominator = torch.sum((post_fun - x_mean) ** 2, dim=0) * torch.sum((y - y_mean)[:, None, None] ** 2, dim=0)
        r2 = numerator / (denominator + 1e-4)
        r2 = torch.nan_to_num(r2)

        best_id = torch.argmax(r2)
        a_id, b_id = torch.div(best_id, grid_number, rounding_mode="floor"), best_id % grid_number

        if a_id == 0 or a_id == grid_number - 1 or b_id == 0 or b_id == grid_number - 1:
            if _ == 0 and verbose is True:
                print("Best value at boundary.")
            if a_id == 0:
                a_range = [a_[0], a_[1]]
            if a_id == grid_number - 1:
                a_range = [a_[-2], a_[-1]]
            if b_id == 0:
                b_range = [b_[0], b_[1]]
            if b_id == grid_number - 1:
                b_range = [b_[-2], b_[-1]]

        else:
            a_range = [a_[a_id - 1], a_[a_id + 1]]
            b_range = [b_[b_id - 1], b_[b_id + 1]]

    a_best = a_[a_id]
    b_best = b_[b_id]
    post_fun = fun(a_best * x + b_best)
    r2_best = r2[a_id, b_id]

    if verbose is True:
        print(f"r2 is {r2_best}")
        if r2_best < 0.9:
            print(f"r2 ({r2_best}) is not very high, please double check if you are choosing the correct symbolic function.")

    post_fun = torch.nan_to_num(post_fun)
    reg = LinearRegression().fit(post_fun[:, None].detach().cpu().numpy(), y.detach().cpu().numpy())
    c_best = torch.from_numpy(reg.coef_)[0]
    d_best = torch.from_numpy(np.array(reg.intercept_))
    return torch.stack([a_best, b_best, c_best, d_best]), r2_best


def suggest_symbolic(x, y, a_range=(-10, 10), b_range=(-10, 10), lib=None, topk=5, verbose=True):
    """suggest the symbolic candidates of phi(l,i,j)

    Args:
    -----
        l : int
            layer index
        i : int
            input neuron index
        j : int
            output neuron index
        lib : dic
            library of symbolic bases. If lib = None, the global default library will be used.
        topk : int
            display the top k symbolic functions (according to r2)
        verbose : bool
            If True, more information will be printed.

    Returns:
    --------
        None
    """
    r2s = []

    if lib is None:
        symbolic_lib = SYMBOLIC_LIB
    else:
        symbolic_lib = {}
        for item in lib:
            symbolic_lib[item] = SYMBOLIC_LIB[item]

    for name, fun in symbolic_lib.items():
        _, r2 = fit_params(x, y, fun[0], a_range=a_range, b_range=b_range, verbose=False)
        r2s.append(r2.item())

    sorted_ids = np.argsort(r2s)[::-1][:topk]
    r2s = np.array(r2s)[sorted_ids][:topk]
    topk = np.minimum(topk, len(symbolic_lib))
    if verbose is True:
        print("function", ",", "r2")
        for i in range(topk):
            print(list(symbolic_lib.items())[sorted_ids[i]][0], ",", r2s[i])

    best_name = list(symbolic_lib.items())[sorted_ids[0]][0]
    best_fun = list(symbolic_lib.items())[sorted_ids[0]][1]
    best_r2 = r2s[0]
    return best_name, best_fun, best_r2
