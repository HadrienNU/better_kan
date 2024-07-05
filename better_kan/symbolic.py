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

    Example
    -------
    >>> model = KAN(width=[2,5,1], grid=5, k=3, noise_scale=0.1, seed=0)
    >>> f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
    >>> dataset = create_dataset(f, n_var=2)
    >>> model.train(dataset, opt='LBFGS', steps=50, lamb=0.01);
    >>> model = model.prune()
    >>> model(dataset['train_input'])
    >>> model.suggest_symbolic(0,0,0)
    function , r2
    sin , 0.9994412064552307
    gaussian , 0.9196369051933289
    tanh , 0.8608126044273376
    sigmoid , 0.8578218817710876
    arctan , 0.842217743396759
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


class Symbolic_KANLayer(nn.Module):
    """
    KANLayer class

    Attributes:
    -----------
        in_features: int
            input dimension
        out_features: int
            output dimension
        funs: 2D array of torch functions (or lambda functions)
            symbolic functions (torch)
        funs_name: 2D arry of str
            names of symbolic functions
        funs_sympy: 2D array of sympy functions (or lambda functions)
            symbolic functions (sympy)
        affine: 3D array of floats
            affine transformations of inputs and outputs

    Methods:
    --------
        __init__():
            initialize a Symbolic_KANLayer
        forward():
            forward
        get_subset():
            get subset of the KANLayer (used for pruning)
        fix_symbolic():
            fix an activation function to be symbolic
    """

    def __init__(self, in_features=3, out_features=2, mask=None):
        """
        initialize a Symbolic_KANLayer (activation functions are initialized to be identity functions)

        Args:
        -----
            in_features : int
                input dimension
            out_features : int
                output dimension

        Returns:
        --------
            self

        Example
        -------
        >>> sb = Symbolic_KANLayer(in_features=3, out_features=3)
        >>> len(sb.funs), len(sb.funs[0])
        (3, 3)
        """
        super(Symbolic_KANLayer, self).__init__()
        self.out_features = out_features
        self.in_features = in_features

        if mask is not None:
            raise NotImplementedError()
            self.reduced_in_dim = mask.shape[0]
            torch._assert(mask.shape[1] == self.in_features, "  Mask should be defined for all inputs")

        else:
            self.reduced_in_dim = self.in_features
            mask = torch.eye(self.in_features)

        self.register_buffer("mask", mask)  #  shape: (self.reduced_in_dim, self.in_features)
        self.register_buffer("inv_mask", torch.linalg.pinv(mask))

        # torch
        self.funs = [[lambda x: x for i in range(self.in_features)] for j in range(self.out_features)]
        # name
        self.funs_name = [["" for i in range(self.in_features)] for j in range(self.out_features)]
        # sympy
        self.funs_sympy = [["" for i in range(self.in_features)] for j in range(self.out_features)]

        self.affine = torch.nn.Parameter(torch.zeros(out_features, self.reduced_in_dim, 4))  # parameters for c*f(a*x+b)+d

    def forward(self, x):
        """
        forward

        Args:
        -----
            x : 2D array
                inputs, shape (batch, input dimension)

        Returns:
        --------
            y : 2D array
                outputs, shape (batch, output dimension)
            postacts : 3D array
                activations after activation functions but before summing on nodes

        Example
        -------
        >>> sb = Symbolic_KANLayer(in_features=3, out_features=5)
        >>> x = torch.normal(0,1,size=(100,3))
        >>> y, postacts = sb(x)
        >>> y.shape, postacts.shape
        (torch.Size([100, 5]), torch.Size([100, 5, 3]))
        """

        out_acts = self.activations_eval(x)

        self.min_vals = torch.min(x, dim=0).values
        self.max_vals = torch.max(x, dim=0).values
        self.l1_norm = torch.mean(torch.abs(out_acts), dim=0) / (self.max_vals - self.min_vals)  # out_features x in_features

        output = torch.sum(out_acts, dim=2)

        return output

    def activations_eval(self, x):
        """
        forward

        Args:
        -----
            x : 2D array
                inputs, shape (batch, input dimension)

        Returns:
        --------
            y : 2D array
                outputs, shape (batch, output dimension)
            postacts : 3D array
                activations after activation functions but before summing on nodes

        Example
        -------
        >>> sb = Symbolic_KANLayer(in_features=3, out_features=5)
        >>> x = torch.normal(0,1,size=(100,3))
        >>> y, postacts = sb(x)
        >>> y.shape, postacts.shape
        (torch.Size([100, 5]), torch.Size([100, 5, 3]))
        """

        out_acts = []

        for i in range(self.in_features):  # Since symbolic function are R -> R function, we call it one by one
            out_acts_ = []
            for j in range(self.out_features):
                xij = self.affine[j, i, 2] * self.funs[j][i](self.affine[j, i, 0] * x[:, [i]] + self.affine[j, i, 1]) + self.affine[j, i, 3]
                # postacts_.append(self.mask[j][i] * xij)  # If mask is set
                out_acts_.append(xij)
            out_acts.append(torch.stack(out_acts_))

        out_acts = torch.stack(out_acts)
        out_acts = out_acts.permute(2, 1, 0, 3)[:, :, :, 0]

        return out_acts

    def get_subset(self, in_id, out_id):
        """
        get a smaller Symbolic_KANLayer from a larger Symbolic_KANLayer (used for pruning)

        Args:
        -----
            in_id : list
                id of selected input neurons
            out_id : list
                id of selected output neurons

        Returns:
        --------
            spb : Symbolic_KANLayer

        Example
        -------
        >>> sb_large = Symbolic_KANLayer(in_features=10, out_features=10)
        >>> sb_small = sb_large.get_subset([0,9],[1,2,3])
        >>> sb_small.in_features, sb_small.out_features
        (2, 3)
        """
        sbb = Symbolic_KANLayer(self.in_features, self.out_features)
        sbb.in_features = len(in_id)
        sbb.out_features = len(out_id)
        sbb.funs = [[self.funs[j][i] for i in in_id] for j in out_id]
        sbb.funs_sympy = [[self.funs_sympy[j][i] for i in in_id] for j in out_id]
        sbb.funs_name = [[self.funs_name[j][i] for i in in_id] for j in out_id]
        sbb.affine.data = self.affine.data[out_id][:, in_id]
        return sbb

    def set_from_another_layer(
        self, parent, fun_names, in_id=None, out_id=None, fit_params_bool=True, a_range=(-10, 10), b_range=(-10, 10), verbose=True, random=False, lib=None
    ):
        """
        set a smaller KANLayer from a larger KANLayer (used for pruning)


        Args:
        -----
            newlayer : kan_layer
                An input KANLayer to be set as a subset of this one
            fun_name : array of str
                function names
            in_id : list
                id of selected input neurons from the parent
            out_id : list
                id of selected output neurons from the parent
            fit_params_bool : bool
                obtaining affine parameters through fitting (True) or setting default values (False)
            a_range : tuple
                sweeping range of a
            b_range : tuple
                sweeping range of b
            verbose : bool
                If True, more information is printed.
            random : bool
                initialize affine parameteres randomly or as [1,0,1,0]

        Returns:
        --------
            newlayer : KANLayer
        """
        if in_id is None:
            in_id = torch.arange(parent.in_features)
        if out_id is None:
            out_id = torch.arange(parent.out_features)

        torch._assert(len(in_id) == self.in_features, "Subset size should match layer size")
        torch._assert(len(out_id) == self.out_features, "Subset size should match layer size")

        if not fit_params_bool:
            for i in in_id:
                for j in out_id:
                    self.fix_symbolic(i, j, fun_names[j][i], verbose=verbose, random=random)
            return None
        else:
            x = parent.grid
            y = parent.activations_eval(x)
            r2 = torch.zeros(self.out_features, self.in_features)
            for i in in_id:
                for j in out_id:
                    if fun_names[j][i] == "auto":
                        name, _, r2_auto = suggest_symbolic(x[:, i], y[:, j, i], a_range=a_range, b_range=b_range, lib=lib, verbose=False)
                        if verbose:
                            print(f"({i},{j}): auto symbolic suggest {name} with r2 {r2_auto}")
                    else:
                        name = fun_names[j][i]
                    r2[j, i] = self.fix_symbolic(i, j, name, x[:, i], y[:, j, i], a_range=a_range, b_range=b_range, verbose=verbose)
            return r2

    def fix_symbolic(self, i, j, fun_name, x=None, y=None, random=False, a_range=(-10, 10), b_range=(-10, 10), verbose=True):
        """
        fix an activation function to be symbolic

        Args:
        -----
            i : int
                the id of input neuron
            j : int
                the id of output neuron
            fun_name : str
                the name of the symbolic functions
            x : 1D array
                preactivations
            y : 1D array
                postactivations
            a_range : tuple
                sweeping range of a
            b_range : tuple
                sweeping range of a
            verbose : bool
                print more information if True

        Returns:
        --------
            r2 (coefficient of determination)

        Example 1
        ---------
        >>> # when x & y are not provided. Affine parameters are set to a = 1, b = 0, c = 1, d = 0
        >>> sb = Symbolic_KANLayer(in_features=3, out_features=2)
        >>> sb.fix_symbolic(2,1,'sin')
        >>> print(sb.funs_name)
        >>> print(sb.affine)
        [['', '', ''], ['', '', 'sin']]
        Parameter containing:
        tensor([[0., 0., 0., 0.],
                 [0., 0., 0., 0.],
                 [1., 0., 1., 0.]], requires_grad=True)
        Example 2
        ---------
        >>> # when x & y are provided, fit_params() is called to find the best fit coefficients
        >>> sb = Symbolic_KANLayer(in_features=3, out_features=2)
        >>> batch = 100
        >>> x = torch.linspace(-1,1,steps=batch)
        >>> noises = torch.normal(0,1,(batch,)) * 0.02
        >>> y = 5.0*torch.sin(3.0*x + 2.0) + 0.7 + noises
        >>> sb.fix_symbolic(2,1,'sin',x,y)
        >>> print(sb.funs_name)
        >>> print(sb.affine[1,2,:].data)
        r2 is 0.9999701976776123
        [['', '', ''], ['', '', 'sin']]
        tensor([2.9981, 1.9997, 5.0039, 0.6978])
        """
        if isinstance(fun_name, str):
            fun = SYMBOLIC_LIB[fun_name][0]
            fun_sympy = SYMBOLIC_LIB[fun_name][1]
            self.funs_sympy[j][i] = fun_sympy
            self.funs_name[j][i] = fun_name
            if x is None or y is None:
                # initialize from just fun
                self.funs[j][i] = fun
                if random is False:
                    self.affine.data[j][i] = torch.tensor([1.0, 0.0, 1.0, 0.0])
                else:
                    self.affine.data[j][i] = torch.rand(4) * 2 - 1
                return None
            else:
                # initialize from x & y and fun
                params, r2 = fit_params(x, y, fun, a_range=a_range, b_range=b_range, verbose=verbose)
                self.funs[j][i] = fun
                self.affine.data[j][i] = params
                return r2
        else:
            # if fun_name itself is a function
            fun = fun_name
            fun_sympy = fun_name
            self.funs_sympy[j][i] = fun_sympy
            self.funs_name[j][i] = "anonymous"

            self.funs[j][i] = fun
            if random is False:
                self.affine.data[j][i] = torch.tensor([1.0, 0.0, 1.0, 0.0])
            else:
                self.affine.data[j][i] = torch.rand(4) * 2 - 1
            return None
