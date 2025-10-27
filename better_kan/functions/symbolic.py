import torch
import torch.nn as nn

from .base import BasisFunction


class SymbolicFunction(BasisFunction):
    """

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

    def set_from_another_layer(self, parent, fun_names, in_id=None, out_id=None, fit_params_bool=True, a_range=(-10, 10), b_range=(-10, 10), verbose=True, random=False, lib=None):
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
