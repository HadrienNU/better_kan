# An Efficient Implementation of Kolmogorov-Arnold Network

This repository contains an efficient implementation of Kolmogorov-Arnold Network (KAN).
The original implementation of KAN is available [here](https://github.com/KindXiaoming/pykan).

It borrows a lot from [efficient-kan](https://github.com/Blealtan/efficient-kan), but include also plotting, locks and pruning (WIP).  Radial basis function are also implemented and allow for extra speedup.

Contrary to efficient-kan, it use the original L1 regularization of pykan without sacrifying too much too performance.


## Note on RBF

There is some fast implementations of KAN that include RBF, but there are not equivalent to the original paper when there is more than one input. Indeed, most RBF implementation use for node $q$, inputs $x_p$ and centers $c_i$

$$ \sum_i c_{q,i} \phi\left(\sqrt{\sum_p (x_p-c_i)^2 }\right)  $$

whereas the correct formule should be

$$ \sum_p  \sum_i  c_{q,i} \phi \left(  |x_p-c_i|\right)  .$$

The last formula is the one implemented here.


## Current benchmark (20/06/2024)

The splines implementation is not as fast as efficient-kan but still allow for significant speedup with respect to pykan (around 77ms).  Based on [this benchmarck](https://github.com/Jerry-Master/KAN-benchmarking)

|                           |      forward  |     backward  |      forward  |     backward  |   num params  |  num trainable params
|---------------------------|---------------|---------------|---------------|---------------|---------------|----------------------
|effkan-cpu                 |     11.74 ms  |     19.18 ms  |       nan GB  |       nan GB  |         4500  |                  4500
|effkan-gpu                 |      3.41 ms  |      5.77 ms  |      0.07 GB  |      0.07 GB  |         4500  |                  4500
|mlp-cpu                    |      1.17 ms  |      1.59 ms  |       nan GB  |       nan GB  |         6001  |                  6001
|mlp-gpu                    |      0.28 ms  |      0.60 ms  |      0.03 GB  |      0.03 GB  |         6001  |                  6001
|rbf-better_kan-cpu         |      4.21 ms  |      3.74 ms  |       nan GB  |       nan GB  |         5265  |                  4201
|rbf-better_kan-gpu         |      1.48 ms  |      1.32 ms  |      0.04 GB  |      0.04 GB  |         5265  |                  4201
|splines-better_kan-cpu    |     13.51 ms  |     18.50 ms  |       nan GB  |       nan GB  |         4651  |                  4651
|splines-better_kan-gpu     |      3.75 ms  |      5.73 ms  |      0.07 GB  |      0.07 GB  |         4651  |                  4651
