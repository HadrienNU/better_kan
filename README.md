# An Efficient Implementation of Kolmogorov-Arnold Network

This repository contains an efficient implementation of Kolmogorov-Arnold Network (KAN).
The original implementation of KAN is available [here](https://github.com/KindXiaoming/pykan).

It borrows a lot from [efficient-kan](https://github.com/Blealtan/efficient-kan), but include also plotting, locks and pruning.  Radial basis function are also implemented and allow for extra speedup.

Contrary to efficient-kan, it use the original L1 regularization of pykan without sacrifying too much too performance.


## Note on RBF

There is some fast implementations of KAN that include RBF, but there are not equivalent to the original paper when there is more than one input. Indeed, most RBF implementation use for node $q$, inputs $x_p$ and centers $c_i$

$$ \sum_i c_{q,i} \phi\left(\sqrt{\sum_p (x_p-c_i)^2 }\right)  $$

whereas the correct formule should be

$$ \sum_p  \sum_i  c_{q,i} \phi \left(  |x_p-c_i|\right)  .$$

The last formula is the one implemented here.


## Current benchmark (24/06/2024)

The splines implementation is not as fast as efficient-kan but still allow for significant speedup with respect to pykan.  Based on [this benchmarck](https://github.com/Jerry-Master/KAN-benchmarking)

``
python3 benchmark.py --batch-size 1000 --inp-size 100 --hid-size 1000 --reps 50
``


|                        |      forward  |     backward  |      forward  |     backward  |   num params  |  num trainable params
|------------------------|---------------|---------------|---------------|---------------|---------------|----------------------
|effkan-cpu              |     94.05 ms  |    177.52 ms  |       nan GB  |       nan GB  |      1010000  |               1010000
|effkan-gpu              |     20.63 ms  |     36.95 ms  |      0.36 GB  |      0.37 GB  |      1010000  |               1010000
|mlp-cpu                 |     10.29 ms  |     13.92 ms  |       nan GB  |       nan GB  |      1020001  |               1020001
|mlp-gpu                 |      1.95 ms  |      3.17 ms  |      0.10 GB  |      0.14 GB  |      1020001  |               1020001
|rbf-better_kan-cpu      |    290.91 ms  |    130.82 ms  |       nan GB  |       nan GB  |      1019801  |               1011001
|rbf-better_kan-gpu      |     46.62 ms  |     29.09 ms  |      1.53 GB  |      0.79 GB  |      1019801  |               1011001
|splines-better_kan-cpu  |    343.72 ms  |    282.10 ms  |       nan GB  |       nan GB  |      1011001  |               1011001
|splines-better_kan-gpu  |     61.39 ms  |     63.29 ms  |      1.53 GB  |      0.79 GB  |      1011001  |               1011001
|cheby-better_kan-cpu    |    245.98 ms  |    113.21 ms  |       nan GB  |       nan GB  |      1213001  |               1213001
|cheby-better_kan-gpu    |     44.12 ms  |     23.91 ms  |      1.53 GB  |      0.79 GB  |      1213001  |               1213001


Comparing to pykan (with smaller number of parameters)

``
python3 benchmark.py --batch-size 1000 --inp-size 100 --hid-size 100 --reps 10 --method pykan
``


|           |      forward  |     backward  |      forward  |     backward  |   num params  |  num trainable params
|-----------|---------------|---------------|---------------|---------------|---------------|----------------------
|pykan-cpu  |   2092.33 ms  |   2032.41 ms  |       nan GB  |       nan GB  |       222301  |                141501
|pykan-gpu  |   1742.72 ms  |   3726.45 ms  |      1.51 GB  |      0.66 GB  |       222301  |                141501


However, when swicthing to fast version, speed becomes comparable to faster KAN implementation. But this fast version does not allow to use the original regularization niether the plotting and pruning utilities.

``
python3 benchmark.py --batch-size 1000 --inp-size 100 --hid-size 1000 --reps 50 --fast_better_kan
``

|                        |      forward  |     backward  |      forward  |     backward  |   num params  |  num trainable params
|------------------------|---------------|---------------|---------------|---------------|---------------|----------------------
|effkan-cpu              |     92.39 ms  |    172.42 ms  |       nan GB  |       nan GB  |      1010000  |               1010000
|effkan-gpu              |     20.54 ms  |     36.70 ms  |      0.36 GB  |      0.37 GB  |      1010000  |               1010000
|mlp-cpu                 |     10.52 ms  |     14.48 ms  |       nan GB  |       nan GB  |      1020001  |               1020001
|mlp-gpu                 |      1.93 ms  |      3.14 ms  |      0.10 GB  |      0.14 GB  |      1020001  |               1020001
|rbf-better_kan-cpu      |     30.39 ms  |     49.31 ms  |       nan GB  |       nan GB  |      1019801  |               1011001
|rbf-better_kan-gpu      |      5.01 ms  |      6.97 ms  |      0.16 GB  |      0.19 GB  |      1019801  |               1011001
|splines-better_kan-cpu  |     90.66 ms  |    169.52 ms  |       nan GB  |       nan GB  |      1011001  |               1011001
|splines-better_kan-gpu  |     20.49 ms  |     34.98 ms  |      0.37 GB  |      0.38 GB  |      1011001  |               1011001
|cheby-better_kan-cpu    |     27.88 ms  |     50.15 ms  |       nan GB  |       nan GB  |      1213001  |               1213001
|cheby-better_kan-gpu    |      5.27 ms  |     11.40 ms  |      0.16 GB  |      0.31 GB  |      1213001  |               1213001
