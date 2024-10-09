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


## Current benchmark (19/07/2024)

The implementation is not as fast as efficient-kan when using slow version but still allow for significant speedup with respect to pykan.  Based on [this benchmarck](https://github.com/Jerry-Master/KAN-benchmarking)

``
python3 benchmark.py --batch-size 1000 --inp-size 100 --hid-size 1000 --reps 50
``


|                        |      forward  |     backward  |      forward  |     backward  |   num params  |  num trainable params
|------------------------|---------------|---------------|---------------|---------------|---------------|----------------------
|effkan-cpu              |     94.50 ms  |    175.48 ms  |       nan GB  |       nan GB  |      1010000  |               1010000
|effkan-gpu              |     20.82 ms  |     37.40 ms  |      0.36 GB  |      0.37 GB  |      1010000  |               1010000
|mlp-cpu                 |     12.22 ms  |     16.95 ms  |       nan GB  |       nan GB  |      1020001  |               1020001
|mlp-gpu                 |      1.93 ms  |      3.18 ms  |      0.10 GB  |      0.14 GB  |      1020001  |               1020001
|rbf-better_kan-cpu      |    279.77 ms  |    126.55 ms  |       nan GB  |       nan GB  |       928602  |                911002
|rbf-better_kan-gpu      |     45.15 ms  |     28.37 ms  |      1.52 GB  |      0.78 GB  |       928602  |                911002
|splines-better_kan-cpu  |    329.53 ms  |    266.63 ms  |       nan GB  |       nan GB  |       924202  |                911002
|splines-better_kan-gpu  |     61.03 ms  |     63.09 ms  |      1.52 GB  |      0.78 GB  |       924202  |                911002
|cheby-better_kan-cpu    |    231.81 ms  |    106.88 ms  |       nan GB  |       nan GB  |      1118502  |               1113002
|cheby-better_kan-gpu    |     44.15 ms  |     23.81 ms  |      1.52 GB  |      0.78 GB  |      1118502  |               1113002



However, when swicthing to fast version, speed becomes comparable to faster KAN implementation. But this fast version does not allow to use the original regularization neither the plotting and pruning utilities. Comparaison with pykan use model.speed()

``
python3 benchmark.py --batch-size 1000 --inp-size 100 --hid-size 1000 --reps 50 --fast
``

|                        |      forward  |     backward  |      forward  |     backward  |   num params  |  num trainable params
|------------------------|---------------|---------------|---------------|---------------|---------------|----------------------
|effkan-cpu              |     90.73 ms  |    171.64 ms  |       nan GB  |       nan GB  |      1010000  |               1010000
|effkan-gpu              |     20.48 ms  |     36.68 ms  |      0.36 GB  |      0.37 GB  |      1010000  |               1010000
|mlp-cpu                 |     10.27 ms  |     14.00 ms  |       nan GB  |       nan GB  |      1020001  |               1020001
|mlp-gpu                 |      1.94 ms  |      3.16 ms  |      0.10 GB  |      0.14 GB  |      1020001  |               1020001
|rbf-better_kan-cpu      |     27.54 ms  |     28.01 ms  |       nan GB  |       nan GB  |       928602  |                911002
|rbf-better_kan-gpu      |      4.79 ms  |      6.58 ms  |      0.15 GB  |      0.18 GB  |       928602  |                911002
|splines-better_kan-cpu  |     90.05 ms  |    165.65 ms  |       nan GB  |       nan GB  |       924202  |                911002
|splines-better_kan-gpu  |     20.12 ms  |     34.32 ms  |      0.36 GB  |      0.37 GB  |       924202  |                911002
|cheby-better_kan-cpu    |     30.28 ms  |     57.30 ms  |       nan GB  |       nan GB  |      1118502  |               1113002
|cheby-better_kan-gpu    |      5.34 ms  |     11.27 ms  |      0.15 GB  |      0.30 GB  |      1118502  |               1113002
|pykan-cpu               |    377.03 ms  |    450.12 ms  |       nan GB  |       nan GB  |      1633204  |               1414000
|pykan-gpu               |     79.88 ms  |     94.09 ms  |      1.90 GB  |      1.53 GB  |      1633204  |               1414000





Comparing to pykan v1 (with smaller number of parameters)

``
python3 benchmark.py --batch-size 1000 --inp-size 100 --hid-size 100 --reps 10 --method pykan
``


|           |      forward  |     backward  |      forward  |     backward  |   num params  |  num trainable params
|-----------|---------------|---------------|---------------|---------------|---------------|----------------------
|pykan-cpu  |   2092.33 ms  |   2032.41 ms  |       nan GB  |       nan GB  |       222301  |                141501
|pykan-gpu  |   1742.72 ms  |   3726.45 ms  |      1.51 GB  |      0.66 GB  |       222301  |                141501


Comparing to pykan v2 (with smaller number of parameters)

``
python3 benchmark.py --batch-size 1000 --inp-size 100 --hid-size 100 --reps 10 --method pykan
``

|           |      forward  |     backward  |      forward  |     backward  |   num params  |  num trainable params
|-----------|---------------|---------------|---------------|---------------|---------------|----------------------
|pykan-cpu  |   1032.11 ms  |   2014.33 ms  |       nan GB  |       nan GB  |       164404  |                141400
|pykan-gpu  |   1682.30 ms  |   3523.50 ms  |      0.32 GB  |      0.32 GB  |       164404  |                141400
