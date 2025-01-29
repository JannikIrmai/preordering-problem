# Algorithms for the preorder problem

This repository contains implementations of the algorithm described in the paper **The Preordering Problem: A Hybrid of Corrleation Clustering and Partial Ordering**.

![image](https://github.com/JannikIrmai/preordering-problem/blob/main/ego-network-preorder-example.png)


## Requirements

The python packages listed in the `requirement.txt` file are required.
In order to solve the ILPs with gurobi a license is required.
Free academic licenses can be obtained [here](https://www.gurobi.com/academia/academic-program-and-licenses/).

## Installation

pip install -r requirements.txt

## Usage

```python
import numpy as np
import matplotlib.pyplot as plt
from preordering_problem.ilp_solver import Preorder
from preordering_problem.drawing import PreorderPlot

# create a value function
values = np.array([
    [ 0,  2,  0,  0,  0, -1],
    [ 3,  0, -2,  0,  0,  4],
    [ 0,  5,  0, -1,  1,  0],
    [ 0,  0, -2,  0,  2, -1],
    [ 0, -3,  0, -2,  0,  3],
    [-3,  0,  0,  0, -1,  0]
])
# create an instance of the preorder problem
preorder = Preorder(values, binary=True)
# solve ILP
preorder.solve()

# plot results
plotter = PreorderPlot(preorder.get_variable_values(), values)
plotter.plot()
plt.show()
```

## Experiments

To reproduce the experiments from the article, run the script from the `experiments` directory:

```
cd experiments
python congress.py
python ego-network.py
```

You may want to modfy the `main` function in the scripts to run only some of the experiments.