This repo provides the official implementation of the paper "Optimal Abstractions for Verifying Properties of Kolmogorov-Arnold Networks (KANs)"

# Overview
We present a novel approach for verifying properties of Kolmogorov-Arnold Networks (KANs). Our method creates mathematical "abstractions" by replacing each KAN unit with a piecewise affine (PWA) function. These abstractions enable property verification by encoding the problem as a mixed integer linear program (MILP), determining whether outputs satisfy specified properties when inputs belong to a given set. By combining dynamic programming with
a knapsack optimization across the network, we minimize the total number of pieces while guaranteeing specified error bounds. This approach determines the optimal approximation strategy for each unit while maintaining overall accuracy
requirements.

![KAN Verification Visualization](.\images\full_method.png)

# Setup 
Simply run `pip3 install -r requirements.txt`.

To reproduce the results seen in the paper, use the provided [python script](kan_verification.py) according to the instructions below. For a walkthrough of the code and method used in the paper, refer to the [Jupyter Notebook](kan_verification.ipynb).

# How to reproduce the results shown in the paper
We provide 
### Input Range Experiment
To run the input range experiment for a KAN of size [4,15,2] (shown in section 6.1), use: 
``` shell
python3 kan_verification.py --size '[4,15,2]' --experiment_type input_range --timeout_seconds 4000 --segments_per_curve 10
```
Change the flag `--size` to test KANs of a different shape.

### Bound Tightness Experiment
To run the bound tightness experiment for a KAN of size [4,15,2] (shown in section 6.1), use: 
``` shell
python3 kan_verification.py --size '[4,15,2]' --experiment_type input_range --timeout_seconds 4000 --segments_per_curve 10
```
Change the flag `--size` to test KANs of a different shape.

### MNIST Experiment
To obtain the verification times for a KAN trained on the MNIST dataset (shown in section 6.1), use:
``` shell
python3 kan_verification.py --size '[784,16,10]' --experiment_type single_run --input_bounds '[-5.0, 5.0]' --segments_per_curve 10
```

### CIFAR10 Experiment
To obtain the verification times for a KAN trained on the MNIST dataset (shown in section 6.1), use:
``` shell
python3 kan_verification.py --size '[[3072,16,10]]' --experiment_type single_run --input_bounds '[-5.0, 5.0]' --segments_per_curve 10
```

![Results](.\images\results.png)

