import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pulp import (
    GUROBI_CMD, LpBinary, LpMaximize, LpMinimize, LpProblem, 
    LpStatus, LpVariable, lpSum
)
import scipy as sc
from sklearn.metrics import r2_score
from src.custom_fastkan import FastKAN, FastKANLayer
import src.kan_verification_grapher as kan_verification_grapher
import ssl
import time
from torch import autograd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Optional, Union
import os
import sys
import argparse
import ast

# ============================================================================
# TRAINING KANS
# ============================================================================
# 1. Duffing Oscillator
class LBFGS_duffing_oscillator(torch.optim.LBFGS):
    def __init__(self, params, lr=1, history_size=10, tolerance_grad=1e-32, tolerance_change=1e-32, tolerance_ys=1e-32):
        super().__init__(params, lr=lr, history_size=history_size, tolerance_grad=tolerance_grad, 
                         tolerance_change=tolerance_change)
        self.tolerance_ys = tolerance_ys

def compute_jacobian_duffing_oscillator(model, x):
    def ret_sum(x):
        return model(x)
    jacb = autograd.functional.jacobian(ret_sum, x, create_graph=True)
    jacb = jacb[:, :, :, 0]
    jacb = jacb.sum(dim=-1).unsqueeze(-1) 
    return jacb

def compute_ode_duffing_oscillator(y_pred, x, device):
    def ode(init_cond, d):
        return init_cond[:, 1], init_cond[:, 0] - init_cond[:, 0]**3 - d*init_cond[:, 1]
    grad = ode(y_pred, x[:, 3])
    grad = torch.tensor([[dxdt, dydt] for dxdt, dydt in zip(grad[0], grad[1])], device=device).unsqueeze(-1) 
    return grad

def compute_forward_integrals_duffing_oscillator(dataset, duration, t_eval, device):
    def system_cont(t, x, d):
        dxdt = [x[1], x[0] - x[0]**3 - d*x[1]]
        return dxdt 
    sol_set = []
    for init_cond in dataset.detach().cpu().numpy():
        t, x0, y0, d = init_cond[0], init_cond[1], init_cond[2], init_cond[3]
        sol = sc.integrate.solve_ivp(system_cont, [0, duration], (x0, y0), args=(d,), t_eval=t_eval)
        sol_set.extend(np.array(sol.y.T))
    sol_set = torch.tensor(np.array(sol_set), dtype=torch.float32, device=device)
    return sol_set

# Loss Function
def ode_residuals_duffing_oscillator(model, coords, steps, duration, t_eval, device, alpha=1.0, beta=1.0, gamma=1.0):
    coords = coords.clone().detach().requires_grad_(True)
    y_pred = model(coords) 
    y_gt = compute_forward_integrals_duffing_oscillator(coords[::steps], duration, t_eval.cpu().numpy(), device)
    kan_autograd = compute_jacobian_duffing_oscillator(model, coords)
    kan_ode_grad = compute_ode_duffing_oscillator(y_pred, coords, device)
    init_kan_grad = y_pred[::steps]
    init_ode_grad = coords[::steps][:, 1:3]
    dyn_loss = torch.mean((kan_autograd - kan_ode_grad)**2)
    integral_loss = torch.mean((y_pred - y_gt)**2)
    init_loss = torch.mean((init_kan_grad - init_ode_grad)**2)
    batch_loss = alpha*dyn_loss + beta*integral_loss + beta*init_loss
    return batch_loss

def train_loop_duffing_oscillator(model, dataset, steps, duration, t_eval, device):
    optimizer = LBFGS_duffing_oscillator(model.parameters(), lr=1, history_size=10, tolerance_grad=1e-32, 
                     tolerance_change=1e-32, tolerance_ys=1e-32)
    iter_ = 10
    for i in range(iter_):
        def closure():
            optimizer.zero_grad()
            loss = ode_residuals_duffing_oscillator(model, dataset, steps, duration, t_eval, device, alpha=0.2, beta=0.6, gamma=0.2)
            loss.backward()
            return loss
        optimizer.step(closure)
        if i % 1 == 0:
            current_loss = closure().item()
            print(f"Iteration {i}, Loss: {current_loss}")

# Create the dataset for the model
def train_duffing_oscillator():
    device = "cpu"
    s_size = 50
    x_sample = np.random.uniform(-1, 1, size=s_size)
    y_sample = np.random.uniform(-1, 1, size=s_size)
    d_sample = np.random.uniform(0.1, 0.5, size=s_size)
    x_sample = torch.tensor(x_sample, dtype=torch.float32, device=device)
    y_sample = torch.tensor(y_sample, dtype=torch.float32, device=device)
    d_sample = torch.tensor(d_sample, dtype=torch.float32, device=device)
    duration = 1
    time_interval = 0.2
    steps = int(duration / time_interval)
    print(f"Steps: {steps}")
    t_eval = torch.linspace(0, duration, steps=steps)
    dataset = torch.stack([x_sample, y_sample, d_sample], dim=1).to(device)
    dataset = dataset.repeat_interleave(steps, dim=0)
    t_eval_set = t_eval.T.repeat(s_size).view(-1, 1).to(device)
    dataset = torch.cat([t_eval_set, dataset], dim=1)
    dataset.requires_grad = True
    print(f"Dataset shape: {dataset.shape}")
    test_size = 5
    test_dataset = dataset[(s_size-test_size)*steps:]
    dataset = dataset[:(s_size-test_size)*steps]
    print(f"Train dataset shape: {dataset.shape}, Test dataset shape: {test_dataset.shape}")

    # Base Update and Layernorm must be set to false for verificaton 
    kan_duffing = FastKAN(layers_hidden=[4, 15, 2], grid_min=-2., grid_max=2., num_grids=15, use_base_update = False, use_layernorm=False)
    print("Starting training...")

    train_loop_duffing_oscillator(kan_duffing, dataset, steps, duration, t_eval, device)
    print("\nTest set predictions:")
    kan_verification_grapher.graph_ode_results(test_dataset, kan_duffing, steps, t_eval, duration)
    print("\nTrain set predictions:")
    kan_verification_grapher.graph_ode_results(dataset, kan_duffing, steps, t_eval, duration)

    print("\nComputing error metrics on test set:")
    sol_map = []
    for init_conds in test_dataset[::steps].detach().cpu().numpy():
        sol, _ = kan_verification_grapher.system_eq_dis(init_conds[1:], t_eval.numpy(), duration)
        sol_map.extend(sol)
    pred = kan_duffing(test_dataset).detach().numpy()
    res_score = r2_score(sol_map, pred)
    print(f"Overall R2-Score (test dataset): {res_score}")
    print("\nComputing error metrics on train set:")
    sol_map = []
    for init_conds in dataset[::steps].detach().cpu().numpy():
        sol, _ = kan_verification_grapher.system_eq_dis(init_conds[1:], t_eval.numpy(), duration)
        sol_map.extend(sol)
    pred = kan_duffing(dataset).detach().numpy()
    res_score = r2_score(sol_map, pred)
    print(f"Overall R2-Score (train dataset): {res_score}")
    return kan_duffing

# 2. Dampened Pendulum
def compute_jacobian_dampened_pendulum(model, x):
    def ret_sum(x):
        return model(x)
    jacb = autograd.functional.jacobian(ret_sum, x, create_graph=True)
    jacb = jacb[:, :, :, 0]
    jacb = jacb.sum(dim=-1).unsqueeze(-1)
    return jacb

def compute_ode_dampened_pendulum(y_pred, x, device):
    def ode(init_cond, a, b, c, d):
        return -a*init_cond[:, 0] + b*init_cond[:, 1] + c*init_cond[:, 2], \
               (d - a)*init_cond[:, 0] - 2*b*init_cond[:, 1], \
               d*init_cond[:, 0] + b*init_cond[:, 1] - c*init_cond[:, 2]

    grad = ode(y_pred, x[:, 4], x[:, 5], x[:, 6], x[:, 7])
    grad = torch.tensor([[dxdt, dydt, dzdt] for dxdt, dydt, dzdt in zip(grad[0], grad[1], grad[2])], 
                        device=device).unsqueeze(-1) 
    return grad

def compute_forward_integrals_dampened_pendulum(dataset, duration, t_eval, device):
    def system_cont(t, x, a, b, c, d):
        dxdt = [-a*x[0] + b*x[1] + c*x[2], 
                (d - a)*x[0] - 2*b*x[1], 
                d*x[0] + b*x[1] - c*x[2]]
        return dxdt 
    
    sol_set = []
    for init_cond in dataset.detach().cpu().numpy():
        t, x0, y0, z0, a, b, c, d = init_cond[0], init_cond[1], init_cond[2], init_cond[3], \
                                    init_cond[4], init_cond[5], init_cond[6], init_cond[7]
        sol = sc.integrate.solve_ivp(system_cont, [0, duration], (x0, y0, z0), 
                                     args=(a, b, c, d), t_eval=t_eval)
        sol_set.extend(np.array(sol.y.T))
    
    sol_set = torch.tensor(np.array(sol_set), dtype=torch.float32, device=device)
    return sol_set

def ode_residuals_dampened_pendulum(model, coords, steps, duration, t_eval, device, alpha=1.0, beta=1.0, gamma=1.0):
    coords = coords.clone().detach().requires_grad_(True)
    y_pred = model(coords)
    y_gt = compute_forward_integrals_dampened_pendulum(coords[::steps], duration, t_eval, device)
    kan_autograd = compute_jacobian_dampened_pendulum(model, coords)
    kan_ode_grad = compute_ode_dampened_pendulum(y_pred, coords, device) 
    init_kan_grad = y_pred[::steps]
    init_ode_grad = coords[::steps][:, 1:4]

    dyn_loss = torch.mean((kan_autograd - kan_ode_grad)**2)
    integral_loss = torch.mean((y_pred - y_gt)**2)
    init_loss = torch.mean((init_kan_grad - init_ode_grad)**2)
    
    batch_loss = alpha*dyn_loss + beta*integral_loss + gamma*init_loss
    
    return batch_loss

def train_loop_dampened_pendulum(model,  dataset, steps, duration, t_eval, device, iterations=2):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    for i in range(iterations):
        optimizer.zero_grad()
        loss = ode_residuals_dampened_pendulum(model, dataset, steps, duration, t_eval, device)
        loss.backward()
        optimizer.step()
        
        if i % 1 == 0:
            print(f"Iteration {i}, Loss: {loss.item()}")
    
    return model

def system_eq_dis_dampened_pendulum(cond_input, t_eval, time):
    def system_cont(t, x, a, b, c, d):
        dxdt = [-a*x[0] + b*x[1] + c*x[2], 
                (d - a)*x[0] - 2*b*x[1], 
                d*x[0] + b*x[1] - c*x[2]]
        return dxdt
    
    x0, y0, z0, a, b, c, d = cond_input
    sol = sc.integrate.solve_ivp(system_cont, [0, time], (x0, y0, z0), 
                                args=(a, b, c, d), t_eval=t_eval)
    return sol.y.T, sol.t.T

def train_dampened_pendulum():
    device = "cpu"
    s_size = 30
    x_sample = torch.tensor(np.random.uniform(0, 1, size=s_size), dtype=torch.float32, device=device)
    y_sample = torch.tensor(np.random.uniform(0, 1, size=s_size), dtype=torch.float32, device=device)
    z_sample = torch.tensor(np.random.uniform(0, 1, size=s_size), dtype=torch.float32, device=device)
    a_sample = torch.tensor(np.random.uniform(0, 1, size=s_size), dtype=torch.float32, device=device)
    b_sample = torch.tensor(np.random.uniform(0, 1, size=s_size), dtype=torch.float32, device=device)
    c_sample = torch.tensor(np.random.uniform(0, 1, size=s_size), dtype=torch.float32, device=device)
    d_sample = torch.tensor(np.random.uniform(0, 1, size=s_size), dtype=torch.float32, device=device)
    duration = 1
    time_interval = 0.2
    steps = int(duration / time_interval)
    print(f"Number of steps: {steps}")
    t_eval = torch.linspace(0, duration, steps=int(steps))
    dataset = torch.stack([x_sample, y_sample, z_sample, a_sample, b_sample, c_sample, d_sample], dim=1).to(device)
    dataset = dataset.repeat_interleave(steps, dim=0)
    t_eval_set = t_eval.T.repeat(s_size).view(-1, 1).to(device)

    dataset = torch.cat([t_eval_set, dataset], dim=1)
    dataset.requires_grad = True
    print(f"Dataset shape: {dataset.shape}")
    test_size = 5
    test_dataset = dataset[(s_size-test_size)*steps:]
    dataset = dataset[:(s_size-test_size)*steps]
    print(f"Train dataset shape: {dataset.shape}, Test dataset shape: {test_dataset.shape}")

    kan_dampened_pendulum = FastKAN(
        layers_hidden=[8, 24, 3],
        grid_min=-2.0,
        grid_max=2.0,
        num_grids=15,
        use_base_update=False,
        use_layernorm=False
    )
    kan_dampened_pendulum.to(device)

    print(f"Jacobian shape: {compute_jacobian_dampened_pendulum(kan_dampened_pendulum, dataset).shape}")
    print("Training model...")
    kan_dampened_pendulum = train_loop_dampened_pendulum(kan_dampened_pendulum,  dataset, steps, duration, t_eval, device, iterations=10)

    print("\nTest set predictions:")
    kan_verification_grapher.graph_ode_results(test_dataset, kan_dampened_pendulum, steps, t_eval, duration)
    print("\nTrain set predictions:")
    kan_verification_grapher.graph_ode_results(dataset, kan_dampened_pendulum, steps, t_eval, duration)

    print("\nEvaluating test set performance:")
    sol_map = []
    for init_conds in test_dataset[::steps].detach().cpu().numpy():
        sol, _ = system_eq_dis_dampened_pendulum(init_conds[1:], t_eval, duration)
        sol_map.extend(sol)

    pred = kan_dampened_pendulum(test_dataset).detach().cpu().numpy()
    res_score = r2_score(sol_map, pred)
    print(f"Overall R2-Score (test_dataset): {res_score}")

    print("\nEvaluating train set performance:")
    sol_map = []
    for init_conds in dataset[::steps].detach().cpu().numpy():
        sol, _ = system_eq_dis_dampened_pendulum(init_conds[1:], t_eval, duration)
        sol_map.extend(sol)

    pred = kan_dampened_pendulum(dataset).detach().cpu().numpy()
    res_score = r2_score(sol_map, pred)
    print(f"Overall R2-Score (train_dataset): {res_score}")
    return kan_dampened_pendulum

# 3. MNIST
def train_mnist():
    ssl._create_default_https_context = ssl._create_unverified_context
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    trainset = torchvision.datasets.MNIST(
        root="./data_mnist", train=True, download=True, transform=transform
    )
    valset = torchvision.datasets.MNIST(
        root="./data_mnist", train=False, download=True, transform=transform
    )
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    valloader = DataLoader(valset, batch_size=64, shuffle=False)

    model = FastKAN([28 * 28, 16, 10], use_base_update=False, use_layernorm=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

    criterion = nn.CrossEntropyLoss()
    for epoch in range(10):
        model.train()
        with tqdm(trainloader) as pbar:
            for i, (images, labels) in enumerate(pbar):
                images = images.view(-1, 28 * 28).to(device)
                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, labels.to(device))
                loss.backward()
                optimizer.step()
                accuracy = (output.argmax(dim=1) == labels.to(device)).float().mean()
                pbar.set_postfix(loss=loss.item(), accuracy=accuracy.item(), lr=optimizer.param_groups[0]['lr'])

        model.eval()
        val_loss = 0
        val_accuracy = 0
        with torch.no_grad():
            for images, labels in valloader:
                images = images.view(-1, 28 * 28).to(device)
                output = model(images)
                val_loss += criterion(output, labels.to(device)).item()
                val_accuracy += (
                    (output.argmax(dim=1) == labels.to(device)).float().mean().item()
                )
        val_loss /= len(valloader)
        val_accuracy /= len(valloader)
        scheduler.step()

        print(
            f"Epoch {epoch + 1}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}"
        )
    return model

# 4. CIFAR10
def train_cifar10():
    ssl._create_default_https_context = ssl._create_unverified_context
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    trainset = torchvision.datasets.CIFAR10(
        root="./data_cifar10", train=True, download=True, transform=transform
    )
    valset = torchvision.datasets.CIFAR10(
        root="./data_cifar10", train=False, download=True, transform=transform
    )
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    valloader = DataLoader(valset, batch_size=64, shuffle=False)

    model = FastKAN([3072, 16, 10])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

    criterion = nn.CrossEntropyLoss()
    for epoch in range(10):
        model.train()
        with tqdm(trainloader) as pbar:
            for i, (images, labels) in enumerate(pbar):
                images = images.view(-1, 3072).to(device)
                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, labels.to(device))
                loss.backward()
                optimizer.step()
                accuracy = (output.argmax(dim=1) == labels.to(device)).float().mean()
                pbar.set_postfix(loss=loss.item(), accuracy=accuracy.item(), lr=optimizer.param_groups[0]['lr'])

        model.eval()
        val_loss = 0
        val_accuracy = 0
        with torch.no_grad():
            for images, labels in valloader:
                images = images.view(-1, 3072).to(device)
                output = model(images)
                val_loss += criterion(output, labels.to(device)).item()
                val_accuracy += (
                    (output.argmax(dim=1) == labels.to(device)).float().mean().item()
                )
        val_loss /= len(valloader)
        val_accuracy /= len(valloader)
        scheduler.step()

        print(
            f"Epoch {epoch + 1}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}"
        )
    return model

# ============================================================================
# FITTING LINEAR SEGMENTS
# ============================================================================

def fit_line_through_points(x1, y1, x2, y2):
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return slope, intercept

def find_bspline_segments_given_max_segments(layer, input_index, output_index, max_segments, min_x, max_x):
    """
    Used in both the vanilla fitting method and DP method.
    Given a single B-Spline in the network and a budget of segments,
    use DP to fit them optimally (minimizing the max error).
    """
    # Sample points from the spline (within defined x range)
    n_samples = 25
    x_points = np.linspace(min_x, max_x, n_samples)
    x_tensor, y_tensor = layer.plot_curve(input_index, output_index, num_pts=n_samples)
    x_np = x_tensor.detach().cpu().numpy()
    y_np = y_tensor.detach().cpu().numpy()
    mask = (x_np >= min_x) & (x_np <= max_x)
    x_points = x_np[mask]
    y_points = y_np[mask]
    n = len(x_points)
    
    # Precompute errors for all possible segments
    errors = np.full((n, n), np.inf)
    for start_idx in range(n-1):
        max_len = min(n - start_idx, int(np.round((max_x - min_x) / ((max_x - min_x) / n))))
        x_start = x_points[start_idx]
        y_start = y_points[start_idx]
        for end_idx in range(start_idx + 1, start_idx + max_len):
            if end_idx >= n:
                break
            x_end = x_points[end_idx]
            y_end = y_points[end_idx]
            if abs(x_end - x_start) <= 1e-10:
                continue
            slope = (y_end - y_start) / (x_end - x_start)
            intercept = y_start - slope * x_start
            segment_x = x_points[start_idx:end_idx+1]
            segment_y = y_points[start_idx:end_idx+1]
            predicted_y = slope * segment_x + intercept
            segment_error = np.max(np.abs(predicted_y - segment_y))
            errors[start_idx, end_idx] = segment_error
    
    # Dynamic programming to find optimal segmentation
    dp = np.full((max_segments, n), np.inf)
    back = np.zeros((max_segments, n), dtype=int)
    for j in range(1, n):  # Base Case: one segment from start to j
        dp[0, j] = errors[0, j] 
    # TODO: add in recurrence relation
    for i in range(1, max_segments):
        for j in range(i+1, n):
            for k in range(i, j):
                if dp[i-1, k] == np.inf:
                    continue
                if errors[k, j] == np.inf:
                    continue
                curr_error = max(dp[i-1, k], errors[k, j])
                if curr_error < dp[i, j]:
                    dp[i, j] = curr_error
                    back[i, j] = k
    
    # Reconstruct the segments
    segments = []
    curr_seg = n - 1
    if dp[max_segments-1, curr_seg] == np.inf:
        return [], np.inf
    for i in range(max_segments-1, -1, -1):
        if i > 0:
            prev_seg = back[i, curr_seg]
        else:
            prev_seg = 0
        x1, y1 = x_points[prev_seg], y_points[prev_seg]
        x2, y2 = x_points[curr_seg], y_points[curr_seg]
        slope, intercept = fit_line_through_points(x1, y1, x2, y2)
        segments.insert(0, (x1, x2, slope, intercept))
        curr_seg = prev_seg
        if curr_seg <= 0:
            break
        
    return segments, dp[max_segments-1, n-1]

def fit_kan_vanilla(kan_model, segments_per_curve, min_x, max_x):
    """
    Our baseline fitting method for the KAN model.
    We assign the same number of segments to each curve for approximation.
    Then, use Gurobi to solve for output bounds and track time to get a target to beat.
    """
    all_segments = []
    for layer_idx, layer in enumerate(kan_model.layers):
        layer.cpu()
        input_dim = layer.input_dim
        output_dim = layer.output_dim
        print(f"Fitting Layer {layer_idx} ({input_dim} -> {output_dim})")
        for input_index in range(input_dim):
            for output_index in range(output_dim):
                segments, max_error = find_bspline_segments_given_max_segments(layer, input_index, output_index, segments_per_curve, min_x, max_x)
                all_segments.append({
                    'layer_idx': layer_idx,
                    'input_idx': input_index,
                    'output_idx': output_index,
                    'segments': segments,
                    'max_error': max_error
                })
    print(f"Generated data for {len(all_segments)} connections.")
    return all_segments

def calculate_bspline_lipschitz_constant(layer: 'FastKANLayer', input_idx: int, output_idx: int, num_pts: int = 1000, min_x: float = None, max_x: float = None) -> float:
    if min_x is None:
        min_x = layer.rbf.grid_min
    if max_x is None:
        max_x = layer.rbf.grid_max
    x_tensor, y_tensor = layer.plot_curve(input_idx, output_idx, num_pts=num_pts)
    x_np = x_tensor.detach().cpu().numpy()
    y_np = y_tensor.detach().cpu().numpy()
    mask = (x_np >= min_x) & (x_np <= max_x)
    x_np = x_np[mask]
    y_np = y_np[mask]

    dx = np.diff(x_np)
    dy = np.diff(y_np)
    nonzero_dx = dx != 0
    slopes = np.zeros_like(dx)
    slopes[nonzero_dx] = dy[nonzero_dx] / dx[nonzero_dx]
    lipschitz_constant = np.max(np.abs(slopes))
    return lipschitz_constant

def compute_dp_tables_lipschitz(kan_model, min_x=-5.0, max_x=5.0, max_segments=15):
    """
    Compute the error tables, segments tables, and Lipschitz constants for all B-splines in the KAN model.
    This is used for the DP method later.
    """
    error_tables = {}
    segments_tables = {}
    lipschitz_constants = {}
    
    # Iterate through all layers
    for layer_idx, layer in enumerate(kan_model.layers):
        input_dim = layer.input_dim
        output_dim = layer.output_dim
        print(f"Analyzing Layer {layer_idx}: {input_dim} inputs -> {output_dim} outputs")
        # Iterate through all B-splines in this layer
        for input_idx in range(input_dim):
            for output_idx in range(output_dim):
                spline_key = (layer_idx, input_idx, output_idx)
                layer = kan_model.layers[layer_idx]
                error_table = {}
                segments_table = {}
                for num_segments in range(1, max_segments + 1):
                    segments, error = find_bspline_segments_given_max_segments(layer, input_idx, output_idx, num_segments, min_x, max_x)
                    error_table[num_segments] = error
                    segments_table[num_segments] = segments
                lipschitz_constant = calculate_bspline_lipschitz_constant(layer=layer, input_idx=input_idx, output_idx=output_idx)
                error_tables[spline_key] = error_table
                segments_tables[spline_key] = segments_table
                lipschitz_constants[spline_key] = lipschitz_constant
    
    return error_tables, segments_tables, lipschitz_constants

def get_all_downstream_splines(kan_model, layer_idx, output_idx, visited=None):
    """
    Find all downstream splines in the KAN model starting from a given spline (at position layer_idx, output_idx).
    This is used to compute the Lipschitz constant for a certain spline.
    """
    if visited is None:
        visited = set()
    # Base Case: last layer has no downstream splines
    if layer_idx >= len(kan_model.layers) - 1:
        return set()
    # The output of current spline becomes an input in the next layer
    next_input_idx = output_idx
    next_layer_idx = layer_idx + 1
    next_layer = kan_model.layers[next_layer_idx]
    downstream_splines = set()
    for next_output_idx in range(next_layer.output_dim):
        next_spline = (next_layer_idx, next_input_idx, next_output_idx)
        if next_spline not in visited:
            visited.add(next_spline)
            downstream_splines.add(next_spline)
            further_downstream = get_all_downstream_splines(kan_model, next_layer_idx, next_output_idx, visited)
            downstream_splines.update(further_downstream)
    return downstream_splines

def weight_dp_tables_lipschitz(kan_model, min_x=-5.0, max_x=5.0, max_segments=15):
    """
    Weight the error tables for every B-Spline by the product of Lipschitz constants of all downstream splines.
    (ie. weight the error tables by their overall effect on the network output bound)
    """
    error_tables, segments_tables, lipschitz_constants = compute_dp_tables_lipschitz(kan_model, min_x, max_x, max_segments)
    weighted_error_tables = {}
    # Iterate through all splines in the network
    for layer_idx, layer in enumerate(kan_model.layers):
        input_dim = layer.input_dim
        output_dim = layer.output_dim
        print(f"Weighing Layer {layer_idx}: {input_dim} inputs -> {output_dim} outputs")
        for input_idx in range(input_dim):
            for output_idx in range(output_dim):
                spline_key = (layer_idx, input_idx, output_idx)
                downstream_splines = get_all_downstream_splines(kan_model, layer_idx, output_idx)
                # Weight the error table by computing the product of Lipschitz constants of all downstream splines
                lipschitz_product = 1.0
                for downstream_spline in downstream_splines:
                    if downstream_spline in lipschitz_constants:
                        lipschitz_product *= lipschitz_constants[downstream_spline]
                weighted_table = {}
                for num_segments, error in error_tables[spline_key].items():
                    weighted_table[num_segments] = error * lipschitz_product
                weighted_error_tables[spline_key] = weighted_table
    return weighted_error_tables, segments_tables, lipschitz_constants

def fit_kan_optimally_lipschitz(weighted_error_tables, segments_tables, max_error, max_segments=50):
    """
    Use DP to find the optimal allocation of segments to splines in the KAN model.
    (Use the weighted error tables computed above).
    """
    used_backup_allocation = False
    all_splines = list(weighted_error_tables.keys())
    num_splines = len(all_splines)
    max_total_segments = num_splines * max_segments
    
    # Initialization: dp[i][j] = minimum error achievable when allocating j total segments to the first i splines
    dp = np.full((num_splines + 1, max_total_segments + 1), float('inf'))
    dp[0, 0] = 0  # Base case: 0 error when allocating 0 segments to 0 splines
    backtrack = np.zeros((num_splines + 1, max_total_segments + 1), dtype=int)
    for i in range(1, num_splines + 1):
        spline_key = all_splines[i-1]
        for j in range(max_total_segments + 1):
            # Try different allocations for the current spline (updating allocation if we find a better error)
            for segs in range(min(j + 1, max_segments + 1)):
                if segs not in weighted_error_tables[spline_key]:
                    continue
                spline_error = weighted_error_tables[spline_key][segs]
                prev_error = dp[i-1, j-segs]
                if prev_error != float('inf') and max(prev_error, spline_error) < dp[i, j]:
                    dp[i, j] = max(prev_error, spline_error)
                    backtrack[i, j] = segs
    
    # Backtrack: Find the min total segments to achieve error <= max_error
    optimal_total_segments = None
    achieved_error = float('inf')
    for j in range(max_total_segments + 1):
        if dp[num_splines, j] <= max_error:
            optimal_total_segments = j
            achieved_error = dp[num_splines, j]
            break
    if optimal_total_segments is None:
        min_error_idx = np.argmin(dp[num_splines, :])
        optimal_total_segments = min_error_idx
        achieved_error = dp[num_splines, min_error_idx]
        print(f"Could not find allocation satisfying the max error constraint ({max_error}).")
        print(f"Returning best allocation with error: {achieved_error}")
        used_backup_allocation = True
    optimal_allocation = {}
    actual_segments = {}
    remaining_segments = optimal_total_segments
    for i in range(num_splines, 0, -1):
        spline_key = all_splines[i-1]
        # Get the segment allocation for this spline and store the coordinates of its linear segments
        segs_for_spline = int(backtrack[i, remaining_segments])
        optimal_allocation[spline_key] = segs_for_spline
        actual_segments[spline_key] = segments_tables[spline_key].get(segs_for_spline, [])
        remaining_segments -= segs_for_spline
    total_segments = sum(optimal_allocation.values())
    print(f"  Achieved error: {achieved_error}")
    print(f"  Total segments allocated: {total_segments}")

    return optimal_allocation, actual_segments, total_segments, achieved_error, dp, used_backup_allocation


def fit_kan_optimally_lipschitz_memory_efficient(weighted_error_tables, segments_tables, max_error, max_segments=50):

    
    """
    Memory-efficient version of fit_kan_optimally_lipschitz using space-optimized DP.
    """
    problem_flag = False
    all_splines = list(weighted_error_tables.keys())
    num_splines = len(all_splines)

    if num_splines == 0:
        print("Warning: No splines found to optimize.")
        return {}, {}, 0, 0.0, False

    print(f"Optimizing segment allocation for {num_splines} B-splines...")


    estimated_total = num_splines * 5
    cap = 200000
    max_total_segments = min(estimated_total, cap)
    print(f"  Max_total_segments not provided. Using heuristic default: {max_total_segments}")

    print(f"  Max segments per spline: {max_segments}")

    prev_row = np.full(max_total_segments + 1, np.inf)
    prev_row[0] = 0.0
    backtrack_decisions = {}
    for i in range(1, num_splines + 1):
        spline_key = all_splines[i - 1]
        spline_weighted_errors = weighted_error_tables.get(spline_key, {})
        curr_row = np.full(max_total_segments + 1, np.inf)
        curr_backtrack = np.zeros(max_total_segments + 1, dtype=np.int16)

        if not spline_weighted_errors:
             curr_row = prev_row.copy()
             print(f"Warning: Spline {spline_key} has no weighted error entries. Allocating 0 segments.")

        else:
            for j in range(max_total_segments + 1):
                 min_achieved_error_for_j = np.inf
                 k = 0
                 error_k0 = spline_weighted_errors.get(0, np.inf)
                 prev_error_k0 = prev_row[j]
                 if prev_error_k0 != np.inf:
                      current_max_error = max(prev_error_k0, error_k0)
                      if current_max_error < min_achieved_error_for_j:
                           min_achieved_error_for_j = current_max_error
                           curr_backtrack[j] = 0
                 max_k_for_spline = min(j, max_segments)
                 for k in range(1, max_k_for_spline + 1):
                      spline_error_k = spline_weighted_errors.get(k, None)
                      if spline_error_k is None:
                          continue
                      prev_error = prev_row[j - k]
                      if prev_error != np.inf:
                          current_max_error = max(prev_error, spline_error_k)
                          if current_max_error < min_achieved_error_for_j:
                              min_achieved_error_for_j = current_max_error
                              curr_backtrack[j] = k
                 curr_row[j] = min_achieved_error_for_j

        backtrack_decisions[i] = curr_backtrack.copy()
        prev_row = curr_row
        if i % 100 == 0 or i == num_splines:
            print(f"  DP progress: Processed spline {i}/{num_splines}")

    final_dp_row = prev_row
    optimal_total_segments = -1
    achieved_error = np.inf
    for j in range(max_total_segments + 1):
        if final_dp_row[j] <= max_error:
            optimal_total_segments = j
            achieved_error = final_dp_row[j]
            break

    if optimal_total_segments == -1:
        problem_flag = True
        min_error_idx = np.argmin(final_dp_row)
        if np.isinf(final_dp_row[min_error_idx]):
             print(f"ERROR: Could not find *any* valid segment allocation within {max_total_segments} total segments.")
             return {}, {}, 0, np.inf, True
        else:
             optimal_total_segments = min_error_idx
             achieved_error = final_dp_row[optimal_total_segments]
             print(f"WARNING: Could not meet max_error constraint ({max_error:.4f}).")
             print(f"         Returning best possible allocation with {optimal_total_segments} segments and error {achieved_error:.4f}")

    optimal_allocation = {}
    actual_segments = {}
    remaining_segments = optimal_total_segments
    print(f"\nReconstructing allocation for {optimal_total_segments} total segments...")
    for i in range(num_splines, 0, -1):
        spline_key = all_splines[i - 1]
        if remaining_segments < 0:
             print(f"ERROR: Backtracking resulted in negative remaining segments at spline index {i}. Aborting reconstruction.")
             return optimal_allocation, actual_segments, -1, achieved_error, True
        try:
             segs_for_spline = int(backtrack_decisions[i][remaining_segments])
        except IndexError:
             print(f"ERROR: IndexError during backtracking. i={i}, remaining_segments={remaining_segments}. Max total segments might be too low or DP issue.")
             return optimal_allocation, actual_segments, -1, achieved_error, True
        except KeyError:
              print(f"ERROR: KeyError during backtracking. Spline index {i} not found in backtrack_decisions. DP issue.")
              return optimal_allocation, actual_segments, -1, achieved_error, True


        optimal_allocation[spline_key] = segs_for_spline
        spline_segs_table = segments_tables.get(spline_key, {})
        actual_segments[spline_key] = spline_segs_table.get(segs_for_spline, [])
        remaining_segments -= segs_for_spline


    if remaining_segments != 0:
        print(f"Warning: Backtracking finished with {remaining_segments} remaining segments != 0. Check DP logic.")


    total_segments_used = sum(optimal_allocation.values())

    print(f"\nOptimization Complete:")
    print(f"  Target Max Weighted Error: {max_error:.4f}")
    print(f"  Achieved Max Weighted Error: {achieved_error:.4f}")
    print(f"  Total Segments Used: {total_segments_used} (Constraint: {max_total_segments})")
    if problem_flag:
        print(f"  NOTE: Target error was not met.")

    return optimal_allocation, actual_segments, total_segments_used, achieved_error, None, problem_flag

# ============================================================================
# MILP VERIFICATION
# ============================================================================

class SplineApproximation:
    """
    Class to represent a spline approximation for a single connection in the KAN model.
    """
    def __init__(self, segments: List[Tuple[float, float, float, float]], max_error: float):
        self.segments = sorted(segments, key=lambda s: s[0]) if segments else []
        self.original_max_error = max_error
        self.max_error_mip = max_error if np.isfinite(max_error) else 1e7 # For MIP solver stability

    def is_valid(self):
        return bool(self.segments) and np.isfinite(self.original_max_error)


def prep_segments_data(kan_segments_data: List[dict], input_dim: int, hidden_dim: int, output_dim: int):
    """
    Seperate the segments data by layer to make verification easier.
    """
    layer0_approximations = [[None for _ in range(hidden_dim)] for _ in range(input_dim)]
    layer1_approximations = [[None for _ in range(output_dim)] for _ in range(hidden_dim)]
    processed_indices_l0 = set()
    processed_indices_l1 = set()
    segments_layer0 = [s for s in kan_segments_data if s['layer_idx'] == 0]
    segments_layer1 = [s for s in kan_segments_data if s['layer_idx'] == 1]

    for seg_data in segments_layer0:
        i = seg_data['input_idx']
        h = seg_data['output_idx']
        approx = SplineApproximation(seg_data['segments'], seg_data['max_error'])
        layer0_approximations[i][h] = approx
        processed_indices_l0.add((i, h))

    for seg_data in segments_layer1:
        h = seg_data['input_idx']
        o = seg_data['output_idx']
        approx = SplineApproximation(seg_data['segments'], seg_data['max_error'])
        layer1_approximations[h][o] = approx
        processed_indices_l1.add((h, o))

    return layer0_approximations, layer1_approximations

def MIP_interval_analysis(
    layer0_approximations: List[List[Optional[SplineApproximation]]],
    layer1_approximations: List[List[Optional[SplineApproximation]]],
    x_min_vec: Union[List[float], np.ndarray],
    x_max_vec: Union[List[float], np.ndarray],
    M: float = 1e4,
):
    """
    Set up the MIP problem for the KAN model using the provided layer approximations and input bounds.
    Returns the results of the optimization for each output dimension.
    """
    if (not layer0_approximations or not layer0_approximations[0]) or (not layer1_approximations or not layer1_approximations[0]):
        print("Error: layer_approximations is empty or invalid.")
        return []
    input_dim = len(layer0_approximations)
    hidden_dim = len(layer0_approximations[0])
    output_dim = len(layer1_approximations[0])
    if len(x_min_vec) != input_dim or len(x_max_vec) != input_dim:
        raise ValueError(f"Input bounds vector length mismatch. Expected {input_dim}, got min:{len(x_min_vec)}, max:{len(x_max_vec)}")
    print(f"Setting up MIP for KAN [{input_dim}, {hidden_dim}, {output_dim}]...")
    print(f"Input bounds: {list(zip(x_min_vec, x_max_vec))}")


    # Define Variables
    x_vars = [LpVariable(f"x_{i}", lowBound=x_min_vec[i], upBound=x_max_vec[i]) for i in range(input_dim)]
    layer0_spline_outputs = [[LpVariable(f"l0_spline_{i}_{h}") for h in range(hidden_dim)] for i in range(input_dim)]
    hidden_activations = [LpVariable(f"hidden_act_{h}") for h in range(hidden_dim)]
    layer1_spline_outputs = [[LpVariable(f"l1_spline_{h}_{o}") for o in range(output_dim)] for h in range(hidden_dim)]
    output_vars = [LpVariable(f"output_{o}") for o in range(output_dim)] # Bounds are the goal


    # Binary indicators for segment selection (layer 0 defined first, then layer 1)
    layer0_indicators = []
    for i in range(input_dim):
        indicators_i = []
        for h in range(hidden_dim):
            approx = layer0_approximations[i][h]
            if approx and approx.segments:
                indicators_i.append([LpVariable(f"l0_ind_{i}_{h}_seg_{j}", cat=LpBinary) for j in range(len(approx.segments))])
            else:
                indicators_i.append([]) # No indicators if no segments
        layer0_indicators.append(indicators_i)
    layer1_indicators = []
    for h in range(hidden_dim):
        indicators_h = []
        for o in range(output_dim):
            approx = layer1_approximations[h][o]
            if approx and approx.segments:
                indicators_h.append([LpVariable(f"l1_ind_{h}_{o}_seg_{j}", cat=LpBinary) for j in range(len(approx.segments))])
            else:
                indicators_h.append([])
        layer1_indicators.append(indicators_h)


    # Define Constraints
    base_constraints = []
    # Layer 0 (Input -> Hidden)
    for i in range(input_dim):
        for h in range(hidden_dim):
            approx = layer0_approximations[i][h]
            x_input = x_vars[i] # Input for this spline connection
            spline_output_var = layer0_spline_outputs[i][h]
            indicators = layer0_indicators[i][h]
            if indicators: # 1. One segment must be active
                base_constraints.append(lpSum(indicators) == 1)
            for j, (start_x, end_x, slope, intercept) in enumerate(approx.segments): # 2. Constrain x bounds and output value
                indicator = indicators[j]
                y_approx = slope * x_input + intercept
                base_constraints.append(x_input >= start_x - M * (1 - indicator))
                base_constraints.append(x_input <= end_x + M * (1 - indicator))
                error = approx.max_error_mip
                base_constraints.append(spline_output_var <= y_approx + error + M * (1 - indicator))
                base_constraints.append(spline_output_var >= y_approx - error - M * (1 - indicator))
    # Hidden Layer
    for h in range(hidden_dim):
        incoming_splines = [layer0_spline_outputs[i][h] for i in range(input_dim)]
        base_constraints.append(hidden_activations[h] == lpSum(incoming_splines)) # Value of activation h in hidden layer = sum over incoming splines
    # Layer 1 (Hidden -> Output)
    for h in range(hidden_dim):
        for o in range(output_dim):
            approx = layer1_approximations[h][o]
            hidden_input = hidden_activations[h] # Input for this spline connection
            spline_output_var = layer1_spline_outputs[h][o]
            indicators = layer1_indicators[h][o]
            if indicators: # 1. One segment must be active
                 base_constraints.append(lpSum(indicators) == 1)
            for j, (start_x, end_x, slope, intercept) in enumerate(approx.segments): # 2. Constrain x bounds and output value
                indicator = indicators[j]
                y_approx = slope * hidden_input + intercept
                base_constraints.append(hidden_input >= start_x - M * (1 - indicator))
                base_constraints.append(hidden_input <= end_x + M * (1 - indicator))
                error = approx.max_error_mip
                base_constraints.append(spline_output_var <= y_approx + error + M * (1 - indicator))
                base_constraints.append(spline_output_var >= y_approx - error - M * (1 - indicator))
    # Final Output
    for o in range(output_dim):
        incoming_splines = [layer1_spline_outputs[h][o] for h in range(hidden_dim)]
        base_constraints.append(output_vars[o] == lpSum(incoming_splines))# Value of activation o in output layer = sum over incoming splines
    print(f"Defined {len(base_constraints)} base constraints.")


    # Solve for Min/Max of each Output
    results = []
    solver = GUROBI_CMD(path=None, keepFiles=False, mip=True, msg=True)
    for o in range(output_dim):
        print(f"\n--- Optimizing for Output Dimension {o} ---")
        target_output_var = output_vars[o]
        min_val = None
        max_val = None
    # Minimize
        prob_min = LpProblem(f"KAN_Output_{o}_Min", LpMinimize)
        prob_min += target_output_var # Objective
        for constraint in base_constraints:
            prob_min += constraint
        print(f"Solving for Min Output {o} using Gurobi...")
        prob_min.solve(solver)
        status_min = LpStatus[prob_min.status]
        if status_min == 'Optimal':
            min_val = target_output_var.varValue
            print(f"Min Output {o} found: {min_val:.6f}")
        else:
            print(f"Min Output {o} Optimization failed. Status: {status_min}")
    # Maximize
        prob_max = LpProblem(f"KAN_Output_{o}_Max", LpMaximize)
        prob_max += target_output_var # Objective
        for constraint in base_constraints:
            prob_max += constraint
        print(f"Solving for Max Output {o} using Gurobi...")
        prob_max.solve(solver) 
        status_max = LpStatus[prob_max.status]
        if status_max == 'Optimal':
            max_val = target_output_var.varValue
            print(f"Max Output {o} found: {max_val:.6f}")
        else:
            print(f"Max Output {o} Optimization failed. Status: {status_max}")
        results.append((min_val, max_val))

    return results

def compute_propagated_error(kan_model, all_segments):
    """
    Compute the overall error for the vanilla fitting method.
    This is used to set the target for the DP fitting.
    """
    segment_dict = {}
    for segment_data in all_segments:
        layer_idx = segment_data['layer_idx']
        input_idx = segment_data['input_idx']
        output_idx = segment_data['output_idx']
        spline_key = (layer_idx, input_idx, output_idx)
        segment_dict[spline_key] = segment_data
    
    # Calculate Lipschitz constants for each B-spline, then compute propagated error
    lipschitz_constants = {}
    for layer_idx, layer in enumerate(kan_model.layers):
        for input_idx in range(layer.input_dim):
            for output_idx in range(layer.output_dim):
                spline_key = (layer_idx, input_idx, output_idx)
                lip_const = calculate_bspline_lipschitz_constant(layer=layer, input_idx=input_idx,output_idx=output_idx)
                lipschitz_constants[spline_key] = lip_const
    spline_errors = {}
    for layer_idx, layer in enumerate(kan_model.layers):
        for input_idx in range(layer.input_dim):
            for output_idx in range(layer.output_dim):
                spline_key = (layer_idx, input_idx, output_idx)
                if spline_key not in segment_dict:
                    continue
                local_error = segment_dict[spline_key]['max_error']
                downstream_splines = get_all_downstream_splines(kan_model, layer_idx, output_idx)
                # Compute the product of Lipschitz constants of all downstream splines
                lipschitz_product = 1.0
                for ds_spline in downstream_splines:
                    if ds_spline in lipschitz_constants:
                        lipschitz_product *= lipschitz_constants[ds_spline]
                propagated_error = local_error * lipschitz_product
                spline_errors[spline_key] = propagated_error
    # Compute overall propagated error (max across all splines)
    if spline_errors:
        total_propagated_error = max(spline_errors.values())
    else:
        total_propagated_error = float('inf')
    
    return total_propagated_error, spline_errors, lipschitz_constants

def check_bounds(
    mip_min: Optional[float],
    mip_max: Optional[float],
    sampled_min: float,
    sampled_max: float,
    abs_tol: float = 1e-4,
    rel_tol: float = 1e-3
) -> Tuple[bool, bool, str]:
    message = []
    lower_bound_ok = False
    upper_bound_ok = False

    if mip_min is None:
        message.append("MIP Min bound is None.")
    else:
        # Check if MIP lower bound is less than or equal to sampled lower bound (allowing tolerance)
        # Tolerance = abs_tol + rel_tol * |sampled_min|
        lower_tol = abs_tol + rel_tol * abs(sampled_min)
        if mip_min <= sampled_min + lower_tol:
            lower_bound_ok = True
        else:
            message.append(f"Lower bound failed: MIP_min ({mip_min:.6f}) > Sampled_min ({sampled_min:.6f}) + Tol ({lower_tol:.6f})")

    if mip_max is None:
        message.append("MIP Max bound is None.")
    else:
        # Check if MIP upper bound is greater than or equal to sampled upper bound (allowing tolerance)
        # Tolerance = abs_tol + rel_tol * |sampled_max|
        upper_tol = abs_tol + rel_tol * abs(sampled_max)
        if mip_max >= sampled_max - upper_tol:
            upper_bound_ok = True
        else:
            message.append(f"Upper bound failed: MIP_max ({mip_max:.6f}) < Sampled_max ({sampled_max:.6f}) - Tol ({upper_tol:.6f})")

    return lower_bound_ok, upper_bound_ok, ". ".join(message)

# ============================================================================
# RUNNING EXPERIMENTS
# ============================================================================

def run_kan_input_range_experiment(
    kan_model,
    starting_range=0.05,
    range_increment=0.05,
    max_range=1.0,
    timeout_seconds=600,
    segments_per_curve=5,
    output_root_dir=None,
    validate_bounds=True,
    num_validation_samples=5000,
    min_x=-5.0,
    max_x=5.0,
    seed=42
):
    input_dim = kan_model.layers[0].input_dim
    hidden_dim = kan_model.layers[0].output_dim  
    output_dim = kan_model.layers[-1].output_dim
    
    if output_root_dir is None:
        model_size = f"{input_dim}_{hidden_dim}_{output_dim}"
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root_dir = os.path.join("output", f"{model_size}_range_experiment_{timestamp}")
    
    os.makedirs(output_root_dir, exist_ok=True)
    
    results_data = []
    
    current_range = starting_range
    iteration = 1
    timed_out = False
    constant = 2.5  
    
    lipschitz_max_segments = int(segments_per_curve * 3)  
    
    while current_range <= max_range and not timed_out:
        print(f"\n{'=' * 80}")
        print(f"ITERATION {iteration}: Testing with input range [-{current_range}, {current_range}]")
        print(f"{'=' * 80}")
        
        input_bounds = [(-current_range, current_range)] * input_dim
        
        range_str = f"{current_range:.2f}".replace('.', 'p')  
        iter_output_dir = os.path.join(output_root_dir, f"range_{range_str}")
        os.makedirs(iter_output_dir, exist_ok=True)
        
        iter_start_time = time.time()
        
        try:
            fitting_results = run_kan_fitting_only(
                kan_model=kan_model,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                vanilla_segments_per_curve=segments_per_curve,  
                max_segments_lipschitz=lipschitz_max_segments,
                min_x=min_x,
                max_x=max_x,
                seed=seed
            )
            
            verification_results, tightness_metrics = run_kan_verification_with_fitting(
                kan_model=kan_model,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                input_bounds=input_bounds,
                fitting_results=fitting_results,
                validate_bounds=validate_bounds,
                num_validation_samples=num_validation_samples,
                seed=seed,
                output_dir=iter_output_dir
            )
            
            total_time = time.time() - iter_start_time
            
            if total_time > timeout_seconds:
                print(f"\nTIMEOUT: Iteration took {total_time:.2f} seconds, exceeding limit of {timeout_seconds} seconds")
                timed_out = True
            
            vanilla_fit_time = fitting_results["vanilla"]["fit_time"]
            vanilla_verify_time = verification_results["verification"].get("vanilla_total_time", 0)
            vanilla_total_time = vanilla_fit_time + vanilla_verify_time
            
            lipschitz_fit_time = fitting_results["lipschitz"].get("total_fit_time", 0)
            lipschitz_verify_time = verification_results["verification"].get("lipschitz_total_time", 0)
            lipschitz_total_time = lipschitz_fit_time + lipschitz_verify_time
            
            max_tightness = tightness_metrics.get("max_tightness", 0)
            vanilla_max_tightness = tightness_metrics.get("vanilla_max_tightness", 0)
            lipschitz_max_tightness = tightness_metrics.get("lipschitz_max_tightness", 0)
            
            bound_width = 2 * current_range  
            
            result_entry = {
                "bound_width": bound_width,
                "input_range": current_range,
                "vanilla_fit_time": vanilla_fit_time,
                "vanilla_verify_time": vanilla_verify_time,
                "vanilla_total_time": vanilla_total_time,
                "vanilla_total_segments": fitting_results["vanilla"]["total_segments"],
                "vanilla_max_tightness": vanilla_max_tightness,
                "lipschitz_fit_time": lipschitz_fit_time,
                "lipschitz_verify_time": lipschitz_verify_time,
                "lipschitz_total_time": lipschitz_total_time,
                "lipschitz_total_segments": fitting_results["lipschitz"].get("total_segments", 0),
                "lipschitz_max_tightness": lipschitz_max_tightness,
                "max_tightness": max_tightness,
                "total_time": total_time,
                "timed_out": timed_out,
                "segments_per_curve": segments_per_curve  
            }
            
            results_data.append(result_entry)
            
            print(f"\nIteration {iteration} complete: {total_time:.2f} seconds")
            print(f"  - Input Range: [-{current_range}, {current_range}], Width: {bound_width}")
            print(f"  - Vanilla total time: {vanilla_total_time:.2f} seconds")
            print(f"  - Lipschitz total time: {lipschitz_total_time:.2f} seconds")
            print(f"  - Maximum tightness: {max_tightness:.2f}%")
            
        except Exception as e:
            print(f"ERROR in iteration {iteration}: {str(e)}")
            import traceback
            traceback.print_exc()
            
            result_entry = {
                "bound_width": 2 * current_range,
                "input_range": current_range,
                "error": str(e),
                "timed_out": False,
                "segments_per_curve": segments_per_curve
            }
            results_data.append(result_entry)
        
        current_range += range_increment
        iteration += 1
    
    results_df = pd.DataFrame(results_data)
    
    csv_path = os.path.join(output_root_dir, "experiment_results.csv")
    results_df.to_csv(csv_path, index=False)
    
    kan_verification_grapher.create_input_range_summary_plot(results_df, output_root_dir, segments_per_curve)
    
    return results_df, output_root_dir

def run_kan_fitting_only(
    kan_model, 
    input_dim: int,
    hidden_dim: int, 
    output_dim: int,
    vanilla_segments_per_curve: int = 5,
    max_segments_lipschitz: int = 15,
    min_x: float = -5.0,
    max_x: float = 5.0,
    seed: int = 42,
    memory_efficient: bool = False,
) -> Dict[str, Any]:
    
    print("=" * 80)
    print("KAN SEGMENT FITTING SETTINGS")
    print("=" * 80)
    print(f"Input Dimensions: {input_dim}")
    print(f"Hidden Dimensions: {hidden_dim}")
    print(f"Output Dimensions: {output_dim}")
    print(f"Vanilla Segments Per Curve: {vanilla_segments_per_curve}")
    print(f"Max Segments Lipschitz: {max_segments_lipschitz}")
    print(f"Min X: {min_x}")
    print(f"Max X: {max_x}")
    print(f"Random Seed: {seed}")
    
    print("=" * 80)
    print("RUNNING KAN SEGMENT FITTING")
    print("=" * 80)
    
    results = {
        "vanilla": {},
        "lipschitz": {}
    }
    
    
    print("\n" + "-" * 50)
    print("METHOD 1: VANILLA FITTING (EQUAL SEGMENTS)")
    print("-" * 50)
    
    
    vanilla_start = time.time()
    vanilla_segments = fit_kan_vanilla(kan_model, segments_per_curve=vanilla_segments_per_curve, min_x=min_x, max_x=max_x)
    total_propagated_error, _, _ = compute_propagated_error(kan_model, vanilla_segments)
    max_error_threshold = total_propagated_error / 2
    print("Max Error Threshold Changed to: ", total_propagated_error / 2)
    vanilla_fit_time = time.time() - vanilla_start
    
    
    total_vanilla_segments = sum(len(segment_data['segments']) for segment_data in vanilla_segments)
    max_vanilla_error = max(segment_data['max_error'] for segment_data in vanilla_segments 
                          if np.isfinite(segment_data['max_error']))
    inf_error_count_vanilla = sum(1 for segment_data in vanilla_segments 
                                if not np.isfinite(segment_data['max_error']))
    
    
    results["vanilla"]["fit_time"] = vanilla_fit_time
    results["vanilla"]["total_segments"] = total_vanilla_segments
    results["vanilla"]["max_error"] = max_vanilla_error
    results["vanilla"]["inf_error_count"] = inf_error_count_vanilla
    results["vanilla"]["segments_data"] = vanilla_segments
    
    print(f"Vanilla Fitting Time: {vanilla_fit_time:.4f} seconds")
    print(f"Total Segments: {total_vanilla_segments}")
    print(f"Max Error: {max_vanilla_error:.6f}")
    print(f"Connections with Infinite Error: {inf_error_count_vanilla}")
    
    
    print("\n" + "-" * 50)
    print("METHOD 2: LIPSCHITZ-WEIGHTED FITTING")
    print("-" * 50)
    
    problem = False
    lipschitz_segments = None
    
    
    lipschitz_dp_start = time.time()
    try:
        
        error_tables, segments_tables, lipschitz_constants = compute_dp_tables_lipschitz(
            kan_model, min_x, max_x, max_segments_lipschitz
        )
        lipschitz_dp_time = time.time() - lipschitz_dp_start
        
        
        weighting_start = time.time()
        weighted_error_tables, _, _ = weight_dp_tables_lipschitz(
            kan_model, min_x, max_x, max_segments_lipschitz
        )
        weighting_time = time.time() - weighting_start
        
        
        allocation_start = time.time()
        if memory_efficient:
            optimal_allocation, actual_segments, total_lipschitz_segments, achieved_error, _, used_backup_allocation = fit_kan_optimally_lipschitz_memory_efficient(
                weighted_error_tables, 
                segments_tables,
                max_error_threshold,
                max_segments=max_segments_lipschitz
            )
        else:
            optimal_allocation, actual_segments, total_lipschitz_segments, achieved_error, _, used_backup_allocation = fit_kan_optimally_lipschitz(
                weighted_error_tables, 
                segments_tables,
                max_error_threshold,
                max_segments=max_segments_lipschitz
            )
        allocation_time = time.time() - allocation_start
        
        
        lipschitz_segments = []
        for (layer_idx, input_idx, output_idx), num_segments in optimal_allocation.items():
            segments = actual_segments.get((layer_idx, input_idx, output_idx), [])
            max_error = error_tables.get((layer_idx, input_idx, output_idx), {}).get(num_segments, np.inf)
            lipschitz_segments.append({
                'layer_idx': layer_idx,
                'input_idx': input_idx,
                'output_idx': output_idx,
                'segments': segments,
                'max_error': max_error
            })
        
        
        lipschitz_fit_time = lipschitz_dp_time + weighting_time + allocation_time
        inf_error_count_lipschitz = sum(1 for segment_data in lipschitz_segments 
                                      if not np.isfinite(segment_data['max_error']))
        
        
        print("LIPSCHITZ ERROR: " + str(achieved_error if achieved_error is not None else "N/A"))
        results["lipschitz"]["dp_time"] = lipschitz_dp_time
        results["lipschitz"]["weighting_time"] = weighting_time
        results["lipschitz"]["allocation_time"] = allocation_time
        results["lipschitz"]["total_fit_time"] = lipschitz_fit_time
        results["lipschitz"]["total_segments"] = total_lipschitz_segments
        results["lipschitz"]["achieved_error"] = achieved_error
        results["lipschitz"]["inf_error_count"] = inf_error_count_lipschitz
        results["lipschitz"]["segments_data"] = lipschitz_segments
        
        print(f"Lipschitz DP Computation Time: {lipschitz_dp_time:.4f} seconds")
        print(f"Weighting Time: {weighting_time:.4f} seconds")
        print(f"Allocation Time: {allocation_time:.4f} seconds")
        print(f"Total Lipschitz Fitting Time: {lipschitz_fit_time:.4f} seconds")
        print(f"Total Segments: {total_lipschitz_segments}")
        print(f"Achieved Error: {achieved_error:.6f}")
        print(f"Connections with Infinite Error: {inf_error_count_lipschitz}")
    except Exception as e:
        print(f"ERROR: Lipschitz method failed: {str(e)}")
        results["lipschitz"]["error"] = str(e)
        
        lipschitz_segments = vanilla_segments  
    
    if problem:
        print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!WARNING: LIPSCHITZ error too small, used best possible error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        print("\nLIPSCHITZ error was valid")
    
    return results

def calculate_tightness_metrics(results):
    metrics = {}
    
    if "validation" not in results or "sampled_min" not in results["validation"]:
        return metrics
    
    sampled_min = results["validation"]["sampled_min"]
    sampled_max = results["validation"]["sampled_max"]
    
    if "vanilla_results" in results["verification"]:
        vanilla_results = results["verification"]["vanilla_results"]
        vanilla_tightness = []
        
        for i in range(len(vanilla_results)):
            mip_min, mip_max = vanilla_results[i]
            samp_min, samp_max = sampled_min[i], sampled_max[i]
            
            mip_width = mip_max - mip_min
            sampled_width = samp_max - samp_min
            
            tightness = (sampled_width / mip_width * 100) if mip_width > 0 else 100.0
            vanilla_tightness.append(tightness)
        
        metrics["vanilla_tightness"] = vanilla_tightness
        metrics["vanilla_max_tightness"] = max(vanilla_tightness) if vanilla_tightness else 0
        metrics["vanilla_avg_tightness"] = sum(vanilla_tightness) / len(vanilla_tightness) if vanilla_tightness else 0
    
    if "lipschitz_results" in results["verification"]:
        lipschitz_results = results["verification"]["lipschitz_results"]
        lipschitz_tightness = []
        
        for i in range(len(lipschitz_results)):
            mip_min, mip_max = lipschitz_results[i]
            samp_min, samp_max = sampled_min[i], sampled_max[i]
            
            mip_width = mip_max - mip_min
            sampled_width = samp_max - samp_min
            
            tightness = (sampled_width / mip_width * 100) if mip_width > 0 else 100.0
            lipschitz_tightness.append(tightness)
        
        metrics["lipschitz_tightness"] = lipschitz_tightness
        metrics["lipschitz_max_tightness"] = max(lipschitz_tightness) if lipschitz_tightness else 0
        metrics["lipschitz_avg_tightness"] = sum(lipschitz_tightness) / len(lipschitz_tightness) if lipschitz_tightness else 0
    
    max_tightness = max(
        metrics.get("vanilla_max_tightness", 0),
        metrics.get("lipschitz_max_tightness", 0)
    )
    metrics["max_tightness"] = max_tightness
    
    return metrics

def run_kan_verification_with_fitting(
    kan_model,
    input_dim,
    hidden_dim,
    output_dim,
    input_bounds,
    fitting_results,
    time_limit_seconds=6000,
    validate_bounds=True,
    num_validation_samples=5000,
    abs_tol=1e-3,
    rel_tol=1e-2,
    seed=42,
    output_dir=None
):
    
    if output_dir is None:
        model_size = f"{input_dim}_{hidden_dim}_{output_dim}"
        if input_bounds:
            input_range = f"{input_bounds[0][0]}_{input_bounds[0][1]}"
            input_range = input_range.replace('-', 'neg')  
            input_range = input_range.replace('.', 'p')    
        else:
            input_range = "unknown"
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("output", f"{model_size}_{input_range}_{timestamp}")
    
    
    os.makedirs(output_dir, exist_ok=True)
    
    original_stdout = sys.stdout
    output_md_path = os.path.join(output_dir, "output.md")
    
    class Tee:
        def __init__(self, *files):
            self.files = files
        
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        
        def flush(self):
            for f in self.files:
                f.flush()
    
    
    x_min_vec = [bound[0] for bound in input_bounds]
    x_max_vec = [bound[1] for bound in input_bounds]
    
    
    vanilla_segments = fitting_results["vanilla"]["segments_data"]
    if "error" not in fitting_results["lipschitz"]:
        lipschitz_segments = fitting_results["lipschitz"]["segments_data"]
        lipschitz_successful = True
    else:
        lipschitz_segments = vanilla_segments  
        lipschitz_successful = False
    
    
    results = {
        "vanilla": fitting_results["vanilla"],
        "lipschitz": fitting_results["lipschitz"],
        "verification": {},
        "validation": {}
    }
    
    with open(output_md_path, 'w') as f:
        
        sys.stdout = Tee(original_stdout, f)
        
        print("=" * 80)
        print("VERIFICATION SETTINGS")
        print("=" * 80)
        print(f"Input Bounds: {input_bounds}")
        print(f"Time Limit (seconds): {time_limit_seconds}")
        print(f"Validate Bounds: {validate_bounds}")
        print(f"Number of Validation Samples: {num_validation_samples}")
        print(f"Absolute Tolerance: {abs_tol}")
        print(f"Relative Tolerance: {rel_tol}")
        
        
        print("\n" + "-" * 50)
        print("VERIFICATION COMPARISON")
        print("-" * 50)
        
        lipschitz_all_bounds_valid = True
        vanilla_all_bounds_valid = True
        
        
        if lipschitz_successful:
            
            lipschitz_start_verify = time.time()
            print("\nConverting Lipschitz segments to verification format...")
            layer0_lipschitz, layer1_lipschitz = prep_segments_data(lipschitz_segments, input_dim, hidden_dim, output_dim)
            
            
            print("\nRunning MIP verification for Lipschitz segments...")
            lipschitz_verify_start = time.time()
            lipschitz_results = MIP_interval_analysis(layer0_lipschitz, layer1_lipschitz, x_min_vec, x_max_vec, M=1e4)
            lipschitz_verify_time = time.time() - lipschitz_verify_start
            total_lipschitz_time = time.time() - lipschitz_start_verify
            
            
            results["verification"]["lipschitz_conversion_time"] = lipschitz_verify_start - lipschitz_start_verify
            results["verification"]["lipschitz_mip_time"] = lipschitz_verify_time
            results["verification"]["lipschitz_total_time"] = total_lipschitz_time
            results["verification"]["lipschitz_results"] = lipschitz_results
            
            print(f"Lipschitz Verification Time: {lipschitz_verify_time:.4f} seconds")
            print(f"Lipschitz Total Time (conversion + verification): {total_lipschitz_time:.4f} seconds")
        
        
        vanilla_start_verify = time.time()
        print("\nConverting vanilla segments to verification format...")
        layer0_vanilla, layer1_vanilla = prep_segments_data(vanilla_segments, input_dim, hidden_dim, output_dim)
        
        
        print("\nRunning MIP verification for vanilla segments...")
        vanilla_verify_start = time.time()
        vanilla_results = MIP_interval_analysis(layer0_vanilla, layer1_vanilla, x_min_vec, x_max_vec, M=1e4)
        vanilla_verify_time = time.time() - vanilla_verify_start
        total_vanilla_time = time.time() - vanilla_start_verify
        
        
        results["verification"]["vanilla_conversion_time"] = vanilla_verify_start - vanilla_start_verify
        results["verification"]["vanilla_mip_time"] = vanilla_verify_time
        results["verification"]["vanilla_total_time"] = total_vanilla_time
        results["verification"]["vanilla_results"] = vanilla_results
        
        print(f"Vanilla Verification Time: {vanilla_verify_time:.4f} seconds")
        print(f"Vanilla Total Time (conversion + verification): {total_vanilla_time:.4f} seconds")
        
        
        if validate_bounds:
            print("\n" + "-" * 50)
            print("VALIDATING MIP BOUNDS")
            print("-" * 50)
            
            
            print(f"\nGenerating {num_validation_samples} validation samples...")
            validation_start = time.time()
            
            
            np.random.seed(seed)
            x_test_np = np.random.uniform(
                low=x_min_vec,
                high=x_max_vec,
                size=(num_validation_samples, input_dim)
            )
            x_test = torch.tensor(x_test_np, dtype=torch.float32)
            
            
            print("Running KAN model on validation samples...")
            with torch.no_grad():
                kan_model.eval()
                y_test = kan_model(x_test).detach().cpu().numpy()
            
            
            sampled_min = np.min(y_test, axis=0)
            sampled_max = np.max(y_test, axis=0)
            
            
            results["validation"]["x_samples"] = x_test_np
            results["validation"]["y_samples"] = y_test
            results["validation"]["sampled_min"] = sampled_min
            results["validation"]["sampled_max"] = sampled_max
            
            
            print("\nValidating vanilla MIP bounds...")
            vanilla_bound_checks = []
            vanilla_all_bounds_valid = True
            
            for o in range(output_dim):
                mip_min, mip_max = vanilla_results[o]
                samp_min, samp_max = sampled_min[o], sampled_max[o]
                
                print(f"Output {o}:")
                print(f"  MIP bounds:    [{mip_min:.6f}, {mip_max:.6f}]")
                print(f"  Sampled bounds: [{samp_min:.6f}, {samp_max:.6f}]")
                
                lower_ok, upper_ok, message = check_bounds(
                    mip_min, mip_max, samp_min, samp_max, abs_tol, rel_tol
                )
                
                bound_valid = lower_ok and upper_ok
                vanilla_all_bounds_valid = vanilla_all_bounds_valid and bound_valid
                vanilla_bound_checks.append({
                    "output_dim": o,
                    "mip_bounds": (mip_min, mip_max),
                    "sampled_bounds": (samp_min, samp_max),
                    "lower_valid": lower_ok,
                    "upper_valid": upper_ok,
                    "message": message,
                    "valid": bound_valid
                })
                
                if bound_valid:
                    print(f"  ✓ Bounds valid")
                else:
                    print(f"  ✗ Bounds invalid: {message}")
            
            results["validation"]["vanilla_bound_checks"] = vanilla_bound_checks
            results["validation"]["vanilla_all_bounds_valid"] = vanilla_all_bounds_valid
            
            if vanilla_all_bounds_valid:
                print("\n✓ All vanilla MIP bounds are valid")
            else:
                print("\n✗ Some vanilla MIP bounds are invalid")
            
            
            if lipschitz_successful:
                print("\nValidating Lipschitz MIP bounds...")
                lipschitz_bound_checks = []
                lipschitz_all_bounds_valid = True
                
                for o in range(output_dim):
                    mip_min, mip_max = lipschitz_results[o]
                    samp_min, samp_max = sampled_min[o], sampled_max[o]
                    
                    print(f"Output {o}:")
                    print(f"  MIP bounds:    [{mip_min:.6f}, {mip_max:.6f}]")
                    print(f"  Sampled bounds: [{samp_min:.6f}, {samp_max:.6f}]")
                    
                    lower_ok, upper_ok, message = check_bounds(
                        mip_min, mip_max, samp_min, samp_max, abs_tol, rel_tol
                    )
                    
                    bound_valid = lower_ok and upper_ok
                    lipschitz_all_bounds_valid = lipschitz_all_bounds_valid and bound_valid
                    lipschitz_bound_checks.append({
                        "output_dim": o,
                        "mip_bounds": (mip_min, mip_max),
                        "sampled_bounds": (samp_min, samp_max),
                        "lower_valid": lower_ok,
                        "upper_valid": upper_ok,
                        "message": message,
                        "valid": bound_valid
                    })
                    
                    if bound_valid:
                        print(f"  ✓ Bounds valid")
                    else:
                        print(f"  ✗ Bounds invalid: {message}")
                
                results["validation"]["lipschitz_bound_checks"] = lipschitz_bound_checks
                results["validation"]["lipschitz_all_bounds_valid"] = lipschitz_all_bounds_valid
                
                if lipschitz_all_bounds_valid:
                    print("\n✓ All Lipschitz MIP bounds are valid")
                else:
                    print("\n✗ Some Lipschitz MIP bounds are invalid")
        
        if not (lipschitz_all_bounds_valid and vanilla_all_bounds_valid):
            print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!WARNING: VERIFICATION FAILED FOR ONE OR BOTH METHODS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        else:  
            print("\nVERIFICATION PASSED FOR BOTH METHODS")
    
    
    sys.stdout = original_stdout
    
    
    kan_verification_grapher.visualize_comparison_results(results, output_dir)
    kan_verification_grapher.detailed_metrics_visualization(results, output_dir)
    if validate_bounds and "create_relative_width_ratio_graph" in globals():
        kan_verification_grapher.create_relative_width_ratio_graph(results, output_dir)
    
    print(f"Results saved to {output_dir}")
    print(f"  - Console output: {output_md_path}")
    print(f"  - Comparison chart: {os.path.join(output_dir, 'output.png')}")
    print(f"  - Detailed metrics: {os.path.join(output_dir, 'detailed_metrics.png')}")
    
    
    tightness_metrics = calculate_tightness_metrics(results) if validate_bounds else {}
    
    return results, tightness_metrics

def run_kan_bound_tightness_experiment(
    kan_model,
    input_bounds,
    timeout_seconds,
    starting_segments=3,
    segment_increment=5,
    max_segments=10,
    output_root_dir=None,
    validate_bounds=True,
    num_validation_samples=5000,
    min_x=-5.0,
    max_x=5.0,
    seed=42
):
    input_dim = kan_model.layers[0].input_dim
    hidden_dim = kan_model.layers[0].output_dim  
    output_dim = kan_model.layers[-1].output_dim
    
    if output_root_dir is None:
        model_size = f"{input_dim}_{hidden_dim}_{output_dim}"
        if input_bounds:
            input_range = f"{input_bounds[0][0]}_{input_bounds[0][1]}"
            input_range = input_range.replace('-', 'neg')  
            input_range = input_range.replace('.', 'p')    
        else:
            input_range = "unknown"
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root_dir = os.path.join("output", f"{model_size}_{input_range}_{timestamp}")
    
    os.makedirs(output_root_dir, exist_ok=True)
    
    results_data = []
    
    segments = starting_segments
    iteration = 3
    timed_out = False
    constant = 2.5
    while segments <= max_segments and not timed_out:
        print(f"\n{'=' * 80}")
        print(f"ITERATION {iteration}: Testing with {segments} segments")
        print(f"{'=' * 80}")
        
        iter_output_dir = os.path.join(output_root_dir, f"segments_{segments}")
        os.makedirs(iter_output_dir, exist_ok=True)
        
        iter_start_time = time.time()
        
        try:
            lipschitz_max_segments = int(segments * 3 * (iteration + 0.4)**0.185)
            constant = constant * 2 * (iteration + 0.4)**0.075
            
            fitting_results = run_kan_fitting_only(
                kan_model=kan_model,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                vanilla_segments_per_curve=segments,
                max_segments_lipschitz=lipschitz_max_segments,
                min_x=min_x,
                max_x=max_x,
                seed=seed,
            )
            
            verification_results, tightness_metrics = run_kan_verification_with_fitting(
                kan_model=kan_model,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                input_bounds=input_bounds,
                fitting_results=fitting_results,
                validate_bounds=validate_bounds,
                num_validation_samples=num_validation_samples,
                seed=seed,
                output_dir=iter_output_dir
            )
            
            total_time = time.time() - iter_start_time
            
            if total_time > timeout_seconds:
                print(f"\nTIMEOUT: Iteration took {total_time:.2f} seconds, exceeding limit of {timeout_seconds} seconds")
                timed_out = True
            
            vanilla_fit_time = fitting_results["vanilla"]["fit_time"]
            vanilla_verify_time = verification_results["verification"].get("vanilla_total_time", 0)
            vanilla_total_time = vanilla_fit_time + vanilla_verify_time
            
            lipschitz_fit_time = fitting_results["lipschitz"].get("total_fit_time", 0)
            lipschitz_verify_time = verification_results["verification"].get("lipschitz_total_time", 0)
            lipschitz_total_time = lipschitz_fit_time + lipschitz_verify_time
            
            max_tightness = tightness_metrics.get("max_tightness", 0)
            vanilla_max_tightness = tightness_metrics.get("vanilla_max_tightness", 0)
            lipschitz_max_tightness = tightness_metrics.get("lipschitz_max_tightness", 0)
            
            result_entry = {
                "segments": segments,
                "vanilla_fit_time": vanilla_fit_time,
                "vanilla_verify_time": vanilla_verify_time,
                "vanilla_total_time": vanilla_total_time,
                "vanilla_total_segments": fitting_results["vanilla"]["total_segments"],
                "vanilla_max_tightness": vanilla_max_tightness,
                "lipschitz_fit_time": lipschitz_fit_time,
                "lipschitz_verify_time": lipschitz_verify_time,
                "lipschitz_total_time": lipschitz_total_time,
                "lipschitz_total_segments": fitting_results["lipschitz"].get("total_segments", 0),
                "lipschitz_max_tightness": lipschitz_max_tightness,
                "max_tightness": max_tightness,
                "total_time": total_time,
                "timed_out": timed_out
            }
            
            results_data.append(result_entry)
            
            print(f"\nIteration {iteration} complete: {total_time:.2f} seconds")
            print(f"  - Vanilla total time: {vanilla_total_time:.2f} seconds")
            print(f"  - Lipschitz total time: {lipschitz_total_time:.2f} seconds")
            print(f"  - Maximum tightness: {max_tightness:.2f}%")
            
        except Exception as e:
            print(f"ERROR in iteration {iteration}: {str(e)}")
            import traceback
            traceback.print_exc()
            
            result_entry = {
                "segments": segments,
                "error": str(e),
                "timed_out": False
            }
            results_data.append(result_entry)
        
        segments += segment_increment
        iteration += 1
    
    results_df = pd.DataFrame(results_data)
    
    csv_path = os.path.join(output_root_dir, "experiment_results.csv")
    results_df.to_csv(csv_path, index=False)
    
    kan_verification_grapher.create_experiment_summary_plot(results_df, output_root_dir)
    
    return results_df, output_root_dir
def main(args):
    size = args.size
    experiment_type = args.experiment_type
    input_bounds = args.input_bounds
    segments_per_curve = args.segments_per_curve
    timeout_seconds = args.timeout_seconds
    if experiment_type == "input_range" and segments_per_curve is None:
        raise ValueError("--segments_per_curve is required for experiment_type 'input_range'")
    if experiment_type == "bound_tightness" and input_bounds is None:
        raise ValueError("--input_bounds is required for experiment_type 'bound_tightness'")
    if experiment_type == "single_run":
        if segments_per_curve is None:
            raise ValueError("--segments_per_curve is required for experiment_type 'single_run'")
        if input_bounds is None:
            raise ValueError("--input_bounds is required for experiment_type 'single_run'")

    kan = None
    if not isinstance(size, list):
        try:
            size = ast.literal_eval(size)
            if not isinstance(size, list):
                raise ValueError
        except (ValueError, SyntaxError):
            print(f"Error: --size argument must be a list (e.g., '[4,15,2]'). Received: {size}")
            return

    if input_bounds and not isinstance(input_bounds, list):
        try:
            input_bounds = ast.literal_eval(input_bounds)
            if not (isinstance(input_bounds, list) and len(input_bounds) == 2 and all(isinstance(x, (int, float)) for x in input_bounds)):
                raise ValueError
        except (ValueError, SyntaxError):
            print(f"Error: --input_bounds argument must be a list of two floats (e.g., '[-1.0, 1.0]'). Received: {input_bounds}")
            return

    if size == [4,15,2]:
        print(f"Training KAN of size: {size} on the Duffing Oscillator ODE. \nThen running experiment: {experiment_type}")
        kan = train_duffing_oscillator()
    elif size == [8,24,3]:
        print(f"Training KAN of size: {size} on the Dampened Pendulum ODE. \nThen running experiment: {experiment_type}")
        kan = train_dampened_pendulum()
    elif size == [784, 16, 10]:
        print(f"Training KAN of size: {size} on the MNIST dataset. \nThen running experiment: {experiment_type}")
        kan = train_mnist()
    elif size == [3072, 16, 10]:
        print(f"Training KAN of size: {size} on the CIFAR10 dataset. \nThen running experiment: {experiment_type}")
        kan = train_cifar10()
    else:
        print(f"Misc. size chosen, randomly instantiating KAN of size: {size}\nThen running experiment: {experiment_type}")
        kan = FastKAN(layers_hidden=size, grid_min=-2., grid_max=2., num_grids=15, use_base_update = False, use_layernorm=False)

    if kan is None:
        print("Error: KAN model was not initialized.")
        return
    if not kan.layers:
        print("Error: KAN model layers are not properly initialized for dimension access.")
        if hasattr(kan, 'layers') and not kan.layers and size:
             kan.layers.append(type('Layer', (), {'input_dim': size[0] if size else 1, 'output_dim': size[-1] if len(size)>1 else 10})())
        else:
            return

    # Convert input_bounds to the required format: [(-1.0, 1.0)] * input_dim
    if input_bounds is not None:
        input_dim = kan.layers[0].input_dim
        input_bounds = [(input_bounds[0], input_bounds[1])] * input_dim
        print(f"Converted input_bounds to format: {input_bounds}")

    if experiment_type == "input_range":
        results_df, output_dir = run_kan_input_range_experiment(
            kan_model=kan,
            starting_range=0.05,
            range_increment=0.1,
            max_range=10.0,
            timeout_seconds=timeout_seconds, # Usually timeout_seconds is ~4000 for paper tests
            segments_per_curve=segments_per_curve,
            validate_bounds=True,
            num_validation_samples=5000
        )
        print(f"\nExperiment complete!")
        print(f"Results saved to: {output_dir}")
        print(f"Results overview:")

    elif experiment_type == "bound_tightness":
        results_df, output_dir = run_kan_bound_tightness_experiment(
            kan_model=kan,
            input_bounds=input_bounds,
            timeout_seconds=timeout_seconds,
            starting_segments=3,
            segment_increment=3,
            max_segments=30,
            validate_bounds=True,
            num_validation_samples=5000
        )

        print(f"\nExperiment complete!")
        print(f"Results saved to: {output_dir}")
        print(f"Results overview:")
        print(results_df[["segments", "vanilla_total_time", "vanilla_max_tightness",
                            "lipschitz_total_time", "lipschitz_max_tightness", "timed_out"]].to_string(index=False))

    elif experiment_type == "single_run": #
        print("Running verification once using Vanilla fitting method and once using Dynamic Programming fitting method.")
        if not kan.layers:
            print("Error: KAN model layers are not defined for single_run. Cannot determine dimensions.")
            return
        input_dim = kan.layers[0].input_dim
        hidden_dim = kan.layers[-1].input_dim
        output_dim = kan.layers[-1].output_dim

        if size == [784, 16, 10] or size == [3072, 16, 10]:
            print("Running with memory efficient fitting")
            fitting_results = run_kan_fitting_only(
                kan_model=kan,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                vanilla_segments_per_curve=segments_per_curve,
                max_segments_lipschitz=segments_per_curve*3,
                min_x=-5.0,
                max_x=5.0,
                memory_efficient=True
            )
        else:
            fitting_results = run_kan_fitting_only(
                kan_model=kan,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                vanilla_segments_per_curve=segments_per_curve,
                max_segments_lipschitz=segments_per_curve*3,
                min_x=-5.0,
                max_x=5.0,
                memory_efficient=True
            )

        run_kan_verification_with_fitting(
            kan_model=kan,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            input_bounds=input_bounds,
            fitting_results=fitting_results,
            output_dir=None
        )
    else:
        print(f"Error: Unknown experiment_type '{experiment_type}'. "
              "Choose from 'input_range', 'bound_tightness', 'single_run'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run KAN experiments.")

    parser.add_argument("--size", type=str, required=True,
                        help="List of integers defining KAN layer sizes (e.g., '[4,15,2]'). String format.")
    parser.add_argument("--experiment_type", type=str, required=True,
                        choices=["input_range", "bound_tightness", "single_run"],
                        help="Type of experiment to run.")
    parser.add_argument("--timeout_seconds", type=int, default=None,
                        help="Timeout in seconds for some experiments.")
    parser.add_argument("--input_bounds", type=str, default=None,
                        help="List of two floats for input bounds (e.g., '[-1.0, 1.0]'). String format Required for 'bound_tightness' and 'single_run'.")
    parser.add_argument("--segments_per_curve", type=int, default=None,
                        help="Number of segments per curve. Required for 'input_range' and 'single_run' experiment.")

    parsed_args = parser.parse_args()

    processed_args = {}
    processed_args['experiment_type'] = parsed_args.experiment_type
    processed_args['timeout_seconds'] = parsed_args.timeout_seconds

    try:
        evaluated_size = ast.literal_eval(parsed_args.size)
        if not isinstance(evaluated_size, list) or not all(isinstance(x, int) for x in evaluated_size):
            raise ValueError("Size must be a list of integers.")
        processed_args['size'] = evaluated_size
    except (ValueError, SyntaxError) as e:
        parser.error(f"Invalid format for --size: {parsed_args.size}. Expected a list of integers (e.g., '[4,15,2]'). Error: {e}")

    if parsed_args.input_bounds:
        try:
            evaluated_bounds = ast.literal_eval(parsed_args.input_bounds)
            if not (isinstance(evaluated_bounds, list) and len(evaluated_bounds) == 2 and all(isinstance(x, (float, int)) for x in evaluated_bounds)):
                raise ValueError("Input bounds must be a list of two numbers.")
            processed_args['input_bounds'] = [float(x) for x in evaluated_bounds] # Ensure float
        except (ValueError, SyntaxError) as e:
            parser.error(f"Invalid format for --input_bounds: {parsed_args.input_bounds}. Expected a list of two floats (e.g., '[-1.0,1.0]'). Error: {e}")
    else:
        processed_args['input_bounds'] = None

    processed_args['segments_per_curve'] = parsed_args.segments_per_curve

    args_namespace = argparse.Namespace(**processed_args)

    main(args_namespace)